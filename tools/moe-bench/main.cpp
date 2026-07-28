// MoE offload benchmark tool.
// Runs prefill + decode loops and prints the MVP profiling summary.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include "common.h"
#include "speculative.h"

#include "moe-offload/runtime.h"
#include "moe-offload/slot_pool.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <psapi.h>
#else
#  include <sys/resource.h>
#endif

#ifndef LLAMA_MOE_OFFLOAD
#  error "llama-moe-bench requires a build configured with LLAMA_MOE_OFFLOAD=ON"
#endif

struct bench_params {
    std::string model;
    std::string prompt;
    int n_prompt = 1024;
    int n_gen = 256;
    int n_repeat = 3;
    int moe_cache_mb = 8000;
    std::string moe_predictor = "eamc";
    std::string moe_eamc_path;
    std::string moe_profile_csv;
    std::string moe_profile_summary;
    std::string token_trace_path;
    int n_gpu_layers = 99;
    int n_ctx = 4096;
    int n_ubatch = 0;
    bool moe_reset_cache_between_repeats = false;
    bool moe_warm_cache = false;
    bool moe_hot_start = false;
    std::string spec_type = "none";
    int spec_draft_n_max = 3;
    int spec_draft_n_min = 0;
    float spec_draft_p_min = 0.0f;
    int spec_draft_ngl = -1;
    ggml_type spec_draft_type_k = GGML_TYPE_F16;
    ggml_type spec_draft_type_v = GGML_TYPE_F16;
};

static bool parse_args(int argc, char ** argv, bench_params & p) {
    bool ok = true;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value_for = [&](const char * name, std::string & value) -> bool {
            const std::string key(name);
            if (arg == key && i + 1 < argc) {
                value = argv[++i];
                return true;
            }
            if (arg.rfind(key + "=", 0) == 0) {
                value = arg.substr(key.size() + 1);
                return true;
            }
            return false;
        };
        auto int_for = [&](const char * name, int & value) -> bool {
            std::string text;
            if (!value_for(name, text)) {
                return false;
            }
            value = std::atoi(text.c_str());
            return true;
        };
        auto layers_for = [&](const char * name, int & value) -> bool {
            std::string text;
            if (!value_for(name, text)) {
                return false;
            }
            if (text == "auto") {
                value = -1;
            } else if (text == "all") {
                value = -2;
            } else {
                value = std::atoi(text.c_str());
            }
            return true;
        };
        auto float_for = [&](const char * name, float & value) -> bool {
            std::string text;
            if (!value_for(name, text)) {
                return false;
            }
            value = std::atof(text.c_str());
            return true;
        };
        auto type_for = [&](const char * name, ggml_type & value) -> bool {
            std::string text;
            if (!value_for(name, text)) {
                return false;
            }
            for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
                if (text == ggml_type_name((ggml_type) t)) {
                    value = (ggml_type) t;
                    return true;
                }
            }
            fprintf(stderr, "unknown ggml type for %s: %s\n", name, text.c_str());
            ok = false;
            return true;
        };

        if (value_for("--model", p.model)) {}
        else if (int_for("--pp", p.n_prompt)) {}
        else if (int_for("--tg", p.n_gen)) {}
        else if (int_for("--repeat", p.n_repeat)) {}
        else if (int_for("--moe-cache-vram-mb", p.moe_cache_mb)) {}
        else if (value_for("--moe-predictor", p.moe_predictor)) {}
        else if (value_for("--moe-eamc-path", p.moe_eamc_path)) {}
        else if (value_for("--moe-profile-csv", p.moe_profile_csv)) {}
        else if (value_for("--moe-profile-summary", p.moe_profile_summary)) {}
        else if (value_for("--token-trace", p.token_trace_path)) {}
        else if (value_for("--output-token-trace", p.token_trace_path)) {}
        else if (layers_for("-ngl", p.n_gpu_layers)) {}
        else if (layers_for("--gpu-layers", p.n_gpu_layers)) {}
        else if (layers_for("--n-gpu-layers", p.n_gpu_layers)) {}
        else if (int_for("-c", p.n_ctx)) {}
        else if (int_for("-ub", p.n_ubatch)) {}
        else if (int_for("--ubatch", p.n_ubatch)) {}
        else if (int_for("--ubatch-size", p.n_ubatch)) {}
        else if (value_for("--spec-type", p.spec_type)) {}
        else if (int_for("--spec-draft-n-max", p.spec_draft_n_max)) {}
        else if (int_for("--spec-draft-n-min", p.spec_draft_n_min)) {}
        else if (float_for("--spec-draft-p-min", p.spec_draft_p_min)) {}
        else if (layers_for("--spec-draft-ngl", p.spec_draft_ngl)) {}
        else if (layers_for("-ngld", p.spec_draft_ngl)) {}
        else if (layers_for("--gpu-layers-draft", p.spec_draft_ngl)) {}
        else if (layers_for("--n-gpu-layers-draft", p.spec_draft_ngl)) {}
        else if (type_for("--spec-draft-type-k", p.spec_draft_type_k)) {}
        else if (type_for("--spec-draft-type-v", p.spec_draft_type_v)) {}
        else if (arg == "--moe-reset-cache-between-repeats") { p.moe_reset_cache_between_repeats = true; }
        else if (arg == "--moe-warm-cache") { p.moe_warm_cache = true; }
        else if (arg == "--moe-hot-start") { p.moe_hot_start = true; }
        else if (value_for("-p", p.prompt)) {}
    }
    if (p.n_repeat < 1) p.n_repeat = 1;
    if (p.n_prompt < 1) p.n_prompt = 1;
    if (p.n_gen < 1) p.n_gen = 1;
    if (p.spec_draft_n_max < 0) p.spec_draft_n_max = 0;
    if (p.spec_draft_n_min < 0) p.spec_draft_n_min = 0;
    return ok && !p.model.empty();
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double bytes_to_gib(uint64_t bytes) {
    return (double) bytes / (1024.0 * 1024.0 * 1024.0);
}

static double bytes_to_mib(uint64_t bytes) {
    return (double) bytes / (1024.0 * 1024.0);
}

static double bytes_to_mib(double bytes) {
    return bytes / (1024.0 * 1024.0);
}

static std::vector<std::vector<int>> rank_hot_start_scores(
        const std::vector<std::vector<double>> & scores,
        int n_slots) {
    std::vector<std::vector<int>> ranked(scores.size());
    for (size_t layer = 0; layer < scores.size(); ++layer) {
        std::vector<int> order(scores[layer].size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return scores[layer][(size_t) a] > scores[layer][(size_t) b];
        });
        if (n_slots > 0 && (size_t) n_slots < order.size()) {
            order.resize((size_t) n_slots);
        }
        ranked[layer] = std::move(order);
    }
    return ranked;
}

static std::vector<std::vector<int>> load_hot_start_from_eamc(
        const std::string & path,
        int n_experts,
        int n_slots) {
    std::vector<std::vector<int>> result;
    if (path.empty() || n_experts <= 0) {
        return result;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return result;
    }

    char magic[4] = {};
    uint32_t file_n_layers = 0;
    uint32_t file_n_experts = 0;
    uint64_t file_capacity = 0;
    uint64_t file_top_k = 0;
    uint64_t rows = 0;
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char *>(&file_n_layers), sizeof(file_n_layers));
    in.read(reinterpret_cast<char *>(&file_n_experts), sizeof(file_n_experts));
    in.read(reinterpret_cast<char *>(&file_capacity), sizeof(file_capacity));
    in.read(reinterpret_cast<char *>(&file_top_k), sizeof(file_top_k));
    in.read(reinterpret_cast<char *>(&rows), sizeof(rows));

    if (!in || std::string(magic, 4) != "EAM1" ||
            file_n_layers == 0 ||
            file_n_experts != (uint32_t) n_experts ||
            rows == 0 || file_capacity == 0 || file_top_k == 0) {
        return {};
    }

    const int n_layers = (int) file_n_layers;
    std::vector<std::vector<double>> scores((size_t) n_layers, std::vector<double>((size_t) n_experts, 0.0));
    std::vector<float> row((size_t) n_layers * (size_t) n_experts);
    for (uint64_t r = 0; r < rows; ++r) {
        in.read(reinterpret_cast<char *>(row.data()), (std::streamsize) (row.size() * sizeof(float)));
        if (!in) {
            return {};
        }
        for (int layer = 0; layer < n_layers; ++layer) {
            for (int expert = 0; expert < n_experts; ++expert) {
                scores[(size_t) layer][(size_t) expert] +=
                    row[(size_t) layer * (size_t) n_experts + (size_t) expert];
            }
        }
    }

    result = rank_hot_start_scores(scores, n_slots);
    return result;
}

static std::string basename_of(const std::string & path) {
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

static std::string default_eamc_path_of(const std::string & model_path) {
    const size_t slash = model_path.find_last_of("/\\");
    const size_t dot = model_path.find_last_of('.');
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
        return model_path.substr(0, dot) + ".eamc";
    }
    return model_path + ".eamc";
}

static std::string storage_label_of(const std::string & path) {
    if (path.size() >= 2 && path[1] == ':') {
        return path.substr(0, 2);
    }
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? "." : path.substr(0, pos);
}

static uint64_t process_dram_peak_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX counters = {};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &counters, sizeof(counters))) {
        return (uint64_t) counters.PeakWorkingSetSize;
    }
    return 0;
#else
    struct rusage usage = {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
#  if defined(__APPLE__)
    return (uint64_t) usage.ru_maxrss;
#  else
    return (uint64_t) usage.ru_maxrss * 1024ull;
#  endif
#endif
}

struct vram_sample {
    uint64_t used_bytes = 0;
    uint64_t total_bytes = 0;
};

static vram_sample sample_vram() {
    vram_sample sample;
    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        if (total_bytes >= free_bytes) {
            sample.used_bytes += (uint64_t) (total_bytes - free_bytes);
            sample.total_bytes += (uint64_t) total_bytes;
        }
    }
    return sample;
}

static uint64_t vram_delta_from_baseline(uint64_t used_bytes, uint64_t baseline_bytes) {
    return used_bytes > baseline_bytes ? used_bytes - baseline_bytes : 0;
}

static double hit_rate_percent(const llama_moe::profile_phase_stats & stats) {
    return stats.required == 0 ? 0.0 : 100.0 * (double) stats.hits / (double) stats.required;
}

static double us_per_token_ms(int64_t usec, int tokens, int repeat) {
    const int denom = std::max(1, tokens * repeat);
    return (double) usec / 1000.0 / (double) denom;
}

static std::string build_summary(
        const bench_params & p,
        const std::string & model_desc,
        int n_prompt_tokens,
        double avg_ttft,
        double avg_tpot,
        double avg_total,
        const llama_moe::profile_snapshot & profile,
        uint64_t vram_peak_bytes,
        uint64_t vram_total_bytes,
        uint64_t dram_peak_bytes) {
    const double prefill_tok_s = n_prompt_tokens > 0 ? n_prompt_tokens / (avg_ttft / 1000.0) : 0.0;
    const double decode_total_ms = avg_tpot * p.n_gen;
    const double decode_tok_s = avg_tpot > 0.0 ? 1000.0 / avg_tpot : 0.0;
    const uint64_t decode_tokens_total = (uint64_t) p.n_gen * (uint64_t) p.n_repeat;
    const uint64_t decode_ssd_bytes = profile.decode.ssd_bytes;
    const double decode_bytes_per_token = decode_tokens_total == 0 ? 0.0 : (double) decode_ssd_bytes / (double) decode_tokens_total;
    const double avg_read_mib = profile.decode.ssd_reads == 0 ? 0.0 : bytes_to_mib((double) profile.decode.ssd_bytes / (double) profile.decode.ssd_reads);
    const double avg_read_latency_ms = profile.decode.ssd_reads == 0 ? 0.0 : (double) profile.decode.ssd_read_us / 1000.0 / (double) profile.decode.ssd_reads;
    const double expert_cache_gib = (double) p.moe_cache_mb / 1024.0;
    const double vram_peak_gib = bytes_to_gib(vram_peak_bytes);
    const double vram_other_gib = std::max(0.0, vram_peak_gib - expert_cache_gib);

    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    out << "model: " << (model_desc.empty() ? basename_of(p.model) : model_desc) << '\n';
    out << "predictor: " << std::left << std::setw(8) << p.moe_predictor << std::right
        << "  cache: " << p.moe_cache_mb << " MB"
        << "   ssd: " << storage_label_of(p.model) << '\n';
    out << "n_prompt: " << n_prompt_tokens << "  n_gen: " << p.n_gen << "  repeats: " << p.n_repeat << "\n\n";

    out << "phase     tokens   total_ms   per_token_ms   tok/s\n";
    out << "prefill   " << std::setw(6) << n_prompt_tokens
        << "   " << std::setw(8) << std::setprecision(1) << avg_ttft
        << "        " << std::setw(6) << std::setprecision(2) << (n_prompt_tokens > 0 ? avg_ttft / n_prompt_tokens : 0.0)
        << "   " << std::setw(6) << std::setprecision(0) << prefill_tok_s << '\n';
    out << std::fixed << std::setprecision(1);
    out << "decode    " << std::setw(6) << p.n_gen
        << "   " << std::setw(8) << decode_total_ms
        << "        " << std::setw(6) << std::setprecision(2) << avg_tpot
        << "   " << std::setw(6) << std::setprecision(0) << decode_tok_s << "\n\n";

    out << std::fixed << std::setprecision(1);
    out << "cache hit rate (prefill): " << hit_rate_percent(profile.prefill) << "%\n";
    out << "cache hit rate (decode): " << hit_rate_percent(profile.decode) << "%\n";
    out << "SSD bytes read (decode): " << std::setprecision(2) << bytes_to_gib(decode_ssd_bytes)
        << " GB  (avg " << bytes_to_mib(decode_bytes_per_token) << " MB/token)\n";
    out << "TTFT: " << std::setprecision(1) << avg_ttft << " ms\n";
    out << "TPOT: " << std::setprecision(2) << avg_tpot << " ms\n";
    out << "total: " << std::setprecision(1) << avg_total << " ms\n\n";

    out << "I/O breakdown (decode, mean per token):\n";
    out << "  ssd_read       " << std::setw(8) << std::setprecision(2) << us_per_token_ms(profile.decode.ssd_read_us, p.n_gen, p.n_repeat) << " ms\n";
    out << "  h2d            " << std::setw(8) << us_per_token_ms(profile.decode.h2d_us, p.n_gen, p.n_repeat) << " ms\n";
    out << "  gpu_compute    " << std::setw(8) << us_per_token_ms(profile.decode.compute_us, p.n_gen, p.n_repeat) << " ms\n";
    out << "  stall (overlap loss) " << std::setw(8) << us_per_token_ms(profile.decode.stall_us, p.n_gen, p.n_repeat) << " ms\n";
    out << "  predictor      " << std::setw(8) << us_per_token_ms(profile.decode.pred_us, p.n_gen, p.n_repeat) << " ms\n\n";

    out << "VRAM peak (process approx): " << std::setprecision(2) << vram_peak_gib << " GB";
    if (vram_total_bytes > 0) {
        out << " / " << bytes_to_gib(vram_total_bytes) << " GB";
    }
    out << "  (baseline-adjusted; experts budget: " << expert_cache_gib
        << " GB, non-expert/process overhead: " << vram_other_gib << " GB)\n";
    out << "DRAM peak (process): " << bytes_to_gib(dram_peak_bytes) << " GB\n";
    out << "SSD reads: " << profile.decode.ssd_reads
        << " (avg " << avg_read_mib << " MB each, avg latency " << avg_read_latency_ms << " ms)\n";
    out << "profile rows: prefill=" << profile.prefill.rows << " decode=" << profile.decode.rows << '\n';
    return out.str();
}

static llama_token greedy_token_ith(llama_context * ctx, int vocab_size, int32_t idx) {
    float * logits = llama_get_logits_ith(ctx, idx);
    if (!logits) {
        return 0;
    }
    llama_token token = 0;
    float max_logit = logits[0];
    for (int v = 1; v < vocab_size; ++v) {
        if (logits[v] > max_logit) {
            max_logit = logits[v];
            token = v;
        }
    }
    return token;
}

static std::string join_tokens_for_csv(const std::vector<llama_token> & tokens) {
    std::ostringstream out;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            out << ' ';
        }
        out << tokens[i];
    }
    return out.str();
}

static void fill_token_batch(
        llama_batch & batch,
        const llama_token * tokens,
        int n_tokens,
        llama_pos pos0,
        bool logits_all) {
    common_batch_clear(batch);
    for (int i = 0; i < n_tokens; ++i) {
        common_batch_add(batch, tokens[i], pos0 + i, { 0 }, logits_all || i == n_tokens - 1);
    }
}

static void safe_context_seq_rm_tail(llama_context * ctx, llama_seq_id seq_id, llama_pos p0) {
    if (!ctx) {
        return;
    }
    const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id);
    if (pos_max < 0 || p0 > pos_max) {
        return;
    }
    common_context_seq_rm(ctx, seq_id, p0, -1);
}

int main(int argc, char ** argv) {
    bench_params p;
    if (!parse_args(argc, argv, p)) {
        fprintf(stderr, "Usage: llama-moe-bench --model <path> --pp N --tg N [--repeat N] [--moe-cache-vram-mb MB] [--moe-predictor lru|eamc] [--moe-eamc-path PATH] [--moe-profile-csv PATH] [--moe-profile-summary PATH] [--token-trace PATH] [--moe-reset-cache-between-repeats] [--moe-warm-cache] [--moe-hot-start] [-ub N] [--spec-type draft-mtp] [--spec-draft-n-max N] [--spec-draft-n-min N] [--spec-draft-p-min P] [--spec-draft-type-k TYPE] [--spec-draft-type-v TYPE]\n");
        return 1;
    }
    const bool use_mtp = p.spec_type == "draft-mtp";
    if (p.spec_type != "none" && !use_mtp) {
        fprintf(stderr, "unsupported --spec-type '%s' (currently supported: none, draft-mtp)\n", p.spec_type.c_str());
        return 1;
    }

    std::ofstream token_trace;
    if (!p.token_trace_path.empty()) {
        token_trace.open(p.token_trace_path, std::ios::out | std::ios::trunc);
        if (!token_trace) {
            fprintf(stderr, "[moe-bench] ERROR: failed to open token trace file: %s\n", p.token_trace_path.c_str());
            return 1;
        }
        token_trace << "repeat,iteration,phase,generated_before,n_past,token_in,emitted_tokens,draft_tokens,draft_accepted,next_token,generated_after\n";
    }
    auto write_token_trace = [&](int repeat_idx,
            int iteration,
            const char * phase,
            int generated_before,
            int n_past,
            llama_token token_in,
            const std::vector<llama_token> & emitted_tokens,
            const std::vector<llama_token> & draft_tokens,
            int n_draft_accepted,
            llama_token next_token,
            int generated_after) {
        if (!token_trace.is_open()) {
            return;
        }
        token_trace << repeat_idx << ','
            << iteration << ','
            << phase << ','
            << generated_before << ','
            << n_past << ','
            << token_in << ','
            << '"' << join_tokens_for_csv(emitted_tokens) << '"' << ','
            << '"' << join_tokens_for_csv(draft_tokens) << '"' << ','
            << n_draft_accepted << ','
            << next_token << ','
            << generated_after << '\n';
    };

    std::string prompt_text = p.prompt;
    if (prompt_text.empty()) {
        prompt_text.reserve((size_t) p.n_prompt * 8);
        for (int i = 0; i < p.n_prompt; ++i) {
            prompt_text += "Hello. ";
        }
    }

    llama_backend_init();
    vram_sample vram_baseline = sample_vram();
    uint64_t vram_total_bytes = vram_baseline.total_bytes;
    uint64_t vram_device_baseline_bytes = vram_baseline.used_bytes;
    uint64_t vram_device_peak_bytes = vram_baseline.used_bytes;
    uint64_t vram_peak_bytes = 0;
    uint64_t dram_peak_bytes = process_dram_peak_bytes();

    auto update_vram_peak = [&]() {
        const vram_sample sample = sample_vram();
        if (sample.total_bytes > 0) {
            vram_total_bytes = sample.total_bytes;
        }
        vram_device_peak_bytes = std::max(vram_device_peak_bytes, sample.used_bytes);
        vram_peak_bytes = std::max(vram_peak_bytes,
                vram_delta_from_baseline(sample.used_bytes, vram_device_baseline_bytes));
    };

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = p.n_gpu_layers;
    model_params.use_mmap = false;
    model_params.moe_offload = true;
    model_params.moe_cache_vram_mb = (uint64_t) p.moe_cache_mb;
    model_params.moe_predictor = p.moe_predictor.c_str();
    model_params.moe_eamc_path = p.moe_eamc_path.empty() ? nullptr : p.moe_eamc_path.c_str();
    model_params.moe_profile_csv = p.moe_profile_csv.empty() ? nullptr : p.moe_profile_csv.c_str();
    // The runtime end_request() summary writer emits the legacy aggregate
    // format. llama-moe-bench owns --moe-profile-summary so it can write the
    // full §4.7 benchmark report after all repeats are complete.
    model_params.moe_profile_summary = nullptr;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = p.n_ctx;
    ctx_params.n_batch = std::max(2048, p.n_prompt);
    if (use_mtp) {
        ctx_params.n_rs_seq = (uint32_t) std::max(1, p.spec_draft_n_max);
    }
    if (p.n_ubatch > 0) {
        ctx_params.n_ubatch = (uint32_t) p.n_ubatch;
    }

    llama_model * model = llama_model_load_from_file(p.model.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        llama_backend_free();
        return 1;
    }
    update_vram_peak();
    dram_peak_bytes = std::max(dram_peak_bytes, process_dram_peak_bytes());

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    update_vram_peak();
    dram_peak_bytes = std::max(dram_peak_bytes, process_dram_peak_bytes());

    llama_context * ctx_dft = nullptr;
    common_speculative * spec = nullptr;
    if (use_mtp) {
        if (llama_model_n_layer_nextn(model) <= 0) {
            fprintf(stderr, "--spec-type draft-mtp requested, but model has no MTP/nextn layers\n");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        llama_context_params ctx_params_dft = ctx_params;
        ctx_params_dft.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
        ctx_params_dft.ctx_other = ctx;
        // Match the server/common speculative setup: rollback snapshots are
        // required by the hybrid target context, not by the MTP-only context.
        ctx_params_dft.n_rs_seq  = 0;
        ctx_params_dft.type_k    = p.spec_draft_type_k;
        ctx_params_dft.type_v    = p.spec_draft_type_v;

        ctx_dft = llama_init_from_model(model, ctx_params_dft);
        if (!ctx_dft) {
            fprintf(stderr, "failed to create MTP draft context\n");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        common_params_speculative spec_params;
        spec_params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        spec_params.draft.ctx_tgt = ctx;
        spec_params.draft.ctx_dft = ctx_dft;
        spec_params.draft.n_max = p.spec_draft_n_max;
        spec_params.draft.n_min = p.spec_draft_n_min;
        spec_params.draft.p_min = p.spec_draft_p_min;
        spec_params.draft.n_gpu_layers = p.spec_draft_ngl;
        spec_params.draft.cache_type_k = p.spec_draft_type_k;
        spec_params.draft.cache_type_v = p.spec_draft_type_v;

        spec = common_speculative_init(spec_params, /*n_seq=*/ 1);
        if (!spec) {
            fprintf(stderr, "failed to initialize MTP speculative decoder\n");
            llama_free(ctx_dft);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        update_vram_peak();
        dram_peak_bytes = std::max(dram_peak_bytes, process_dram_peak_bytes());
    }

    char model_desc_buf[512] = {};
    std::string model_desc;
    if (llama_model_desc(model, model_desc_buf, sizeof(model_desc_buf)) > 0) {
        model_desc = model_desc_buf;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int vocab_size = llama_vocab_n_tokens(vocab);
    const int n_tokens_max = p.n_prompt + 128;
    std::vector<llama_token> prompt_tokens((size_t) n_tokens_max);
    int n_prompt_tokens = llama_tokenize(vocab,
            prompt_text.c_str(), (int) prompt_text.size(),
            prompt_tokens.data(), n_tokens_max, true, true);
    if (n_prompt_tokens < 0) {
        n_prompt_tokens = p.n_prompt;
        for (int i = 0; i < n_prompt_tokens; ++i) {
            prompt_tokens[i] = (i % 32000) + 1;
        }
    }
    if (n_prompt_tokens > p.n_prompt) {
        n_prompt_tokens = p.n_prompt;
    }

    if (p.moe_hot_start) {
        const uint32_t n_slots = llama_moe::n_slots_per_layer();
        const uint32_t n_experts = llama_moe::n_experts_per_layer();
        const std::string hot_start_path = p.moe_eamc_path.empty() ? default_eamc_path_of(p.model) : p.moe_eamc_path;
        const std::vector<std::vector<int>> preload = load_hot_start_from_eamc(
                hot_start_path,
                (int) n_experts,
                (int) n_slots);
        if (preload.empty()) {
            fprintf(stderr, "[moe-bench] warning: --moe-hot-start requested but no compatible EAMC sidecar was loaded from %s\n",
                    hot_start_path.c_str());
        } else if (!llama_moe::slot_pool_hot_start(preload)) {
            fprintf(stderr, "[moe-bench] ERROR: MoE hot-start preload failed\n");
            if (spec) {
                common_speculative_free(spec);
            }
            if (ctx_dft) {
                llama_free(ctx_dft);
            }
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
    }

    llama_batch batch = llama_batch_init((int32_t) ctx_params.n_batch, 0, 1);

    std::vector<double> ttft_ms;
    std::vector<double> prefill_target_ms;
    std::vector<double> prefill_mtp_process_ms;
    std::vector<double> cold_ttft_ms;
    std::vector<double> warm_ttft_ms;
    std::vector<double> tpot_ms;
    std::vector<double> total_ms;
    ttft_ms.reserve((size_t) p.n_repeat);
    prefill_target_ms.reserve((size_t) p.n_repeat);
    prefill_mtp_process_ms.reserve((size_t) p.n_repeat);
    cold_ttft_ms.reserve((size_t) p.n_repeat);
    warm_ttft_ms.reserve((size_t) p.n_repeat);
    tpot_ms.reserve((size_t) p.n_repeat);
    total_ms.reserve((size_t) p.n_repeat);

    uint64_t spec_draft_tokens = 0;
    uint64_t spec_draft_accepted = 0;

    int exit_code = 0;

    auto average_or_zero = [](const std::vector<double> & values) -> double {
        if (values.empty()) {
            return 0.0;
        }
        double total = 0.0;
        for (double value : values) {
            total += value;
        }
        return total / (double) values.size();
    };

    auto write_summary = [&](bool print_stdout) -> llama_moe::profile_snapshot {
        llama_moe::profile_summary_context summary_ctx;
        summary_ctx.model = model_desc.empty() ? basename_of(p.model) : model_desc;
        summary_ctx.predictor = p.moe_predictor;
        summary_ctx.storage = storage_label_of(p.model);
        summary_ctx.cache_mb = (uint64_t) p.moe_cache_mb;
        summary_ctx.n_prompt = n_prompt_tokens;
        summary_ctx.n_gen = p.n_gen;
        summary_ctx.n_repeat = (int) std::max<size_t>(1, std::max(ttft_ms.size(), total_ms.size()));
        summary_ctx.n_ubatch_requested = (int) ctx_params.n_ubatch;
        summary_ctx.n_ubatch = ctx ? (int) llama_n_ubatch(ctx) : 0;
        summary_ctx.n_slots = llama_moe::n_slots_per_layer();
        summary_ctx.n_active_slots = llama_moe::active_slots_per_layer();
        summary_ctx.n_experts = llama_moe::n_experts_per_layer();
        summary_ctx.streaming = llama_moe::streaming_mode();
        summary_ctx.cache_reset_between_repeats = p.moe_reset_cache_between_repeats;
        summary_ctx.warm_cache = p.moe_warm_cache;
        summary_ctx.hot_start = p.moe_hot_start;
        summary_ctx.spec_type = use_mtp ? p.spec_type : "none";
        summary_ctx.spec_draft_n_max = use_mtp ? p.spec_draft_n_max : 0;
        summary_ctx.spec_draft_tokens = spec_draft_tokens;
        summary_ctx.spec_draft_accepted = spec_draft_accepted;
        summary_ctx.ttft_ms = average_or_zero(ttft_ms);
        summary_ctx.prefill_target_ms = average_or_zero(prefill_target_ms);
        summary_ctx.prefill_mtp_process_ms = average_or_zero(prefill_mtp_process_ms);
        summary_ctx.cold_ttft_ms = average_or_zero(cold_ttft_ms);
        summary_ctx.warm_ttft_ms = average_or_zero(warm_ttft_ms);
        summary_ctx.tpot_ms = average_or_zero(tpot_ms);
        summary_ctx.total_ms = total_ms.empty() ? summary_ctx.ttft_ms : average_or_zero(total_ms);
        summary_ctx.cold_prefill_count = (int) cold_ttft_ms.size();
        summary_ctx.warm_prefill_count = (int) warm_ttft_ms.size();
        summary_ctx.vram_peak_bytes = vram_peak_bytes;
        summary_ctx.vram_total_bytes = vram_total_bytes;
        summary_ctx.vram_device_baseline_bytes = vram_device_baseline_bytes;
        summary_ctx.vram_device_peak_bytes = vram_device_peak_bytes;
        summary_ctx.dram_peak_bytes = dram_peak_bytes;

        const llama_moe::profile_snapshot profile = llama_moe::get_profile_snapshot();
        const std::string summary = llama_moe::format_summary(summary_ctx, profile);

        if (print_stdout) {
            fputc('\n', stdout);
            fputs(summary.c_str(), stdout);
            fflush(stdout);
        }

        if (!p.moe_profile_summary.empty()) {
            fprintf(stderr, "[moe-bench] writing summary to: %s\n", p.moe_profile_summary.c_str());
            std::ofstream out(p.moe_profile_summary, std::ios::out | std::ios::trunc);
            if (!out) {
                fprintf(stderr, "[moe-bench] ERROR: failed to open summary file: %s\n", p.moe_profile_summary.c_str());
            } else {
                out << summary;
                out.close();
                if (out.fail()) {
                    fprintf(stderr, "[moe-bench] ERROR: write failed for summary file: %s\n", p.moe_profile_summary.c_str());
                } else if (print_stdout) {
                    fprintf(stderr, "[moe-bench] summary written successfully to: %s\n", p.moe_profile_summary.c_str());
                }
            }
        }

        return profile;
    };

    if (p.moe_warm_cache) {
        fprintf(stderr, "[moe-bench] warming MoE cache before measured repeats\n");
        llama_memory_clear(llama_get_memory(ctx), true);
        if (ctx_dft) {
            llama_memory_clear(llama_get_memory(ctx_dft), true);
        }
        llama_moe::set_profile_request_context(-1, -1, "warmup");
        fill_token_batch(batch, prompt_tokens.data(), n_prompt_tokens, 0, use_mtp);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "warm-cache prefill decode failed\n");
            llama_moe::slot_pool_shutdown_io();
            llama_batch_free(batch);
            if (spec) {
                common_speculative_free(spec);
            }
            if (ctx_dft) {
                llama_free(ctx_dft);
            }
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        if (spec && !common_speculative_process(spec, batch)) {
            fprintf(stderr, "warm-cache speculative process failed\n");
            llama_moe::slot_pool_shutdown_io();
            llama_batch_free(batch);
            common_speculative_free(spec);
            if (ctx_dft) {
                llama_free(ctx_dft);
            }
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        update_vram_peak();
        dram_peak_bytes = std::max(dram_peak_bytes, process_dram_peak_bytes());
        llama_moe::reset_profile();
    }

    for (int rep = 0; rep < p.n_repeat; ++rep) {
        llama_memory_clear(llama_get_memory(ctx), true);
        if (ctx_dft) {
            llama_memory_clear(llama_get_memory(ctx_dft), true);
        }
        if (p.moe_reset_cache_between_repeats) {
            llama_moe::slot_pool_reset_cache();
        }
        llama_perf_context_reset(ctx);
        if (ctx_dft) {
            llama_perf_context_reset(ctx_dft);
        }

        const double t0 = now_ms();
        llama_moe::set_profile_request_context(rep, 0, "prefill_target");
        fill_token_batch(batch, prompt_tokens.data(), n_prompt_tokens, 0, use_mtp);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "prefill decode failed (rep %d)\n", rep);
            exit_code = 1;
            break;
        }
        const double t_target = now_ms();
        if (spec) {
            llama_moe::set_profile_request_context(rep, 0, "prefill_mtp_process");
        }
        if (spec && !common_speculative_process(spec, batch)) {
            fprintf(stderr, "prefill speculative process failed (rep %d)\n", rep);
            exit_code = 1;
            break;
        }
        update_vram_peak();
        dram_peak_bytes = std::max(dram_peak_bytes, process_dram_peak_bytes());
        const double t1 = now_ms();
        const double measured_ttft_ms = t1 - t0;
        const double measured_prefill_target_ms = t_target - t0;
        const double measured_prefill_mtp_process_ms = spec ? t1 - t_target : 0.0;
        ttft_ms.push_back(measured_ttft_ms);
        prefill_target_ms.push_back(measured_prefill_target_ms);
        prefill_mtp_process_ms.push_back(measured_prefill_mtp_process_ms);
        const bool measured_prefill_warm = !p.moe_reset_cache_between_repeats && (p.moe_warm_cache || rep > 0);
        if (measured_prefill_warm) {
            warm_ttft_ms.push_back(measured_ttft_ms);
        } else {
            cold_ttft_ms.push_back(measured_ttft_ms);
        }
        write_summary(false);

        llama_token token = greedy_token_ith(ctx, vocab_size, n_prompt_tokens - 1);
        if (use_mtp) {
            llama_tokens history(prompt_tokens.begin(), prompt_tokens.begin() + n_prompt_tokens);
            common_speculative_begin(spec, 0, history);

            int generated = 0;
            int n_past = n_prompt_tokens;
            std::vector<llama_token> draft;

            while (generated < p.n_gen) {
                const int iteration = generated + 1;
                const int generated_before = generated;
                const int remaining = p.n_gen - generated;
                const int n_draft_max = std::min(p.spec_draft_n_max, std::max(0, remaining - 1));

                draft.clear();
                if (n_draft_max > 0) {
                    auto & dp = common_speculative_get_draft_params(spec, 0);
                    dp = {
                        /* .drafting = */ true,
                        /* .n_max    = */ n_draft_max,
                        /* .n_past   = */ n_past,
                        /* .id_last  = */ token,
                        /* .prompt   = */ &history,
                        /* .result   = */ &draft,
                    };
                    llama_moe::set_profile_request_context(rep, iteration, "decode_mtp_draft");
                    common_speculative_draft(spec);
                    if ((int) draft.size() > n_draft_max) {
                        draft.resize((size_t) n_draft_max);
                    }
                    spec_draft_tokens += draft.size();

                    safe_context_seq_rm_tail(ctx_dft, 0, n_past);
                }

                common_batch_clear(batch);
                common_batch_add(batch, token, n_past, { 0 }, true);
                for (size_t i = 0; i < draft.size(); ++i) {
                    common_batch_add(batch, draft[i], n_past + 1 + (llama_pos) i, { 0 }, true);
                }

                llama_moe::set_profile_request_context(rep, iteration, "decode_target");
                if (llama_decode(ctx, batch) != 0) {
                    fprintf(stderr, "decode failed at generated=%d (rep %d)\n", generated, rep);
                    exit_code = 1;
                    break;
                }
                llama_moe::set_profile_request_context(rep, iteration, "decode_mtp_process");
                if (!common_speculative_process(spec, batch)) {
                    fprintf(stderr, "decode speculative process failed at generated=%d (rep %d)\n", generated, rep);
                    exit_code = 1;
                    break;
                }

                std::vector<llama_token> accepted;
                accepted.reserve(draft.size() + 1);

                int n_draft_accepted = 0;
                for (size_t i = 0; i < draft.size(); ++i) {
                    const llama_token id = greedy_token_ith(ctx, vocab_size, (int32_t) i);
                    accepted.push_back(id);
                    if (id != draft[i]) {
                        break;
                    }
                    ++n_draft_accepted;
                }
                if (n_draft_accepted == (int) draft.size()) {
                    accepted.push_back(greedy_token_ith(ctx, vocab_size, (int32_t) draft.size()));
                }

                if (!draft.empty()) {
                    spec_draft_accepted += (uint64_t) n_draft_accepted;
                    common_speculative_accept(spec, 0, (uint16_t) n_draft_accepted);
                }

                std::vector<llama_token> emitted;
                emitted.reserve((size_t) 1 + (size_t) n_draft_accepted);
                emitted.push_back(token);
                for (int i = 0; i < n_draft_accepted; ++i) {
                    emitted.push_back(draft[(size_t) i]);
                }
                const llama_token next_token = accepted.empty() ? 0 : accepted.back();
                const int generated_after = generated + (int) emitted.size();
                write_token_trace(rep, iteration, "mtp", generated_before, n_past,
                        token, emitted, draft, n_draft_accepted, next_token, generated_after);

                history.push_back(token);
                for (int i = 0; i < n_draft_accepted; ++i) {
                    history.push_back(draft[(size_t) i]);
                }
                n_past += (int) emitted.size();

                safe_context_seq_rm_tail(ctx, 0, n_past);
                safe_context_seq_rm_tail(ctx_dft, 0, n_past);

                token = next_token;
                generated = generated_after;

                update_vram_peak();
                dram_peak_bytes = std::max(dram_peak_bytes, process_dram_peak_bytes());
            }
        } else {
            for (int gen = 0; gen < p.n_gen; ++gen) {
                const llama_pos n_past = n_prompt_tokens + gen;
                const llama_token token_in = token;
                fill_token_batch(batch, &token_in, 1, n_past, true);
                llama_moe::set_profile_request_context(rep, gen + 1, "decode_target");
                if (llama_decode(ctx, batch) != 0) {
                    fprintf(stderr, "decode failed at gen %d (rep %d)\n", gen, rep);
                    exit_code = 1;
                    break;
                }
                token = greedy_token_ith(ctx, vocab_size, 0);
                const std::vector<llama_token> emitted = { token_in };
                const std::vector<llama_token> empty_draft;
                write_token_trace(rep, gen + 1, "target", gen, n_past,
                        token_in, emitted, empty_draft, 0, token, gen + 1);
                update_vram_peak();
                dram_peak_bytes = std::max(dram_peak_bytes, process_dram_peak_bytes());
            }
        }
        if (exit_code != 0) {
            break;
        }

        const double t2 = now_ms();
        tpot_ms.push_back((t2 - t1) / p.n_gen);
        total_ms.push_back(t2 - t0);
        update_vram_peak();
        dram_peak_bytes = std::max(dram_peak_bytes, process_dram_peak_bytes());
        write_summary(false);
    }

    if (!llama_moe::flush_predictor()) {
        fprintf(stderr, "[moe-bench] ERROR: failed to flush MoE predictor state\n");
        exit_code = 1;
    }

    fprintf(stderr, "[moe-bench] computing summary...\n");
    const llama_moe::profile_snapshot profile = write_summary(true);
    if (profile.prefill.rows + profile.prefill_target.rows + profile.prefill_mtp_process.rows + profile.decode.rows == 0) {
        fprintf(stderr, "warning: no MoE profile rows were recorded; check that the model is a repacked *.moe.gguf and that the run entered streaming mode\n");
    }

    fprintf(stderr, "[moe-bench] done. prefill_rows=%llu decode_rows=%llu\n",
            (unsigned long long) profile.prefill.rows,
            (unsigned long long) profile.decode.rows);

    if (token_trace.is_open()) {
        token_trace.flush();
    }

    llama_batch_free(batch);
    llama_moe::slot_pool_shutdown_io();
    if (spec) {
        common_speculative_free(spec);
    }
    if (ctx_dft) {
        llama_free(ctx_dft);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return exit_code;
}
