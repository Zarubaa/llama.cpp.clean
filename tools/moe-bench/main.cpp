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
    std::string output_tokens;
    int n_gpu_layers = 99;
    int n_ctx = 4096;
    int n_ubatch = 0;
    std::string spec_type = "none";
    int spec_draft_n_max = 16;
    int spec_draft_n_min = 4;
    int spec_ngram_size_n = 12;
    int spec_ngram_min_hits = 1;
    int spec_n_rs_seq = -1;
    int spec_stage3_calibration_tokens = 16;
    bool moe_offload = true;
    bool spec_stage3 = false;
    bool moe_reset_cache_between_repeats = false;
    bool moe_warm_cache = false;
    bool moe_hot_start = false;
};

static bool parse_args(int argc, char ** argv, bench_params & p) {
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

        if (value_for("--model", p.model)) {}
        else if (int_for("--pp", p.n_prompt)) {}
        else if (int_for("--tg", p.n_gen)) {}
        else if (int_for("--repeat", p.n_repeat)) {}
        else if (int_for("--moe-cache-vram-mb", p.moe_cache_mb)) {}
        else if (value_for("--moe-predictor", p.moe_predictor)) {}
        else if (value_for("--moe-eamc-path", p.moe_eamc_path)) {}
        else if (value_for("--moe-profile-csv", p.moe_profile_csv)) {}
        else if (value_for("--moe-profile-summary", p.moe_profile_summary)) {}
        else if (value_for("--output-tokens", p.output_tokens)) {}
        else if (int_for("-ngl", p.n_gpu_layers)) {}
        else if (int_for("-c", p.n_ctx)) {}
        else if (int_for("-ub", p.n_ubatch)) {}
        else if (int_for("--ubatch", p.n_ubatch)) {}
        else if (int_for("--ubatch-size", p.n_ubatch)) {}
        else if (value_for("--spec-type", p.spec_type)) {}
        else if (int_for("--spec-draft-n-max", p.spec_draft_n_max)) {}
        else if (int_for("--spec-draft-n-min", p.spec_draft_n_min)) {}
        else if (int_for("--spec-ngram-size-n", p.spec_ngram_size_n)) {}
        else if (int_for("--spec-ngram-min-hits", p.spec_ngram_min_hits)) {}
        else if (int_for("--spec-n-rs-seq", p.spec_n_rs_seq)) {}
        else if (int_for("--spec-stage3-calibration-tokens", p.spec_stage3_calibration_tokens)) {}
        else if (arg == "--spec-stage3") { p.spec_stage3 = true; }
        else if (arg == "--no-moe-offload") { p.moe_offload = false; }
        else if (arg == "--moe-reset-cache-between-repeats") { p.moe_reset_cache_between_repeats = true; }
        else if (arg == "--moe-warm-cache") { p.moe_warm_cache = true; }
        else if (arg == "--moe-hot-start") { p.moe_hot_start = true; }
        else if (value_for("-p", p.prompt)) {}
    }
    if (p.n_repeat < 1) p.n_repeat = 1;
    if (p.n_prompt < 1) p.n_prompt = 1;
    if (p.n_gen < 1) p.n_gen = 1;
    if (p.spec_draft_n_max < 1) p.spec_draft_n_max = 1;
    if (p.spec_draft_n_min < 0) p.spec_draft_n_min = 0;
    if (p.spec_draft_n_min > p.spec_draft_n_max) p.spec_draft_n_min = p.spec_draft_n_max;
    if (p.spec_ngram_size_n < 1) p.spec_ngram_size_n = 1;
    if (p.spec_ngram_min_hits < 1) p.spec_ngram_min_hits = 1;
    if (p.spec_n_rs_seq < -1) p.spec_n_rs_seq = -1;
    if (p.spec_stage3_calibration_tokens < 1) p.spec_stage3_calibration_tokens = 1;
    return !p.model.empty();
}

static bool make_speculative_params(
        const bench_params & p,
        common_params_speculative & params,
        std::string & error) {
    const common_speculative_type type = common_speculative_type_from_name(p.spec_type);

    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
            break;
        case COMMON_SPECULATIVE_TYPE_COUNT:
            error = "unknown speculative type: " + p.spec_type;
            return false;
        default:
            error = "llama-moe-bench Stage 1/2 supports n-gram speculative types only";
            return false;
    }

    params.types = { type };
    params.draft.n_max = p.spec_draft_n_max;
    params.draft.n_min = p.spec_draft_n_min;

    params.ngram_mod.n_match = p.spec_ngram_size_n;
    params.ngram_mod.n_max = p.spec_draft_n_max;
    params.ngram_mod.n_min = p.spec_draft_n_min;

    params.ngram_simple.size_n = (uint16_t) p.spec_ngram_size_n;
    params.ngram_simple.size_m = (uint16_t) p.spec_draft_n_max;
    params.ngram_simple.min_hits = (uint16_t) p.spec_ngram_min_hits;
    params.ngram_map_k = params.ngram_simple;
    params.ngram_map_k4v = params.ngram_simple;

    return true;
}

static uint64_t hash_tokens(const llama_tokens & tokens) {
    uint64_t hash = 1469598103934665603ull;
    for (llama_token token : tokens) {
        const uint32_t value = (uint32_t) token;
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= (value >> shift) & 0xffu;
            hash *= 1099511628211ull;
        }
    }
    return hash;
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

static llama_token greedy_token(llama_context * ctx, int idx, int vocab_size) {
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

static llama_tokens greedy_sample_and_accept_n(
        llama_context * ctx,
        const llama_tokens & draft,
        int vocab_size) {
    llama_tokens result;
    result.reserve(draft.size() + 1);

    size_t i = 0;
    for (; i < draft.size(); ++i) {
        const llama_token id = greedy_token(ctx, (int) i, vocab_size);
        result.push_back(id);
        if (draft[i] != id) {
            break;
        }
    }
    if (i == draft.size()) {
        result.push_back(greedy_token(ctx, (int) i, vocab_size));
    }

    return result;
}

int main(int argc, char ** argv) {
    bench_params p;
    if (!parse_args(argc, argv, p)) {
        fprintf(stderr, "Usage: llama-moe-bench --model <path> --pp N --tg N [--repeat N] [--moe-cache-vram-mb MB] [--moe-predictor lru|eamc] [--moe-eamc-path PATH] [--moe-profile-csv PATH] [--moe-profile-summary PATH] [--output-tokens PATH] [--no-moe-offload] [--moe-reset-cache-between-repeats] [--moe-warm-cache] [--moe-hot-start] [-ub N] [--spec-type none|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod] [--spec-draft-n-max N] [--spec-draft-n-min N] [--spec-ngram-size-n N] [--spec-ngram-min-hits N] [--spec-n-rs-seq N] [--spec-stage3] [--spec-stage3-calibration-tokens N]\n");
        return 1;
    }

    common_params_speculative speculative_params;
    std::string speculative_error;
    if (!make_speculative_params(p, speculative_params, speculative_error)) {
        fprintf(stderr, "invalid speculative configuration: %s\n", speculative_error.c_str());
        return 1;
    }
    const bool speculative_enabled = speculative_params.types[0] != COMMON_SPECULATIVE_TYPE_NONE;
    if (p.spec_stage3 && !speculative_enabled) {
        fprintf(stderr, "--spec-stage3 requires a speculative n-gram type\n");
        return 1;
    }
    const int speculative_n_max = speculative_enabled
        ? common_speculative_n_max(&speculative_params)
        : 0;

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
    model_params.moe_offload = p.moe_offload;
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
    if (speculative_enabled) {
        ctx_params.n_outputs_max = (uint32_t) std::max(1, 1 + speculative_n_max);
        ctx_params.n_rs_seq = p.spec_n_rs_seq >= 0
            ? (uint32_t) p.spec_n_rs_seq
            : (uint32_t) speculative_n_max;
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

    common_context_seq_rm_type seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_PART;
    if (speculative_enabled) {
        seq_rm_type = common_context_can_seq_rm(ctx);
        if (seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            fprintf(stderr, "speculative decoding requires context sequence rollback support\n");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        // The capability probe evaluates two tokens and must not affect the
        // measured cache or profiler state.
        if (p.moe_offload) {
            llama_moe::slot_pool_reset_cache();
            llama_moe::reset_profile();
        }
    }
    llama_moe::slot_pool_set_speculative_aware(p.spec_stage3 && p.moe_offload);

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

    if (p.moe_hot_start && p.moe_offload) {
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
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
    }

    std::vector<double> ttft_ms;
    std::vector<double> cold_ttft_ms;
    std::vector<double> warm_ttft_ms;
    std::vector<double> tpot_ms;
    std::vector<double> total_ms;
    uint64_t draft_tokens_generated = 0;
    uint64_t draft_tokens_accepted = 0;
    uint64_t verification_steps = 0;
    uint64_t target_tokens_evaluated = 0;
    double draft_time_ms = 0.0;
    double target_verify_time_ms = 0.0;
    uint64_t generation_token_hash = 0;
    bool generation_tokens_consistent = true;
    uint64_t stage3_gate_enabled_repeats = 0;
    uint64_t stage3_gate_disabled_repeats = 0;
    uint64_t stage3_spec_calibration_outputs = 0;
    uint64_t stage3_base_calibration_outputs = 0;
    double stage3_spec_calibration_ms = 0.0;
    double stage3_base_calibration_ms = 0.0;
    ttft_ms.reserve((size_t) p.n_repeat);
    cold_ttft_ms.reserve((size_t) p.n_repeat);
    warm_ttft_ms.reserve((size_t) p.n_repeat);
    tpot_ms.reserve((size_t) p.n_repeat);
    total_ms.reserve((size_t) p.n_repeat);

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
        summary_ctx.predictor = p.moe_offload ? p.moe_predictor : "none";
        summary_ctx.storage = storage_label_of(p.model);
        summary_ctx.cache_mb = p.moe_offload ? (uint64_t) p.moe_cache_mb : 0;
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
        summary_ctx.ttft_ms = average_or_zero(ttft_ms);
        summary_ctx.cold_ttft_ms = average_or_zero(cold_ttft_ms);
        summary_ctx.warm_ttft_ms = average_or_zero(warm_ttft_ms);
        summary_ctx.tpot_ms = average_or_zero(tpot_ms);
        summary_ctx.total_ms = total_ms.empty() ? summary_ctx.ttft_ms : average_or_zero(total_ms);
        summary_ctx.cold_prefill_count = (int) cold_ttft_ms.size();
        summary_ctx.warm_prefill_count = (int) warm_ttft_ms.size();
        summary_ctx.speculative_type = p.spec_type;
        summary_ctx.draft_tokens_generated = draft_tokens_generated;
        summary_ctx.draft_tokens_accepted = draft_tokens_accepted;
        summary_ctx.verification_steps = verification_steps;
        summary_ctx.target_tokens_evaluated = target_tokens_evaluated;
        summary_ctx.draft_time_ms = draft_time_ms;
        summary_ctx.target_verify_time_ms = target_verify_time_ms;
        summary_ctx.generation_token_hash = generation_token_hash;
        summary_ctx.generation_tokens_consistent = generation_tokens_consistent;
        summary_ctx.speculative_stage3 = p.spec_stage3;
        summary_ctx.stage3_gate_enabled_repeats = stage3_gate_enabled_repeats;
        summary_ctx.stage3_gate_disabled_repeats = stage3_gate_disabled_repeats;
        summary_ctx.stage3_spec_calibration_outputs = stage3_spec_calibration_outputs;
        summary_ctx.stage3_base_calibration_outputs = stage3_base_calibration_outputs;
        summary_ctx.stage3_spec_calibration_ms = stage3_spec_calibration_ms;
        summary_ctx.stage3_base_calibration_ms = stage3_base_calibration_ms;
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
        llama_moe::set_profile_request_context(-1, -1, "warmup");
        llama_batch warm_batch = llama_batch_get_one(prompt_tokens.data(), n_prompt_tokens);
        if (llama_decode(ctx, warm_batch) != 0) {
            fprintf(stderr, "warm-cache prefill decode failed\n");
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
        if (p.moe_offload && p.moe_reset_cache_between_repeats) {
            llama_moe::slot_pool_reset_cache();
        }
        llama_perf_context_reset(ctx);

        const double t0 = now_ms();
        llama_moe::set_profile_request_context(rep, 0, "prefill");
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt_tokens);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "prefill decode failed (rep %d)\n", rep);
            exit_code = 1;
            break;
        }
        update_vram_peak();
        dram_peak_bytes = std::max(dram_peak_bytes, process_dram_peak_bytes());
        const double t1 = now_ms();
        const double measured_ttft_ms = t1 - t0;
        ttft_ms.push_back(measured_ttft_ms);
        const bool measured_prefill_warm = !p.moe_reset_cache_between_repeats && (p.moe_warm_cache || rep > 0);
        if (measured_prefill_warm) {
            warm_ttft_ms.push_back(measured_ttft_ms);
        } else {
            cold_ttft_ms.push_back(measured_ttft_ms);
        }
        write_summary(false);

        llama_token id_last = greedy_token(ctx, 0, vocab_size);

        llama_tokens prompt_tgt(prompt_tokens.begin(), prompt_tokens.begin() + n_prompt_tokens);
        prompt_tgt.reserve((size_t) p.n_ctx);

        common_speculative_ptr spec;
        if (speculative_enabled) {
            spec.reset(common_speculative_init(speculative_params, 1));
            if (!spec) {
                fprintf(stderr, "failed to initialize speculative decoding\n");
                exit_code = 1;
                break;
            }
            common_speculative_begin(spec.get(), 0, prompt_tgt);
        }

        llama_batch batch_tgt = llama_batch_init(std::max(1, 1 + speculative_n_max), 0, 1);
        llama_tokens draft;
        llama_tokens generated;
        generated.reserve((size_t) p.n_gen);
        common_prompt_checkpoint ckpt;
        size_t n_draft = 0;
        size_t n_accepted_from_original = 0;
        bool replaying_checkpoint = false;
        bool use_checkpoint = false;
        int n_past = n_prompt_tokens;
        uint64_t verification_steps_rep = 0;
        bool decode_failed = false;

        enum class stage3_gate_phase {
            spec_calibration,
            base_calibration,
            enabled,
            disabled,
        };
        stage3_gate_phase gate_phase = p.spec_stage3
            ? stage3_gate_phase::spec_calibration
            : stage3_gate_phase::enabled;
        uint64_t gate_spec_outputs_rep = 0;
        uint64_t gate_base_outputs_rep = 0;
        double gate_spec_ms_rep = 0.0;
        double gate_base_ms_rep = 0.0;
        double cycle_start_ms = now_ms();

        while ((int) generated.size() < p.n_gen) {
            if (draft.empty()) {
                cycle_start_ms = now_ms();
                ckpt.update_pos(
                        prompt_tgt.size(),
                        llama_memory_seq_pos_min(llama_get_memory(ctx), 0),
                        llama_memory_seq_pos_max(llama_get_memory(ctx), 0));

                const int remaining = p.n_gen - (int) generated.size();
                const bool gate_allows_draft = gate_phase == stage3_gate_phase::spec_calibration ||
                    gate_phase == stage3_gate_phase::enabled;
                if (spec && gate_allows_draft && remaining > 1) {
                    const double t_draft_start = now_ms();
                    common_speculative_get_draft_params(spec.get(), 0) = {
                        /* .drafting = */ true,
                        /* .n_max    = */ remaining - 1,
                        /* .n_past   = */ n_past,
                        /* .id_last  = */ id_last,
                        /* .prompt   = */ &prompt_tgt,
                        /* .result   = */ &draft,
                    };
                    common_speculative_draft(spec.get());
                    draft_time_ms += now_ms() - t_draft_start;
                }

                n_draft = draft.size();
                n_accepted_from_original = 0;
                replaying_checkpoint = false;
                draft_tokens_generated += n_draft;
                use_checkpoint = !draft.empty() &&
                    (seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                     (seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx)));

                if (use_checkpoint) {
                    ckpt.update_tgt(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }

            common_batch_clear(batch_tgt);
            common_batch_add(batch_tgt, id_last, n_past, { 0 }, true);
            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + 1 + (llama_pos) i, { 0 }, true);
            }

            llama_moe::set_profile_request_context(rep, (int) verification_steps_rep + 1, "decode");
            const double t_verify_start = now_ms();
            if (llama_decode(ctx, batch_tgt) != 0) {
                fprintf(stderr, "decode failed at output token %zu (rep %d)\n", generated.size(), rep);
                exit_code = 1;
                decode_failed = true;
                break;
            }
            if (!common_speculative_process(spec.get(), batch_tgt)) {
                fprintf(stderr, "speculative batch processing failed at output token %zu (rep %d)\n", generated.size(), rep);
                exit_code = 1;
                decode_failed = true;
                break;
            }
            target_verify_time_ms += now_ms() - t_verify_start;
            ++verification_steps;
            ++verification_steps_rep;
            target_tokens_evaluated += (uint64_t) batch_tgt.n_tokens;

            llama_tokens ids = greedy_sample_and_accept_n(ctx, draft, vocab_size);
            GGML_ASSERT(!ids.empty());

            if (!replaying_checkpoint) {
                n_accepted_from_original = ids.size() - 1;
            }

            const uint32_t n_rollback = (uint32_t) draft.size() + 1u - (uint32_t) ids.size();
            if (use_checkpoint && n_rollback > 0) {
                draft = std::move(ids);
                replaying_checkpoint = true;
                ckpt.load_tgt(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                common_context_seq_rm(ctx, 0, ckpt.pos_max + 1, -1);
                n_past = (int) prompt_tgt.size();
                continue;
            }

            if (spec && n_draft > 0) {
                common_speculative_accept(spec.get(), 0, (uint16_t) n_accepted_from_original);
                draft_tokens_accepted += n_accepted_from_original;
            }

            for (llama_token id : ids) {
                prompt_tgt.push_back(id_last);
                id_last = id;
                generated.push_back(id);
            }
            n_past = (int) prompt_tgt.size();

            if (n_rollback > 0) {
                common_context_seq_rm(ctx, 0, n_past, -1);
            }

            if (p.spec_stage3) {
                const double cycle_ms = now_ms() - cycle_start_ms;
                if (gate_phase == stage3_gate_phase::spec_calibration && n_draft > 0) {
                    gate_spec_outputs_rep += ids.size();
                    gate_spec_ms_rep += cycle_ms;
                    if (gate_spec_outputs_rep >= (uint64_t) p.spec_stage3_calibration_tokens) {
                        gate_phase = stage3_gate_phase::base_calibration;
                    }
                } else if (gate_phase == stage3_gate_phase::base_calibration) {
                    gate_base_outputs_rep += ids.size();
                    gate_base_ms_rep += cycle_ms;
                    if (gate_base_outputs_rep >= (uint64_t) p.spec_stage3_calibration_tokens) {
                        const double spec_ms_per_output = gate_spec_ms_rep / (double) gate_spec_outputs_rep;
                        const double base_ms_per_output = gate_base_ms_rep / (double) gate_base_outputs_rep;
                        const bool enable = spec_ms_per_output < 0.98 * base_ms_per_output;
                        gate_phase = enable ? stage3_gate_phase::enabled : stage3_gate_phase::disabled;
                        if (enable) {
                            ++stage3_gate_enabled_repeats;
                        } else {
                            ++stage3_gate_disabled_repeats;
                        }
                        fprintf(stderr, "[moe-bench] Stage 3 gate rep=%d spec=%.2f ms/token base=%.2f ms/token decision=%s\n",
                                rep, spec_ms_per_output, base_ms_per_output, enable ? "enabled" : "disabled");
                    }
                }
            }

            draft.clear();
            update_vram_peak();
            dram_peak_bytes = std::max(dram_peak_bytes, process_dram_peak_bytes());
        }

        llama_batch_free(batch_tgt);
        if (decode_failed) {
            break;
        }

        stage3_spec_calibration_outputs += gate_spec_outputs_rep;
        stage3_base_calibration_outputs += gate_base_outputs_rep;
        stage3_spec_calibration_ms += gate_spec_ms_rep;
        stage3_base_calibration_ms += gate_base_ms_rep;

        const uint64_t token_hash = hash_tokens(generated);
        if (generation_token_hash == 0) {
            generation_token_hash = token_hash;
        } else if (generation_token_hash != token_hash) {
            generation_tokens_consistent = false;
        }

        if (rep == 0 && !p.output_tokens.empty()) {
            std::ofstream out(p.output_tokens, std::ios::out | std::ios::trunc);
            if (!out) {
                fprintf(stderr, "failed to open token output: %s\n", p.output_tokens.c_str());
                exit_code = 1;
                break;
            }
            for (size_t i = 0; i < generated.size(); ++i) {
                out << i << ',' << generated[i] << '\n';
            }
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
    if (p.moe_offload && profile.prefill.rows + profile.decode.rows == 0) {
        fprintf(stderr, "warning: no MoE profile rows were recorded; check that the model is a repacked *.moe.gguf and that the run entered streaming mode\n");
    }

    fprintf(stderr, "[moe-bench] done. prefill_rows=%llu decode_rows=%llu\n",
            (unsigned long long) profile.prefill.rows,
            (unsigned long long) profile.decode.rows);

    llama_moe::slot_pool_shutdown_io();
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return exit_code;
}
