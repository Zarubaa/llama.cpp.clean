// MoE offload benchmark tool.
// Runs prefill + decode loops and prints the MVP profiling summary.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include "moe-offload/runtime.h"
#include "moe-offload/slot_pool.h"
#include "moe-offload/host_cache.h"
#include "page-cache.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
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
    std::string prompt_file;
    int n_prompt = 1024;
    int n_gen = 256;
    int n_repeat = 3;
    int moe_cache_mb = 8000;
    std::string moe_predictor = "eamc";
    std::string moe_eamc_path;
    std::string moe_profile_csv;
    std::string moe_profile_summary;
    std::string moe_host_cache = "off";
    std::string moe_host_cache_preload = "none";
    std::string moe_sere_path;
    std::string moe_sere_policy = "paper";
    int moe_sere_top_k = 0;
    float moe_sere_threshold = 0.0f;
    bool moe_sere_shadow = false;
    std::string page_cache_policy = "natural";
    std::string token_trace;
    std::string logits_bin;
    int n_gpu_layers = 99;
    int n_ctx = 4096;
    int n_ubatch = 0;
    bool moe_reset_cache_between_repeats = false;
    bool moe_warm_cache = false;
    bool moe_hot_start = false;
};

static bool parse_args(int argc, char ** argv, bench_params & p) {
    bool parse_error = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value_for = [&](const char * name, std::string & value) -> bool {
            const std::string key(name);
            if (arg == key) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "missing value for %s\n", name);
                    parse_error = true;
                    value.clear();
                    return true;
                }
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
            const char * begin = text.data();
            const char * end = begin + text.size();
            const auto parsed = std::from_chars(begin, end, value);
            if (text.empty() || parsed.ec != std::errc() || parsed.ptr != end) {
                fprintf(stderr, "invalid integer value for %s: %s\n", name, text.c_str());
                parse_error = true;
            }
            return true;
        };
        auto float_for = [&](const char * name, float & value) -> bool {
            std::string text;
            if (!value_for(name, text)) {
                return false;
            }
            errno = 0;
            char * end = nullptr;
            const float parsed = std::strtof(text.c_str(), &end);
            if (text.empty() || end != text.c_str() + text.size() || errno == ERANGE ||
                    !std::isfinite(parsed)) {
                fprintf(stderr, "invalid floating-point value for %s: %s\n", name, text.c_str());
                parse_error = true;
            } else {
                value = parsed;
            }
            return true;
        };

        if (value_for("--model", p.model)) {}
        else if (value_for("--prompt-file", p.prompt_file)) {}
        else if (int_for("--pp", p.n_prompt)) {}
        else if (int_for("--tg", p.n_gen)) {}
        else if (int_for("--repeat", p.n_repeat)) {}
        else if (int_for("--moe-cache-vram-mb", p.moe_cache_mb)) {}
        else if (value_for("--moe-predictor", p.moe_predictor)) {}
        else if (value_for("--moe-eamc-path", p.moe_eamc_path)) {}
        else if (value_for("--moe-profile-csv", p.moe_profile_csv)) {}
        else if (value_for("--moe-profile-summary", p.moe_profile_summary)) {}
        else if (value_for("--moe-host-cache", p.moe_host_cache)) {}
        else if (value_for("--moe-host-cache-preload", p.moe_host_cache_preload)) {}
        else if (value_for("--moe-sere-path", p.moe_sere_path)) {}
        else if (int_for("--moe-sere-top-k", p.moe_sere_top_k)) {}
        else if (value_for("--moe-sere-policy", p.moe_sere_policy)) {}
        else if (float_for("--moe-sere-threshold", p.moe_sere_threshold)) {}
        else if (arg == "--moe-sere-shadow") { p.moe_sere_shadow = true; }
        else if (value_for("--page-cache-policy", p.page_cache_policy)) {}
        else if (value_for("--token-trace", p.token_trace)) {}
        else if (value_for("--logits-bin", p.logits_bin)) {}
        else if (int_for("-ngl", p.n_gpu_layers)) {}
        else if (int_for("-c", p.n_ctx)) {}
        else if (int_for("-ub", p.n_ubatch)) {}
        else if (int_for("--ubatch", p.n_ubatch)) {}
        else if (int_for("--ubatch-size", p.n_ubatch)) {}
        else if (arg == "--moe-reset-cache-between-repeats") { p.moe_reset_cache_between_repeats = true; }
        else if (arg == "--moe-warm-cache") { p.moe_warm_cache = true; }
        else if (arg == "--moe-hot-start") { p.moe_hot_start = true; }
        else if (value_for("-p", p.prompt)) {}
    }
    if (p.n_repeat < 1) p.n_repeat = 1;
    if (p.n_prompt < 1) p.n_prompt = 1;
    if (p.n_gen < 1) p.n_gen = 1;
    return !parse_error && !p.model.empty();
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

struct process_memory_sample {
    uint64_t rss_bytes = 0;
    uint64_t hwm_bytes = 0;
    uint64_t locked_bytes = 0;
};

static process_memory_sample process_memory_status() {
    process_memory_sample sample;
#if !defined(_WIN32)
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream row(line);
        std::string key;
        uint64_t kib = 0;
        std::string unit;
        if (!(row >> key >> kib >> unit)) {
            continue;
        }
        if (key == "VmRSS:") sample.rss_bytes = kib * 1024ull;
        else if (key == "VmHWM:") sample.hwm_bytes = kib * 1024ull;
        else if (key == "VmLck:") sample.locked_bytes = kib * 1024ull;
    }
#endif
    return sample;
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

static bool greedy_token(llama_context * ctx, int vocab_size, llama_token & token) {
    float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) {
        return false;
    }
    token = 0;
    float max_logit = logits[0];
    for (int v = 1; v < vocab_size; ++v) {
        if (logits[v] > max_logit) {
            max_logit = logits[v];
            token = v;
        }
    }
    return true;
}

static bool write_logits_record(std::ofstream & out, llama_context * ctx, int repeat, int step, int vocab_size) {
    if (!out.is_open()) {
        return true;
    }
    float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) {
        return false;
    }
    const int32_t header[] = {repeat, step, vocab_size};
    out.write(reinterpret_cast<const char *>(header), sizeof(header));
    out.write(reinterpret_cast<const char *>(logits), (std::streamsize) vocab_size * sizeof(float));
    return out.good();
}

struct page_cache_runtime_metrics {
    page_cache_prepare_result prepare;
    page_cache_sample after_model_load;
    page_cache_sample after_prefill;
    page_cache_sample after_decode;
    process_io_sample process_start;
    process_io_sample after_prepare;
    process_io_sample after_model_load_io;
    process_io_sample after_prefill_io;
    process_io_sample after_decode_io;
    double model_init_ms = 0.0;
    double process_start_to_prefill_end_ms = 0.0;
};

static std::string format_page_cache_metrics(const page_cache_runtime_metrics & metrics) {
    auto read_delta = [](const process_io_sample & before, const process_io_sample & after) {
        return process_io_delta(before.read_bytes, after.read_bytes);
    };
    auto rchar_delta = [](const process_io_sample & before, const process_io_sample & after) {
        return process_io_delta(before.rchar, after.rchar);
    };
    std::ostringstream out;
    out << std::fixed << std::setprecision(4);
    out << "Page cache policy: " << page_cache_policy_name(metrics.prepare.policy) << '\n';
    out << "Page cache prepare: time=" << (double) metrics.prepare.elapsed_us / 1000.0
        << " ms attempts=" << metrics.prepare.attempts
        << " bytes_read=" << metrics.prepare.bytes_read << '\n';
    out << "Page cache resident bytes: before=" << metrics.prepare.before.resident_bytes
        << " after_prepare=" << metrics.prepare.after.resident_bytes
        << " after_model_load=" << metrics.after_model_load.resident_bytes
        << " after_prefill=" << metrics.after_prefill.resident_bytes
        << " after_decode=" << metrics.after_decode.resident_bytes
        << " total=" << metrics.prepare.after.total_bytes << '\n';
    out << "Page cache resident pct: before=" << page_cache_resident_percent(metrics.prepare.before)
        << " after_prepare=" << page_cache_resident_percent(metrics.prepare.after)
        << " after_model_load=" << page_cache_resident_percent(metrics.after_model_load)
        << " after_prefill=" << page_cache_resident_percent(metrics.after_prefill)
        << " after_decode=" << page_cache_resident_percent(metrics.after_decode) << '\n';
    out << "Page cache sample valid: before=" << (metrics.prepare.before.valid ? 1 : 0)
        << " after_prepare=" << (metrics.prepare.after.valid ? 1 : 0)
        << " after_model_load=" << (metrics.after_model_load.valid ? 1 : 0)
        << " after_prefill=" << (metrics.after_prefill.valid ? 1 : 0)
        << " after_decode=" << (metrics.after_decode.valid ? 1 : 0) << '\n';
    out << "Process IO read_bytes: prepare=" << read_delta(metrics.process_start, metrics.after_prepare)
        << " model_init=" << read_delta(metrics.after_prepare, metrics.after_model_load_io)
        << " prefill=" << read_delta(metrics.after_model_load_io, metrics.after_prefill_io)
        << " decode=" << read_delta(metrics.after_prefill_io, metrics.after_decode_io)
        << " total=" << read_delta(metrics.process_start, metrics.after_decode_io) << '\n';
    out << "Process IO rchar: prepare=" << rchar_delta(metrics.process_start, metrics.after_prepare)
        << " model_init=" << rchar_delta(metrics.after_prepare, metrics.after_model_load_io)
        << " prefill=" << rchar_delta(metrics.after_model_load_io, metrics.after_prefill_io)
        << " decode=" << rchar_delta(metrics.after_prefill_io, metrics.after_decode_io)
        << " total=" << rchar_delta(metrics.process_start, metrics.after_decode_io) << '\n';
    out << "Process IO sample valid: start=" << (metrics.process_start.valid ? 1 : 0)
        << " after_prepare=" << (metrics.after_prepare.valid ? 1 : 0)
        << " after_model_load=" << (metrics.after_model_load_io.valid ? 1 : 0)
        << " after_prefill=" << (metrics.after_prefill_io.valid ? 1 : 0)
        << " after_decode=" << (metrics.after_decode_io.valid ? 1 : 0) << '\n';
    out << "Page cache timing: prepare_ms=" << (double) metrics.prepare.elapsed_us / 1000.0
        << " model_init_ms=" << metrics.model_init_ms
        << " process_start_to_prefill_end_ms=" << metrics.process_start_to_prefill_end_ms << '\n';
    return out.str();
}

int main(int argc, char ** argv) {
    bench_params p;
    if (!parse_args(argc, argv, p)) {
        fprintf(stderr, "Usage: llama-moe-bench --model <path> [--prompt-file PATH | -p TEXT] --pp N --tg N [--repeat N] [--moe-cache-vram-mb MB] [--moe-predictor lru|eamc] [--moe-eamc-path PATH] [--moe-sere-path PATH --moe-sere-top-k N --moe-sere-threshold F --moe-sere-policy paper|miss [--moe-sere-shadow]] [--moe-profile-csv PATH] [--moe-profile-summary PATH] [--moe-host-cache off|pageable|pinned] [--moe-host-cache-preload none|all] [--page-cache-policy natural|cold|hot] [--token-trace PATH] [--logits-bin PATH] [--moe-reset-cache-between-repeats] [--moe-warm-cache] [--moe-hot-start] [-ub N]\n");
        return 1;
    }
    if ((p.moe_host_cache != "off" && p.moe_host_cache != "pageable" && p.moe_host_cache != "pinned") ||
            (p.moe_host_cache_preload != "none" && p.moe_host_cache_preload != "all") ||
            (p.moe_host_cache == "off" && p.moe_host_cache_preload != "none")) {
        fprintf(stderr, "invalid host cache configuration\n");
        return 1;
    }
    if (p.moe_sere_top_k < 0 ||
            (p.moe_sere_top_k > 0 && p.moe_sere_path.empty()) ||
            !std::isfinite(p.moe_sere_threshold) ||
            p.moe_sere_threshold < 0.0f || p.moe_sere_threshold > 1.0f ||
            (p.moe_sere_policy != "paper" && p.moe_sere_policy != "miss") ||
            (p.moe_sere_shadow && (p.moe_sere_top_k <= 0 || p.moe_sere_path.empty()))) {
        fprintf(stderr, "invalid SERE configuration\n");
        return 1;
    }
    page_cache_policy cache_policy;
    if (!page_cache_policy_from_string(p.page_cache_policy, cache_policy)) {
        fprintf(stderr, "invalid --page-cache-policy value: %s\n", p.page_cache_policy.c_str());
        return 1;
    }

    if (!p.prompt.empty() && !p.prompt_file.empty()) {
        fprintf(stderr, "use only one of --prompt-file and -p\n");
        return 1;
    }
    std::string prompt_text = p.prompt;
    if (!p.prompt_file.empty()) {
        std::ifstream prompt_stream(p.prompt_file, std::ios::in | std::ios::binary);
        if (!prompt_stream) {
            fprintf(stderr, "failed to open prompt file: %s\n", p.prompt_file.c_str());
            return 1;
        }
        std::ostringstream contents;
        contents << prompt_stream.rdbuf();
        prompt_text = contents.str();
        if (!prompt_stream.good() && !prompt_stream.eof()) {
            fprintf(stderr, "failed to read prompt file: %s\n", p.prompt_file.c_str());
            return 1;
        }
    }
    if (prompt_text.empty()) {
        prompt_text.reserve((size_t) p.n_prompt * 8);
        for (int i = 0; i < p.n_prompt; ++i) {
            prompt_text += "Hello. ";
        }
    }

    const double process_start_ms = now_ms();
    page_cache_runtime_metrics page_cache_metrics;
    page_cache_metrics.process_start = process_io_current();
    if (!page_cache_prepare(p.model, cache_policy, page_cache_metrics.prepare)) {
        fprintf(stderr, "page-cache preparation failed for %s: %s\n",
                p.model.c_str(), page_cache_metrics.prepare.error.c_str());
        return 2;
    }
    page_cache_metrics.after_prepare = process_io_current();
    const double model_init_begin_ms = now_ms();

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
    model_params.moe_host_cache = p.moe_host_cache.c_str();
    model_params.moe_host_cache_preload = p.moe_host_cache_preload.c_str();
    model_params.moe_sere_path = p.moe_sere_path.empty() ? nullptr : p.moe_sere_path.c_str();
    model_params.moe_sere_policy = p.moe_sere_policy.c_str();
    model_params.moe_sere_top_k = p.moe_sere_top_k;
    model_params.moe_sere_threshold = p.moe_sere_threshold;
    model_params.moe_sere_shadow = p.moe_sere_shadow;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = p.n_ctx;
    ctx_params.n_batch = std::max(2048, p.n_prompt);
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

    char model_desc_buf[512] = {};
    std::string model_desc;
    if (llama_model_desc(model, model_desc_buf, sizeof(model_desc_buf)) > 0) {
        model_desc = model_desc_buf;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int vocab_size = llama_vocab_n_tokens(vocab);
    const int source_prompt_tokens = -llama_tokenize(vocab,
            prompt_text.c_str(), (int) prompt_text.size(),
            nullptr, 0, true, true);
    if (source_prompt_tokens <= 0) {
        fprintf(stderr, "failed to size prompt tokenization\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    std::vector<llama_token> prompt_tokens((size_t) source_prompt_tokens);
    int n_prompt_tokens = llama_tokenize(vocab,
            prompt_text.c_str(), (int) prompt_text.size(),
            prompt_tokens.data(), (int) prompt_tokens.size(), true, true);
    if (n_prompt_tokens != source_prompt_tokens) {
        fprintf(stderr, "failed to tokenize prompt: required=%d result=%d\n",
                source_prompt_tokens, n_prompt_tokens);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    if (n_prompt_tokens > p.n_prompt) {
        n_prompt_tokens = p.n_prompt;
        prompt_tokens.resize((size_t) n_prompt_tokens);
    }
    uint64_t prompt_token_hash = 1469598103934665603ULL;
    for (llama_token prompt_token : prompt_tokens) {
        const uint32_t value = (uint32_t) prompt_token;
        for (int byte = 0; byte < 4; ++byte) {
            prompt_token_hash ^= (uint8_t) (value >> (byte * 8));
            prompt_token_hash *= 1099511628211ULL;
        }
    }
    fprintf(stderr, "[moe-bench] prompt source=%s bytes=%zu source_tokens=%d used_tokens=%d token_hash=%016llx\n",
            p.prompt_file.empty() ? "inline" : p.prompt_file.c_str(),
            prompt_text.size(), source_prompt_tokens, n_prompt_tokens,
            (unsigned long long) prompt_token_hash);

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
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
    }

    page_cache_metrics.model_init_ms = now_ms() - model_init_begin_ms;
    page_cache_metrics.after_model_load = page_cache_residency(p.model);
    page_cache_metrics.after_model_load_io = process_io_current();

    std::vector<double> ttft_ms;
    std::vector<double> cold_ttft_ms;
    std::vector<double> warm_ttft_ms;
    std::vector<double> tpot_ms;
    std::vector<double> total_ms;
    uint64_t host_ready_bytes_after_prefill = 0;
    uint64_t host_ready_bytes_after_decode = 0;
    ttft_ms.reserve((size_t) p.n_repeat);
    cold_ttft_ms.reserve((size_t) p.n_repeat);
    warm_ttft_ms.reserve((size_t) p.n_repeat);
    tpot_ms.reserve((size_t) p.n_repeat);
    total_ms.reserve((size_t) p.n_repeat);

    int exit_code = 0;
    std::ofstream token_trace;
    if (!p.token_trace.empty()) {
        token_trace.open(p.token_trace, std::ios::out | std::ios::trunc);
        if (!token_trace) {
            fprintf(stderr, "failed to open token trace: %s\n", p.token_trace.c_str());
            exit_code = 1;
        } else {
            token_trace << "repeat,step,token\n";
            if (!token_trace) {
                fprintf(stderr, "failed to write token trace header: %s\n", p.token_trace.c_str());
                exit_code = 1;
            }
        }
    }
    std::ofstream logits_bin;
    if (!p.logits_bin.empty()) {
        logits_bin.open(p.logits_bin, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!logits_bin) {
            fprintf(stderr, "failed to open logits output: %s\n", p.logits_bin.c_str());
            exit_code = 1;
        } else {
            const char magic[8] = {'M', 'O', 'E', 'L', 'O', 'G', '1', '\0'};
            logits_bin.write(magic, sizeof(magic));
            if (!logits_bin) {
                fprintf(stderr, "failed to write logits output header: %s\n", p.logits_bin.c_str());
                exit_code = 1;
            }
        }
    }

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
        summary_ctx.ttft_ms = average_or_zero(ttft_ms);
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
        const process_memory_sample mem = process_memory_status();
        const llama_moe::host_cache_snapshot host = llama_moe::host_cache_get_snapshot();
        summary_ctx.dram_rss_bytes = mem.rss_bytes;
        summary_ctx.dram_locked_bytes = mem.locked_bytes;
        summary_ctx.host_cache_mode = host.mode;
        summary_ctx.host_cache_preload = host.preload;
        summary_ctx.sere_policy = p.moe_sere_policy;
        summary_ctx.sere_top_k = p.moe_sere_top_k;
        summary_ctx.sere_threshold = p.moe_sere_threshold;
        summary_ctx.sere_shadow = p.moe_sere_shadow;
        summary_ctx.host_cache_capacity_bytes = host.capacity_bytes;
        summary_ctx.host_cache_data_bytes = host.data_bytes;
        summary_ctx.host_cache_ready_blobs = host.ready_blobs;
        summary_ctx.host_cache_total_blobs = host.total_blobs;
        summary_ctx.host_cache_ready_bytes_after_prefill = host_ready_bytes_after_prefill;
        summary_ctx.host_cache_ready_bytes_after_decode = host_ready_bytes_after_decode;
        summary_ctx.host_cache_preload_alloc_us = host.preload_alloc_us;
        summary_ctx.host_cache_preload_read_us = host.preload_read_us;
        summary_ctx.host_cache_preload_total_us = host.preload_total_us;
        summary_ctx.host_cache_preload_bytes = host.preload_bytes;
        summary_ctx.host_cache_verified_blobs = host.verified_blobs;
        summary_ctx.host_cache_verification_failures = host.verification_failures;
        summary_ctx.service_cold_start_ttft_ms = summary_ctx.ttft_ms + (double) host.preload_total_us / 1000.0;

        const llama_moe::profile_snapshot profile = llama_moe::get_profile_snapshot();
        std::string summary = llama_moe::format_summary(summary_ctx, profile);
        summary += format_page_cache_metrics(page_cache_metrics);
        summary += "prompt source tokens: " + std::to_string(source_prompt_tokens) + '\n';
        summary += "prompt used tokens: " + std::to_string(n_prompt_tokens) + '\n';
        std::ostringstream token_hash_stream;
        token_hash_stream << std::hex << std::setw(16) << std::setfill('0') << prompt_token_hash;
        summary += "prompt token hash: " + token_hash_stream.str() + '\n';

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

    for (int rep = 0; exit_code == 0 && rep < p.n_repeat; ++rep) {
        llama_memory_clear(llama_get_memory(ctx), true);
        if (p.moe_reset_cache_between_repeats) {
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
        llama_token token = 0;
        if (!greedy_token(ctx, vocab_size, token)) {
            fprintf(stderr, "failed to retrieve final prefill logits (rep %d)\n", rep);
            exit_code = 1;
            break;
        }
        const double t1 = now_ms();
        const double measured_ttft_ms = t1 - t0;

        update_vram_peak();
        dram_peak_bytes = std::max(dram_peak_bytes, process_dram_peak_bytes());
        host_ready_bytes_after_prefill = llama_moe::host_cache_get_snapshot().ready_bytes;
        page_cache_metrics.after_prefill = page_cache_residency(p.model);
        page_cache_metrics.after_prefill_io = process_io_current();
        page_cache_metrics.process_start_to_prefill_end_ms = t1 - process_start_ms;

        if (!write_logits_record(logits_bin, ctx, rep, 0, vocab_size)) {
            fprintf(stderr, "failed to write prefill logits\n");
            exit_code = 1;
            break;
        }
        ttft_ms.push_back(measured_ttft_ms);
        const bool measured_prefill_warm = !p.moe_reset_cache_between_repeats && (p.moe_warm_cache || rep > 0);
        if (measured_prefill_warm) {
            warm_ttft_ms.push_back(measured_ttft_ms);
        } else {
            cold_ttft_ms.push_back(measured_ttft_ms);
        }
        write_summary(false);

        std::vector<llama_token> trace_tokens;
        if (token_trace) {
            trace_tokens.resize((size_t) p.n_gen + 1);
            trace_tokens[0] = token;
        }
        const double decode_t0 = now_ms();
        int completed_gen = 0;
        for (int gen = 0; gen < p.n_gen; ++gen) {
            batch = llama_batch_get_one(&token, 1);
            llama_moe::set_profile_request_context(rep, gen + 1, "decode");
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "decode failed at gen %d (rep %d)\n", gen, rep);
                exit_code = 1;
                break;
            }
            if (!greedy_token(ctx, vocab_size, token)) {
                fprintf(stderr, "failed to retrieve decode logits at gen %d (rep %d)\n", gen, rep);
                exit_code = 1;
                break;
            }
            if (!write_logits_record(logits_bin, ctx, rep, gen + 1, vocab_size)) {
                fprintf(stderr, "failed to write decode logits at gen %d\n", gen);
                exit_code = 1;
                break;
            }
            if (token_trace) {
                trace_tokens[(size_t) gen + 1] = token;
            }
            ++completed_gen;
        }

        const double t2 = now_ms();
        if (exit_code != 0 || completed_gen != p.n_gen) {
            break;
        }
        if (token_trace) {
            for (size_t step = 0; step < trace_tokens.size(); ++step) {
                token_trace << rep << ',' << step << ',' << trace_tokens[step] << '\n';
            }
            if (!token_trace) {
                fprintf(stderr, "failed to write token trace for rep %d\n", rep);
                exit_code = 1;
                break;
            }
        }
        tpot_ms.push_back((t2 - decode_t0) / completed_gen);
        total_ms.push_back(measured_ttft_ms + (t2 - decode_t0));

        update_vram_peak();
        dram_peak_bytes = std::max(dram_peak_bytes, process_dram_peak_bytes());
        host_ready_bytes_after_decode = llama_moe::host_cache_get_snapshot().ready_bytes;
        page_cache_metrics.after_decode = page_cache_residency(p.model);
        page_cache_metrics.after_decode_io = process_io_current();
        write_summary(false);
    }

    if (token_trace.is_open()) {
        token_trace.flush();
        if (!token_trace) {
            fprintf(stderr, "failed to flush token trace: %s\n", p.token_trace.c_str());
            exit_code = 1;
        }
    }
    if (logits_bin.is_open()) {
        logits_bin.flush();
        if (!logits_bin) {
            fprintf(stderr, "failed to flush logits output: %s\n", p.logits_bin.c_str());
            exit_code = 1;
        }
    }

    if (!llama_moe::flush_predictor()) {
        fprintf(stderr, "[moe-bench] ERROR: failed to flush MoE predictor state\n");
        exit_code = 1;
    }
    if (!llama_moe::host_cache_verify_ready(256)) {
        fprintf(stderr, "[moe-bench] ERROR: host cache verification failed\n");
        exit_code = 1;
    }

    fprintf(stderr, "[moe-bench] computing summary...\n");
    const llama_moe::profile_snapshot profile = write_summary(true);
    if (profile.prefill.rows + profile.decode.rows == 0) {
        fprintf(stderr, "warning: no MoE profile rows were recorded; check that the model is a repacked *.moe.gguf and that the run entered streaming mode\n");
    }

    fprintf(stderr, "[moe-bench] done. prefill_rows=%llu decode_rows=%llu\n",
            (unsigned long long) profile.prefill.rows,
            (unsigned long long) profile.decode.rows);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return exit_code;
}
