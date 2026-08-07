#pragma once

#include "llama.h"

#include <cstdint>
#include <fstream>
#include <string>

namespace llama_moe {

struct profile_row {
    uint64_t request_idx = 0;
    int repeat_idx = -1;
    int batch_idx = -1;
    uint64_t token_idx = 0;
    std::string phase = "decode";
    int layer = -1;
    int k_required = 0;
    int k_hit = 0;
    int k_miss = 0;
    int64_t ssd_read_us = 0;
    int64_t h2d_us = 0;
    int64_t compute_us = 0;
    int64_t stall_us = 0;
    int64_t pred_us = 0;
    int64_t pred_observe_us = 0;
    int64_t pred_score_us = 0;
    int64_t callback_wall_us = 0;
    int64_t topk_d2h_us = 0;
    int64_t slot_ids_h2d_us = 0;
    int64_t slot_table_h2d_us = 0;
    uint64_t eamc_rows_scored = 0;
    int64_t eamc_cosine_us = 0;
    int64_t eamc_score_materialize_us = 0;
    uint64_t eamc_score_cache_hits = 0;
    uint64_t eamc_score_cache_misses = 0;
    uint64_t ssd_bytes = 0;
    uint64_t ssd_reads = 0;
    uint64_t host_cache_hits = 0;
    uint64_t host_cache_misses = 0;
    uint64_t host_cache_hit_bytes = 0;
    uint64_t host_cache_miss_bytes = 0;
    int64_t host_cache_lookup_us = 0;
    int64_t host_cache_fill_us = 0;
    int64_t host_memcpy_us = 0;
    uint64_t host_memcpy_bytes = 0;
    uint64_t h2d_bytes = 0;
    int64_t pinned_staging_wait_us = 0;
    int cache_resident_experts = 0;
    std::string predictor = "lru";
    uint64_t routes_required = 0;
    uint64_t routes_hit = 0;
    uint64_t routes_persistent = 0;
    uint64_t k_empty_admit = 0;
    uint64_t k_victim_admit = 0;
    uint64_t k_scratch = 0;
    int64_t route_rank_us = 0;
    uint64_t route_hash = 0;
    uint64_t sere_secondary_routes = 0;
    uint64_t sere_rerouted_routes = 0;
    uint64_t sere_original_miss_routes = 0;
    uint64_t sere_original_unique_required = 0;
    uint64_t sere_original_unique_misses = 0;
    uint64_t sere_rerouted_miss_routes = 0;
    uint64_t sere_threshold_rejects = 0;
    double sere_similarity_sum = 0.0;
};

struct profile_request_row {
    uint64_t request_idx = 0;
    int repeat_idx = -1;
    int batch_idx = -1;
    std::string phase = "unknown";
    int64_t request_wall_us = 0;
    int64_t request_end_us = 0;
    int64_t predictor_end_us = 0;
    int64_t predictor_save_us = 0;
    int64_t profile_flush_us = 0;
    uint64_t sidecar_write_bytes = 0;
};

struct profile_phase_stats {
    uint64_t rows = 0;
    uint64_t requests = 0;
    uint64_t required = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t ssd_bytes = 0;
    uint64_t ssd_reads = 0;
    uint64_t host_cache_hits = 0;
    uint64_t host_cache_misses = 0;
    uint64_t host_cache_hit_bytes = 0;
    uint64_t host_cache_miss_bytes = 0;
    int64_t host_cache_lookup_us = 0;
    int64_t host_cache_fill_us = 0;
    int64_t host_memcpy_us = 0;
    uint64_t host_memcpy_bytes = 0;
    uint64_t h2d_bytes = 0;
    int64_t pinned_staging_wait_us = 0;
    int64_t ssd_read_us = 0;
    int64_t h2d_us = 0;
    int64_t compute_us = 0;
    int64_t stall_us = 0;
    int64_t pred_us = 0;
    int64_t pred_observe_us = 0;
    int64_t pred_score_us = 0;
    int64_t callback_wall_us = 0;
    int64_t topk_d2h_us = 0;
    int64_t slot_ids_h2d_us = 0;
    int64_t slot_table_h2d_us = 0;
    uint64_t eamc_rows_scored = 0;
    int64_t eamc_cosine_us = 0;
    int64_t eamc_score_materialize_us = 0;
    uint64_t eamc_score_cache_hits = 0;
    uint64_t eamc_score_cache_misses = 0;
    int64_t request_wall_us = 0;
    int64_t request_end_us = 0;
    int64_t predictor_end_us = 0;
    int64_t predictor_save_us = 0;
    int64_t profile_flush_us = 0;
    uint64_t sidecar_write_bytes = 0;
    int cache_resident_peak = 0;
    uint64_t routes_required = 0;
    uint64_t routes_hit = 0;
    uint64_t routes_persistent = 0;
    uint64_t k_empty_admit = 0;
    uint64_t k_victim_admit = 0;
    uint64_t k_scratch = 0;
    int64_t route_rank_us = 0;
    uint64_t sere_secondary_routes = 0;
    uint64_t sere_rerouted_routes = 0;
    uint64_t sere_original_miss_routes = 0;
    uint64_t sere_original_unique_required = 0;
    uint64_t sere_original_unique_misses = 0;
    uint64_t sere_rerouted_miss_routes = 0;
    uint64_t sere_threshold_rejects = 0;
    double sere_similarity_sum = 0.0;
};

struct profile_snapshot {
    profile_phase_stats prefill;
    profile_phase_stats decode;
};

struct profile_summary_context {
    std::string model;
    std::string predictor;
    std::string storage;
    uint64_t cache_mb = 0;
    int n_prompt = 0;
    int n_gen = 0;
    int n_repeat = 1;
    int n_ubatch_requested = 0;
    int n_ubatch = 0;
    uint32_t n_slots = 0;
    uint32_t n_active_slots = 0;
    uint32_t n_experts = 0;
    bool streaming = false;
    bool cache_reset_between_repeats = false;
    bool warm_cache = false;
    bool hot_start = false;
    double ttft_ms = 0.0;
    double cold_ttft_ms = 0.0;
    double warm_ttft_ms = 0.0;
    double tpot_ms = 0.0;
    double total_ms = 0.0;
    int cold_prefill_count = 0;
    int warm_prefill_count = 0;
    // Baseline-adjusted VRAM peak for this benchmark process. This is computed
    // from whole-device memory samples, so it removes pre-existing allocations
    // on the visible GPU(s) but can still include concurrent allocations that
    // appear after the benchmark starts.
    uint64_t vram_peak_bytes = 0;
    uint64_t vram_total_bytes = 0;
    uint64_t vram_device_baseline_bytes = 0;
    uint64_t vram_device_peak_bytes = 0;
    uint64_t dram_peak_bytes = 0;
    uint64_t dram_rss_bytes = 0;
    uint64_t dram_locked_bytes = 0;
    std::string host_cache_mode = "off";
    std::string host_cache_preload = "none";
    std::string sere_policy = "paper";
    int sere_top_k = 0;
    float sere_threshold = 0.0f;
    bool sere_shadow = false;
    uint64_t host_cache_capacity_bytes = 0;
    uint64_t host_cache_data_bytes = 0;
    uint64_t host_cache_ready_blobs = 0;
    uint64_t host_cache_total_blobs = 0;
    uint64_t host_cache_ready_bytes_after_prefill = 0;
    uint64_t host_cache_ready_bytes_after_decode = 0;
    int64_t host_cache_preload_alloc_us = 0;
    int64_t host_cache_preload_read_us = 0;
    int64_t host_cache_preload_total_us = 0;
    uint64_t host_cache_preload_bytes = 0;
    uint64_t host_cache_verified_blobs = 0;
    uint64_t host_cache_verification_failures = 0;
    double service_cold_start_ttft_ms = 0.0;
};

class profiler {
public:
    profiler() = default;
    explicit profiler(const std::string & csv_path);
    ~profiler();

    bool open(const std::string & csv_path);
    void reset(const std::string & csv_path);
    void record(const profile_row & row);
    void record_request(const profile_request_row & row);
    void flush();
    profile_snapshot snapshot() const;
    std::string summary() const;

private:
    void write_header();
    profile_phase_stats total() const;

    std::ofstream csv;
    profile_phase_stats prefill_stats;
    profile_phase_stats decode_stats;
};

LLAMA_API std::string format_summary(
    const profile_summary_context & ctx,
    const profile_snapshot & profile);

} // namespace llama_moe
