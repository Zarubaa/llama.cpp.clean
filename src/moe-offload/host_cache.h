#pragma once

#include "loader.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace llama_moe {

struct host_cache_access {
    const void * data = nullptr;
    uint64_t lease = 0;
    bool hit = false;
    bool miss = false;
    bool admitted = false;
    bool bypassed = false;
    int64_t lookup_us = 0;
    int64_t fill_us = 0;
    uint64_t source_reads = 0;
    uint64_t source_bytes = 0;
    int64_t source_read_us = 0;
};

struct host_cache_snapshot {
    std::string mode = "off";
    std::string preload = "none";
    uint64_t configured_capacity_bytes = 0;
    uint64_t capacity_bytes = 0;
    uint64_t data_bytes = 0;
    uint64_t total_blobs = 0;
    uint64_t ready_bytes = 0;
    uint64_t ready_blobs = 0;
    uint64_t ready_experts = 0;
    uint64_t slots_per_layer = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t hit_bytes = 0;
    uint64_t miss_bytes = 0;
    uint64_t admissions = 0;
    uint64_t evictions = 0;
    uint64_t bypasses = 0;
    uint64_t active_leases = 0;
    uint64_t gpu_overlap_removals = 0;
    int64_t lookup_us = 0;
    int64_t fill_us = 0;
    int64_t preload_alloc_us = 0;
    int64_t preload_read_us = 0;
    int64_t preload_total_us = 0;
    uint64_t preload_bytes = 0;
    uint64_t verified_blobs = 0;
    uint64_t verification_failures = 0;
};

// capacity_mb == 0 preserves the historical unbounded cache behavior.
bool host_cache_init(
        const manifest & mf,
        const std::string & mode,
        const std::string & preload,
        uint64_t capacity_mb = 0,
        const std::string & hotset_path = "");
void host_cache_shutdown();
bool host_cache_enabled();
bool host_cache_is_pinned();
bool host_cache_is_bounded();

// A cache miss is not an I/O failure. When admit is false, or no evictable
// Host slot is available, this returns true with access.data == nullptr so the
// caller can use the ordinary pinned-staging path.
bool host_cache_get_or_fill(
        uint32_t logical_layer,
        uint32_t expert,
        expert_kind kind,
        FILE * source,
        bool admit,
        double heat,
        host_cache_access & access);

// The returned pointer remains valid until its lease is released. This is
// required for direct pinned-cache H2D, which can outlive the I/O worker call.
void host_cache_release(uint64_t lease);

// Stable GPU tiers and the bounded Host tier are exclusive. Removal is delayed
// while a Host entry has an H2D lease or a blob fill in progress.
void host_cache_mark_gpu_resident(uint32_t logical_layer, uint32_t expert);
bool host_cache_contains(uint32_t logical_layer, uint32_t expert);

host_cache_snapshot host_cache_get_snapshot();
bool host_cache_verify_ready(size_t max_samples);

} // namespace llama_moe
