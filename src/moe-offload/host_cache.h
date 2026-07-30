#pragma once

#include "loader.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace llama_moe {

struct host_cache_access {
    const void * data = nullptr;
    bool hit = false;
    bool miss = false;
    int64_t lookup_us = 0;
    int64_t fill_us = 0;
    uint64_t source_reads = 0;
    uint64_t source_bytes = 0;
    int64_t source_read_us = 0;
};

struct host_cache_snapshot {
    std::string mode = "off";
    std::string preload = "none";
    uint64_t capacity_bytes = 0;
    uint64_t data_bytes = 0;
    uint64_t total_blobs = 0;
    uint64_t ready_bytes = 0;
    uint64_t ready_blobs = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t hit_bytes = 0;
    uint64_t miss_bytes = 0;
    int64_t lookup_us = 0;
    int64_t fill_us = 0;
    int64_t preload_alloc_us = 0;
    int64_t preload_read_us = 0;
    int64_t preload_total_us = 0;
    uint64_t preload_bytes = 0;
    uint64_t verified_blobs = 0;
    uint64_t verification_failures = 0;
};

bool host_cache_init(const manifest & mf, const std::string & mode, const std::string & preload);
void host_cache_shutdown();
bool host_cache_enabled();
bool host_cache_is_pinned();
bool host_cache_get_or_fill(
        uint32_t logical_layer,
        uint32_t expert,
        expert_kind kind,
        FILE * source,
        host_cache_access & access);
host_cache_snapshot host_cache_get_snapshot();
bool host_cache_verify_ready(size_t max_samples);

} // namespace llama_moe
