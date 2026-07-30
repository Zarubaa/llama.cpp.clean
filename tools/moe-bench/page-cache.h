#pragma once

#include <cstdint>
#include <string>

enum class page_cache_policy {
    natural,
    cold,
    hot,
};

struct page_cache_sample {
    uint64_t resident_bytes = 0;
    uint64_t total_bytes = 0;
    uint64_t resident_pages = 0;
    uint64_t total_pages = 0;
    bool valid = false;
};

struct page_cache_prepare_result {
    page_cache_policy policy = page_cache_policy::natural;
    page_cache_sample before;
    page_cache_sample after;
    int64_t elapsed_us = 0;
    int attempts = 0;
    uint64_t bytes_read = 0;
    std::string error;
};

struct process_io_sample {
    uint64_t rchar = 0;
    uint64_t read_bytes = 0;
    uint64_t syscr = 0;
    bool valid = false;
};

bool page_cache_policy_from_string(const std::string & text, page_cache_policy & policy);
const char * page_cache_policy_name(page_cache_policy policy);
page_cache_sample page_cache_residency(const std::string & path);
bool page_cache_prepare(const std::string & path, page_cache_policy policy, page_cache_prepare_result & result);
double page_cache_resident_percent(const page_cache_sample & sample);
process_io_sample process_io_current();
uint64_t process_io_delta(uint64_t before, uint64_t after);
