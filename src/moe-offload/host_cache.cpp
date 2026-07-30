#include "host_cache.h"

#include "io.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <vector>

#if defined(_WIN32)
#  define host_cache_fseek(f, off, whence) _fseeki64((f), (off), (whence))
#else
#  define host_cache_fseek(f, off, whence) fseeko((f), (off), (whence))
#endif

namespace llama_moe {
namespace {

constexpr size_t kBlockBytes = 256ull * 1024ull * 1024ull;
constexpr size_t kAlignment = 64;

enum class entry_state : uint8_t {
    empty,
    loading,
    ready,
    failed,
};

struct cache_block {
    void * data = nullptr;
    size_t size = 0;
    size_t used = 0;
};

struct cache_entry {
    void * data = nullptr;
    uint64_t file_offset = 0;
    size_t size = 0;
    entry_state state = entry_state::empty;
};

struct cache_state {
    std::mutex mutex;
    std::mutex file_mutex;
    std::condition_variable cv;
    bool initialized = false;
    bool pinned = false;
    manifest mf;
    std::string mode = "off";
    std::string preload = "none";
    std::vector<cache_block> blocks;
    std::vector<cache_entry> entries;
    host_cache_snapshot stats;
};

cache_state & state() {
    static cache_state instance;
    return instance;
}

int64_t elapsed_us(
        const std::chrono::steady_clock::time_point & begin,
        const std::chrono::steady_clock::time_point & end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
}

size_t align_up(size_t value) {
    return (value + kAlignment - 1) & ~(kAlignment - 1);
}

size_t entry_index(const manifest & mf, uint32_t logical_layer, uint32_t expert, expert_kind kind) {
    return ((size_t) logical_layer * mf.n_experts_per_layer + expert) * EXPERT_KIND_COUNT + (size_t) kind;
}

void free_blocks(cache_state & s) {
    for (cache_block & block : s.blocks) {
        if (s.pinned) {
            io_pinned_free(block.data);
        } else {
            free(block.data);
        }
    }
    s.blocks.clear();
}

bool allocate_block(cache_state & s, size_t minimum) {
    const size_t bytes = std::max(kBlockBytes, align_up(minimum));
    void * data = s.pinned ? io_pinned_alloc(bytes) : malloc(bytes);
    if (!data) {
        return false;
    }
    s.blocks.push_back(cache_block{data, bytes, 0});
    s.stats.capacity_bytes += bytes;
    return true;
}

bool read_entry(cache_state & s, cache_entry & entry, FILE * source, int64_t & read_us) {
    const auto begin = std::chrono::steady_clock::now();
    size_t got = 0;
    {
        std::lock_guard<std::mutex> file_lock(s.file_mutex);
        if (host_cache_fseek(source, (int64_t) entry.file_offset, SEEK_SET) != 0) {
            return false;
        }
        got = fread(entry.data, 1, entry.size, source);
    }
    const auto end = std::chrono::steady_clock::now();
    read_us = elapsed_us(begin, end);
    return got == entry.size;
}

} // namespace

bool host_cache_init(const manifest & mf, const std::string & mode, const std::string & preload) {
    host_cache_shutdown();
    cache_state & s = state();
    const auto total_begin = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(s.mutex);
    s.mode = mode;
    s.preload = preload;
    s.pinned = mode == "pinned";
    s.mf = mf;
    s.stats = {};
    s.stats.mode = mode;
    s.stats.preload = preload;

    if (mode == "off") {
        s.initialized = true;
        return true;
    }

    const auto alloc_begin = std::chrono::steady_clock::now();
    s.entries.resize(mf.experts.size());
    for (size_t i = 0; i < mf.experts.size(); ++i) {
        const expert_record & record = mf.experts[i];
        if (record.size == 0) {
            continue;
        }
        if (s.blocks.empty() || align_up(s.blocks.back().used) + record.size > s.blocks.back().size) {
            if (!allocate_block(s, (size_t) record.size)) {
                free_blocks(s);
                s.entries.clear();
                s.stats = {};
                s.stats.mode = mode;
                s.stats.preload = preload;
                return false;
            }
        }
        cache_block & block = s.blocks.back();
        block.used = align_up(block.used);
        cache_entry & entry = s.entries[i];
        entry.data = (char *) block.data + block.used;
        entry.file_offset = mf.data_offset + record.rel_offset;
        entry.size = (size_t) record.size;
        block.used += entry.size;
        s.stats.data_bytes += entry.size;
        ++s.stats.total_blobs;
    }
    const auto alloc_end = std::chrono::steady_clock::now();
    s.stats.preload_alloc_us = elapsed_us(alloc_begin, alloc_end);
    s.initialized = true;

    if (preload == "all") {
        FILE * source = fopen(mf.source_path.c_str(), "rb");
        if (!source) {
            free_blocks(s);
            s.entries.clear();
            s.initialized = false;
            return false;
        }
        int64_t read_total_us = 0;
        std::vector<size_t> order(s.entries.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return s.entries[a].file_offset < s.entries[b].file_offset;
        });
        for (size_t index : order) {
            cache_entry & entry = s.entries[index];
            if (entry.size == 0) {
                continue;
            }
            entry.state = entry_state::loading;
            lock.unlock();
            int64_t read_us = 0;
            const bool ok = read_entry(s, entry, source, read_us);
            lock.lock();
            read_total_us += read_us;
            if (!ok) {
                entry.state = entry_state::failed;
                fclose(source);
                free_blocks(s);
                s.entries.clear();
                s.initialized = false;
                return false;
            }
            entry.state = entry_state::ready;
            s.stats.ready_bytes += entry.size;
            s.stats.preload_bytes += entry.size;
            ++s.stats.ready_blobs;
        }
        fclose(source);
        s.stats.preload_read_us = read_total_us;
    }

    const auto total_end = std::chrono::steady_clock::now();
    s.stats.preload_total_us = elapsed_us(total_begin, total_end);
    return true;
}

void host_cache_shutdown() {
    cache_state & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    free_blocks(s);
    s.entries.clear();
    s.mf = {};
    s.initialized = false;
    s.pinned = false;
    s.mode = "off";
    s.preload = "none";
    s.stats = {};
}

bool host_cache_enabled() {
    cache_state & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.initialized && s.mode != "off";
}

bool host_cache_is_pinned() {
    cache_state & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.initialized && s.pinned;
}

bool host_cache_get_or_fill(
        uint32_t logical_layer,
        uint32_t expert,
        expert_kind kind,
        FILE * source,
        host_cache_access & access) {
    cache_state & s = state();
    access = {};
    const auto lookup_begin = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(s.mutex);
    if (!s.initialized || s.mode == "off" || logical_layer >= s.mf.n_layers ||
            expert >= s.mf.n_experts_per_layer || !source) {
        return false;
    }
    const size_t index = entry_index(s.mf, logical_layer, expert, kind);
    if (index >= s.entries.size() || s.entries[index].size == 0) {
        return false;
    }
    cache_entry & entry = s.entries[index];
    while (entry.state == entry_state::loading) {
        s.cv.wait(lock);
    }
    const auto lookup_end = std::chrono::steady_clock::now();
    access.lookup_us = elapsed_us(lookup_begin, lookup_end);
    s.stats.lookup_us += access.lookup_us;

    if (entry.state == entry_state::ready) {
        access.data = entry.data;
        access.hit = true;
        ++s.stats.hits;
        s.stats.hit_bytes += entry.size;
        return true;
    }
    if (entry.state == entry_state::failed) {
        return false;
    }

    entry.state = entry_state::loading;
    access.miss = true;
    ++s.stats.misses;
    s.stats.miss_bytes += entry.size;
    lock.unlock();

    const auto fill_begin = std::chrono::steady_clock::now();
    int64_t source_read_us = 0;
    const bool ok = read_entry(s, entry, source, source_read_us);
    const auto fill_end = std::chrono::steady_clock::now();

    lock.lock();
    access.fill_us = elapsed_us(fill_begin, fill_end);
    access.source_read_us = source_read_us;
    access.source_reads = 1;
    access.source_bytes = entry.size;
    s.stats.fill_us += access.fill_us;
    entry.state = ok ? entry_state::ready : entry_state::failed;
    if (ok) {
        access.data = entry.data;
        s.stats.ready_bytes += entry.size;
        ++s.stats.ready_blobs;
    }
    lock.unlock();
    s.cv.notify_all();
    return ok;
}

host_cache_snapshot host_cache_get_snapshot() {
    cache_state & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.stats;
}

bool host_cache_verify_ready(size_t max_samples) {
    cache_state & s = state();
    struct sample_entry {
        const void * data;
        uint64_t file_offset;
        size_t size;
    };
    std::vector<sample_entry> ready;
    std::string source_path;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        if (!s.initialized || s.mode == "off") {
            return true;
        }
        source_path = s.mf.source_path;
        for (const cache_entry & entry : s.entries) {
            if (entry.state == entry_state::ready && entry.size > 0) {
                ready.push_back(sample_entry{entry.data, entry.file_offset, entry.size});
            }
        }
    }
    if (ready.empty() || max_samples == 0) {
        return true;
    }
    FILE * source = fopen(source_path.c_str(), "rb");
    if (!source) {
        return false;
    }
    const size_t sample_count = std::min(max_samples, ready.size());
    std::vector<uint8_t> buffer;
    uint64_t verified = 0;
    uint64_t failures = 0;
    for (size_t i = 0; i < sample_count; ++i) {
        const size_t index = sample_count == 1 ? 0 : i * (ready.size() - 1) / (sample_count - 1);
        const sample_entry & entry = ready[index];
        buffer.resize(entry.size);
        if (host_cache_fseek(source, (int64_t) entry.file_offset, SEEK_SET) != 0 ||
                fread(buffer.data(), 1, entry.size, source) != entry.size ||
                memcmp(buffer.data(), entry.data, entry.size) != 0) {
            ++failures;
        } else {
            ++verified;
        }
    }
    fclose(source);
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.stats.verified_blobs += verified;
        s.stats.verification_failures += failures;
    }
    return failures == 0;
}

} // namespace llama_moe
