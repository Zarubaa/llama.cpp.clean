#include "host_cache.h"

#include "io.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
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

enum class blob_state : uint8_t {
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

struct cache_blob {
    void * data = nullptr;
    uint64_t file_offset = 0;
    size_t size = 0;
    blob_state state = blob_state::empty;
};

struct cache_slot {
    uint32_t layer = 0;
    int32_t expert = -1;
    uint32_t generation = 1;
    uint32_t leases = 0;
    bool pending_remove = false;
    double heat = 0.0;
    uint64_t touch = 0;
    std::array<cache_blob, EXPERT_KIND_COUNT> blobs;
};

struct hotset_entry {
    uint32_t layer = 0;
    uint32_t expert = 0;
    double heat = 0.0;
};

struct cache_state {
    std::mutex mutex;
    std::mutex file_mutex;
    std::condition_variable cv;
    bool initialized = false;
    bool pinned = false;
    bool bounded = false;
    manifest mf;
    std::string mode = "off";
    std::string preload = "none";
    std::string hotset_path;
    uint64_t capacity_limit = 0;
    uint64_t touch_clock = 0;
    std::vector<cache_block> blocks;
    std::vector<cache_slot> slots;
    std::vector<std::vector<uint32_t>> layer_slots;
    std::vector<int32_t> expert_to_slot;
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

size_t expert_index(const manifest & mf, uint32_t logical_layer, uint32_t expert) {
    return (size_t) logical_layer * mf.n_experts_per_layer + expert;
}

bool slot_busy(const cache_slot & slot) {
    if (slot.leases != 0) {
        return true;
    }
    for (const cache_blob & blob : slot.blobs) {
        if (blob.state == blob_state::loading) {
            return true;
        }
    }
    return false;
}

bool slot_all_ready(const cache_slot & slot) {
    for (const cache_blob & blob : slot.blobs) {
        if (blob.size > 0 && blob.state != blob_state::ready) {
            return false;
        }
    }
    return slot.expert >= 0;
}

void account_remove_ready(cache_state & s, cache_slot & slot) {
    const bool expert_ready = slot_all_ready(slot);
    for (cache_blob & blob : slot.blobs) {
        if (blob.state == blob_state::ready) {
            s.stats.ready_bytes -= blob.size;
            --s.stats.ready_blobs;
        }
        blob.state = blob_state::empty;
    }
    if (expert_ready) {
        --s.stats.ready_experts;
    }
}

void clear_slot_mapping(cache_state & s, cache_slot & slot, bool count_eviction) {
    if (slot.expert < 0) {
        return;
    }
    const size_t old_index = expert_index(s.mf, slot.layer, (uint32_t) slot.expert);
    if (old_index < s.expert_to_slot.size() && s.expert_to_slot[old_index] >= 0) {
        s.expert_to_slot[old_index] = -1;
    }
    account_remove_ready(s, slot);
    slot.expert = -1;
    slot.pending_remove = false;
    slot.heat = 0.0;
    slot.touch = 0;
    ++slot.generation;
    if (slot.generation == 0) {
        slot.generation = 1;
    }
    if (count_eviction) {
        ++s.stats.evictions;
    }
}

void maybe_remove_pending(cache_state & s, cache_slot & slot) {
    if (slot.pending_remove && !slot_busy(slot)) {
        clear_slot_mapping(s, slot, false);
        ++s.stats.gpu_overlap_removals;
    }
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

bool allocate_block(cache_state & s, size_t bytes) {
    void * data = s.pinned ? io_pinned_alloc(bytes) : malloc(bytes);
    if (!data) {
        return false;
    }
    s.blocks.push_back(cache_block{data, bytes, 0});
    s.stats.capacity_bytes += bytes;
    return true;
}

void * arena_allocate(cache_state & s, size_t bytes) {
    bytes = align_up(bytes);
    if (bytes == 0) {
        return nullptr;
    }
    if (s.blocks.empty() || align_up(s.blocks.back().used) + bytes > s.blocks.back().size) {
        const uint64_t allocated = s.stats.capacity_bytes;
        if (allocated >= s.capacity_limit) {
            return nullptr;
        }
        const size_t remaining = (size_t) (s.capacity_limit - allocated);
        const size_t block_bytes = std::min(kBlockBytes, remaining);
        if (block_bytes < bytes || !allocate_block(s, block_bytes)) {
            return nullptr;
        }
    }
    cache_block & block = s.blocks.back();
    block.used = align_up(block.used);
    void * result = (char *) block.data + block.used;
    block.used += bytes;
    return result;
}

std::vector<std::array<size_t, EXPERT_KIND_COUNT>> layer_blob_maxima(const manifest & mf) {
    std::vector<std::array<size_t, EXPERT_KIND_COUNT>> result(mf.n_layers);
    for (uint32_t layer = 0; layer < mf.n_layers; ++layer) {
        for (uint32_t expert = 0; expert < mf.n_experts_per_layer; ++expert) {
            for (int kind = 0; kind < EXPERT_KIND_COUNT; ++kind) {
                result[layer][kind] = std::max(result[layer][kind],
                        (size_t) mf.at(layer, expert, (expert_kind) kind).size);
            }
        }
    }
    return result;
}

uint64_t simulated_bytes(
        const std::vector<std::array<size_t, EXPERT_KIND_COUNT>> & maxima,
        uint32_t slots_per_layer,
        uint64_t capacity) {
    uint64_t allocated = 0;
    uint64_t block_used = 0;
    uint64_t block_size = 0;
    for (size_t layer = 0; layer < maxima.size(); ++layer) {
        for (uint32_t slot = 0; slot < slots_per_layer; ++slot) {
            for (int kind = 0; kind < EXPERT_KIND_COUNT; ++kind) {
                const uint64_t bytes = align_up(maxima[layer][kind]);
                if (bytes == 0) continue;
                const uint64_t aligned_used = align_up(block_used);
                if (block_size == 0 || aligned_used + bytes > block_size) {
                    if (allocated >= capacity) return std::numeric_limits<uint64_t>::max();
                    block_size = std::min<uint64_t>(kBlockBytes, capacity - allocated);
                    if (block_size < bytes) return std::numeric_limits<uint64_t>::max();
                    allocated += block_size;
                    block_used = 0;
                } else {
                    block_used = aligned_used;
                }
                block_used += bytes;
            }
        }
    }
    return allocated;
}

bool initialize_slots(cache_state & s) {
    const auto maxima = layer_blob_maxima(s.mf);
    uint32_t per_layer = s.mf.n_experts_per_layer;
    if (s.bounded) {
        while (per_layer > 0 && simulated_bytes(maxima, per_layer, s.capacity_limit) > s.capacity_limit) {
            --per_layer;
        }
        if (per_layer == 0) {
            return false;
        }
    }

    s.layer_slots.assign(s.mf.n_layers, {});
    s.expert_to_slot.assign((size_t) s.mf.n_layers * s.mf.n_experts_per_layer, -1);
    s.stats.slots_per_layer = per_layer;
    s.stats.total_blobs = (uint64_t) per_layer * s.mf.n_layers * EXPERT_KIND_COUNT;
    s.slots.reserve((size_t) per_layer * s.mf.n_layers);
    for (uint32_t layer = 0; layer < s.mf.n_layers; ++layer) {
        s.layer_slots[layer].reserve(per_layer);
        for (uint32_t local = 0; local < per_layer; ++local) {
            cache_slot slot;
            slot.layer = layer;
            for (int kind = 0; kind < EXPERT_KIND_COUNT; ++kind) {
                slot.blobs[kind].size = maxima[layer][kind];
                slot.blobs[kind].data = arena_allocate(s, maxima[layer][kind]);
                if (maxima[layer][kind] > 0 && !slot.blobs[kind].data) {
                    return false;
                }
                s.stats.data_bytes += maxima[layer][kind];
            }
            s.layer_slots[layer].push_back((uint32_t) s.slots.size());
            s.slots.push_back(slot);
        }
    }

    if (!s.bounded) {
        for (uint32_t layer = 0; layer < s.mf.n_layers; ++layer) {
            for (uint32_t expert = 0; expert < s.mf.n_experts_per_layer; ++expert) {
                const uint32_t slot_index = s.layer_slots[layer][expert];
                cache_slot & slot = s.slots[slot_index];
                slot.expert = (int32_t) expert;
                for (int kind = 0; kind < EXPERT_KIND_COUNT; ++kind) {
                    const expert_record & rec = s.mf.at(layer, expert, (expert_kind) kind);
                    slot.blobs[kind].file_offset = s.mf.data_offset + rec.rel_offset;
                    slot.blobs[kind].size = (size_t) rec.size;
                }
                s.expert_to_slot[expert_index(s.mf, layer, expert)] = (int32_t) slot_index;
            }
        }
    }
    return true;
}

void bind_slot_to_expert(cache_state & s, cache_slot & slot, uint32_t expert, double heat) {
    slot.expert = (int32_t) expert;
    slot.pending_remove = false;
    slot.heat = heat;
    slot.touch = ++s.touch_clock;
    for (int kind = 0; kind < EXPERT_KIND_COUNT; ++kind) {
        const expert_record & rec = s.mf.at(slot.layer, expert, (expert_kind) kind);
        slot.blobs[kind].file_offset = s.mf.data_offset + rec.rel_offset;
        slot.blobs[kind].size = (size_t) rec.size;
        slot.blobs[kind].state = blob_state::empty;
    }
    s.expert_to_slot[expert_index(s.mf, slot.layer, expert)] = (int32_t) (&slot - s.slots.data());
    ++s.stats.admissions;
}

int32_t choose_slot(cache_state & s, uint32_t layer, double heat) {
    int32_t victim = -1;
    double victim_heat = std::numeric_limits<double>::infinity();
    uint64_t victim_touch = std::numeric_limits<uint64_t>::max();
    for (uint32_t index : s.layer_slots[layer]) {
        cache_slot & slot = s.slots[index];
        maybe_remove_pending(s, slot);
        if (slot.expert < 0) {
            return (int32_t) index;
        }
        if (slot_busy(slot) || slot.pending_remove) {
            continue;
        }
        if (slot.heat < victim_heat || (slot.heat == victim_heat && slot.touch < victim_touch)) {
            victim = (int32_t) index;
            victim_heat = slot.heat;
            victim_touch = slot.touch;
        }
    }
    if (victim >= 0 && heat <= victim_heat) {
        return -1;
    }
    return victim;
}

bool read_blob(cache_state & s, cache_blob & blob, FILE * source, int64_t & read_us) {
    const auto begin = std::chrono::steady_clock::now();
    size_t got = 0;
    {
        std::lock_guard<std::mutex> file_lock(s.file_mutex);
        if (host_cache_fseek(source, (int64_t) blob.file_offset, SEEK_SET) != 0) {
            return false;
        }
        got = fread(blob.data, 1, blob.size, source);
    }
    const auto end = std::chrono::steady_clock::now();
    read_us = elapsed_us(begin, end);
    return got == blob.size;
}

uint64_t make_lease(uint32_t slot_index, uint32_t generation) {
    return ((uint64_t) generation << 32) | ((uint64_t) slot_index + 1);
}

bool parse_hotset(const std::string & path, std::vector<hotset_entry> & entries) {
    std::ifstream input(path);
    if (!input) {
        return false;
    }
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream stream(line);
        hotset_entry entry;
        if (stream >> entry.layer >> entry.expert >> entry.heat) {
            entries.push_back(entry);
        }
    }
    return !entries.empty();
}

bool preload_slot(cache_state & s, std::unique_lock<std::mutex> & lock, cache_slot & slot, FILE * source) {
    bool was_ready = slot_all_ready(slot);
    for (int kind = 0; kind < EXPERT_KIND_COUNT; ++kind) {
        cache_blob & blob = slot.blobs[kind];
        if (blob.size == 0 || blob.state == blob_state::ready) continue;
        blob.state = blob_state::loading;
        lock.unlock();
        int64_t read_us = 0;
        const bool ok = read_blob(s, blob, source, read_us);
        lock.lock();
        s.stats.preload_read_us += read_us;
        if (!ok) {
            blob.state = blob_state::failed;
            return false;
        }
        blob.state = blob_state::ready;
        s.stats.ready_bytes += blob.size;
        s.stats.preload_bytes += blob.size;
        ++s.stats.ready_blobs;
    }
    if (!was_ready && slot_all_ready(slot)) {
        ++s.stats.ready_experts;
    }
    return true;
}

} // namespace

bool host_cache_init(
        const manifest & mf,
        const std::string & mode,
        const std::string & preload,
        uint64_t capacity_mb,
        const std::string & hotset_path) {
    host_cache_shutdown();
    cache_state & s = state();
    const auto total_begin = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(s.mutex);
    s.mode = mode;
    s.preload = preload;
    s.hotset_path = hotset_path;
    s.pinned = mode == "pinned";
    s.bounded = capacity_mb > 0;
    s.mf = mf;
    s.stats = {};
    s.stats.mode = mode;
    s.stats.preload = preload;
    s.stats.configured_capacity_bytes = capacity_mb * 1024ull * 1024ull;

    if (mode == "off") {
        s.initialized = true;
        return true;
    }

    if (s.bounded) {
        s.capacity_limit = s.stats.configured_capacity_bytes;
    } else {
        uint64_t full = 0;
        for (const expert_record & record : mf.experts) {
            full += align_up((size_t) record.size);
        }
        s.capacity_limit = full + kBlockBytes;
    }

    const auto alloc_begin = std::chrono::steady_clock::now();
    if (!initialize_slots(s)) {
        free_blocks(s);
        s.slots.clear();
        s.layer_slots.clear();
        s.expert_to_slot.clear();
        return false;
    }
    const auto alloc_end = std::chrono::steady_clock::now();
    s.stats.preload_alloc_us = elapsed_us(alloc_begin, alloc_end);
    s.initialized = true;

    if (preload == "all" || preload == "hotset") {
        FILE * source = fopen(mf.source_path.c_str(), "rb");
        if (!source) {
            return false;
        }
        bool ok = true;
        if (preload == "hotset") {
            std::vector<hotset_entry> entries;
            if (!s.bounded || !parse_hotset(hotset_path, entries)) {
                fclose(source);
                return false;
            }
            std::vector<uint32_t> counts(mf.n_layers, 0);
            for (const hotset_entry & entry : entries) {
                if (entry.layer >= mf.n_layers || entry.expert >= mf.n_experts_per_layer ||
                        counts[entry.layer] >= s.stats.slots_per_layer) {
                    continue;
                }
                const uint32_t slot_index = s.layer_slots[entry.layer][counts[entry.layer]++];
                cache_slot & slot = s.slots[slot_index];
                bind_slot_to_expert(s, slot, entry.expert, entry.heat);
                if (!preload_slot(s, lock, slot, source)) {
                    ok = false;
                    break;
                }
            }
            for (uint32_t count : counts) {
                if (count != s.stats.slots_per_layer) {
                    ok = false;
                }
            }
        } else {
            for (cache_slot & slot : s.slots) {
                if (slot.expert < 0 || !preload_slot(s, lock, slot, source)) {
                    ok = false;
                    break;
                }
            }
        }
        fclose(source);
        if (!ok) {
            return false;
        }
    }

    s.stats.preload_total_us = elapsed_us(total_begin, std::chrono::steady_clock::now());
    return true;
}

void host_cache_shutdown() {
    cache_state & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    free_blocks(s);
    s.slots.clear();
    s.layer_slots.clear();
    s.expert_to_slot.clear();
    s.mf = {};
    s.initialized = false;
    s.pinned = false;
    s.bounded = false;
    s.mode = "off";
    s.preload = "none";
    s.hotset_path.clear();
    s.capacity_limit = 0;
    s.touch_clock = 0;
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

bool host_cache_is_bounded() {
    cache_state & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.initialized && s.bounded;
}

bool host_cache_get_or_fill(
        uint32_t logical_layer,
        uint32_t expert,
        expert_kind kind,
        FILE * source,
        bool admit,
        double heat,
        host_cache_access & access) {
    cache_state & s = state();
    access = {};
    const auto lookup_begin = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(s.mutex);
    if (!s.initialized || s.mode == "off" || logical_layer >= s.mf.n_layers ||
            expert >= s.mf.n_experts_per_layer || !source || kind < 0 || kind >= EXPERT_KIND_COUNT) {
        return false;
    }

    const size_t mapping_index = expert_index(s.mf, logical_layer, expert);
    int32_t slot_index = s.expert_to_slot[mapping_index];
    if (slot_index < 0) {
        access.miss = true;
        ++s.stats.misses;
        s.stats.miss_bytes += s.mf.at(logical_layer, expert, kind).size;
        if (!admit && s.bounded) {
            access.bypassed = true;
            ++s.stats.bypasses;
            access.lookup_us = elapsed_us(lookup_begin, std::chrono::steady_clock::now());
            s.stats.lookup_us += access.lookup_us;
            return true;
        }
        slot_index = choose_slot(s, logical_layer, heat);
        if (slot_index < 0) {
            access.bypassed = true;
            ++s.stats.bypasses;
            access.lookup_us = elapsed_us(lookup_begin, std::chrono::steady_clock::now());
            s.stats.lookup_us += access.lookup_us;
            return true;
        }
        cache_slot & selected = s.slots[(size_t) slot_index];
        if (selected.expert >= 0) {
            clear_slot_mapping(s, selected, true);
        }
        bind_slot_to_expert(s, selected, expert, heat);
        access.admitted = true;
    }

    cache_slot & slot = s.slots[(size_t) slot_index];
    slot.heat = std::max(slot.heat, heat);
    slot.touch = ++s.touch_clock;
    cache_blob & blob = slot.blobs[(size_t) kind];
    while (blob.state == blob_state::loading && slot.expert == (int32_t) expert) {
        s.cv.wait(lock);
    }
    if (slot.expert != (int32_t) expert) {
        access.bypassed = true;
        ++s.stats.bypasses;
        return true;
    }

    access.lookup_us = elapsed_us(lookup_begin, std::chrono::steady_clock::now());
    s.stats.lookup_us += access.lookup_us;
    if (blob.state == blob_state::ready) {
        access.data = blob.data;
        access.hit = true;
        if (!access.miss) {
            ++s.stats.hits;
            s.stats.hit_bytes += blob.size;
        }
        ++slot.leases;
        ++s.stats.active_leases;
        access.lease = make_lease((uint32_t) slot_index, slot.generation);
        return true;
    }
    if (blob.state == blob_state::failed) {
        return false;
    }

    if (!access.miss) {
        access.miss = true;
        ++s.stats.misses;
        s.stats.miss_bytes += blob.size;
    }
    blob.state = blob_state::loading;
    lock.unlock();
    const auto fill_begin = std::chrono::steady_clock::now();
    int64_t source_read_us = 0;
    const bool ok = read_blob(s, blob, source, source_read_us);
    const auto fill_end = std::chrono::steady_clock::now();
    lock.lock();

    access.fill_us = elapsed_us(fill_begin, fill_end);
    access.source_read_us = source_read_us;
    access.source_reads = 1;
    access.source_bytes = blob.size;
    s.stats.fill_us += access.fill_us;
    blob.state = ok ? blob_state::ready : blob_state::failed;
    if (ok) {
        access.data = blob.data;
        s.stats.ready_bytes += blob.size;
        ++s.stats.ready_blobs;
        if (slot_all_ready(slot)) {
            ++s.stats.ready_experts;
        }
        ++slot.leases;
        ++s.stats.active_leases;
        access.lease = make_lease((uint32_t) slot_index, slot.generation);
    }
    lock.unlock();
    s.cv.notify_all();
    return ok;
}

void host_cache_release(uint64_t lease) {
    if (lease == 0) return;
    cache_state & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const uint32_t slot_encoded = (uint32_t) lease;
    const uint32_t generation = (uint32_t) (lease >> 32);
    if (slot_encoded == 0 || slot_encoded - 1 >= s.slots.size()) return;
    cache_slot & slot = s.slots[slot_encoded - 1];
    if (slot.generation != generation || slot.leases == 0) return;
    --slot.leases;
    --s.stats.active_leases;
    maybe_remove_pending(s, slot);
}

void host_cache_mark_gpu_resident(uint32_t logical_layer, uint32_t expert) {
    cache_state & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.initialized || !s.bounded || logical_layer >= s.mf.n_layers || expert >= s.mf.n_experts_per_layer) {
        return;
    }
    const int32_t index = s.expert_to_slot[expert_index(s.mf, logical_layer, expert)];
    if (index < 0) return;
    cache_slot & slot = s.slots[(size_t) index];
    slot.pending_remove = true;
    maybe_remove_pending(s, slot);
}

bool host_cache_contains(uint32_t logical_layer, uint32_t expert) {
    cache_state & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.initialized || logical_layer >= s.mf.n_layers || expert >= s.mf.n_experts_per_layer) return false;
    return s.expert_to_slot[expert_index(s.mf, logical_layer, expert)] >= 0;
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
        if (!s.initialized || s.mode == "off") return true;
        source_path = s.mf.source_path;
        for (const cache_slot & slot : s.slots) {
            if (slot.expert < 0) continue;
            for (const cache_blob & blob : slot.blobs) {
                if (blob.state == blob_state::ready && blob.size > 0) {
                    ready.push_back(sample_entry{blob.data, blob.file_offset, blob.size});
                }
            }
        }
    }
    if (ready.empty() || max_samples == 0) return true;
    FILE * source = fopen(source_path.c_str(), "rb");
    if (!source) return false;
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
