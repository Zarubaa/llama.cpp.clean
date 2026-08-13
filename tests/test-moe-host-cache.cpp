#include "moe-offload/host_cache.h"

#include <chrono>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace llama_moe;

int main() {
    manifest mf;
    mf.present = true;
    mf.n_layers = 2;
    mf.n_experts_per_layer = 4;
    mf.data_offset = 0;
    mf.experts.resize((size_t) mf.n_layers * mf.n_experts_per_layer * EXPERT_KIND_COUNT);

    std::vector<std::vector<uint8_t>> expected(mf.experts.size());
    uint64_t offset = 0;
    for (size_t i = 0; i < mf.experts.size(); ++i) {
        const size_t size = 65536 + i;
        mf.experts[i] = expert_record{offset, size};
        expected[i].resize(size);
        for (size_t j = 0; j < size; ++j) {
            expected[i][j] = (uint8_t) ((i * 17 + j * 13) & 0xff);
        }
        offset += size;
    }

    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("test-moe-host-cache-" + std::to_string(stamp) + ".bin");
    mf.source_path = path.string();
    {
        std::ofstream out(path, std::ios::binary);
        for (const auto & blob : expected) {
            out.write(reinterpret_cast<const char *>(blob.data()), (std::streamsize) blob.size());
        }
        if (!out) {
            return 1;
        }
    }

    bool ok = host_cache_init(mf, "pageable", "none");
    std::vector<bool> thread_ok(4, false);
    std::vector<std::thread> threads;
    for (size_t i = 0; i < thread_ok.size(); ++i) {
        threads.emplace_back([&, i] {
            FILE * source = fopen(path.string().c_str(), "rb");
            host_cache_access access;
            thread_ok[i] = source && host_cache_get_or_fill(0, 0, EXPERT_GATE, source, true, 1.0, access) &&
                    access.data && memcmp(access.data, expected[0].data(), expected[0].size()) == 0;
            host_cache_release(access.lease);
            if (source) fclose(source);
        });
    }
    for (auto & thread : threads) {
        thread.join();
    }
    for (bool thread_result : thread_ok) {
        ok = ok && thread_result;
    }
    host_cache_snapshot snapshot = host_cache_get_snapshot();
    if (!(snapshot.misses == 1 && snapshot.hits == 3 && snapshot.ready_blobs == 1)) {
        std::fprintf(stderr, "concurrent stats misses=%llu hits=%llu ready=%llu\n",
                (unsigned long long) snapshot.misses, (unsigned long long) snapshot.hits,
                (unsigned long long) snapshot.ready_blobs);
    }
    ok = ok && snapshot.misses == 1 && snapshot.hits == 3 && snapshot.ready_blobs == 1;
    ok = ok && host_cache_verify_ready(1);
    host_cache_shutdown();

    // A 1 MiB bounded cache fits two complete experts per layer for this
    // manifest. Verify whole-expert eviction and delayed GPU-overlap removal.
    ok = ok && host_cache_init(mf, "pageable", "none", 1);
    snapshot = host_cache_get_snapshot();
    if (!(snapshot.capacity_bytes <= 1024 * 1024 && snapshot.slots_per_layer == 2)) {
        std::fprintf(stderr, "bounded shape capacity=%llu slots=%llu\n",
                (unsigned long long) snapshot.capacity_bytes,
                (unsigned long long) snapshot.slots_per_layer);
    }
    ok = ok && snapshot.capacity_bytes <= 1024 * 1024 && snapshot.slots_per_layer == 2;
    auto fill_expert = [&](uint32_t expert, double heat, uint64_t * held_lease = nullptr) {
        bool filled = true;
        for (int kind = 0; kind < EXPERT_KIND_COUNT; ++kind) {
            FILE * source = fopen(path.string().c_str(), "rb");
            host_cache_access access;
            filled = filled && source && host_cache_get_or_fill(
                    0, expert, (expert_kind) kind, source, true, heat, access) && access.data;
            if (held_lease && kind == 0) {
                *held_lease = access.lease;
            } else {
                host_cache_release(access.lease);
            }
            if (source) fclose(source);
        }
        return filled;
    };
    uint64_t held = 0;
    ok = ok && fill_expert(0, 1.0, &held);
    ok = ok && fill_expert(1, 2.0);
    ok = ok && fill_expert(2, 3.0);
    if (!(host_cache_contains(0, 0) && !host_cache_contains(0, 1) && host_cache_contains(0, 2))) {
        std::fprintf(stderr, "bounded mapping c0=%d c1=%d c2=%d\n",
                (int) host_cache_contains(0, 0), (int) host_cache_contains(0, 1), (int) host_cache_contains(0, 2));
    }
    ok = ok && host_cache_contains(0, 0) && !host_cache_contains(0, 1) && host_cache_contains(0, 2);
    host_cache_mark_gpu_resident(0, 0);
    ok = ok && host_cache_contains(0, 0);
    host_cache_release(held);
    ok = ok && !host_cache_contains(0, 0);
    snapshot = host_cache_get_snapshot();
    if (!(snapshot.active_leases == 0 && snapshot.evictions == 1 && snapshot.gpu_overlap_removals == 1)) {
        std::fprintf(stderr, "bounded stats leases=%llu evictions=%llu overlap=%llu\n",
                (unsigned long long) snapshot.active_leases,
                (unsigned long long) snapshot.evictions,
                (unsigned long long) snapshot.gpu_overlap_removals);
    }
    ok = ok && snapshot.active_leases == 0 && snapshot.evictions == 1 && snapshot.gpu_overlap_removals == 1;
    host_cache_shutdown();

    const bool full_init = host_cache_init(mf, "pageable", "all");
    if (!full_init) std::fprintf(stderr, "full init failed\n");
    ok = ok && full_init;
    snapshot = host_cache_get_snapshot();
    ok = ok && snapshot.ready_blobs == mf.experts.size() && snapshot.ready_bytes == offset;
    ok = ok && host_cache_verify_ready(mf.experts.size());
    snapshot = host_cache_get_snapshot();
    ok = ok && snapshot.verified_blobs == mf.experts.size() && snapshot.verification_failures == 0;
    host_cache_shutdown();

    std::error_code error;
    std::filesystem::remove(path, error);
    if (!ok) {
        std::fprintf(stderr, "host-cache test failed\n");
    }
    return ok && !error ? 0 : 1;
}
