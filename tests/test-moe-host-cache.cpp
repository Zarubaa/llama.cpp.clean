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
        const size_t size = 257 + i;
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
            thread_ok[i] = source && host_cache_get_or_fill(0, 0, EXPERT_GATE, source, access) &&
                access.data && memcmp(access.data, expected[0].data(), expected[0].size()) == 0;
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
    ok = ok && snapshot.misses == 1 && snapshot.hits == 3 && snapshot.ready_blobs == 1;
    ok = ok && host_cache_verify_ready(1);
    host_cache_shutdown();

    ok = ok && host_cache_init(mf, "pageable", "all");
    snapshot = host_cache_get_snapshot();
    ok = ok && snapshot.ready_blobs == mf.experts.size() && snapshot.ready_bytes == offset;
    ok = ok && host_cache_verify_ready(mf.experts.size());
    snapshot = host_cache_get_snapshot();
    ok = ok && snapshot.verified_blobs == mf.experts.size() && snapshot.verification_failures == 0;
    host_cache_shutdown();

    std::error_code error;
    std::filesystem::remove(path, error);
    return ok && !error ? 0 : 1;
}
