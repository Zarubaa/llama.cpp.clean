#ifdef NDEBUG
#undef NDEBUG
#endif

#include "moe-offload/sere.h"

#if !defined(_WIN32)
#  include <unistd.h>
#endif

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void write_sidecar(
        const char * path,
        const std::vector<float> & matrix,
        const char magic[8] = "MOESERE1",
        uint32_t version = 1,
        uint32_t layers = 1,
        uint32_t experts = 4) {
    const uint32_t metric = 1;

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(magic, 8);
    output.write(reinterpret_cast<const char *>(&version), sizeof(version));
    output.write(reinterpret_cast<const char *>(&layers), sizeof(layers));
    output.write(reinterpret_cast<const char *>(&experts), sizeof(experts));
    output.write(reinterpret_cast<const char *>(&metric), sizeof(metric));
    output.write(reinterpret_cast<const char *>(matrix.data()),
            (std::streamsize) (matrix.size() * sizeof(float)));
    assert(output.good());
}

int main() {
#if defined(_WIN32)
    return 0;
#else
    char path[] = "/tmp/test-moe-sere-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    close(fd);

    std::vector<float> matrix(16, 0.0f);
    for (int i = 0; i < 4; ++i) {
        matrix[(size_t) i * 4 + i] = 1.0f;
    }
    matrix[2 * 4 + 0] = 0.8f;
    matrix[2 * 4 + 1] = 0.9f;
    matrix[3 * 4 + 0] = 0.7f;
    matrix[3 * 4 + 1] = 0.7f;
    write_sidecar(path, matrix);

    llama_moe::sere_similarity_matrix similarity;
    std::string error;
    assert(similarity.load(path, error));
    assert(error.empty());
    assert(similarity.n_layers() == 1);
    assert(similarity.n_experts() == 4);
    assert(similarity.metric() == 1);
    assert(std::fabs(similarity.at(0, 2, 1) - 0.9f) < 1e-6f);

    const std::vector<int32_t> ids = {0, 1, 2, 3};
    const std::vector<uint8_t> resident = {1, 1, 1, 0};
    llama_moe::sere_route_stats stats;

    auto rerouted = llama_moe::sere_reroute_decode(
            similarity, 0, ids, 2, 0.75f,
            llama_moe::sere_policy::paper, resident, stats);
    assert((rerouted == std::vector<int32_t>{0, 1, 1, 3}));
    assert(stats.secondary_routes == 2);
    assert(stats.original_miss_routes == 1);
    assert(stats.original_unique_required == 4);
    assert(stats.original_unique_misses == 1);
    assert(stats.rerouted_routes == 1);
    assert(stats.rerouted_miss_routes == 0);
    assert(stats.threshold_rejects == 1);
    assert(std::fabs(stats.similarity_sum - 0.9) < 1e-6);

    rerouted = llama_moe::sere_reroute_decode(
            similarity, 0, ids, 2, 0.6f,
            llama_moe::sere_policy::miss, resident, stats);
    assert((rerouted == std::vector<int32_t>{0, 1, 2, 0}));
    assert(stats.secondary_routes == 2);
    assert(stats.original_miss_routes == 1);
    assert(stats.original_unique_required == 4);
    assert(stats.original_unique_misses == 1);
    assert(stats.rerouted_routes == 1);
    assert(stats.rerouted_miss_routes == 1);
    assert(stats.threshold_rejects == 0);
    assert(std::fabs(stats.similarity_sum - 0.7) < 1e-6);

    // The paper kernel scans the primary mask by expert ID, so expert 0 wins
    // an exact tie even when expert 1 has the earlier router rank.
    const std::vector<int32_t> tie_ids = {1, 0, 3, 2};
    rerouted = llama_moe::sere_reroute_decode(
            similarity, 0, tie_ids, 2, 0.7f,
            llama_moe::sere_policy::paper, resident, stats);
    assert((rerouted == std::vector<int32_t>{1, 0, 0, 1}));

    rerouted = llama_moe::sere_reroute_decode(
            similarity, 0, ids, 0, 0.0f,
            llama_moe::sere_policy::paper, resident, stats);
    assert(rerouted == ids);
    assert(stats.rerouted_routes == 0);

    assert(llama_moe::parse_sere_policy("paper") == llama_moe::sere_policy::paper);
    assert(std::strcmp(llama_moe::sere_policy_name(llama_moe::sere_policy::miss), "miss") == 0);

    bool policy_rejected = false;
    try {
        (void) llama_moe::parse_sere_policy("invalid");
    } catch (const std::invalid_argument &) {
        policy_rejected = true;
    }
    assert(policy_rejected);

    matrix[0] = std::numeric_limits<float>::quiet_NaN();
    write_sidecar(path, matrix);
    assert(!similarity.load(path, error));
    assert(similarity.empty());
    assert(!error.empty());

    matrix[0] = 1.0f;
    write_sidecar(path, matrix);
    {
        std::ofstream output(path, std::ios::binary | std::ios::app);
        const char extra = 0;
        output.write(&extra, 1);
    }
    assert(!similarity.load(path, error));
    assert(similarity.empty());

    const char bad_magic[8] = {'B', 'A', 'D', 'M', 'A', 'G', 'I', 'C'};
    write_sidecar(path, matrix, bad_magic);
    assert(!similarity.load(path, error));
    assert(similarity.empty());
    assert(error == "invalid similarity sidecar header");

    write_sidecar(path, matrix, "MOESERE1", 2);
    assert(!similarity.load(path, error));
    assert(error == "unsupported similarity sidecar version: 2");

    write_sidecar(path, {}, "MOESERE1", 1, 0, 4);
    assert(!similarity.load(path, error));
    assert(error == "similarity sidecar dimensions must be non-zero");

    write_sidecar(
            path, {}, "MOESERE1", 1,
            std::numeric_limits<uint32_t>::max(),
            std::numeric_limits<uint32_t>::max());
    assert(!similarity.load(path, error));
    assert(error.find("overflow") != std::string::npos);

    std::remove(path);
    assert(!similarity.load(path, error));
    assert(error.find("failed to open") != std::string::npos);
    return 0;
#endif
}
