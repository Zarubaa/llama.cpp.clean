#include "sere.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace llama_moe {

namespace {

constexpr char SERE_MAGIC[8] = {'M', 'O', 'E', 'S', 'E', 'R', 'E', '1'};
constexpr uint32_t SERE_VERSION = 1;
constexpr uint64_t SERE_HEADER_SIZE = 8 + 4 * sizeof(uint32_t);

template<typename T>
bool read_value(std::ifstream & input, T & value) {
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    return input.good();
}

} // namespace

sere_policy parse_sere_policy(const std::string & value) {
    if (value == "paper") {
        return sere_policy::paper;
    }
    if (value == "miss") {
        return sere_policy::miss;
    }
    throw std::invalid_argument("invalid SERE policy: " + value);
}

const char * sere_policy_name(sere_policy policy) {
    switch (policy) {
        case sere_policy::paper: return "paper";
        case sere_policy::miss:  return "miss";
    }
    return "unknown";
}

bool sere_similarity_matrix::load(const std::string & path, std::string & error) {
    clear();
    error.clear();

    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        error = "failed to open similarity sidecar: " + path;
        return false;
    }
    const std::streamoff end = input.tellg();
    if (end < 0 || (uint64_t) end < SERE_HEADER_SIZE) {
        error = "similarity sidecar is shorter than its header";
        return false;
    }
    input.seekg(0, std::ios::beg);

    char magic[sizeof(SERE_MAGIC)] = {};
    uint32_t version = 0;
    uint32_t file_layers = 0;
    uint32_t file_experts = 0;
    uint32_t file_metric = 0;
    input.read(magic, sizeof(magic));
    if (!input || std::memcmp(magic, SERE_MAGIC, sizeof(magic)) != 0 ||
            !read_value(input, version) ||
            !read_value(input, file_layers) ||
            !read_value(input, file_experts) ||
            !read_value(input, file_metric)) {
        error = "invalid similarity sidecar header";
        return false;
    }
    if (version != SERE_VERSION) {
        error = "unsupported similarity sidecar version: " + std::to_string(version);
        return false;
    }
    if (file_layers == 0 || file_experts == 0) {
        error = "similarity sidecar dimensions must be non-zero";
        return false;
    }

    uint64_t matrix_count = file_layers;
    if (file_experts > std::numeric_limits<uint64_t>::max() / matrix_count) {
        error = "similarity sidecar dimensions overflow uint64";
        return false;
    }
    matrix_count *= file_experts;
    if (file_experts > std::numeric_limits<uint64_t>::max() / matrix_count) {
        error = "similarity sidecar dimensions overflow uint64";
        return false;
    }
    matrix_count *= file_experts;
    if (matrix_count > (uint64_t) std::numeric_limits<size_t>::max() / sizeof(float)) {
        error = "similarity sidecar dimensions overflow host address space";
        return false;
    }
    const uint64_t payload_size = matrix_count * sizeof(float);
    if (payload_size > std::numeric_limits<uint64_t>::max() - SERE_HEADER_SIZE ||
            payload_size > (uint64_t) std::numeric_limits<std::streamsize>::max()) {
        error = "similarity sidecar payload size overflows the file reader";
        return false;
    }
    const uint64_t expected_size = SERE_HEADER_SIZE + payload_size;
    if ((uint64_t) end != expected_size) {
        error = "similarity sidecar size mismatch: expected " + std::to_string(expected_size) +
            " bytes, got " + std::to_string((uint64_t) end);
        return false;
    }

    std::vector<float> loaded((size_t) matrix_count);
    input.read(reinterpret_cast<char *>(loaded.data()), (std::streamsize) payload_size);
    if (!input) {
        error = "failed to read similarity sidecar matrix data";
        return false;
    }
    for (float value : loaded) {
        if (!std::isfinite(value)) {
            error = "similarity sidecar contains a non-finite value";
            return false;
        }
    }

    layers = file_layers;
    experts = file_experts;
    metric_id = file_metric;
    values = std::move(loaded);
    return true;
}

void sere_similarity_matrix::clear() {
    layers = 0;
    experts = 0;
    metric_id = 0;
    values.clear();
}

bool sere_similarity_matrix::empty() const {
    return values.empty();
}

uint32_t sere_similarity_matrix::n_layers() const {
    return layers;
}

uint32_t sere_similarity_matrix::n_experts() const {
    return experts;
}

uint32_t sere_similarity_matrix::metric() const {
    return metric_id;
}

float sere_similarity_matrix::at(uint32_t layer, uint32_t source, uint32_t target) const {
    if (layer >= layers || source >= experts || target >= experts) {
        throw std::out_of_range("SERE similarity index out of range");
    }
    return values[((size_t) layer * experts + source) * experts + target];
}

std::vector<int32_t> sere_reroute_decode(
        const sere_similarity_matrix & similarity,
        uint32_t layer,
        const std::vector<int32_t> & original_ids,
        uint32_t select_top_k,
        float threshold,
        sere_policy policy,
        const std::vector<uint8_t> & resident,
        sere_route_stats & stats) {
    stats = {};
    std::vector<int32_t> result = original_ids;
    if (similarity.empty() || layer >= similarity.n_layers() ||
            original_ids.empty() || select_top_k == 0) {
        return result;
    }

    const uint32_t n_experts = similarity.n_experts();
    std::vector<uint8_t> seen(n_experts, 0);
    for (int32_t expert : original_ids) {
        if (expert < 0 || (uint32_t) expert >= n_experts) {
            continue;
        }
        const bool is_resident = (size_t) expert < resident.size() && resident[(size_t) expert] != 0;
        if (!is_resident) {
            ++stats.original_miss_routes;
        }
        if (!seen[(size_t) expert]) {
            seen[(size_t) expert] = 1;
            ++stats.original_unique_required;
            if (!is_resident) {
                ++stats.original_unique_misses;
            }
        }
    }
    const size_t primary_count = std::min<size_t>(select_top_k, original_ids.size());
    std::vector<int32_t> primary;
    primary.reserve(primary_count);
    std::vector<uint8_t> is_primary(n_experts, 0);
    for (size_t i = 0; i < primary_count; ++i) {
        const int32_t expert = original_ids[i];
        if (expert < 0 || (uint32_t) expert >= n_experts || is_primary[(size_t) expert]) {
            continue;
        }
        is_primary[(size_t) expert] = 1;
        primary.push_back(expert);
    }
    if (primary.empty()) {
        return result;
    }
    // Match the paper CUDA kernel, which scans its primary mask by expert ID.
    std::sort(primary.begin(), primary.end());

    for (size_t i = primary_count; i < original_ids.size(); ++i) {
        const int32_t source = original_ids[i];
        if (source < 0 || (uint32_t) source >= n_experts || is_primary[(size_t) source]) {
            continue;
        }

        ++stats.secondary_routes;
        const bool source_resident = (size_t) source < resident.size() && resident[(size_t) source] != 0;
        if (policy == sere_policy::miss && source_resident) {
            continue;
        }

        int32_t best = primary.front();
        float best_similarity = similarity.at(layer, (uint32_t) source, (uint32_t) best);
        for (size_t j = 1; j < primary.size(); ++j) {
            const int32_t candidate = primary[j];
            const float candidate_similarity = similarity.at(layer, (uint32_t) source, (uint32_t) candidate);
            if (candidate_similarity > best_similarity) {
                best = candidate;
                best_similarity = candidate_similarity;
            }
        }

        if (threshold > 0.0f && best_similarity < threshold) {
            ++stats.threshold_rejects;
            continue;
        }
        if (best != source) {
            result[i] = best;
            ++stats.rerouted_routes;
            if (!source_resident) {
                ++stats.rerouted_miss_routes;
            }
            stats.similarity_sum += best_similarity;
        }
    }
    return result;
}

} // namespace llama_moe
