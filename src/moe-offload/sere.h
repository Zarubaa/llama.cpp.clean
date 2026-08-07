#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llama_moe {

enum class sere_policy {
    paper,
    miss,
};

sere_policy parse_sere_policy(const std::string & value);
const char * sere_policy_name(sere_policy policy);

class sere_similarity_matrix {
public:
    bool load(const std::string & path, std::string & error);
    void clear();

    bool empty() const;
    uint32_t n_layers() const;
    uint32_t n_experts() const;
    uint32_t metric() const;
    float at(uint32_t layer, uint32_t source, uint32_t target) const;

private:
    uint32_t layers = 0;
    uint32_t experts = 0;
    uint32_t metric_id = 0;
    std::vector<float> values;
};

struct sere_route_stats {
    // Secondary routes that are not themselves members of the primary set.
    uint64_t secondary_routes = 0;
    uint64_t rerouted_routes = 0;
    // All original top-k routes absent from the cache before rerouting.
    uint64_t original_miss_routes = 0;
    uint64_t original_unique_required = 0;
    uint64_t original_unique_misses = 0;
    uint64_t rerouted_miss_routes = 0;
    uint64_t threshold_rejects = 0;
    double similarity_sum = 0.0;
};

std::vector<int32_t> sere_reroute_decode(
        const sere_similarity_matrix & similarity,
        uint32_t layer,
        const std::vector<int32_t> & original_ids,
        uint32_t select_top_k,
        float threshold,
        sere_policy policy,
        const std::vector<uint8_t> & resident,
        sere_route_stats & stats);

} // namespace llama_moe
