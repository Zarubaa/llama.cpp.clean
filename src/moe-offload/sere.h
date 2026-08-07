#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace llama_moe {

struct manifest;

constexpr uint32_t SERE_SIDECAR_VERSION = 2;
constexpr uint32_t SERE_BINDING_SCHEMA = 1;
constexpr uint32_t SERE_SIDECAR_HEADER_SIZE = 80;

enum class sere_policy {
    paper,
    miss,
};

sere_policy parse_sere_policy(const std::string & value);
const char * sere_policy_name(sere_policy policy);

struct sere_model_binding {
    uint32_t n_layers = 0;
    uint32_t n_experts = 0;
    uint32_t model_top_k = 0;
    uint32_t moe_manifest_version = 0;
    uint32_t general_file_type = 0;
    uint32_t quantization_version = 0;
    std::array<uint8_t, 32> manifest_sha256{};
};

// Builds a content-independent identity from model/repack metadata. It binds
// architecture, router top-k and the quantized expert layout without reading
// the GGUF tensor payload.
bool make_sere_model_binding(
        const manifest & mf,
        sere_model_binding & binding,
        std::string & error);

class sere_similarity_matrix {
public:
    bool load(
            const std::string & path,
            const sere_model_binding & expected,
            std::string & error);
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
