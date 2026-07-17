#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llama_moe {

struct routed_expert {
    int32_t  id = -1;
    uint32_t count = 0;
    size_t   first_index = 0;
};

// Rank valid routed experts by descending token multiplicity. First occurrence
// and expert id provide deterministic tie breaks for a fixed top-k tensor.
std::vector<routed_expert> rank_routed_experts(
        const std::vector<int32_t> & ids,
        uint32_t n_experts);

} // namespace llama_moe
