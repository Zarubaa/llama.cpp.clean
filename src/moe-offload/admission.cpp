#include "admission.h"

#include <algorithm>

namespace llama_moe {

std::vector<routed_expert> rank_routed_experts(
        const std::vector<int32_t> & ids,
        uint32_t n_experts) {
    std::vector<uint32_t> counts(n_experts, 0);
    std::vector<size_t> first_indices(n_experts, 0);
    std::vector<int32_t> seen;
    seen.reserve(std::min(ids.size(), (size_t) n_experts));

    for (size_t i = 0; i < ids.size(); ++i) {
        const int32_t id = ids[i];
        if (id < 0 || (uint32_t) id >= n_experts) {
            continue;
        }
        if (counts[(size_t) id]++ == 0) {
            first_indices[(size_t) id] = i;
            seen.push_back(id);
        }
    }

    std::vector<routed_expert> ranked;
    ranked.reserve(seen.size());
    for (int32_t id : seen) {
        ranked.push_back({ id, counts[(size_t) id], first_indices[(size_t) id] });
    }
    std::sort(ranked.begin(), ranked.end(), [](const routed_expert & a, const routed_expert & b) {
        if (a.count != b.count) {
            return a.count > b.count;
        }
        if (a.first_index != b.first_index) {
            return a.first_index < b.first_index;
        }
        return a.id < b.id;
    });
    return ranked;
}

} // namespace llama_moe
