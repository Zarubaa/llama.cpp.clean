#ifdef NDEBUG
#undef NDEBUG
#endif

#include "moe-offload/admission.h"

#include <cassert>
#include <cstdint>
#include <vector>

using llama_moe::rank_routed_experts;
using llama_moe::routed_expert;

static void assert_ranked(
        const std::vector<routed_expert> & actual,
        const std::vector<routed_expert> & expected) {
    assert(actual.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(actual[i].id == expected[i].id);
        assert(actual[i].count == expected[i].count);
        assert(actual[i].first_index == expected[i].first_index);
    }
}

int main() {
    assert_ranked(
            rank_routed_experts({ 3, 1, 3, 2, 1, 3 }, 4),
            { { 3, 3, 0 }, { 1, 2, 1 }, { 2, 1, 3 } });

    // Equal counts retain the order of first occurrence.
    assert_ranked(
            rank_routed_experts({ 4, 2, 3, 2, 3, 4 }, 5),
            { { 4, 2, 0 }, { 2, 2, 1 }, { 3, 2, 2 } });

    // Invalid ids neither contribute counts nor affect valid first positions.
    assert_ranked(
            rank_routed_experts({ -1, 2, 9, 1, 2 }, 4),
            { { 2, 2, 1 }, { 1, 1, 3 } });

    assert(rank_routed_experts({}, 4).empty());
    assert(rank_routed_experts({ 0, 1, 0 }, 0).empty());

    const std::vector<int32_t> ids = { 5, 1, 5, 3, 1, 3, 5, 2 };
    const auto first = rank_routed_experts(ids, 6);
    for (int i = 0; i < 32; ++i) {
        assert_ranked(rank_routed_experts(ids, 6), first);
    }

    return 0;
}
