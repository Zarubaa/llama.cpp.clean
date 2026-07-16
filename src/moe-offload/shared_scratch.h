#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llama_moe {

enum class shared_slot_kind : uint8_t {
    persistent,
    shared_scratch,
};

struct shared_scratch_config {
    int32_t  n_layers                 = 0;
    int32_t  n_experts                = 0;
    uint32_t n_persistent_slots       = 0;
    uint32_t n_shared_scratch_slots   = 0;

    // When true and n_shared_scratch_slots is zero, use the remaining expert
    // axis as the global miss buffer, e.g. 256 experts - 81 resident slots.
    bool fill_remaining_with_scratch  = true;
};

struct shared_slot_assignment {
    int32_t          expert       = -1;
    shared_slot_kind kind         = shared_slot_kind::persistent;
    uint32_t         slot         = 0;
    uint32_t         unified_slot = 0;
    bool             hit          = false;
    bool             needs_load   = false;
    int32_t          evicted      = -1;
};

struct shared_scratch_plan {
    bool        ok    = true;
    std::string error;

    int32_t  layer                  = -1;
    uint32_t n_unique               = 0;
    uint32_t n_hits                 = 0;
    uint32_t n_misses               = 0;
    uint32_t n_persistent_loads     = 0;
    uint32_t n_shared_scratch_loads = 0;
    uint32_t n_persistent_slots     = 0;
    uint32_t n_shared_scratch_slots = 0;

    // Unique expert IDs in first-use order for the current layer callback.
    std::vector<int32_t> unique_experts;

    // Maps original expert ID -> unified slot ID consumed by the downstream
    // expert op. Missing entries are -1.
    std::vector<int32_t> expert_to_unified_slot;

    // One row per unique expert. Entries with needs_load=true are the SSD/H2D
    // loads required before the expert op can execute.
    std::vector<shared_slot_assignment> assignments;

    uint32_t active_slot_axis() const;
    uint32_t safe_ubatch_for_top_k(uint32_t top_k) const;
};

class shared_scratch_planner {
public:
    explicit shared_scratch_planner(shared_scratch_config config);

    void reset();

    // Seed persistent slots for a layer. Extra or invalid expert IDs are ignored.
    void seed_persistent(int32_t layer, const std::vector<int32_t> & experts);

    shared_scratch_plan plan_layer(int32_t layer, const std::vector<int32_t> & selected_experts) const;
    shared_scratch_plan plan_layer(int32_t layer, const int32_t * selected_experts, size_t n_selected) const;

    // Applies persistent-slot changes from a successful plan. Shared scratch
    // assignments are intentionally transient and are not retained.
    bool commit_layer(const shared_scratch_plan & plan);

    bool write_unified_slot_ids(
            const shared_scratch_plan & plan,
            const int32_t             * selected_experts,
            size_t                      n_selected,
            int32_t                   * out_slot_ids) const;

    uint32_t active_slot_axis() const;
    uint32_t safe_ubatch_for_top_k(uint32_t top_k) const;

    uint32_t persistent_slots() const;
    uint32_t shared_scratch_slots() const;

    std::vector<int32_t> persistent_residency(int32_t layer) const;

private:
    struct layer_state {
        std::vector<int32_t> slot_to_expert;
        std::vector<int32_t> expert_to_slot;
        std::vector<uint64_t> last_use;
    };

    struct slot_pick {
        bool     ok      = false;
        uint32_t slot    = 0;
        int32_t  evicted = -1;
    };

    slot_pick pick_empty_persistent_slot(
            const layer_state       & state,
            const std::vector<bool> & protected_slots) const;

    slot_pick pick_lru_persistent_slot(
            const layer_state       & state,
            const std::vector<bool> & protected_slots) const;

    void validate_layer(int32_t layer) const;

    shared_scratch_config config;
    uint32_t              n_scratch = 0;
    uint64_t              step      = 0;
    std::vector<layer_state> layers;
};

} // namespace llama_moe
