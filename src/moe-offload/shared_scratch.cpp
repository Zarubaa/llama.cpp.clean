#include "shared_scratch.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace llama_moe {

uint32_t shared_scratch_plan::active_slot_axis() const {
    return n_persistent_slots + n_shared_scratch_slots;
}

uint32_t shared_scratch_plan::safe_ubatch_for_top_k(uint32_t top_k) const {
    if (top_k == 0) {
        return 0;
    }
    if (!expert_to_unified_slot.empty() && active_slot_axis() >= expert_to_unified_slot.size()) {
        return std::numeric_limits<uint32_t>::max();
    }
    return active_slot_axis() / top_k;
}

shared_scratch_planner::shared_scratch_planner(shared_scratch_config config) :
    config(config),
    layers((size_t) config.n_layers) {
    if (config.n_layers <= 0) {
        throw std::invalid_argument("MoE shared scratch requires n_layers > 0");
    }
    if (config.n_experts <= 0) {
        throw std::invalid_argument("MoE shared scratch requires n_experts > 0");
    }
    if (config.n_persistent_slots > (uint32_t) config.n_experts) {
        throw std::invalid_argument("MoE persistent slots exceed expert count");
    }

    n_scratch = config.n_shared_scratch_slots;
    if (n_scratch == 0 && config.fill_remaining_with_scratch) {
        n_scratch = (uint32_t) config.n_experts - config.n_persistent_slots;
    }
    if (config.n_persistent_slots + n_scratch == 0) {
        throw std::invalid_argument("MoE shared scratch requires at least one active slot");
    }
    if (config.n_persistent_slots + n_scratch > (uint32_t) config.n_experts) {
        throw std::invalid_argument("MoE active slot axis exceeds expert count");
    }

    reset();
}

void shared_scratch_planner::reset() {
    step = 0;
    for (auto & layer : layers) {
        layer.slot_to_expert.assign((size_t) config.n_persistent_slots, -1);
        layer.expert_to_slot.assign((size_t) config.n_experts, -1);
        layer.last_use.assign((size_t) config.n_persistent_slots, 0);
    }
}

void shared_scratch_planner::seed_persistent(int32_t layer, const std::vector<int32_t> & experts) {
    validate_layer(layer);

    auto & state = layers[(size_t) layer];
    std::fill(state.slot_to_expert.begin(), state.slot_to_expert.end(), -1);
    std::fill(state.expert_to_slot.begin(), state.expert_to_slot.end(), -1);
    std::fill(state.last_use.begin(), state.last_use.end(), 0);

    uint32_t slot = 0;
    for (int32_t expert : experts) {
        if (slot >= config.n_persistent_slots) {
            break;
        }
        if (expert < 0 || expert >= config.n_experts) {
            continue;
        }
        if (state.expert_to_slot[(size_t) expert] >= 0) {
            continue;
        }
        state.slot_to_expert[(size_t) slot] = expert;
        state.expert_to_slot[(size_t) expert] = (int32_t) slot;
        state.last_use[(size_t) slot] = ++step;
        ++slot;
    }
}

shared_scratch_plan shared_scratch_planner::plan_layer(
        int32_t layer, const std::vector<int32_t> & selected_experts) const {
    return plan_layer(layer, selected_experts.data(), selected_experts.size());
}

shared_scratch_plan shared_scratch_planner::plan_layer(
        int32_t layer, const int32_t * selected_experts, size_t n_selected) const {
    validate_layer(layer);

    shared_scratch_plan plan;
    plan.layer                  = layer;
    plan.n_persistent_slots     = config.n_persistent_slots;
    plan.n_shared_scratch_slots = n_scratch;
    plan.expert_to_unified_slot.assign((size_t) config.n_experts, -1);

    if (n_selected > 0 && selected_experts == nullptr) {
        plan.ok = false;
        plan.error = "selected_experts is null";
        return plan;
    }

    std::vector<bool> seen((size_t) config.n_experts, false);
    for (size_t i = 0; i < n_selected; ++i) {
        const int32_t expert = selected_experts[i];
        if (expert < 0 || expert >= config.n_experts) {
            continue;
        }
        if (!seen[(size_t) expert]) {
            seen[(size_t) expert] = true;
            plan.unique_experts.push_back(expert);
        }
    }
    plan.n_unique = (uint32_t) plan.unique_experts.size();

    if (plan.n_unique > active_slot_axis()) {
        plan.ok = false;
        plan.error = "unique experts exceed persistent + shared scratch slot axis";
        return plan;
    }

    const auto & state = layers[(size_t) layer];
    std::vector<bool> protected_slots((size_t) config.n_persistent_slots, false);
    std::vector<int32_t> tmp_slot_to_expert = state.slot_to_expert;
    std::vector<uint64_t> tmp_last_use = state.last_use;

    uint32_t next_scratch = 0;

    for (int32_t expert : plan.unique_experts) {
        const int32_t resident_slot = state.expert_to_slot[(size_t) expert];
        if (resident_slot >= 0) {
            protected_slots[(size_t) resident_slot] = true;
            plan.expert_to_unified_slot[(size_t) expert] = resident_slot;
            plan.assignments.push_back(shared_slot_assignment{
                expert,
                shared_slot_kind::persistent,
                (uint32_t) resident_slot,
                (uint32_t) resident_slot,
                true,
                false,
                -1,
            });
            ++plan.n_hits;
            continue;
        }

        ++plan.n_misses;

        slot_pick dst = pick_empty_persistent_slot(
                layer_state{ tmp_slot_to_expert, state.expert_to_slot, tmp_last_use },
                protected_slots);
        if (dst.ok) {
            protected_slots[(size_t) dst.slot] = true;
            tmp_slot_to_expert[(size_t) dst.slot] = expert;
            tmp_last_use[(size_t) dst.slot] = std::numeric_limits<uint64_t>::max();
            plan.expert_to_unified_slot[(size_t) expert] = (int32_t) dst.slot;
            plan.assignments.push_back(shared_slot_assignment{
                expert,
                shared_slot_kind::persistent,
                dst.slot,
                dst.slot,
                false,
                true,
                dst.evicted,
            });
            ++plan.n_persistent_loads;
            continue;
        }

        if (next_scratch < n_scratch) {
            const uint32_t unified = config.n_persistent_slots + next_scratch;
            plan.expert_to_unified_slot[(size_t) expert] = (int32_t) unified;
            plan.assignments.push_back(shared_slot_assignment{
                expert,
                shared_slot_kind::shared_scratch,
                next_scratch,
                unified,
                false,
                true,
                -1,
            });
            ++next_scratch;
            ++plan.n_shared_scratch_loads;
            continue;
        }

        dst = pick_lru_persistent_slot(
                layer_state{ tmp_slot_to_expert, state.expert_to_slot, tmp_last_use },
                protected_slots);
        if (!dst.ok) {
            plan.ok = false;
            plan.error = "no evictable persistent slot after shared scratch filled";
            return plan;
        }

        protected_slots[(size_t) dst.slot] = true;
        tmp_slot_to_expert[(size_t) dst.slot] = expert;
        tmp_last_use[(size_t) dst.slot] = std::numeric_limits<uint64_t>::max();
        plan.expert_to_unified_slot[(size_t) expert] = (int32_t) dst.slot;
        plan.assignments.push_back(shared_slot_assignment{
            expert,
            shared_slot_kind::persistent,
            dst.slot,
            dst.slot,
            false,
            true,
            dst.evicted,
        });
        ++plan.n_persistent_loads;
    }

    return plan;
}

bool shared_scratch_planner::commit_layer(const shared_scratch_plan & plan) {
    if (!plan.ok) {
        return false;
    }
    validate_layer(plan.layer);

    auto & state = layers[(size_t) plan.layer];
    ++step;
    for (const auto & assignment : plan.assignments) {
        if (assignment.kind != shared_slot_kind::persistent) {
            continue;
        }
        if (assignment.slot >= config.n_persistent_slots) {
            return false;
        }
        if (assignment.expert < 0 || assignment.expert >= config.n_experts) {
            return false;
        }
        if (assignment.evicted >= 0 && assignment.evicted < config.n_experts) {
            state.expert_to_slot[(size_t) assignment.evicted] = -1;
        }
        state.slot_to_expert[(size_t) assignment.slot] = assignment.expert;
        state.expert_to_slot[(size_t) assignment.expert] = (int32_t) assignment.slot;
        state.last_use[(size_t) assignment.slot] = step;
    }
    return true;
}

bool shared_scratch_planner::write_unified_slot_ids(
        const shared_scratch_plan & plan,
        const int32_t             * selected_experts,
        size_t                      n_selected,
        int32_t                   * out_slot_ids) const {
    if (!plan.ok || (n_selected > 0 && (selected_experts == nullptr || out_slot_ids == nullptr))) {
        return false;
    }

    for (size_t i = 0; i < n_selected; ++i) {
        const int32_t expert = selected_experts[i];
        if (expert < 0 || expert >= (int32_t) plan.expert_to_unified_slot.size()) {
            return false;
        }
        const int32_t slot = plan.expert_to_unified_slot[(size_t) expert];
        if (slot < 0) {
            return false;
        }
        out_slot_ids[i] = slot;
    }
    return true;
}

uint32_t shared_scratch_planner::active_slot_axis() const {
    return config.n_persistent_slots + n_scratch;
}

uint32_t shared_scratch_planner::safe_ubatch_for_top_k(uint32_t top_k) const {
    if (top_k == 0) {
        return 0;
    }
    if (active_slot_axis() >= (uint32_t) config.n_experts) {
        return std::numeric_limits<uint32_t>::max();
    }
    return active_slot_axis() / top_k;
}

uint32_t shared_scratch_planner::persistent_slots() const {
    return config.n_persistent_slots;
}

uint32_t shared_scratch_planner::shared_scratch_slots() const {
    return n_scratch;
}

std::vector<int32_t> shared_scratch_planner::persistent_residency(int32_t layer) const {
    validate_layer(layer);
    return layers[(size_t) layer].slot_to_expert;
}

shared_scratch_planner::slot_pick shared_scratch_planner::pick_empty_persistent_slot(
        const layer_state       & state,
        const std::vector<bool> & protected_slots) const {
    for (uint32_t slot = 0; slot < config.n_persistent_slots; ++slot) {
        if (protected_slots[(size_t) slot]) {
            continue;
        }
        if (state.slot_to_expert[(size_t) slot] < 0) {
            return slot_pick{ true, slot, -1 };
        }
    }
    return {};
}

shared_scratch_planner::slot_pick shared_scratch_planner::pick_lru_persistent_slot(
        const layer_state       & state,
        const std::vector<bool> & protected_slots) const {
    uint32_t best_slot = 0;
    uint64_t best_use = std::numeric_limits<uint64_t>::max();
    bool found = false;

    for (uint32_t slot = 0; slot < config.n_persistent_slots; ++slot) {
        if (protected_slots[(size_t) slot]) {
            continue;
        }
        const uint64_t last_use = state.last_use[(size_t) slot];
        if (!found || last_use < best_use) {
            found = true;
            best_slot = slot;
            best_use = last_use;
        }
    }

    if (!found) {
        return {};
    }
    return slot_pick{ true, best_slot, state.slot_to_expert[(size_t) best_slot] };
}

void shared_scratch_planner::validate_layer(int32_t layer) const {
    if (layer < 0 || layer >= config.n_layers) {
        throw std::out_of_range("MoE layer out of range");
    }
}

} // namespace llama_moe
