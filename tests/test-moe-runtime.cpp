#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama.h"
#include "moe-offload/runtime.h"

#include <cassert>
#include <string>

int main() {
    llama_moe::runtime_options defaults;
    assert(defaults.sere_policy == "paper");
    assert(!defaults.sere_shadow);
    assert(llama_moe::profile_summary_context{}.sere_policy == "paper");
    assert(!llama_moe::profile_summary_context{}.sere_shadow);

    const llama_model_params model_defaults = llama_model_default_params();
    assert(model_defaults.moe_sere_policy != nullptr);
    assert(std::string(model_defaults.moe_sere_policy) == "paper");
    assert(!model_defaults.moe_sere_shadow);

    llama_moe::runtime_options options;
    options.enabled = true;
    llama_moe::configure_runtime(options, {});

    // A single-token request without an explicit phase could be a prompt or
    // prefill tail. It must not be classified as decode, because that would
    // make it eligible for SERE rerouting.
    llama_moe::begin_request(1);
    assert(llama_moe::current_profile_request_row().phase == "unknown");
    llama_moe::end_request();

    // Multi-token requests remain safe to infer as prefill for callers that
    // do not provide benchmark phase metadata.
    llama_moe::set_profile_request_context(0, 0, nullptr);
    llama_moe::begin_request(8);
    assert(llama_moe::current_profile_request_row().phase == "prefill");
    llama_moe::end_request();

    llama_moe::set_profile_request_context(0, 1, "decode");
    llama_moe::begin_request(1);
    assert(llama_moe::current_profile_request_row().phase == "decode");
    llama_moe::end_request();

    // Phase metadata is scoped to one request. A missed context update after
    // decode must not leak decode eligibility into the next request.
    llama_moe::begin_request(1);
    assert(llama_moe::current_profile_request_row().phase == "unknown");
    llama_moe::end_request();

    llama_moe::set_profile_request_context(0, 2, "prefill");
    llama_moe::begin_request(1);
    assert(llama_moe::current_profile_request_row().phase == "prefill");
    llama_moe::end_request();

    const llama_moe::profile_snapshot snapshot = llama_moe::get_profile_snapshot();
    assert(snapshot.prefill.requests == 4);
    assert(snapshot.decode.requests == 1);

    llama_moe::configure_runtime({}, {});
    return 0;
}
