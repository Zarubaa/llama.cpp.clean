#ifdef NDEBUG
#undef NDEBUG
#endif

#include "moe-offload/sere.h"
#include "moe-offload/loader.h"

#include "ggml.h"
#include "gguf.h"

#if !defined(_WIN32)
#  include <unistd.h>
#endif

#include <cassert>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void write_u32(std::ofstream & output, uint32_t value) {
    const std::array<uint8_t, 4> bytes{{
        (uint8_t) value,
        (uint8_t) (value >> 8),
        (uint8_t) (value >> 16),
        (uint8_t) (value >> 24),
    }};
    output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

static void write_v2_sidecar(
        const char * path,
        const std::vector<float> & matrix,
        const llama_moe::sere_model_binding & binding,
        const char magic[8] = "MOESERE2",
        uint32_t version = llama_moe::SERE_SIDECAR_VERSION,
        uint32_t layers = 1,
        uint32_t experts = 4,
        uint32_t header_size = llama_moe::SERE_SIDECAR_HEADER_SIZE,
        uint32_t binding_schema = llama_moe::SERE_BINDING_SCHEMA) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(magic, 8);
    write_u32(output, version);
    write_u32(output, header_size);
    write_u32(output, binding_schema);
    write_u32(output, layers);
    write_u32(output, experts);
    write_u32(output, 1); // metric
    write_u32(output, binding.model_top_k);
    write_u32(output, binding.moe_manifest_version);
    write_u32(output, binding.general_file_type);
    write_u32(output, binding.quantization_version);
    output.write(reinterpret_cast<const char *>(binding.manifest_sha256.data()),
            binding.manifest_sha256.size());
    for (float value : matrix) {
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "test requires 32-bit float");
        std::memcpy(&bits, &value, sizeof(bits));
        write_u32(output, bits);
    }
    assert(output.good());
}

static void write_v1_sidecar(const char * path, const std::vector<float> & matrix) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write("MOESERE1", 8);
    write_u32(output, 1);
    write_u32(output, 1);
    write_u32(output, 4);
    write_u32(output, 1);
    output.write(reinterpret_cast<const char *>(matrix.data()),
            (std::streamsize) (matrix.size() * sizeof(float)));
    assert(output.good());
}

static llama_moe::manifest make_manifest(uint64_t size_scale = 1) {
    llama_moe::manifest mf;
    mf.present = true;
    mf.version = 2;
    mf.n_layers = 1;
    mf.n_experts_per_layer = 4;
    mf.n_expert_used = 4;
    mf.expert_blob_size_max = 32 * size_scale;
    mf.source_size = 123456 + size_scale;
    mf.general_file_type = 15;
    mf.quantization_version = 2;
    mf.architecture = "test-moe";
    mf.model_name = "test-model";
    mf.layout = "fused-tensors-page-aligned-v1";
    mf.layer_ids = {7};

    const std::array<uint64_t, 3> bases{{1000, 2000, 4000}};
    const std::array<uint64_t, 3> sizes{{16 * size_scale, 24 * size_scale, 32 * size_scale}};
    mf.experts.resize(1 * 4 * llama_moe::EXPERT_KIND_COUNT);
    mf.expert_tensors.resize(llama_moe::EXPERT_KIND_COUNT);
    for (int kind = 0; kind < llama_moe::EXPERT_KIND_COUNT; ++kind) {
        for (uint32_t expert = 0; expert < mf.n_experts_per_layer; ++expert) {
            const size_t index = ((size_t) expert * llama_moe::EXPERT_KIND_COUNT) + kind;
            mf.experts[index] = {
                bases[(size_t) kind] + expert * sizes[(size_t) kind],
                sizes[(size_t) kind],
            };
        }
        llama_moe::expert_tensor_layout & tensor = mf.expert_tensors[(size_t) kind];
        tensor.name = "blk.7.ffn_" + std::string(llama_moe::expert_kind_name((llama_moe::expert_kind) kind)) +
            "_exps.weight";
        tensor.type = kind == llama_moe::EXPERT_DOWN ? 13 : 12;
        tensor.rel_offset = bases[(size_t) kind];
        tensor.size = sizes[(size_t) kind] * mf.n_experts_per_layer;
        tensor.ne = {{512, 2048, mf.n_experts_per_layer, 1}};
    }
    return mf;
}

int main(int argc, char ** argv) {
#if defined(_WIN32)
    (void) argc;
    (void) argv;
    return 0;
#else
    if (argc == 3) {
        ggml_context * meta_ctx = nullptr;
        gguf_init_params params = {true, &meta_ctx};
        gguf_context * gguf = gguf_init_from_file(argv[1], params);
        if (!gguf || !meta_ctx) {
            std::fprintf(stderr, "failed to read GGUF metadata\n");
            return 1;
        }
        const llama_moe::manifest model_manifest = llama_moe::inspect_manifest(gguf, argv[1]);
        llama_moe::sere_model_binding model_binding;
        std::string model_error;
        const bool binding_ok = llama_moe::make_sere_model_binding(
                model_manifest, model_binding, model_error);
        llama_moe::sere_similarity_matrix model_similarity;
        const bool sidecar_ok = binding_ok && model_similarity.load(
                argv[2], model_binding, model_error);
        gguf_free(gguf);
        ggml_free(meta_ctx);
        if (!sidecar_ok) {
            std::fprintf(stderr, "%s\n", model_error.c_str());
            return 1;
        }
        std::printf("validated MOESERE2: layers=%u experts=%u model_top_k=%u metric=%u\n",
                model_similarity.n_layers(), model_similarity.n_experts(),
                model_binding.model_top_k, model_similarity.metric());
        return 0;
    }
    if (argc != 1) {
        std::fprintf(stderr, "usage: test-moe-sere [MODEL.gguf SIDECAR.sere]\n");
        return 1;
    }

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

    const llama_moe::manifest mf = make_manifest();
    llama_moe::sere_model_binding binding;
    std::string error;
    assert(llama_moe::make_sere_model_binding(mf, binding, error));
    assert(error.empty());
    assert(binding.n_layers == 1);
    assert(binding.n_experts == 4);
    assert(binding.model_top_k == 4);
    const std::array<uint8_t, 32> expected_manifest_sha256{{
        0x28, 0xb9, 0x63, 0xa3, 0x51, 0x38, 0xc7, 0xd4,
        0x23, 0x73, 0xbd, 0x05, 0xb9, 0xb1, 0x3a, 0x92,
        0x00, 0x8d, 0x75, 0xa1, 0xbd, 0x15, 0x8e, 0x81,
        0x8d, 0x2d, 0xa6, 0x04, 0x33, 0xa4, 0x5c, 0x15,
    }};
    assert(binding.manifest_sha256 == expected_manifest_sha256);
    write_v2_sidecar(path, matrix, binding);

    llama_moe::sere_similarity_matrix similarity;
    assert(similarity.load(path, binding, error));
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

    llama_moe::manifest other_mf = mf;
    llama_moe::sere_model_binding other_binding;
    other_mf.n_expert_used = 3;
    assert(llama_moe::make_sere_model_binding(other_mf, other_binding, error));
    assert(!similarity.load(path, other_binding, error));
    assert(error.find("top-k") != std::string::npos);

    other_mf = mf;
    other_mf.general_file_type = 14;
    assert(llama_moe::make_sere_model_binding(other_mf, other_binding, error));
    assert(!similarity.load(path, other_binding, error));
    assert(error.find("quantization metadata") != std::string::npos);

    other_mf = mf;
    other_mf.architecture = "other-moe";
    assert(llama_moe::make_sere_model_binding(other_mf, other_binding, error));
    assert(!similarity.load(path, other_binding, error));
    assert(error.find("fingerprint") != std::string::npos);

    other_mf = mf;
    other_mf.layout = "other-layout";
    assert(llama_moe::make_sere_model_binding(other_mf, other_binding, error));
    assert(!similarity.load(path, other_binding, error));
    assert(error.find("fingerprint") != std::string::npos);

    other_mf = make_manifest(2);
    assert(llama_moe::make_sere_model_binding(other_mf, other_binding, error));
    assert(!similarity.load(path, other_binding, error));
    assert(error.find("fingerprint") != std::string::npos);

    matrix[0] = std::numeric_limits<float>::quiet_NaN();
    write_v2_sidecar(path, matrix, binding);
    assert(!similarity.load(path, binding, error));
    assert(similarity.empty());
    assert(!error.empty());

    matrix[0] = 1.0f;
    write_v2_sidecar(path, matrix, binding);
    {
        std::ofstream output(path, std::ios::binary | std::ios::app);
        const char extra = 0;
        output.write(&extra, 1);
    }
    assert(!similarity.load(path, binding, error));
    assert(similarity.empty());

    const char bad_magic[8] = {'B', 'A', 'D', 'M', 'A', 'G', 'I', 'C'};
    write_v2_sidecar(path, matrix, binding, bad_magic);
    assert(!similarity.load(path, binding, error));
    assert(similarity.empty());
    assert(error == "invalid similarity sidecar header");

    write_v1_sidecar(path, matrix);
    assert(!similarity.load(path, binding, error));
    assert(error.find("not model-bound") != std::string::npos);

    write_v2_sidecar(path, matrix, binding, "MOESERE2", 3);
    assert(!similarity.load(path, binding, error));
    assert(error == "unsupported similarity sidecar version: 3");

    write_v2_sidecar(path, matrix, binding, "MOESERE2", 2, 1, 4, 79);
    assert(!similarity.load(path, binding, error));
    assert(error == "unsupported similarity sidecar header size: 79");

    write_v2_sidecar(path, matrix, binding, "MOESERE2", 2, 1, 4, 80, 2);
    assert(!similarity.load(path, binding, error));
    assert(error == "unsupported SERE model binding schema: 2");

    write_v2_sidecar(path, {}, binding, "MOESERE2", 2, 0, 4);
    llama_moe::sere_model_binding zero_layers = binding;
    zero_layers.n_layers = 0;
    assert(!similarity.load(path, zero_layers, error));
    assert(error == "similarity sidecar dimensions must be non-zero");

    write_v2_sidecar(
            path, {}, binding, "MOESERE2", 2,
            std::numeric_limits<uint32_t>::max(),
            std::numeric_limits<uint32_t>::max());
    llama_moe::sere_model_binding huge = binding;
    huge.n_layers = std::numeric_limits<uint32_t>::max();
    huge.n_experts = std::numeric_limits<uint32_t>::max();
    assert(!similarity.load(path, huge, error));
    assert(error.find("overflow") != std::string::npos);

    std::remove(path);
    assert(!similarity.load(path, binding, error));
    assert(error.find("failed to open") != std::string::npos);
    return 0;
#endif
}
