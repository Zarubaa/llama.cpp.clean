#include "sere.h"

#include "loader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace llama_moe {

namespace {

constexpr char SERE_MAGIC_V1[8] = {'M', 'O', 'E', 'S', 'E', 'R', 'E', '1'};
constexpr char SERE_MAGIC_V2[8] = {'M', 'O', 'E', 'S', 'E', 'R', 'E', '2'};
constexpr char SERE_MANIFEST_DOMAIN[] = "llama.cpp-sere-model-manifest-v1";

struct sha256_context {
    std::array<uint32_t, 8> state{{
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    }};
    std::array<uint8_t, 64> block{};
    uint64_t total_bytes = 0;
    size_t block_size = 0;
};

constexpr std::array<uint32_t, 64> SHA256_K{{
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
}};

uint32_t rotate_right(uint32_t value, uint32_t bits) {
    return (value >> bits) | (value << (32 - bits));
}

void sha256_transform(sha256_context & ctx, const uint8_t * block) {
    std::array<uint32_t, 64> words{};
    for (size_t i = 0; i < 16; ++i) {
        words[i] = ((uint32_t) block[4 * i + 0] << 24) |
                   ((uint32_t) block[4 * i + 1] << 16) |
                   ((uint32_t) block[4 * i + 2] << 8) |
                   ((uint32_t) block[4 * i + 3]);
    }
    for (size_t i = 16; i < words.size(); ++i) {
        const uint32_t s0 = rotate_right(words[i - 15], 7) ^
                            rotate_right(words[i - 15], 18) ^
                            (words[i - 15] >> 3);
        const uint32_t s1 = rotate_right(words[i - 2], 17) ^
                            rotate_right(words[i - 2], 19) ^
                            (words[i - 2] >> 10);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    uint32_t a = ctx.state[0];
    uint32_t b = ctx.state[1];
    uint32_t c = ctx.state[2];
    uint32_t d = ctx.state[3];
    uint32_t e = ctx.state[4];
    uint32_t f = ctx.state[5];
    uint32_t g = ctx.state[6];
    uint32_t h = ctx.state[7];
    for (size_t i = 0; i < words.size(); ++i) {
        const uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const uint32_t choice = (e & f) ^ ((~e) & g);
        const uint32_t temp1 = h + sum1 + choice + SHA256_K[i] + words[i];
        const uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    ctx.state[0] += a;
    ctx.state[1] += b;
    ctx.state[2] += c;
    ctx.state[3] += d;
    ctx.state[4] += e;
    ctx.state[5] += f;
    ctx.state[6] += g;
    ctx.state[7] += h;
}

void sha256_update(sha256_context & ctx, const uint8_t * data, size_t size) {
    ctx.total_bytes += size;
    while (size > 0) {
        const size_t count = std::min(size, ctx.block.size() - ctx.block_size);
        std::memcpy(ctx.block.data() + ctx.block_size, data, count);
        ctx.block_size += count;
        data += count;
        size -= count;
        if (ctx.block_size == ctx.block.size()) {
            sha256_transform(ctx, ctx.block.data());
            ctx.block_size = 0;
        }
    }
}

std::array<uint8_t, 32> sha256_final(sha256_context & ctx) {
    const uint64_t total_bits = ctx.total_bytes * 8;
    ctx.block[ctx.block_size++] = 0x80;
    if (ctx.block_size > 56) {
        std::fill(ctx.block.begin() + ctx.block_size, ctx.block.end(), 0);
        sha256_transform(ctx, ctx.block.data());
        ctx.block_size = 0;
    }
    std::fill(ctx.block.begin() + ctx.block_size, ctx.block.begin() + 56, 0);
    for (size_t i = 0; i < 8; ++i) {
        ctx.block[63 - i] = (uint8_t) (total_bits >> (8 * i));
    }
    sha256_transform(ctx, ctx.block.data());

    std::array<uint8_t, 32> digest{};
    for (size_t i = 0; i < ctx.state.size(); ++i) {
        digest[4 * i + 0] = (uint8_t) (ctx.state[i] >> 24);
        digest[4 * i + 1] = (uint8_t) (ctx.state[i] >> 16);
        digest[4 * i + 2] = (uint8_t) (ctx.state[i] >> 8);
        digest[4 * i + 3] = (uint8_t) ctx.state[i];
    }
    return digest;
}

void append_u32(std::vector<uint8_t> & output, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        output.push_back((uint8_t) (value >> shift));
    }
}

void append_u64(std::vector<uint8_t> & output, uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        output.push_back((uint8_t) (value >> shift));
    }
}

bool append_string(std::vector<uint8_t> & output, const std::string & value) {
    if (value.size() > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    append_u32(output, (uint32_t) value.size());
    output.insert(output.end(), value.begin(), value.end());
    return true;
}

uint32_t read_u32_le(const uint8_t * data) {
    return (uint32_t) data[0] |
           ((uint32_t) data[1] << 8) |
           ((uint32_t) data[2] << 16) |
           ((uint32_t) data[3] << 24);
}

std::string digest_hex(const std::array<uint8_t, 32> & digest) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (uint8_t byte : digest) {
        out << std::setw(2) << (unsigned) byte;
    }
    return out.str();
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

bool make_sere_model_binding(
        const manifest & mf,
        sere_model_binding & binding,
        std::string & error) {
    binding = {};
    error.clear();
    if (!mf.present || mf.n_layers == 0 || mf.n_experts_per_layer == 0 || mf.n_expert_used == 0) {
        error = "model has incomplete MoE manifest dimensions or router top-k";
        return false;
    }
    if (mf.architecture.empty() || mf.model_name.empty() || mf.layout.empty()) {
        error = "model is missing architecture, name, or MoE layout metadata required by SERE V2";
        return false;
    }
    if (mf.source_size == 0 ||
            mf.general_file_type == std::numeric_limits<uint32_t>::max() ||
            mf.quantization_version == std::numeric_limits<uint32_t>::max()) {
        error = "model is missing file size or quantization metadata required by SERE V2";
        return false;
    }
    if (mf.layer_ids.size() != mf.n_layers) {
        error = "model MoE layer ID table does not match its declared layer count";
        return false;
    }
    if (mf.n_experts_per_layer > std::numeric_limits<size_t>::max() / mf.n_layers /
            EXPERT_KIND_COUNT) {
        error = "model expert manifest dimensions overflow host address space";
        return false;
    }
    const size_t tensor_count = (size_t) mf.n_layers * EXPERT_KIND_COUNT;
    const size_t expert_count = tensor_count * mf.n_experts_per_layer;
    if (mf.experts.size() != expert_count || mf.expert_tensors.size() != tensor_count) {
        error = "model expert blob or tensor layout table has an unexpected size";
        return false;
    }

    for (uint32_t logical = 0; logical < mf.n_layers; ++logical) {
        for (int kind = 0; kind < EXPERT_KIND_COUNT; ++kind) {
            const expert_tensor_layout & tensor =
                mf.expert_tensors[(size_t) logical * EXPERT_KIND_COUNT + (size_t) kind];
            const expert_record & first = mf.at(logical, 0, (expert_kind) kind);
            if (tensor.name.empty() || tensor.type == std::numeric_limits<uint32_t>::max() ||
                    tensor.size == 0 || first.size == 0 || tensor.rel_offset != first.rel_offset) {
                error = "model has an incomplete expert tensor layout entry";
                return false;
            }
            if (first.size > std::numeric_limits<uint64_t>::max() / mf.n_experts_per_layer ||
                    tensor.size != first.size * mf.n_experts_per_layer) {
                error = "model expert tensor size does not match its blob table";
                return false;
            }
            for (uint32_t expert = 0; expert < mf.n_experts_per_layer; ++expert) {
                const expert_record & record = mf.at(logical, expert, (expert_kind) kind);
                if (expert > (std::numeric_limits<uint64_t>::max() - first.rel_offset) / first.size ||
                        record.size != first.size ||
                        record.rel_offset != first.rel_offset + (uint64_t) expert * first.size) {
                    error = "model expert blob table is not a contiguous fused-tensor layout";
                    return false;
                }
            }
        }
    }

    std::vector<uint8_t> canonical;
    canonical.insert(canonical.end(), SERE_MANIFEST_DOMAIN,
            SERE_MANIFEST_DOMAIN + sizeof(SERE_MANIFEST_DOMAIN) - 1);
    append_u32(canonical, SERE_BINDING_SCHEMA);
    if (!append_string(canonical, mf.architecture) ||
            !append_string(canonical, mf.model_name) ||
            !append_string(canonical, mf.layout)) {
        error = "model identity string is too long for the SERE manifest";
        return false;
    }
    append_u32(canonical, mf.version);
    append_u32(canonical, mf.n_layers);
    append_u32(canonical, mf.n_experts_per_layer);
    append_u32(canonical, mf.n_expert_used);
    append_u32(canonical, mf.general_file_type);
    append_u32(canonical, mf.quantization_version);
    append_u64(canonical, mf.expert_blob_size_max);
    append_u64(canonical, mf.source_size);
    append_u32(canonical, (uint32_t) mf.layer_ids.size());
    for (uint32_t layer : mf.layer_ids) {
        append_u32(canonical, layer);
    }
    append_u32(canonical, (uint32_t) mf.expert_tensors.size());
    for (const expert_tensor_layout & tensor : mf.expert_tensors) {
        if (!append_string(canonical, tensor.name)) {
            error = "expert tensor name is too long for the SERE manifest";
            return false;
        }
        append_u32(canonical, tensor.type);
        append_u64(canonical, tensor.rel_offset);
        append_u64(canonical, tensor.size);
        for (uint64_t ne : tensor.ne) {
            append_u64(canonical, ne);
        }
    }
    append_u64(canonical, (uint64_t) mf.experts.size());
    for (const expert_record & record : mf.experts) {
        append_u64(canonical, record.rel_offset);
        append_u64(canonical, record.size);
    }

    sha256_context sha;
    sha256_update(sha, canonical.data(), canonical.size());
    binding.n_layers = mf.n_layers;
    binding.n_experts = mf.n_experts_per_layer;
    binding.model_top_k = mf.n_expert_used;
    binding.moe_manifest_version = mf.version;
    binding.general_file_type = mf.general_file_type;
    binding.quantization_version = mf.quantization_version;
    binding.manifest_sha256 = sha256_final(sha);
    return true;
}

bool sere_similarity_matrix::load(
        const std::string & path,
        const sere_model_binding & expected,
        std::string & error) {
    clear();
    error.clear();

    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        error = "failed to open similarity sidecar: " + path;
        return false;
    }
    const std::streamoff end = input.tellg();
    if (end < 0 || (uint64_t) end < sizeof(SERE_MAGIC_V2)) {
        error = "similarity sidecar is shorter than its header";
        return false;
    }
    input.seekg(0, std::ios::beg);

    std::array<uint8_t, SERE_SIDECAR_HEADER_SIZE> header{};
    input.read(reinterpret_cast<char *>(header.data()), sizeof(SERE_MAGIC_V2));
    if (!input) {
        error = "invalid similarity sidecar header";
        return false;
    }
    if (std::memcmp(header.data(), SERE_MAGIC_V1, sizeof(SERE_MAGIC_V1)) == 0) {
        error = "legacy MOESERE1 sidecar is not model-bound; re-export it as MOESERE2";
        return false;
    }
    if (std::memcmp(header.data(), SERE_MAGIC_V2, sizeof(SERE_MAGIC_V2)) != 0) {
        error = "invalid similarity sidecar header";
        return false;
    }
    if ((uint64_t) end < SERE_SIDECAR_HEADER_SIZE) {
        error = "similarity sidecar is shorter than its V2 header";
        return false;
    }
    input.read(reinterpret_cast<char *>(header.data()) + sizeof(SERE_MAGIC_V2),
            SERE_SIDECAR_HEADER_SIZE - sizeof(SERE_MAGIC_V2));
    if (!input) {
        error = "invalid similarity sidecar header";
        return false;
    }

    const uint32_t version = read_u32_le(header.data() + 8);
    const uint32_t header_size = read_u32_le(header.data() + 12);
    const uint32_t binding_schema = read_u32_le(header.data() + 16);
    const uint32_t file_layers = read_u32_le(header.data() + 20);
    const uint32_t file_experts = read_u32_le(header.data() + 24);
    const uint32_t file_metric = read_u32_le(header.data() + 28);
    const uint32_t file_top_k = read_u32_le(header.data() + 32);
    const uint32_t file_manifest_version = read_u32_le(header.data() + 36);
    const uint32_t file_type = read_u32_le(header.data() + 40);
    const uint32_t quantization_version = read_u32_le(header.data() + 44);
    std::array<uint8_t, 32> file_digest{};
    std::copy(header.begin() + 48, header.end(), file_digest.begin());

    if (version != SERE_SIDECAR_VERSION) {
        error = "unsupported similarity sidecar version: " + std::to_string(version);
        return false;
    }
    if (header_size != SERE_SIDECAR_HEADER_SIZE) {
        error = "unsupported similarity sidecar header size: " + std::to_string(header_size);
        return false;
    }
    if (binding_schema != SERE_BINDING_SCHEMA) {
        error = "unsupported SERE model binding schema: " + std::to_string(binding_schema);
        return false;
    }
    if (file_layers == 0 || file_experts == 0) {
        error = "similarity sidecar dimensions must be non-zero";
        return false;
    }
    if (file_layers != expected.n_layers || file_experts != expected.n_experts) {
        error = "SERE sidecar dimensions " + std::to_string(file_layers) + "x" +
            std::to_string(file_experts) + " do not match model " +
            std::to_string(expected.n_layers) + "x" + std::to_string(expected.n_experts);
        return false;
    }
    if (file_top_k != expected.model_top_k) {
        error = "SERE sidecar model top-k " + std::to_string(file_top_k) +
            " does not match model top-k " + std::to_string(expected.model_top_k);
        return false;
    }
    if (file_manifest_version != expected.moe_manifest_version) {
        error = "SERE sidecar MoE manifest version does not match model";
        return false;
    }
    if (file_type != expected.general_file_type ||
            quantization_version != expected.quantization_version) {
        error = "SERE sidecar quantization metadata does not match model";
        return false;
    }
    if (file_digest != expected.manifest_sha256) {
        error = "SERE sidecar manifest fingerprint does not match model: expected " +
            digest_hex(expected.manifest_sha256) + ", got " + digest_hex(file_digest);
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
    if (payload_size > std::numeric_limits<uint64_t>::max() - SERE_SIDECAR_HEADER_SIZE ||
            payload_size > (uint64_t) std::numeric_limits<std::streamsize>::max()) {
        error = "similarity sidecar payload size overflows the file reader";
        return false;
    }
    const uint64_t expected_size = SERE_SIDECAR_HEADER_SIZE + payload_size;
    if ((uint64_t) end != expected_size) {
        error = "similarity sidecar size mismatch: expected " + std::to_string(expected_size) +
            " bytes, got " + std::to_string((uint64_t) end);
        return false;
    }

    std::vector<uint8_t> payload((size_t) payload_size);
    input.read(reinterpret_cast<char *>(payload.data()), (std::streamsize) payload_size);
    if (!input) {
        error = "failed to read similarity sidecar matrix data";
        return false;
    }
    std::vector<float> loaded((size_t) matrix_count);
    for (size_t i = 0; i < loaded.size(); ++i) {
        const uint32_t bits = read_u32_le(payload.data() + i * sizeof(float));
        static_assert(sizeof(bits) == sizeof(loaded[i]), "SERE requires 32-bit IEEE float storage");
        std::memcpy(&loaded[i], &bits, sizeof(bits));
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
