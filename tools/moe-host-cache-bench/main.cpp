#include "moe-offload/io.h"
#include "moe-offload/loader.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
#  define bench_fseek(f, off, whence) _fseeki64((f), (off), (whence))
#else
#  define bench_fseek(f, off, whence) fseeko((f), (off), (whence))
#endif

namespace {

struct params {
    std::string model;
    std::string output;
    int samples = 256;
    int warmups = 3;
    int iterations = 20;
};

struct sample_blob {
    uint64_t offset;
    size_t size;
    std::vector<uint8_t> data;
};

bool parse_args(int argc, char ** argv, params & p) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char * name) -> const char * {
            return arg == name && i + 1 < argc ? argv[++i] : nullptr;
        };
        if (const char * v = value("--model")) p.model = v;
        else if (const char * v = value("--output")) p.output = v;
        else if (const char * v = value("--samples")) p.samples = std::atoi(v);
        else if (const char * v = value("--warmups")) p.warmups = std::atoi(v);
        else if (const char * v = value("--iterations")) p.iterations = std::atoi(v);
    }
    return !p.model.empty() && !p.output.empty() && p.samples > 0 && p.warmups >= 0 && p.iterations > 0;
}

int64_t host_elapsed_us(
        const std::chrono::steady_clock::time_point & begin,
        const std::chrono::steady_clock::time_point & end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
}

void write_row(std::ofstream & out, const char * op, int iteration, size_t sample, size_t bytes, double usec) {
    const double gib_s = usec > 0.0 ? (double) bytes / (1024.0 * 1024.0 * 1024.0) / (usec / 1e6) : 0.0;
    out << op << ',' << iteration << ',' << sample << ',' << bytes << ',' << usec << ',' << gib_s << '\n';
}

bool timed_h2d(void * dst, const void * src, size_t bytes, double & usec) {
    void * begin = nullptr;
    void * end = nullptr;
    if (!llama_moe::io_h2d_async_timed(dst, src, bytes, &begin, &end)) {
        return false;
    }
    const int64_t elapsed = llama_moe::io_event_elapsed_us(begin, end);
    llama_moe::io_event_release(begin);
    llama_moe::io_event_release(end);
    if (elapsed < 0) {
        return false;
    }
    usec = (double) elapsed;
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    params p;
    if (!parse_args(argc, argv, p)) {
        std::cerr << "usage: llama-moe-host-cache-bench --model PATH --output CSV [--samples 256] [--warmups 3] [--iterations 20]\n";
        return 1;
    }

    ggml_context * meta_ctx = nullptr;
    gguf_init_params init_params = {true, &meta_ctx};
    gguf_context * gguf = gguf_init_from_file(p.model.c_str(), init_params);
    if (!gguf || !meta_ctx) {
        std::cerr << "failed to read GGUF metadata\n";
        return 1;
    }
    const llama_moe::manifest mf = llama_moe::inspect_manifest(gguf, p.model);
    if (!mf.present || mf.experts.empty()) {
        std::cerr << "model has no MoE offload manifest\n";
        gguf_free(gguf);
        ggml_free(meta_ctx);
        return 1;
    }

    std::vector<size_t> valid;
    for (size_t i = 0; i < mf.experts.size(); ++i) {
        if (mf.experts[i].size > 0) valid.push_back(i);
    }
    const size_t n_samples = std::min((size_t) p.samples, valid.size());
    std::vector<sample_blob> samples;
    samples.reserve(n_samples);
    FILE * source = fopen(p.model.c_str(), "rb");
    if (!source) {
        return 1;
    }
    size_t max_size = 0;
    for (size_t i = 0; i < n_samples; ++i) {
        const size_t selected = valid[n_samples == 1 ? 0 : i * (valid.size() - 1) / (n_samples - 1)];
        const llama_moe::expert_record & record = mf.experts[selected];
        sample_blob blob{mf.data_offset + record.rel_offset, (size_t) record.size, {}};
        blob.data.resize(blob.size);
        if (bench_fseek(source, (int64_t) blob.offset, SEEK_SET) != 0 ||
                fread(blob.data.data(), 1, blob.size, source) != blob.size) {
            std::cerr << "failed to preload sample blob\n";
            fclose(source);
            return 1;
        }
        max_size = std::max(max_size, blob.size);
        samples.push_back(std::move(blob));
    }

    llama_backend_init();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        std::cerr << "no GPU backend\n";
        fclose(source);
        return 1;
    }
    ggml_backend_buffer_t device_buffer = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), max_size);
    void * device_dst = device_buffer ? ggml_backend_buffer_get_base(device_buffer) : nullptr;
    void * pinned = llama_moe::io_pinned_alloc(max_size);
    if (!device_dst || !pinned) {
        std::cerr << "failed to allocate benchmark buffers\n";
        if (device_buffer) ggml_backend_buffer_free(device_buffer);
        fclose(source);
        return 1;
    }

    std::ofstream out(p.output, std::ios::out | std::ios::trunc);
    out << "operation,iteration,sample,bytes,latency_us,bandwidth_gib_s\n";
    std::vector<size_t> order(samples.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(20260729);
    const int total_iterations = p.warmups + p.iterations;
    for (int iteration = 0; iteration < total_iterations; ++iteration) {
        std::shuffle(order.begin(), order.end(), rng);
        const bool measured = iteration >= p.warmups;
        const int measured_iteration = iteration - p.warmups;
        for (size_t index : order) {
            const sample_blob & blob = samples[index];

            auto begin = std::chrono::steady_clock::now();
            if (bench_fseek(source, (int64_t) blob.offset, SEEK_SET) != 0 ||
                    fread(pinned, 1, blob.size, source) != blob.size) {
                std::cerr << "fread benchmark failed\n";
                return 1;
            }
            auto end = std::chrono::steady_clock::now();
            if (measured) write_row(out, "fread-pinned", measured_iteration, index, blob.size, host_elapsed_us(begin, end));

            begin = std::chrono::steady_clock::now();
            memcpy(pinned, blob.data.data(), blob.size);
            end = std::chrono::steady_clock::now();
            const double copy_us = (double) host_elapsed_us(begin, end);
            if (measured) write_row(out, "pageable-pinned", measured_iteration, index, blob.size, copy_us);

            double h2d_us = 0.0;
            if (!timed_h2d(device_dst, pinned, blob.size, h2d_us)) {
                std::cerr << "H2D benchmark failed\n";
                return 1;
            }
            if (measured) {
                write_row(out, "pinned-h2d", measured_iteration, index, blob.size, h2d_us);
                write_row(out, "pageable-staging-h2d", measured_iteration, index, blob.size, copy_us + h2d_us);
            }
        }
    }

    out.close();
    llama_moe::io_pinned_free(pinned);
    ggml_backend_buffer_free(device_buffer);
    llama_backend_free();
    fclose(source);
    gguf_free(gguf);
    ggml_free(meta_ctx);
    std::cout << "samples=" << samples.size() << " iterations=" << p.iterations << " output=" << p.output << '\n';
    return out.fail() ? 1 : 0;
}
