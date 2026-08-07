#ifdef NDEBUG
#undef NDEBUG
#endif

#include "moe-offload/profiler.h"

#if !defined(_WIN32)
#  include <unistd.h>
#endif

#include <cassert>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<std::string> split_csv(const std::string & line) {
    std::vector<std::string> fields;
    std::istringstream input(line);
    std::string field;
    while (std::getline(input, field, ',')) {
        fields.push_back(field);
    }
    if (!line.empty() && line.back() == ',') {
        fields.emplace_back();
    }
    return fields;
}

int main() {
#if defined(_WIN32)
    return 0;
#else
    char path[] = "/tmp/test-moe-profiler-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    close(fd);

    {
        llama_moe::profiler profile(path);
        llama_moe::profile_row layer;
        layer.phase = "prefill";
        layer.ssd_bytes = 123456;
        layer.ssd_reads = 7;
        layer.route_hash = 987654321;
        layer.sere_rerouted_routes = 3;
        layer.sere_original_unique_required = 8;
        layer.sere_original_unique_misses = 5;
        layer.sere_rerouted_miss_routes = 2;
        layer.sere_similarity_sum = 2.25;
        profile.record(layer);

        llama_moe::profile_request_row request;
        request.phase = "decode";
        profile.record_request(request);
        profile.flush();
    }

    std::ifstream input(path);
    std::string header_line;
    std::string layer_line;
    std::string request_line;
    assert(std::getline(input, header_line));
    assert(std::getline(input, layer_line));
    assert(std::getline(input, request_line));
    const auto header = split_csv(header_line);
    const auto layer = split_csv(layer_line);
    const auto request = split_csv(request_line);
    assert(header.size() == 62);
    assert(layer.size() == header.size());
    assert(request.size() == header.size());
    assert(header[51] == "ssd_bytes");
    assert(header[52] == "ssd_reads");
    assert(header[53] == "route_hash");
    assert(layer[51] == "123456");
    assert(layer[52] == "7");
    assert(layer[53] == "987654321");
    assert(request[51] == "0");
    assert(request[52] == "0");
    assert(request[53] == "0");
    assert(header[54] == "sere_secondary_routes");
    assert(header[55] == "sere_rerouted_routes");
    assert(header[57] == "sere_original_unique_required");
    assert(header[58] == "sere_original_unique_misses");
    assert(header[59] == "sere_rerouted_miss_routes");
    assert(header[61] == "sere_similarity_sum");
    assert(layer[55] == "3");
    assert(layer[57] == "8");
    assert(layer[58] == "5");
    assert(layer[59] == "2");
    assert(layer[61] == "2.25");
    assert(request[54] == "0");
    assert(request[61] == "0");

    llama_moe::profile_summary_context summary_ctx;
    summary_ctx.sere_top_k = 4;
    summary_ctx.sere_shadow = true;
    const std::string summary = llama_moe::format_summary(summary_ctx, {});
    assert(summary.find("mode=shadow") != std::string::npos);
    assert(summary.find("cache hit rate (decode, effective)") == std::string::npos);

    std::remove(path);
    return 0;
#endif
}
