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
    assert(header.size() == 54);
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

    std::remove(path);
    return 0;
#endif
}
