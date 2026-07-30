#include "page-cache.h"

#if !defined(_WIN32)
#  include <fcntl.h>
#  include <unistd.h>
#endif

#include <cstdio>
#include <string>
#include <vector>

int main() {
#if defined(_WIN32)
    return 0;
#else
    char path[] = "/tmp/test-page-cache-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) {
        return 1;
    }
    std::vector<unsigned char> data(32ull * 1024ull * 1024ull, 0x5a);
    size_t written = 0;
    while (written < data.size()) {
        const ssize_t count = write(fd, data.data() + written, data.size() - written);
        if (count <= 0) {
            close(fd);
            std::remove(path);
            return 2;
        }
        written += (size_t) count;
    }
    if (fsync(fd) != 0) {
        close(fd);
        std::remove(path);
        return 3;
    }
    close(fd);

    page_cache_prepare_result result;
    bool ok = page_cache_prepare(path, page_cache_policy::hot, result);
    ok = ok && result.after.valid && page_cache_resident_percent(result.after) >= 99.0;
    ok = ok && result.bytes_read == data.size();

    page_cache_prepare_result natural;
    ok = ok && page_cache_prepare(path, page_cache_policy::natural, natural);
    ok = ok && natural.attempts == 0;

    page_cache_prepare_result cold;
    ok = ok && page_cache_prepare(path, page_cache_policy::cold, cold);
    ok = ok && cold.after.valid && page_cache_resident_percent(cold.after) <= 1.0;

    page_cache_prepare_result missing;
    ok = ok && !page_cache_prepare(std::string(path) + ".missing", page_cache_policy::cold, missing);

    std::remove(path);
    return ok ? 0 : 4;
#endif
}
