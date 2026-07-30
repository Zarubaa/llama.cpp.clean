#include "page-cache.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#if !defined(_WIN32)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace {

constexpr size_t kReadBytes = 16ull * 1024ull * 1024ull;

int64_t elapsed_us(
        const std::chrono::steady_clock::time_point & begin,
        const std::chrono::steady_clock::time_point & end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
}

bool cold_enough(const page_cache_sample & sample) {
    return sample.valid && sample.total_bytes > 0 &&
        sample.resident_bytes * 100ull <= sample.total_bytes;
}

bool hot_enough(const page_cache_sample & sample) {
    return sample.valid && sample.total_bytes > 0 &&
        sample.resident_bytes * 100ull >= sample.total_bytes * 99ull;
}

} // namespace

bool page_cache_policy_from_string(const std::string & text, page_cache_policy & policy) {
    if (text == "natural") {
        policy = page_cache_policy::natural;
        return true;
    }
    if (text == "cold") {
        policy = page_cache_policy::cold;
        return true;
    }
    if (text == "hot") {
        policy = page_cache_policy::hot;
        return true;
    }
    return false;
}

const char * page_cache_policy_name(page_cache_policy policy) {
    switch (policy) {
        case page_cache_policy::natural: return "natural";
        case page_cache_policy::cold:    return "cold";
        case page_cache_policy::hot:     return "hot";
    }
    return "unknown";
}

page_cache_sample page_cache_residency(const std::string & path) {
    page_cache_sample sample;
#if defined(_WIN32)
    (void) path;
    return sample;
#else
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return sample;
    }
    struct stat st = {};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        return sample;
    }
    const long page_size_value = sysconf(_SC_PAGESIZE);
    if (page_size_value <= 0) {
        close(fd);
        return sample;
    }
    const size_t page_size = (size_t) page_size_value;
    const uint64_t file_size = (uint64_t) st.st_size;
    const size_t page_count = (size_t) ((file_size + page_size - 1) / page_size);
    void * mapping = mmap(nullptr, (size_t) file_size, PROT_NONE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) {
        close(fd);
        return sample;
    }
    std::vector<unsigned char> residency(page_count);
    if (mincore(mapping, (size_t) file_size, residency.data()) == 0) {
        sample.total_bytes = file_size;
        sample.total_pages = page_count;
        for (size_t i = 0; i < page_count; ++i) {
            if ((residency[i] & 1u) == 0) {
                continue;
            }
            ++sample.resident_pages;
            const uint64_t offset = (uint64_t) i * page_size;
            sample.resident_bytes += std::min<uint64_t>(page_size, file_size - offset);
        }
        sample.valid = true;
    }
    munmap(mapping, (size_t) file_size);
    close(fd);
    return sample;
#endif
}

bool page_cache_prepare(const std::string & path, page_cache_policy policy, page_cache_prepare_result & result) {
    result = {};
    result.policy = policy;
    result.before = page_cache_residency(path);
    const auto begin = std::chrono::steady_clock::now();
#if defined(_WIN32)
    result.error = "page-cache policies are supported only on POSIX systems";
    result.elapsed_us = elapsed_us(begin, std::chrono::steady_clock::now());
    return false;
#else
    if (!result.before.valid) {
        result.error = "failed to sample page-cache residency before preparation";
        result.elapsed_us = elapsed_us(begin, std::chrono::steady_clock::now());
        return false;
    }
    if (policy == page_cache_policy::natural) {
        result.after = result.before;
        result.elapsed_us = elapsed_us(begin, std::chrono::steady_clock::now());
        return true;
    }

    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        result.error = std::string("open failed: ") + std::strerror(errno);
        result.elapsed_us = elapsed_us(begin, std::chrono::steady_clock::now());
        return false;
    }

    bool ok = false;
    if (policy == page_cache_policy::cold) {
        for (int attempt = 1; attempt <= 3; ++attempt) {
            result.attempts = attempt;
            const int rc = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            if (rc != 0) {
                result.error = std::string("posix_fadvise(DONTNEED) failed: ") + std::strerror(rc);
                break;
            }
            result.after = page_cache_residency(path);
            if (cold_enough(result.after)) {
                ok = true;
                break;
            }
            usleep(100000);
        }
        if (!ok && result.error.empty()) {
            result.error = "model page-cache residency remains above 1%";
        }
    } else {
        result.attempts = 1;
        (void) posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
        std::vector<unsigned char> buffer(kReadBytes);
        off_t offset = 0;
        while (true) {
            const ssize_t got = pread(fd, buffer.data(), buffer.size(), offset);
            if (got < 0) {
                if (errno == EINTR) {
                    continue;
                }
                result.error = std::string("page-cache warm read failed: ") + std::strerror(errno);
                break;
            }
            if (got == 0) {
                ok = true;
                break;
            }
            offset += got;
            result.bytes_read += (uint64_t) got;
        }
        result.after = page_cache_residency(path);
        if (ok && !hot_enough(result.after)) {
            ok = false;
            result.error = "model page-cache residency remains below 99%";
        }
    }
    close(fd);
    result.elapsed_us = elapsed_us(begin, std::chrono::steady_clock::now());
    return ok;
#endif
}

double page_cache_resident_percent(const page_cache_sample & sample) {
    return sample.valid && sample.total_bytes > 0
        ? 100.0 * (double) sample.resident_bytes / (double) sample.total_bytes
        : 0.0;
}

process_io_sample process_io_current() {
    process_io_sample sample;
#if !defined(_WIN32)
    std::ifstream stream("/proc/self/io");
    std::string line;
    while (std::getline(stream, line)) {
        std::istringstream row(line);
        std::string key;
        uint64_t value = 0;
        if (!(row >> key >> value)) {
            continue;
        }
        if (key == "rchar:") sample.rchar = value;
        else if (key == "read_bytes:") sample.read_bytes = value;
        else if (key == "syscr:") sample.syscr = value;
    }
    sample.valid = stream.eof();
#endif
    return sample;
}

uint64_t process_io_delta(uint64_t before, uint64_t after) {
    return after >= before ? after - before : 0;
}
