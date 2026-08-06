#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <spawn.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cerrno>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

extern char ** environ;

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

namespace {

constexpr const char * SERVICE_VERSION = "1";
constexpr size_t MAX_REQUEST_BYTES = 128 * 1024;
constexpr size_t MAX_PROMPT_BYTES = 64 * 1024;
constexpr size_t MAX_TAIL_LINES = 250;
constexpr size_t MAX_EVENTS = 20000;
constexpr size_t MAX_LOG_LINE_BYTES = 64 * 1024;
constexpr int CANCEL_GRACE_MS = 3000;

const std::vector<std::string> ARTIFACT_NAMES = {
    "summary.txt",
    "profile.csv",
    "tokens.csv",
    "stdout.log",
    "stderr.log",
    "command.txt",
    "request.json",
    "metadata.json",
    "result.json",
};

struct server_config {
    std::string host = "127.0.0.1";
    int port = 8088;
    std::string bench = "/workspace/build-moe/bin/llama-moe-bench";
    std::string model = "/workspace/models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf";
    std::string prompt_file = "/workspace/benchmark-results/new-result/prompts/normal-long-prompt.zh-en.txt";
    std::string output_root = "/workspace/benchmark-results/web-bench";
    std::string web_root = "/workspace/tools/moe-bench/webui";
    std::string cuda_visible_devices;
};

struct run_request {
    std::string prompt_mode;
    std::string prompt_text;
    int pp = 1024;
    int tg = 256;
    int repeat = 1;
    int ubatch = 512;
    int cache_vram_mb = 8000;
    std::string predictor = "lru";
    std::string host_cache = "off";
    std::string host_cache_preload = "none";
    std::string page_cache_policy = "cold";
    int context_size = 4096;
    int gpu_layers = 99;
    bool reset_cache_between_repeats = true;
    bool warm_cache = false;
    bool hot_start = false;
};

struct stream_event {
    uint64_t seq = 0;
    std::string name;
    json data;
};

struct bench_run {
    mutable std::mutex mutex;
    mutable std::mutex persistence_mutex;
    std::condition_variable event_cv;

    std::string id;
    fs::path directory;
    std::string status = "queued";
    int64_t created_ms = 0;
    int64_t started_ms = 0;
    int64_t finished_ms = 0;
    json request = json::object();
    std::vector<std::string> command;
    json metrics = json::object();
    std::vector<std::string> validation_errors;
    std::string raw_summary;
    std::deque<std::string> stdout_tail;
    std::deque<std::string> stderr_tail;
    std::deque<stream_event> events;
    uint64_t next_event_seq = 1;

    pid_t pid = -1;
    int exit_code = -1;
    int term_signal = 0;
    bool cancel_requested = false;
    bool term_sent = false;
    std::chrono::steady_clock::time_point term_sent_at;
};

int64_t epoch_ms_now() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string iso_time(int64_t epoch_ms) {
    if (epoch_ms <= 0) {
        return "";
    }
    const std::time_t seconds = (std::time_t) (epoch_ms / 1000);
    std::tm tm = {};
    gmtime_r(&seconds, &tm);
    char date[32] = {};
    std::strftime(date, sizeof(date), "%Y-%m-%dT%H:%M:%S", &tm);
    std::ostringstream out;
    out << date << '.' << std::setw(3) << std::setfill('0') << (epoch_ms % 1000) << 'Z';
    return out.str();
}

int64_t parse_iso_time(const std::string & value) {
    std::tm tm = {};
    int millis = 0;
    if (std::sscanf(value.c_str(), "%d-%d-%dT%d:%d:%d.%dZ",
                &tm.tm_year, &tm.tm_mon, &tm.tm_mday,
                &tm.tm_hour, &tm.tm_min, &tm.tm_sec, &millis) != 7) {
        return 0;
    }
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    const std::time_t seconds = timegm(&tm);
    return seconds < 0 ? 0 : (int64_t) seconds * 1000 + millis;
}

bool is_terminal(const std::string & status) {
    return status == "completed" || status == "failed" || status == "cancelled";
}

std::string join_lines(const std::deque<std::string> & lines) {
    std::ostringstream out;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i != 0) {
            out << '\n';
        }
        out << lines[i];
    }
    return out.str();
}

void push_tail(std::deque<std::string> & tail, const std::string & line) {
    tail.push_back(line);
    while (tail.size() > MAX_TAIL_LINES) {
        tail.pop_front();
    }
}

std::deque<std::string> read_tail(const fs::path & path) {
    std::deque<std::string> result;
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        push_tail(result, line);
    }
    return result;
}

std::string read_file(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return "";
    }
    std::ostringstream out;
    out << input.rdbuf();
    return out.str();
}

bool nonempty_file(const fs::path & path) {
    std::error_code ec;
    return fs::is_regular_file(path, ec) && fs::file_size(path, ec) > 0 && !ec;
}

bool resolve_artifact_path(const fs::path & directory, const std::string & name,
        fs::path & resolved) {
    std::error_code ec;
    const fs::file_status directory_status = fs::symlink_status(directory, ec);
    if (ec || fs::is_symlink(directory_status) || !fs::is_directory(directory_status)) {
        return false;
    }

    const fs::path candidate = directory / name;
    const fs::file_status candidate_status = fs::symlink_status(candidate, ec);
    if (ec || fs::is_symlink(candidate_status) || !fs::is_regular_file(candidate_status)) {
        return false;
    }

    const fs::path canonical_directory = fs::canonical(directory, ec);
    if (ec) {
        return false;
    }
    const fs::path canonical_candidate = fs::canonical(candidate, ec);
    if (ec || canonical_candidate.parent_path() != canonical_directory) {
        return false;
    }
    resolved = canonical_candidate;
    return true;
}

std::optional<std::vector<std::string>> parse_csv_fields(const std::string & line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (size_t index = 0; index < line.size(); ++index) {
        const char ch = line[index];
        if (quoted) {
            if (ch == '"') {
                if (index + 1 < line.size() && line[index + 1] == '"') {
                    field.push_back('"');
                    ++index;
                } else {
                    quoted = false;
                }
            } else {
                field.push_back(ch);
            }
        } else if (ch == ',') {
            fields.push_back(std::move(field));
            field.clear();
        } else if (ch == '"' && field.empty()) {
            quoted = true;
        } else {
            field.push_back(ch);
        }
    }
    if (quoted) {
        return std::nullopt;
    }
    fields.push_back(std::move(field));
    return fields;
}

bool field_has_content(const std::string & field) {
    return field.find_first_not_of(" \t\r") != std::string::npos;
}

struct csv_check_result {
    bool valid = false;
    std::string error;
};

csv_check_result check_csv_artifact(const fs::path & directory, const std::string & name,
        const std::vector<std::string> & required_columns) {
    fs::path path;
    if (!resolve_artifact_path(directory, name, path)) {
        return {false, name + " is missing, unsafe, or not a regular file"};
    }
    std::ifstream input(path, std::ios::binary);
    std::string header_line;
    if (!std::getline(input, header_line)) {
        return {false, name + " is empty"};
    }
    if (!header_line.empty() && header_line.back() == '\r') {
        header_line.pop_back();
    }
    const auto header = parse_csv_fields(header_line);
    if (!header) {
        return {false, name + " has an invalid CSV header"};
    }

    std::vector<size_t> required_indices;
    for (const std::string & required : required_columns) {
        const auto it = std::find(header->begin(), header->end(), required);
        if (it == header->end()) {
            return {false, name + " header is missing column: " + required};
        }
        required_indices.push_back((size_t) std::distance(header->begin(), it));
    }

    std::string row_line;
    while (std::getline(input, row_line)) {
        if (!row_line.empty() && row_line.back() == '\r') {
            row_line.pop_back();
        }
        if (!field_has_content(row_line)) {
            continue;
        }
        const auto row = parse_csv_fields(row_line);
        if (!row || row->size() != header->size()) {
            continue;
        }
        bool complete = true;
        for (size_t index : required_indices) {
            if (index >= row->size() || !field_has_content((*row)[index])) {
                complete = false;
                break;
            }
        }
        if (complete) {
            return {true, ""};
        }
    }
    return {false, name + " has a header but no valid data row"};
}

void write_text(const fs::path & path, const std::string & contents) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to open " + path.string());
    }
    output << contents;
    output.close();
    if (output.fail()) {
        throw std::runtime_error("failed to write " + path.string());
    }
}

void write_json_atomic(const fs::path & path, const json & value) {
    const fs::path temporary = path.string() + ".tmp";
    write_text(temporary, value.dump(2) + "\n");
    std::error_code ec;
    fs::rename(temporary, path, ec);
    if (ec) {
        fs::remove(temporary);
        throw std::runtime_error("failed to replace " + path.string() + ": " + ec.message());
    }
}

json empty_metrics() {
    return {
        {"ttft_ms", nullptr},
        {"prefill_tok_s", nullptr},
        {"tpot_ms", nullptr},
        {"decode_tok_s", nullptr},
        {"total_ms", nullptr},
        {"prefill_hit_pct", nullptr},
        {"decode_hit_pct", nullptr},
        {"source_bytes", nullptr},
        {"ssd_decode_bytes", nullptr},
        {"physical_read_bytes", nullptr},
        {"vram_peak_gib", nullptr},
        {"dram_peak_gib", nullptr},
        {"page_cache_after_decode_pct", nullptr},
    };
}

json request_to_json(const run_request & request) {
    json result = {
        {"prompt_mode", request.prompt_mode},
        {"pp", request.pp},
        {"tg", request.tg},
        {"repeat", request.repeat},
        {"ubatch", request.ubatch},
        {"cache_vram_mb", request.cache_vram_mb},
        {"predictor", request.predictor},
        {"host_cache", request.host_cache},
        {"host_cache_preload", request.host_cache_preload},
        {"page_cache_policy", request.page_cache_policy},
        {"context_size", request.context_size},
        {"gpu_layers", request.gpu_layers},
        {"reset_cache_between_repeats", request.reset_cache_between_repeats},
        {"warm_cache", request.warm_cache},
        {"hot_start", request.hot_start},
    };
    if (request.prompt_mode == "text") {
        result["prompt_text"] = request.prompt_text;
    }
    return result;
}

std::string shell_quote_for_display(const std::string & value) {
    if (!value.empty() && value.find_first_not_of(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:=,+-") == std::string::npos) {
        return value;
    }
    std::string result = "'";
    for (char ch : value) {
        if (ch == '\'') {
            result += "'\\''";
        } else {
            result += ch;
        }
    }
    result += '\'';
    return result;
}

std::string command_for_display(const std::vector<std::string> & command) {
    std::ostringstream out;
    for (size_t i = 0; i < command.size(); ++i) {
        if (i != 0) {
            out << ' ';
        }
        out << shell_quote_for_display(command[i]);
    }
    return out.str() + "\n";
}

std::optional<double> capture_double(const std::string & line, const std::regex & pattern) {
    std::smatch match;
    if (!std::regex_search(line, match, pattern) || match.size() < 2) {
        return std::nullopt;
    }
    try {
        const double value = std::stod(match[1].str());
        return std::isfinite(value) ? std::optional<double>(value) : std::nullopt;
    } catch (...) {
        return std::nullopt;
    }
}

json parse_summary_metrics(const std::string & summary) {
    json metrics = empty_metrics();
    const std::regex ttft_re(R"(^TTFT:\s*([0-9]+(?:\.[0-9]+)?)\s+ms)");
    const std::regex tpot_re(R"(^TPOT:\s*([0-9]+(?:\.[0-9]+)?)\s+ms)");
    const std::regex total_re(R"(^total:\s*([0-9]+(?:\.[0-9]+)?)\s+ms)");
    const std::regex prefill_hit_re(R"(^cache hit rate \(prefill\):\s*([0-9]+(?:\.[0-9]+)?)%)");
    const std::regex decode_hit_re(R"(^cache hit rate \(decode\):\s*([0-9]+(?:\.[0-9]+)?)%)");
    const std::regex source_re(R"(^Source bytes read \(total\):\s*([0-9]+(?:\.[0-9]+)?)\s+GB)");
    const std::regex ssd_re(R"(^SSD bytes read \(decode\):\s*([0-9]+(?:\.[0-9]+)?)\s+GB)");
    const std::regex physical_re(R"(^Process IO read_bytes:.*\btotal=([0-9]+))");
    const std::regex vram_re(R"(^VRAM peak \(process approx\):\s*([0-9]+(?:\.[0-9]+)?)\s+GB)");
    const std::regex dram_re(R"(^DRAM peak \(process\):\s*([0-9]+(?:\.[0-9]+)?)\s+GB)");
    const std::regex page_cache_re(R"(^Page cache resident pct:.*\bafter_decode=([0-9]+(?:\.[0-9]+)?))");

    std::istringstream input(summary);
    std::string line;
    while (std::getline(input, line)) {
        auto assign = [&](const char * key, const std::regex & pattern) {
            if (const auto value = capture_double(line, pattern)) {
                metrics[key] = *value;
            }
        };
        assign("ttft_ms", ttft_re);
        assign("tpot_ms", tpot_re);
        assign("total_ms", total_re);
        assign("prefill_hit_pct", prefill_hit_re);
        assign("decode_hit_pct", decode_hit_re);
        assign("vram_peak_gib", vram_re);
        assign("dram_peak_gib", dram_re);
        assign("page_cache_after_decode_pct", page_cache_re);

        if (const auto value = capture_double(line, source_re)) {
            metrics["source_bytes"] = (uint64_t) std::llround(*value * 1024.0 * 1024.0 * 1024.0);
        }
        if (const auto value = capture_double(line, ssd_re)) {
            metrics["ssd_decode_bytes"] = (uint64_t) std::llround(*value * 1024.0 * 1024.0 * 1024.0);
        }
        if (const auto value = capture_double(line, physical_re)) {
            metrics["physical_read_bytes"] = (uint64_t) *value;
        }

        if (line.rfind("prefill", 0) == 0 || line.rfind("decode", 0) == 0) {
            std::istringstream row(line);
            std::string phase;
            int tokens = 0;
            double phase_total = 0.0;
            double per_token = 0.0;
            double tok_s = 0.0;
            if (row >> phase >> tokens >> phase_total >> per_token >> tok_s &&
                    std::isfinite(tok_s)) {
                metrics[phase == "prefill" ? "prefill_tok_s" : "decode_tok_s"] = tok_s;
            }
        }
    }
    return metrics;
}

std::string artifact_mime(const std::string & name) {
    if (name.size() >= 5 && name.substr(name.size() - 5) == ".json") {
        return "application/json; charset=utf-8";
    }
    if (name.size() >= 4 && name.substr(name.size() - 4) == ".csv") {
        return "text/csv; charset=utf-8";
    }
    return "text/plain; charset=utf-8";
}

void set_json_response(httplib::Response & response, int status, const json & body) {
    response.status = status;
    response.set_header("Cache-Control", "no-store");
    response.set_content(body.dump(), "application/json; charset=utf-8");
}

std::string sse_message(const std::string & name, const json & data,
        const std::optional<uint64_t> & id = std::nullopt) {
    std::ostringstream out;
    if (id) {
        out << "id: " << *id << '\n';
    }
    out << "event: " << name << '\n';
    out << "data: " << data.dump() << "\n\n";
    return out.str();
}

class bench_service {
public:
    explicit bench_service(server_config config) : config_(std::move(config)) {
        fs::create_directories(config_.output_root);
        load_history();
        worker_ = std::thread([this]() { worker_loop(); });
    }

    ~bench_service() {
        shutdown();
    }

    json config_json() const {
        run_request defaults;
        defaults.prompt_mode = nonempty_file(config_.prompt_file) ? "file" : "default";
        return {
            {"service", {
                {"version", SERVICE_VERSION},
                {"host", config_.host},
                {"port", config_.port},
            }},
            {"defaults", request_to_json(defaults)},
            {"limits", {
                {"pp", {{"min", 1}, {"max", 32768}}},
                {"tg", {{"min", 1}, {"max", 16384}}},
                {"repeat", {{"min", 1}, {"max", 20}}},
                {"ubatch", {{"min", 1}, {"max", 8192}}},
                {"cache_vram_mb", {{"min", 0}, {"max", 131072}}},
                {"context_size", {{"min", 128}, {"max", 262144}}},
                {"gpu_layers", {{"min", 0}, {"max", 999}}},
                {"prompt_text", {{"max_bytes", MAX_PROMPT_BYTES}}},
            }},
            {"options", {
                {"prompt_mode", {"default", "file", "text"}},
                {"predictor", {"lru", "eamc"}},
                {"host_cache", {"off", "pageable", "pinned"}},
                {"host_cache_preload", {"none", "all"}},
                {"page_cache_policy", {"natural", "cold", "hot"}},
            }},
            {"paths", {
                {"bench", config_.bench},
                {"model", config_.model},
                {"prompt_file", config_.prompt_file},
                {"output_root", config_.output_root},
                {"web_root", config_.web_root},
            }},
        };
    }

    std::shared_ptr<bench_run> find_run(const std::string & id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = runs_.find(id);
        return it == runs_.end() ? nullptr : it->second;
    }

    json list_runs_json() const {
        std::vector<std::shared_ptr<bench_run>> runs;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            runs.reserve(runs_.size());
            for (const auto & entry : runs_) {
                runs.push_back(entry.second);
            }
        }
        std::sort(runs.begin(), runs.end(), [](const auto & left, const auto & right) {
            return left->created_ms > right->created_ms;
        });
        json result = json::array();
        for (const auto & run : runs) {
            result.push_back(public_run_json(run, false));
        }
        return {{"runs", std::move(result)}};
    }

    json public_run_json(const std::shared_ptr<bench_run> & run, bool detailed = true,
            uint64_t * event_cursor = nullptr) const {
        std::optional<size_t> queue_position;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            size_t position = 1;
            for (const auto & id : queue_) {
                if (id == run->id) {
                    queue_position = position;
                    break;
                }
                ++position;
            }
        }

        std::lock_guard<std::mutex> lock(run->mutex);
        if (event_cursor != nullptr) {
            *event_cursor = run->next_event_seq - 1;
        }
        const int64_t end_ms = run->finished_ms > 0 ? run->finished_ms : epoch_ms_now();
        int64_t elapsed_ms = 0;
        if (run->started_ms > 0) {
            elapsed_ms = std::max<int64_t>(0, end_ms - run->started_ms);
        }

        json result = {
            {"id", run->id},
            {"status", run->status},
            {"queue_position", queue_position ? json(*queue_position) : json(nullptr)},
            {"created_at", iso_time(run->created_ms)},
            {"started_at", run->started_ms > 0 ? json(iso_time(run->started_ms)) : json(nullptr)},
            {"finished_at", run->finished_ms > 0 ? json(iso_time(run->finished_ms)) : json(nullptr)},
            {"elapsed_ms", elapsed_ms},
            {"request", run->request},
            {"metrics", run->metrics},
            {"validation_errors", run->validation_errors},
        };
        if (!detailed) {
            return result;
        }

        json artifacts = json::array();
        for (const std::string & name : ARTIFACT_NAMES) {
            fs::path path;
            if (!resolve_artifact_path(run->directory, name, path)) {
                continue;
            }
            std::error_code ec;
            const uint64_t size = fs::file_size(path, ec);
            if (!ec) {
                artifacts.push_back({
                    {"name", name},
                    {"size", size},
                    {"url", "/api/runs/" + run->id + "/artifacts/" + name},
                });
            }
        }
        result["command"] = run->command;
        result["artifacts"] = std::move(artifacts);
        result["raw_summary"] = run->raw_summary;
        result["stdout_tail"] = join_lines(run->stdout_tail);
        result["stderr_tail"] = join_lines(run->stderr_tail);
        return result;
    }

    bool parse_request_body(const std::string & body, run_request & request,
            std::vector<std::string> & errors) const {
        if (body.size() > MAX_REQUEST_BYTES) {
            errors.push_back("request body exceeds 128 KiB");
            return false;
        }
        const json input = json::parse(body, nullptr, false);
        if (input.is_discarded() || !input.is_object()) {
            errors.push_back("request body must be a JSON object");
            return false;
        }

        const std::set<std::string> allowed = {
            "prompt_mode", "prompt_text", "pp", "tg", "repeat", "ubatch",
            "cache_vram_mb", "predictor", "host_cache", "host_cache_preload",
            "page_cache_policy", "context_size", "gpu_layers",
            "reset_cache_between_repeats", "warm_cache", "hot_start",
        };
        for (const auto & item : input.items()) {
            if (allowed.count(item.key()) == 0) {
                errors.push_back("unknown field: " + item.key());
            }
        }

        request.prompt_mode = nonempty_file(config_.prompt_file) ? "file" : "default";
        auto string_field = [&](const char * name, std::string & target) {
            if (!input.contains(name)) {
                return;
            }
            if (!input[name].is_string()) {
                errors.push_back(std::string(name) + " must be a string");
                return;
            }
            target = input[name].get<std::string>();
        };
        auto int_field = [&](const char * name, int & target, int minimum, int maximum) {
            if (!input.contains(name)) {
                return;
            }
            if (!input[name].is_number_integer() && !input[name].is_number_unsigned()) {
                errors.push_back(std::string(name) + " must be an integer");
                return;
            }
            bool in_range = false;
            int64_t value = 0;
            if (input[name].is_number_unsigned()) {
                const uint64_t unsigned_value = input[name].get<uint64_t>();
                in_range = unsigned_value >= (uint64_t) minimum &&
                    unsigned_value <= (uint64_t) maximum;
                if (in_range) {
                    value = (int64_t) unsigned_value;
                }
            } else {
                value = input[name].get<int64_t>();
                in_range = value >= minimum && value <= maximum;
            }
            if (!in_range) {
                errors.push_back(std::string(name) + " must be between " +
                        std::to_string(minimum) + " and " + std::to_string(maximum));
                return;
            }
            target = (int) value;
        };
        auto bool_field = [&](const char * name, bool & target) {
            if (!input.contains(name)) {
                return;
            }
            if (!input[name].is_boolean()) {
                errors.push_back(std::string(name) + " must be a boolean");
                return;
            }
            target = input[name].get<bool>();
        };

        string_field("prompt_mode", request.prompt_mode);
        string_field("predictor", request.predictor);
        string_field("host_cache", request.host_cache);
        string_field("host_cache_preload", request.host_cache_preload);
        string_field("page_cache_policy", request.page_cache_policy);
        int_field("pp", request.pp, 1, 32768);
        int_field("tg", request.tg, 1, 16384);
        int_field("repeat", request.repeat, 1, 20);
        int_field("ubatch", request.ubatch, 1, 8192);
        int_field("cache_vram_mb", request.cache_vram_mb, 0, 131072);
        int_field("context_size", request.context_size, 128, 262144);
        int_field("gpu_layers", request.gpu_layers, 0, 999);
        bool_field("reset_cache_between_repeats", request.reset_cache_between_repeats);
        bool_field("warm_cache", request.warm_cache);
        bool_field("hot_start", request.hot_start);

        auto enum_check = [&](const char * name, const std::string & value,
                const std::set<std::string> & choices) {
            if (choices.count(value) == 0) {
                errors.push_back(std::string(name) + " has an unsupported value");
            }
        };
        enum_check("prompt_mode", request.prompt_mode, {"default", "file", "text"});
        enum_check("predictor", request.predictor, {"lru", "eamc"});
        enum_check("host_cache", request.host_cache, {"off", "pageable", "pinned"});
        enum_check("host_cache_preload", request.host_cache_preload, {"none", "all"});
        enum_check("page_cache_policy", request.page_cache_policy, {"natural", "cold", "hot"});

        if (request.prompt_mode == "text") {
            if (!input.contains("prompt_text") || !input["prompt_text"].is_string()) {
                errors.push_back("prompt_text must be a string when prompt_mode is text");
            } else {
                request.prompt_text = input["prompt_text"].get<std::string>();
                if (request.prompt_text.empty()) {
                    errors.push_back("prompt_text must not be empty");
                } else if (request.prompt_text.find('\0') != std::string::npos) {
                    errors.push_back("prompt_text must not contain NUL bytes");
                } else if (request.prompt_text.size() > MAX_PROMPT_BYTES) {
                    errors.push_back("prompt_text exceeds 64 KiB");
                }
            }
        } else if (input.contains("prompt_text")) {
            if (!input["prompt_text"].is_string()) {
                errors.push_back("prompt_text must be a string");
            } else if (!input["prompt_text"].get<std::string>().empty()) {
                errors.push_back("prompt_text must be empty unless prompt_mode is text");
            }
        }
        if (request.prompt_mode == "file" && !nonempty_file(config_.prompt_file)) {
            errors.push_back("the server prompt file is unavailable or empty");
        }
        if (request.host_cache == "off" && request.host_cache_preload != "none") {
            errors.push_back("host_cache_preload must be none when host_cache is off");
        }
        if (request.hot_start && request.predictor != "eamc") {
            errors.push_back("hot_start is only allowed with predictor=eamc");
        }
        if ((int64_t) request.context_size < (int64_t) request.pp + request.tg) {
            errors.push_back("context_size must be at least pp + tg");
        }

        return errors.empty();
    }

    std::shared_ptr<bench_run> create_run(const run_request & request) {
        auto run = std::make_shared<bench_run>();
        run->created_ms = epoch_ms_now();
        run->request = request_to_json(request);
        run->metrics = empty_metrics();

        for (;;) {
            const uint64_t sequence = next_id_.fetch_add(1);
            run->id = make_run_id(run->created_ms, sequence);
            run->directory = fs::path(config_.output_root) / run->id;
            std::error_code ec;
            if (fs::create_directory(run->directory, ec)) {
                break;
            }
            if (ec && ec != std::errc::file_exists) {
                throw std::runtime_error("failed to create run directory: " + ec.message());
            }
        }

        run->command = build_command(request, run->directory);
        write_json_atomic(run->directory / "request.json", run->request);
        write_text(run->directory / "command.txt", command_for_display(run->command));
        write_text(run->directory / "stdout.log", "");
        write_text(run->directory / "stderr.log", "");
        write_text(run->directory / "summary.txt", "");
        write_text(run->directory / "profile.csv", "");
        write_text(run->directory / "tokens.csv", "");
        write_metadata(run);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            runs_[run->id] = run;
            queue_.push_back(run->id);
        }
        persist_run(run);
        add_state_event(run);
        queue_cv_.notify_one();
        return run;
    }

    bool cancel_run(const std::shared_ptr<bench_run> & run, std::string & error) {
        bool was_queued = false;
        pid_t pid = -1;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::lock_guard<std::mutex> run_lock(run->mutex);
            if (is_terminal(run->status)) {
                error = "run is already terminal";
                return false;
            }
            run->cancel_requested = true;
            pid = run->pid;
            if (run->status == "queued") {
                const auto it = std::find(queue_.begin(), queue_.end(), run->id);
                if (it != queue_.end()) {
                    queue_.erase(it);
                }
                run->status = "cancelled";
                run->finished_ms = epoch_ms_now();
                run->validation_errors.push_back("cancelled before execution");
                was_queued = true;
            }
        }

        if (was_queued) {
            write_metadata(run);
            persist_run(run);
            add_state_event(run);
        } else if (pid > 0) {
            send_termination(run, pid);
        }
        queue_cv_.notify_all();
        return true;
    }

    void shutdown() {
        std::unique_lock<std::mutex> shutdown_lock(shutdown_mutex_);
        if (shutdown_complete_) {
            return;
        }
        if (shutdown_started_) {
            shutdown_cv_.wait(shutdown_lock, [this]() { return shutdown_complete_; });
            return;
        }
        shutdown_started_ = true;
        if (!stopping_.exchange(true)) {
            std::vector<std::shared_ptr<bench_run>> queued;
            std::shared_ptr<bench_run> active;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                for (const std::string & id : queue_) {
                    const auto it = runs_.find(id);
                    if (it != runs_.end()) {
                        queued.push_back(it->second);
                    }
                }
                queue_.clear();
                active = active_;
            }
            for (const auto & run : queued) {
                {
                    std::lock_guard<std::mutex> lock(run->mutex);
                    run->cancel_requested = true;
                    run->status = "cancelled";
                    run->finished_ms = epoch_ms_now();
                    run->validation_errors.push_back("service shutting down");
                }
                write_metadata(run);
                persist_run(run);
                add_state_event(run);
            }
            if (active) {
                pid_t pid = -1;
                {
                    std::lock_guard<std::mutex> lock(active->mutex);
                    active->cancel_requested = true;
                    pid = active->pid;
                }
                if (pid > 0) {
                    send_termination(active, pid);
                }
            }
            queue_cv_.notify_all();
        }
        shutdown_lock.unlock();
        if (worker_.joinable()) {
            worker_.join();
        }
        shutdown_lock.lock();
        shutdown_complete_ = true;
        shutdown_cv_.notify_all();
    }

private:
    static std::string make_run_id(int64_t epoch_ms, uint64_t sequence) {
        const std::time_t seconds = (std::time_t) (epoch_ms / 1000);
        std::tm tm = {};
        gmtime_r(&seconds, &tm);
        std::ostringstream out;
        out << std::put_time(&tm, "%Y%m%d-%H%M%S") << '-'
            << getpid() << '-' << std::setw(4) << std::setfill('0') << sequence;
        return out.str();
    }

    std::vector<std::string> build_command(const run_request & request,
            const fs::path & directory) const {
        std::vector<std::string> command = {
            config_.bench,
            "--model", config_.model,
        };
        if (request.prompt_mode == "file") {
            command.insert(command.end(), {"--prompt-file", config_.prompt_file});
        } else if (request.prompt_mode == "text") {
            command.insert(command.end(), {"-p", request.prompt_text});
        }
        command.insert(command.end(), {
            "--pp", std::to_string(request.pp),
            "--tg", std::to_string(request.tg),
            "--repeat", std::to_string(request.repeat),
            "--moe-cache-vram-mb", std::to_string(request.cache_vram_mb),
            "--moe-predictor", request.predictor,
            "--moe-profile-csv", (directory / "profile.csv").string(),
            "--moe-profile-summary", (directory / "summary.txt").string(),
            "--moe-host-cache", request.host_cache,
            "--moe-host-cache-preload", request.host_cache_preload,
            "--page-cache-policy", request.page_cache_policy,
            "--token-trace", (directory / "tokens.csv").string(),
            "-ngl", std::to_string(request.gpu_layers),
            "-c", std::to_string(request.context_size),
            "-ub", std::to_string(request.ubatch),
        });
        if (request.reset_cache_between_repeats) {
            command.push_back("--moe-reset-cache-between-repeats");
        }
        if (request.warm_cache) {
            command.push_back("--moe-warm-cache");
        }
        if (request.hot_start) {
            command.push_back("--moe-hot-start");
        }
        return command;
    }

    void load_history() {
        std::error_code ec;
        for (const fs::directory_entry & entry : fs::directory_iterator(config_.output_root, ec)) {
            if (ec || !entry.is_directory() || entry.is_symlink()) {
                continue;
            }
            const fs::path result_path = entry.path() / "result.json";
            std::ifstream input(result_path);
            if (!input) {
                continue;
            }
            json persisted = json::parse(input, nullptr, false);
            if (persisted.is_discarded()) {
                std::cerr << "[moe-bench-server] ignoring invalid " << result_path << '\n';
                continue;
            }
            const json & value = persisted.contains("run") ? persisted["run"] : persisted;
            if (!value.is_object() || !value.contains("id") || !value["id"].is_string()) {
                continue;
            }
            auto run = std::make_shared<bench_run>();
            run->id = value["id"].get<std::string>();
            if (!std::regex_match(run->id, std::regex("[A-Za-z0-9_-]+"))) {
                continue;
            }
            if (entry.path().filename().string() != run->id) {
                continue;
            }
            run->directory = entry.path();
            run->status = value.contains("status") && value["status"].is_string()
                ? value["status"].get<std::string>() : "failed";
            const std::string created_at = value.contains("created_at") && value["created_at"].is_string()
                ? value["created_at"].get<std::string>() : "";
            run->created_ms = parse_iso_time(created_at);
            if (run->created_ms <= 0) {
                run->created_ms = epoch_ms_now();
            }
            if (value.contains("started_at") && value["started_at"].is_string()) {
                run->started_ms = parse_iso_time(value["started_at"].get<std::string>());
            }
            if (value.contains("finished_at") && value["finished_at"].is_string()) {
                run->finished_ms = parse_iso_time(value["finished_at"].get<std::string>());
            }
            run->request = value.contains("request") && value["request"].is_object()
                ? value["request"] : json::object();
            if (value.contains("command") && value["command"].is_array()) {
                for (const json & arg : value["command"]) {
                    if (arg.is_string()) {
                        run->command.push_back(arg.get<std::string>());
                    }
                }
            }
            const json metric_defaults = empty_metrics();
            run->metrics = metric_defaults;
            const json loaded_metrics = value.contains("metrics") && value["metrics"].is_object()
                ? value["metrics"] : json::object();
            for (const auto & item : metric_defaults.items()) {
                if (loaded_metrics.contains(item.key()) &&
                        (loaded_metrics[item.key()].is_number() || loaded_metrics[item.key()].is_null())) {
                    run->metrics[item.key()] = loaded_metrics[item.key()];
                }
            }
            if (value.contains("validation_errors") && value["validation_errors"].is_array()) {
                for (const json & error : value["validation_errors"]) {
                    if (error.is_string()) {
                        run->validation_errors.push_back(error.get<std::string>());
                    }
                }
            }
            run->raw_summary = value.contains("raw_summary") && value["raw_summary"].is_string()
                ? value["raw_summary"].get<std::string>() : read_file(entry.path() / "summary.txt");
            run->stdout_tail = read_tail(entry.path() / "stdout.log");
            run->stderr_tail = read_tail(entry.path() / "stderr.log");
            bool changed = false;
            if (!is_terminal(run->status)) {
                run->status = "failed";
                run->finished_ms = epoch_ms_now();
                run->validation_errors.push_back("service restarted before this run reached a terminal state");
                changed = true;
            }
            runs_[run->id] = run;
            if (changed) {
                write_metadata(run);
                persist_run(run);
            }
        }
    }

    void persist_run(const std::shared_ptr<bench_run> & run) const {
        std::lock_guard<std::mutex> persistence_lock(run->persistence_mutex);
        try {
            write_json_atomic(run->directory / "result.json", {
                {"schema_version", 1},
                {"run", public_run_json(run)},
            });
        } catch (const std::exception & error) {
            std::cerr << "[moe-bench-server] result persistence failed for "
                      << run->id << ": " << error.what() << '\n';
        }
    }

    void write_metadata(const std::shared_ptr<bench_run> & run) const {
        json metadata;
        {
            std::lock_guard<std::mutex> lock(run->mutex);
            metadata = {
                {"service_version", SERVICE_VERSION},
                {"run_id", run->id},
                {"status", run->status},
                {"created_at", iso_time(run->created_ms)},
                {"started_at", run->started_ms > 0 ? json(iso_time(run->started_ms)) : json(nullptr)},
                {"finished_at", run->finished_ms > 0 ? json(iso_time(run->finished_ms)) : json(nullptr)},
                {"pid", run->pid > 0 ? json(run->pid) : json(nullptr)},
                {"exit_code", run->exit_code >= 0 ? json(run->exit_code) : json(nullptr)},
                {"term_signal", run->term_signal > 0 ? json(run->term_signal) : json(nullptr)},
                {"validation_errors", run->validation_errors},
                {"bench", config_.bench},
                {"model", config_.model},
                {"prompt_file", config_.prompt_file},
                {"cuda_visible_devices", config_.cuda_visible_devices},
            };
        }
        try {
            write_json_atomic(run->directory / "metadata.json", metadata);
        } catch (const std::exception & error) {
            std::cerr << "[moe-bench-server] metadata persistence failed for "
                      << run->id << ": " << error.what() << '\n';
        }
    }

    void add_event(const std::shared_ptr<bench_run> & run, const std::string & name, json data) {
        {
            std::lock_guard<std::mutex> lock(run->mutex);
            const uint64_t sequence = run->next_event_seq++;
            data["seq"] = sequence;
            run->events.push_back({sequence, name, std::move(data)});
            while (run->events.size() > MAX_EVENTS) {
                run->events.pop_front();
            }
        }
        run->event_cv.notify_all();
    }

    void add_state_event(const std::shared_ptr<bench_run> & run) {
        add_event(run, "state", {
            {"type", "state"},
            {"run", public_run_json(run)},
        });
    }

    void add_log_event(const std::shared_ptr<bench_run> & run,
            const std::string & stream, const std::string & line) {
        {
            std::lock_guard<std::mutex> lock(run->mutex);
            push_tail(stream == "stdout" ? run->stdout_tail : run->stderr_tail, line);
            const uint64_t sequence = run->next_event_seq++;
            json data = {
                {"type", "log"},
                {"stream", stream},
                {"line", line},
                {"seq", sequence},
            };
            run->events.push_back({sequence, "log", std::move(data)});
            while (run->events.size() > MAX_EVENTS) {
                run->events.pop_front();
            }
        }
        run->event_cv.notify_all();
    }

    void worker_loop() {
        for (;;) {
            std::shared_ptr<bench_run> run;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                queue_cv_.wait(lock, [this]() { return stopping_ || !queue_.empty(); });
                if (queue_.empty()) {
                    if (stopping_) {
                        return;
                    }
                    continue;
                }
                const std::string id = queue_.front();
                queue_.pop_front();
                const auto it = runs_.find(id);
                if (it == runs_.end()) {
                    continue;
                }
                run = it->second;
                active_ = run;
            }

            bool skip = false;
            {
                std::lock_guard<std::mutex> lock(run->mutex);
                if (run->cancel_requested || run->status != "queued") {
                    skip = true;
                } else {
                    run->status = "running";
                    run->started_ms = epoch_ms_now();
                }
            }
            if (!skip) {
                persist_run(run);
                add_state_event(run);
                execute_run(run);
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (active_ == run) {
                    active_.reset();
                }
            }
        }
    }

    void send_termination(const std::shared_ptr<bench_run> & run, pid_t pid) {
        bool should_send = false;
        {
            std::lock_guard<std::mutex> lock(run->mutex);
            if (!run->term_sent) {
                run->term_sent = true;
                run->term_sent_at = std::chrono::steady_clock::now();
                should_send = true;
            }
        }
        if (should_send && kill(-pid, SIGTERM) != 0 && errno != ESRCH) {
            add_log_event(run, "stderr", std::string("[server] SIGTERM failed: ") + std::strerror(errno));
        }
    }

    static void pipe_reader(const std::shared_ptr<bench_run> & run, int fd,
            const fs::path & log_path, const std::string & stream, bench_service * service) {
        std::ofstream log(log_path, std::ios::binary | std::ios::app);
        std::string pending;
        char buffer[8192];
        auto emit_complete_fragments = [&]() {
            for (;;) {
                const size_t newline = pending.find('\n');
                if (newline != std::string::npos && newline <= MAX_LOG_LINE_BYTES) {
                    std::string line = pending.substr(0, newline);
                    pending.erase(0, newline + 1);
                    if (!line.empty() && line.back() == '\r') {
                        line.pop_back();
                    }
                    service->add_log_event(run, stream, line);
                } else if (pending.size() >= MAX_LOG_LINE_BYTES) {
                    service->add_log_event(run, stream, pending.substr(0, MAX_LOG_LINE_BYTES));
                    pending.erase(0, MAX_LOG_LINE_BYTES);
                } else {
                    return;
                }
            }
        };
        for (;;) {
            const ssize_t count = read(fd, buffer, sizeof(buffer));
            if (count > 0) {
                log.write(buffer, count);
                log.flush();
                pending.append(buffer, (size_t) count);
                emit_complete_fragments();
            } else if (count == 0) {
                break;
            } else if (errno != EINTR) {
                service->add_log_event(run, "stderr",
                        std::string("[server] failed reading ") + stream + ": " + std::strerror(errno));
                break;
            }
        }
        if (!pending.empty()) {
            if (!pending.empty() && pending.back() == '\r') {
                pending.pop_back();
            }
            service->add_log_event(run, stream, pending);
        }
        close(fd);
    }

    void execute_run(const std::shared_ptr<bench_run> & run) {
        int stdout_pipe[2] = {-1, -1};
        int stderr_pipe[2] = {-1, -1};
        if (pipe2(stdout_pipe, O_CLOEXEC) != 0 || pipe2(stderr_pipe, O_CLOEXEC) != 0) {
            const int saved_errno = errno;
            if (stdout_pipe[0] >= 0) close(stdout_pipe[0]);
            if (stdout_pipe[1] >= 0) close(stdout_pipe[1]);
            if (stderr_pipe[0] >= 0) close(stderr_pipe[0]);
            if (stderr_pipe[1] >= 0) close(stderr_pipe[1]);
            finish_launch_failure(run, std::string("failed to create output pipes: ") + std::strerror(saved_errno));
            return;
        }

        std::vector<std::string> argv_storage = {
            "/proc/self/exe",
            "--run-child",
            std::to_string(getpid()),
            "--",
        };
        argv_storage.insert(argv_storage.end(), run->command.begin(), run->command.end());
        std::vector<char *> child_argv;
        child_argv.reserve(argv_storage.size() + 1);
        for (std::string & arg : argv_storage) {
            child_argv.push_back(arg.data());
        }
        child_argv.push_back(nullptr);

        std::vector<std::string> environment_storage;
        bool cuda_environment_replaced = false;
        for (char ** variable = environ; variable != nullptr && *variable != nullptr; ++variable) {
            const std::string value(*variable);
            if (!config_.cuda_visible_devices.empty() &&
                    value.rfind("CUDA_VISIBLE_DEVICES=", 0) == 0) {
                environment_storage.push_back("CUDA_VISIBLE_DEVICES=" + config_.cuda_visible_devices);
                cuda_environment_replaced = true;
            } else {
                environment_storage.push_back(value);
            }
        }
        if (!config_.cuda_visible_devices.empty() && !cuda_environment_replaced) {
            environment_storage.push_back("CUDA_VISIBLE_DEVICES=" + config_.cuda_visible_devices);
        }
        std::vector<char *> child_environment;
        child_environment.reserve(environment_storage.size() + 1);
        for (std::string & variable : environment_storage) {
            child_environment.push_back(variable.data());
        }
        child_environment.push_back(nullptr);

        posix_spawn_file_actions_t file_actions;
        posix_spawnattr_t attributes;
        int spawn_error = posix_spawn_file_actions_init(&file_actions);
        const bool file_actions_initialized = spawn_error == 0;
        if (spawn_error == 0) {
            spawn_error = posix_spawn_file_actions_adddup2(&file_actions, stdout_pipe[1], STDOUT_FILENO);
        }
        if (spawn_error == 0) {
            spawn_error = posix_spawn_file_actions_adddup2(&file_actions, stderr_pipe[1], STDERR_FILENO);
        }
        if (spawn_error == 0) {
            spawn_error = posix_spawn_file_actions_addclose(&file_actions, stdout_pipe[0]);
        }
        if (spawn_error == 0) {
            spawn_error = posix_spawn_file_actions_addclose(&file_actions, stderr_pipe[0]);
        }
        if (spawn_error == 0) {
            spawn_error = posix_spawn_file_actions_addclose(&file_actions, stdout_pipe[1]);
        }
        if (spawn_error == 0) {
            spawn_error = posix_spawn_file_actions_addclose(&file_actions, stderr_pipe[1]);
        }
        if (spawn_error == 0) {
            spawn_error = posix_spawnattr_init(&attributes);
        }
        const bool attributes_initialized = spawn_error == 0;
        short spawn_flags = 0;
        if (spawn_error == 0) {
            spawn_flags = (short) (spawn_flags | POSIX_SPAWN_SETPGROUP);
            spawn_error = posix_spawnattr_setflags(&attributes, spawn_flags);
        }
        if (spawn_error == 0) {
            spawn_error = posix_spawnattr_setpgroup(&attributes, 0);
        }

        pid_t pid = -1;
        if (spawn_error == 0) {
            spawn_error = posix_spawnp(&pid, child_argv[0], &file_actions, &attributes,
                    child_argv.data(), child_environment.data());
        }
        if (attributes_initialized) {
            posix_spawnattr_destroy(&attributes);
        }
        if (file_actions_initialized) {
            posix_spawn_file_actions_destroy(&file_actions);
        }
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);
        if (spawn_error != 0) {
            close(stdout_pipe[0]);
            close(stderr_pipe[0]);
            finish_launch_failure(run, std::string("posix_spawnp failed: ") + std::strerror(spawn_error));
            return;
        }

        setpgid(pid, pid);
        {
            std::lock_guard<std::mutex> lock(run->mutex);
            run->pid = pid;
        }
        write_metadata(run);
        add_log_event(run, "stderr", "[server] benchmark process started with pid " + std::to_string(pid));

        std::thread stdout_reader(pipe_reader, run, stdout_pipe[0],
                run->directory / "stdout.log", "stdout", this);
        std::thread stderr_reader(pipe_reader, run, stderr_pipe[0],
                run->directory / "stderr.log", "stderr", this);

        int wait_status = 0;
        bool wait_succeeded = false;
        for (;;) {
            const pid_t result = waitpid(pid, &wait_status, WNOHANG);
            if (result == pid) {
                wait_succeeded = true;
                break;
            }
            if (result < 0 && errno != EINTR) {
                add_log_event(run, "stderr", std::string("[server] waitpid failed: ") + std::strerror(errno));
                break;
            }

            bool cancel_requested = false;
            bool term_sent = false;
            std::chrono::steady_clock::time_point term_sent_at;
            {
                std::lock_guard<std::mutex> lock(run->mutex);
                cancel_requested = run->cancel_requested;
                term_sent = run->term_sent;
                term_sent_at = run->term_sent_at;
            }
            if (cancel_requested && !term_sent) {
                send_termination(run, pid);
            } else if (term_sent &&
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - term_sent_at).count() >= CANCEL_GRACE_MS) {
                if (kill(-pid, SIGKILL) == 0) {
                    add_log_event(run, "stderr", "[server] benchmark did not stop after SIGTERM; sent SIGKILL");
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        stdout_reader.join();
        stderr_reader.join();

        bool cancelled = false;
        {
            std::lock_guard<std::mutex> lock(run->mutex);
            run->pid = -1;
            cancelled = run->cancel_requested;
            if (!wait_succeeded) {
                run->exit_code = 255;
            } else if (WIFEXITED(wait_status)) {
                run->exit_code = WEXITSTATUS(wait_status);
            } else if (WIFSIGNALED(wait_status)) {
                run->term_signal = WTERMSIG(wait_status);
            }
        }

        finalize_run(run, cancelled);
    }

    void finish_launch_failure(const std::shared_ptr<bench_run> & run, const std::string & error) {
        add_log_event(run, "stderr", "[server] " + error);
        {
            std::lock_guard<std::mutex> lock(run->mutex);
            run->status = "failed";
            run->finished_ms = epoch_ms_now();
            run->validation_errors.push_back(error);
        }
        write_metadata(run);
        persist_run(run);
        add_state_event(run);
    }

    void finalize_run(const std::shared_ptr<bench_run> & run, bool cancelled) {
        fs::path summary_path;
        const std::string summary = resolve_artifact_path(run->directory, "summary.txt", summary_path)
            ? read_file(summary_path) : "";
        const json metrics = parse_summary_metrics(summary);
        std::vector<std::string> errors;

        int exit_code = -1;
        int term_signal = 0;
        {
            std::lock_guard<std::mutex> lock(run->mutex);
            exit_code = run->exit_code;
            term_signal = run->term_signal;
        }
        if (!cancelled) {
            if (exit_code != 0) {
                if (term_signal > 0) {
                    errors.push_back("benchmark terminated by signal " + std::to_string(term_signal));
                } else {
                    errors.push_back("benchmark exited with code " + std::to_string(exit_code));
                }
            }
            if (summary.find_first_not_of(" \t\r\n") == std::string::npos) {
                errors.push_back("summary.txt is missing or empty");
            }
            for (const char * key : {"ttft_ms", "prefill_tok_s", "tpot_ms", "decode_tok_s", "total_ms"}) {
                if (!metrics.contains(key) || !metrics[key].is_number() ||
                        !std::isfinite(metrics[key].get<double>()) || metrics[key].get<double>() <= 0.0) {
                    errors.push_back(std::string("summary metric is missing or non-positive: ") + key);
                }
            }
            const csv_check_result profile = check_csv_artifact(
                    run->directory, "profile.csv", {"row_type"});
            if (!profile.valid) {
                errors.push_back(profile.error);
            }
            const csv_check_result tokens = check_csv_artifact(
                    run->directory, "tokens.csv", {"repeat", "step", "token"});
            if (!tokens.valid) {
                errors.push_back(tokens.error);
            }
        }

        {
            std::lock_guard<std::mutex> lock(run->mutex);
            run->raw_summary = summary;
            run->metrics = metrics;
            run->finished_ms = epoch_ms_now();
            if (cancelled) {
                run->status = "cancelled";
                run->validation_errors.push_back("cancelled by user or service shutdown");
            } else if (errors.empty()) {
                run->status = "completed";
            } else {
                run->status = "failed";
                run->validation_errors.insert(run->validation_errors.end(), errors.begin(), errors.end());
            }
        }
        write_metadata(run);
        persist_run(run);
        add_state_event(run);
    }

    server_config config_;
    mutable std::mutex mutex_;
    std::condition_variable queue_cv_;
    std::unordered_map<std::string, std::shared_ptr<bench_run>> runs_;
    std::deque<std::string> queue_;
    std::shared_ptr<bench_run> active_;
    std::thread worker_;
    std::atomic<uint64_t> next_id_{1};
    std::atomic<bool> stopping_{false};
    std::mutex shutdown_mutex_;
    std::condition_variable shutdown_cv_;
    bool shutdown_started_ = false;
    bool shutdown_complete_ = false;
};

bool parse_int_argument(const std::string & value, int minimum, int maximum, int & result) {
    try {
        size_t consumed = 0;
        const long parsed = std::stol(value, &consumed, 10);
        if (consumed != value.size() || parsed < minimum || parsed > maximum) {
            return false;
        }
        result = (int) parsed;
        return true;
    } catch (...) {
        return false;
    }
}

int run_child_wrapper(int argc, char ** argv) {
    int expected_parent = 0;
    if (argc < 5 || std::string(argv[1]) != "--run-child" ||
            !parse_int_argument(argv[2], 1, std::numeric_limits<int>::max(), expected_parent) ||
            std::string(argv[3]) != "--" || argv[4][0] == '\0') {
        std::cerr << "llama-moe-bench-server: invalid internal child invocation\n";
        return 126;
    }
    if (prctl(PR_SET_PDEATHSIG, SIGTERM) != 0) {
        std::cerr << "llama-moe-bench-server: PR_SET_PDEATHSIG failed: "
                  << std::strerror(errno) << '\n';
        return 126;
    }
    if (getppid() != (pid_t) expected_parent) {
        std::cerr << "llama-moe-bench-server: parent exited before child initialization\n";
        return 125;
    }
    execvp(argv[4], argv + 4);
    std::cerr << "llama-moe-bench-server: execvp failed: " << std::strerror(errno) << '\n';
    return 127;
}

void print_usage(const char * program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --host HOST                    Listen address (default: 127.0.0.1)\n"
        << "  --port PORT                    Listen port (default: 8088)\n"
        << "  --bench PATH                   Fixed llama-moe-bench executable\n"
        << "  --model PATH                   Fixed model passed to every run\n"
        << "  --prompt-file PATH             Fixed prompt used by prompt_mode=file\n"
        << "  --output-root PATH             Per-run result directory root\n"
        << "  --web-root PATH                Static frontend directory\n"
        << "  --cuda-visible-devices VALUE   CUDA_VISIBLE_DEVICES for bench children\n"
        << "  -h, --help                     Show this help\n";
}

bool parse_server_args(int argc, char ** argv, server_config & config, bool & show_help,
        std::string & error) {
    show_help = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "-h" || argument == "--help") {
            show_help = true;
            return true;
        }

        auto value_for = [&](const std::string & name, std::string & value) -> bool {
            if (argument == name) {
                if (index + 1 >= argc) {
                    error = "missing value for " + name;
                    return true;
                }
                value = argv[++index];
                return true;
            }
            const std::string prefix = name + "=";
            if (argument.rfind(prefix, 0) == 0) {
                value = argument.substr(prefix.size());
                return true;
            }
            return false;
        };

        std::string value;
        if (value_for("--host", value)) {
            if (!error.empty()) return false;
            config.host = value;
        } else if (value_for("--port", value)) {
            if (!error.empty()) return false;
            if (!parse_int_argument(value, 1, 65535, config.port)) {
                error = "--port must be an integer between 1 and 65535";
                return false;
            }
        } else if (value_for("--bench", value)) {
            if (!error.empty()) return false;
            config.bench = value;
        } else if (value_for("--model", value)) {
            if (!error.empty()) return false;
            config.model = value;
        } else if (value_for("--prompt-file", value)) {
            if (!error.empty()) return false;
            config.prompt_file = value;
        } else if (value_for("--output-root", value)) {
            if (!error.empty()) return false;
            config.output_root = value;
        } else if (value_for("--web-root", value)) {
            if (!error.empty()) return false;
            config.web_root = value;
        } else if (value_for("--cuda-visible-devices", value)) {
            if (!error.empty()) return false;
            config.cuda_visible_devices = value;
        } else {
            error = "unknown option: " + argument;
            return false;
        }
    }
    if (config.host.empty() || config.bench.empty() || config.model.empty() ||
            config.output_root.empty()) {
        error = "--host, --bench, --model, and --output-root must not be empty";
        return false;
    }
    return true;
}

volatile std::sig_atomic_t signal_requested = 0;

void signal_handler(int) {
    signal_requested = 1;
}

}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--run-child") {
        return run_child_wrapper(argc, argv);
    }
    server_config config;
    bool show_help = false;
    std::string argument_error;
    if (!parse_server_args(argc, argv, config, show_help, argument_error)) {
        std::cerr << "llama-moe-bench-server: " << argument_error << '\n';
        print_usage(argv[0]);
        return 2;
    }
    if (show_help) {
        print_usage(argv[0]);
        return 0;
    }

    std::error_code filesystem_error;
    if (!fs::is_regular_file(config.bench, filesystem_error) || access(config.bench.c_str(), X_OK) != 0) {
        std::cerr << "llama-moe-bench-server: bench executable is unavailable or not executable: "
                  << config.bench << '\n';
        return 2;
    }
    filesystem_error.clear();
    if (!fs::is_regular_file(config.model, filesystem_error)) {
        std::cerr << "llama-moe-bench-server: model is unavailable: " << config.model << '\n';
        return 2;
    }

    try {
        bench_service service(config);
        httplib::Server server;
        server.set_read_timeout(10, 0);
        server.set_write_timeout(30, 0);
        server.set_payload_max_length(MAX_REQUEST_BYTES);

        server.set_exception_handler([](const httplib::Request &, httplib::Response & response,
                    std::exception_ptr exception) {
            std::string message = "internal server error";
            try {
                if (exception) {
                    std::rethrow_exception(exception);
                }
            } catch (const std::exception & error) {
                message = error.what();
            } catch (...) {
            }
            set_json_response(response, 500, {{"error", message}});
        });

        server.Get("/api/config", [&](const httplib::Request &, httplib::Response & response) {
            set_json_response(response, 200, service.config_json());
        });

        server.Get("/api/runs", [&](const httplib::Request &, httplib::Response & response) {
            set_json_response(response, 200, service.list_runs_json());
        });

        server.Post("/api/runs", [&](const httplib::Request & request, httplib::Response & response) {
            run_request parsed;
            std::vector<std::string> errors;
            if (!service.parse_request_body(request.body, parsed, errors)) {
                set_json_response(response, 400, {
                    {"error", "request validation failed"},
                    {"validation_errors", errors},
                });
                return;
            }
            try {
                const auto run = service.create_run(parsed);
                set_json_response(response, 201, {{"run", service.public_run_json(run)}});
            } catch (const std::exception & error) {
                set_json_response(response, 500, {{"error", error.what()}});
            }
        });

        server.Get(R"(/api/runs/([A-Za-z0-9_-]+))",
                [&](const httplib::Request & request, httplib::Response & response) {
            const auto run = service.find_run(request.matches[1].str());
            if (!run) {
                set_json_response(response, 404, {{"error", "run not found"}});
                return;
            }
            set_json_response(response, 200, {{"run", service.public_run_json(run)}});
        });

        server.Post(R"(/api/runs/([A-Za-z0-9_-]+)/cancel)",
                [&](const httplib::Request & request, httplib::Response & response) {
            const auto run = service.find_run(request.matches[1].str());
            if (!run) {
                set_json_response(response, 404, {{"error", "run not found"}});
                return;
            }
            std::string error;
            if (!service.cancel_run(run, error)) {
                set_json_response(response, 409, {
                    {"error", error},
                    {"run", service.public_run_json(run)},
                });
                return;
            }
            set_json_response(response, 202, {{"run", service.public_run_json(run)}});
        });

        server.Get(R"(/api/runs/([A-Za-z0-9_-]+)/artifacts/([A-Za-z0-9_.-]+))",
                [&](const httplib::Request & request, httplib::Response & response) {
            const auto run = service.find_run(request.matches[1].str());
            if (!run) {
                set_json_response(response, 404, {{"error", "run not found"}});
                return;
            }
            const std::string name = request.matches[2].str();
            if (std::find(ARTIFACT_NAMES.begin(), ARTIFACT_NAMES.end(), name) == ARTIFACT_NAMES.end()) {
                set_json_response(response, 404, {{"error", "artifact not found"}});
                return;
            }
            fs::path path;
            if (!resolve_artifact_path(run->directory, name, path)) {
                set_json_response(response, 404, {{"error", "artifact not found"}});
                return;
            }
            response.set_header("Cache-Control", "no-store");
            response.set_header("Content-Disposition", "attachment; filename=\"" + name + "\"");
            response.set_file_content(path.string(), artifact_mime(name));
        });

        server.Get(R"(/api/runs/([A-Za-z0-9_-]+)/events)",
                [&](const httplib::Request & request, httplib::Response & response) {
            const auto run = service.find_run(request.matches[1].str());
            if (!run) {
                set_json_response(response, 404, {{"error", "run not found"}});
                return;
            }

            struct sse_state {
                std::shared_ptr<bench_run> run;
                bench_service * service = nullptr;
                uint64_t cursor = 0;
                std::optional<uint64_t> requested_cursor;
                bool initialized = false;
                bool baseline_sent = false;
                bool resumed = false;
                bool terminal_sent = false;
            };
            auto state = std::make_shared<sse_state>();
            state->run = run;
            state->service = &service;
            if (request.has_header("Last-Event-ID")) {
                const std::string header = request.get_header_value("Last-Event-ID");
                try {
                    size_t consumed = 0;
                    const uint64_t cursor = std::stoull(header, &consumed, 10);
                    if (consumed == header.size()) {
                        state->requested_cursor = cursor;
                    }
                } catch (...) {
                }
            }

            response.set_header("Cache-Control", "no-cache, no-transform");
            response.set_header("X-Accel-Buffering", "no");
            response.set_chunked_content_provider("text/event-stream; charset=utf-8",
                [state](size_t, httplib::DataSink & sink) -> bool {
                    if (!state->initialized) {
                        state->initialized = true;
                        if (state->requested_cursor) {
                            std::lock_guard<std::mutex> lock(state->run->mutex);
                            const uint64_t latest = state->run->next_event_seq - 1;
                            const uint64_t earliest = state->run->events.empty()
                                ? state->run->next_event_seq : state->run->events.front().seq;
                            if (*state->requested_cursor <= latest &&
                                    *state->requested_cursor >= earliest - 1) {
                                state->cursor = *state->requested_cursor;
                                state->resumed = true;
                                state->baseline_sent = true;
                            }
                        }
                    }
                    if (!state->baseline_sent) {
                        state->baseline_sent = true;
                        uint64_t snapshot_cursor = 0;
                        const json current = state->service->public_run_json(
                                state->run, true, &snapshot_cursor);
                        state->cursor = snapshot_cursor;
                        state->terminal_sent = is_terminal(current["status"].get<std::string>());
                        const std::string message = sse_message("snapshot", {
                            {"type", "snapshot"},
                            {"cursor", snapshot_cursor},
                            {"run", current},
                        }, snapshot_cursor);
                        return sink.write(message.data(), message.size());
                    }
                    if (state->terminal_sent) {
                        sink.done();
                        return false;
                    }

                    std::vector<stream_event> available;
                    bool terminal = false;
                    bool cursor_gap = false;
                    {
                        std::unique_lock<std::mutex> lock(state->run->mutex);
                        state->run->event_cv.wait_for(lock, std::chrono::seconds(10), [&]() {
                            return (!state->run->events.empty() &&
                                    state->run->events.back().seq > state->cursor) ||
                                is_terminal(state->run->status);
                        });
                        cursor_gap = !state->run->events.empty() &&
                            state->cursor < state->run->events.front().seq - 1;
                        for (const stream_event & event : state->run->events) {
                            if (event.seq > state->cursor) {
                                available.push_back(event);
                            }
                        }
                        terminal = is_terminal(state->run->status);
                    }

                    if (cursor_gap) {
                        uint64_t snapshot_cursor = 0;
                        const json current = state->service->public_run_json(
                                state->run, true, &snapshot_cursor);
                        state->cursor = snapshot_cursor;
                        state->resumed = false;
                        state->terminal_sent = is_terminal(current["status"].get<std::string>());
                        const std::string message = sse_message("snapshot", {
                            {"type", "snapshot"},
                            {"cursor", snapshot_cursor},
                            {"run", current},
                        }, snapshot_cursor);
                        return sink.write(message.data(), message.size());
                    }

                    if (!available.empty()) {
                        std::string messages;
                        bool terminal_state_sent = false;
                        for (const stream_event & event : available) {
                            messages += sse_message(event.name, event.data, event.seq);
                            state->cursor = event.seq;
                            if (terminal && event.name == "state" && event.data.contains("run") &&
                                    event.data["run"].is_object() && event.data["run"].contains("status") &&
                                    event.data["run"]["status"].is_string() &&
                                    is_terminal(event.data["run"]["status"].get<std::string>())) {
                                terminal_state_sent = true;
                            }
                        }
                        if (terminal && !terminal_state_sent) {
                            messages += sse_message("state", {
                                {"type", "state"},
                                {"run", state->service->public_run_json(state->run)},
                            });
                        }
                        if (terminal) {
                            state->terminal_sent = true;
                        }
                        return sink.write(messages.data(), messages.size());
                    }
                    if (terminal) {
                        if (state->resumed) {
                            sink.done();
                            return false;
                        }
                        const std::string message = sse_message("state", {
                            {"type", "state"},
                            {"run", state->service->public_run_json(state->run)},
                        });
                        state->terminal_sent = true;
                        return sink.write(message.data(), message.size());
                    }
                    static const std::string heartbeat = ": keep-alive\n\n";
                    return sink.write(heartbeat.data(), heartbeat.size());
                },
                [](bool) {});
        });

        const bool have_web_root = !config.web_root.empty() &&
            fs::is_directory(config.web_root, filesystem_error);
        if (have_web_root) {
            if (!server.set_mount_point("/", config.web_root)) {
                throw std::runtime_error("failed to mount web root: " + config.web_root);
            }
        } else {
            std::cerr << "[moe-bench-server] warning: web root is unavailable: "
                      << config.web_root << '\n';
            server.Get("/", [](const httplib::Request &, httplib::Response & response) {
                set_json_response(response, 404, {{"error", "frontend is not installed"}});
            });
        }

        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);
        std::signal(SIGPIPE, SIG_IGN);
        std::atomic<bool> watcher_done{false};
        std::thread signal_watcher([&]() {
            while (!watcher_done.load()) {
                if (signal_requested != 0) {
                    server.stop();
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });

        std::cout << "[moe-bench-server] listening on http://" << config.host << ':' << config.port << '\n';
        std::cout << "[moe-bench-server] results: " << config.output_root << '\n';
        const bool listened = server.listen(config.host, config.port);
        watcher_done.store(true);
        signal_watcher.join();
        service.shutdown();
        if (!listened && signal_requested == 0) {
            std::cerr << "llama-moe-bench-server: failed to listen on "
                      << config.host << ':' << config.port << '\n';
            return 1;
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "llama-moe-bench-server: " << error.what() << '\n';
        return 1;
    }
}
