# llama-moe-bench web service

`llama-moe-bench-server` is a local dashboard for the existing
`llama-moe-bench` executable. The service does not keep a model or MoE runtime
alive between runs. Every run starts a new benchmark process so page-cache,
slot-cache, and process-lifetime behavior remain owned by the benchmark.

The service is Linux-only and has no new runtime dependencies. It uses the
repository's existing `cpp-httplib` and `nlohmann/json` copies and serves the
static files in `webui/`.

## Build

From the development container:

```bash
cmake --build /workspace/build-moe --target llama-moe-bench llama-moe-bench-server -j2
```

Building these explicit targets does not build or download `llama-ui`.

## Run

The defaults match the current development container, non-MTP Q4 model, and
normal prompt file:

```bash
/workspace/build-moe/bin/llama-moe-bench-server \
    --host 0.0.0.0 \
    --port 8088 \
    --cuda-visible-devices 3
```

The default paths are:

```text
bench:       /workspace/build-moe/bin/llama-moe-bench
model:       /workspace/models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf
prompt file: /workspace/benchmark-results/new-result/prompts/normal-long-prompt.zh-en.txt
results:     /workspace/benchmark-results/web-bench
web root:    /workspace/tools/moe-bench/webui
```

Use `--bench`, `--model`, `--prompt-file`, `--output-root`, or `--web-root` to
replace a server-owned path. Run requests cannot replace these paths or add
arbitrary command-line arguments.

The default listen address is `127.0.0.1`. Binding `0.0.0.0` is useful inside
a container, but the service has no authentication and should not be exposed
to an untrusted network.

## Run semantics

- One worker executes runs in FIFO order. Concurrent HTTP requests cannot run
  two GPU benchmarks at the same time.
- `prompt_mode=default` passes no prompt option and uses the benchmark's
  generated `Hello. ` input.
- `prompt_mode=file` passes the server-owned prompt file.
- `prompt_mode=text` passes only the submitted text as one argv element.
- Numeric values and enums are checked by the service before process launch.
- `context_size` must be at least `pp + tg`.
- Host-cache preload is disabled when Host cache is off, and hot-start is only
  valid with the EAMC predictor.
- Cancelling a running job sends `SIGTERM` to its process group, then `SIGKILL`
  after a three-second grace period if needed.

The service serializes execution, but it does not provide the GPU-idle wait,
foreign-process monitor, CPU affinity, or NUMA checks used by formal benchmark
runners. Configure and isolate the host separately before treating dashboard
results as controlled performance measurements.

## Results

Each run gets a separate directory under the output root containing:

```text
summary.txt       final human-readable benchmark summary
profile.csv       per-request and per-layer profiler rows
tokens.csv        generated token IDs
stdout.log        complete benchmark stdout
stderr.log        complete benchmark stderr
command.txt       shell-quoted command for display and reproduction
request.json      normalized web request
metadata.json     process and service metadata
result.json       persisted status, parsed metrics, and output tails
```

Run history is restored from `result.json` after a service restart. A run that
was queued or running when the service stopped is restored as failed rather
than silently resumed.

## API

```text
GET  /api/config
GET  /api/runs
POST /api/runs
GET  /api/runs/{id}
GET  /api/runs/{id}/events
POST /api/runs/{id}/cancel
GET  /api/runs/{id}/artifacts/{name}
```

The events endpoint uses Server-Sent Events and emits `snapshot`, `log`, and
`state` events. Artifact names are fixed by the server; arbitrary filesystem
paths are not accepted.

## Tests

The integration suite uses a temporary fake benchmark and does not require a
GPU, model, network access, or third-party Python packages:

```bash
python3 tools/moe-bench/tests/test_web_server.py \
    --server-bin build-moe/bin/llama-moe-bench-server -v
```
