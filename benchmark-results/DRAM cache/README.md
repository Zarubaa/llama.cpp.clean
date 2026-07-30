# DRAM Expert Cache Benchmark

This directory contains the reproducible evaluation of the host DRAM expert
cache added on top of the 0716 EAMC MoE offload path.

## Modes

| Result directory | Host cache | Preload | Data path |
|---|---|---|---|
| `control-0716` | off | none | page cache `fread` -> pinned staging -> H2D |
| `pageable-cold` | pageable | none | first read -> pageable cache -> pinned staging -> H2D |
| `pageable-hot` | pageable | all | pageable cache -> pinned staging -> H2D |
| `pinned-hot` | CUDA pinned | all | pinned cache -> H2D |

The primary cold definition applies only to the application-managed expert
cache. The model file must remain at least 99% resident in the Linux page
cache for every formal run.

## Reproduction

Build and validate first:

```bash
cmake --build build-moe --target llama-moe-bench -j 16
cmake --build build-moe-tests --target test-moe-admission test-moe-host-cache -j 16
build-moe-tests/bin/test-moe-admission
build-moe-tests/bin/test-moe-host-cache
"benchmark-results/DRAM cache/scripts/run-baseline-crosscheck.sh"
GPU_INDEX=0 "benchmark-results/DRAM cache/scripts/validate-correctness.sh"
GPU_INDEX=0 "benchmark-results/DRAM cache/scripts/run-microbench.sh"
```

Run the complete alternating matrix:

```bash
GPU_INDEX=0 RUNS=8 RUN_REAL_PROMPT=1 \
  "benchmark-results/DRAM cache/scripts/run-experiments.sh"
```

The completed experiment used GPU0 with CPU affinity `0-31,64-95`; these are
also the runner defaults. The runner refuses to start while that GPU has an
active compute process.

The runner is resumable. A run is skipped only when its summary, profile,
metadata, stdout, stderr, and token trace exist and the summary reports zero
host-cache verification failures.

Targeted CV extensions can be resumed without touching other cells by setting
`RUN_MAIN`, `RUN_REAL_PROMPT`, `RUN_TARGET_MODES`, `RUN_TARGET_VRAMS`, and
`RUN_TARGET_UBATCHES`.

## Timing boundaries

- `TTFT` starts immediately before measured prefill.
- `service cold-start TTFT` is Host allocation/preload time plus measured TTFT.
- Formal runs use one fresh process and `--repeat 1`.
- Per-run profile CSVs are retained to match the existing benchmark-results
  format. The separate diagnostic run provides a stable canonical profile for
  each case.
- `cudaHostAlloc` memory is verified by allocation mode, zero staging-copy
  bytes, and successful direct async H2D. NVIDIA pinned allocations are not
  necessarily included in Linux `VmLck` on this driver.

Run `scripts/summarize.py` after adding or replacing results. It regenerates
`summary-table.tsv`, every `case-summary.tsv`, and `final-report.md`.
