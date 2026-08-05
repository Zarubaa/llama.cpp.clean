# Repeat-1 LRU benchmark fix and rerun

## Scope

This directory contains the corrected repeat-1 results for the two distinct
workloads represented by the historical 8000 MiB / ubatch 512 results. The
tests use the non-MTP Q4 model, LRU, host cache off, pp=1024, tg=1024, and one
fresh process per case.

The code fix is limited to `tools/moe-bench/main.cpp`:

- select the final output with `llama_get_logits_ith(ctx, -1)` and fail on a
  missing logits row;
- include final-logits synchronization and argmax in TTFT;
- measure decode with a separate timer and record TPOT/total only after all
  requested tokens complete;
- move page-cache, memory, summary, and token-trace work outside measured
  intervals;
- require exact two-pass tokenization and report source/used token counts and
  the used-token hash;
- validate token/logits trace headers and final flushes.

No LRU policy, slot capacity, Host-cache path, page-cache policy, async I/O,
or H2D implementation was changed.

## Environment

- Branch/HEAD: `0805_dram-bench` / `9f0199dbb990e89e600f3dce43629d18ef355bed`
- Container: `llama-cpp-clean-dev`, repository mounted at `/workspace`
- Binary: `/workspace/build-moe/bin/llama-moe-bench`
- Binary SHA256: `d0c6ba506f426610621d33b715bc47ecf1029e5d6c5105d32c68b43ed95dd306`
- Model SHA256: `4674317e422518acafef023522768e25a42b58e22f960e5a83133bedd0cd14b4`
- Prompt SHA256: `78fd02cac487c229f9e2399ea35905847dd1823a641cca7f762facc714ea1e80`
- GPU: physical GPU3, NVIDIA L20, NUMA node1
- CPU affinity: `32-63,96-127`
- Model page cache: targeted eviction followed by a node1-bound full read;
  every measured process started at 100% residency
- GPU isolation: 60-second stable-idle gate plus a host-side process monitor

Each formal command uses the following options; `-ub` is 512 for the original
pair and 1024 for the `ub1024/` pair:

```text
-ngl 99 -c 4096 --pp 1024 --tg 1024 --repeat 1 -ub <512|1024>
--moe-cache-vram-mb 8000 --moe-predictor lru
--moe-host-cache off --moe-host-cache-preload none
--page-cache-policy natural
```

## Results

| ubatch | Case | Input hash | TTFT | Prefill | TPOT | Decode | Decode hit | Logical decode source |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | default prompt | `d702a11d89198696` | 3197.9 ms | 320 tok/s | 9.31 ms | 107 tok/s | 99.7% | 1.48 GiB |
| 512 | normal prompt | `318c1480f039b981` | 6999.3 ms | 146 tok/s | 46.31 ms | 22 tok/s | 81.8% | 105.87 GiB |
| 1024 | default prompt | `d702a11d89198696` | 3136.0 ms | 327 tok/s | 9.55 ms | 105 tok/s | 99.7% | 2.02 GiB |
| 1024 | normal prompt | `318c1480f039b981` | 5262.0 ms | 195 tok/s | 49.92 ms | 20 tok/s | 80.5% | 113.77 GiB |

The default input is the benchmark's generated repeated `Hello. ` text. Its
decode alternates between a very small set of tokens, reaches a 99.7% cache
hit rate, and is not representative of a normal request. The normal prompt
uses the requested file and produces a substantially broader route trace.

The model was 100% resident for all four runs. Three runs report zero process
`read_bytes` during inference; ubatch 1024/default has only 774,144 bytes of
process reads during prefill. This small accounting difference does not
represent a cold SSD pass. The large reported source reads are logical expert
transfers served from the Linux page cache/DRAM, not physical SSD traffic.

The two normal-prompt runs have the same prompt hash and first token, but their
greedy traces diverge at decode step 7 after changing ubatch. Different batch
shapes can change GPU floating-point reduction order. This makes the timings
valid for their respective configurations, but route-dependent cache/source
bytes should not be interpreted as a strict same-token-path comparison.

These corrected traces must not be compared as if they were the same request
as old token-0 results. The previous invalid prefill logits changed the first
generated token and therefore the entire autoregressive route sequence.

## Validation

- The final build completed inside `llama-cpp-clean-dev`.
- The pp=8/tg=4 smoke trace starts with token `95789`, not token 0, and includes
  logits records for every step.
- All four formal traces contain one header plus 1025 token rows.
- Formal first tokens are `21251` (default) and `103859` (normal) for both
  ubatch values.
- Neither formal stderr contains `invalid logits`, `failed`, or `ERROR`.
- The formal profile CSVs have the expected schema and 42065 (ubatch 512) or
  42025 (ubatch 1024) data rows.
- Host-cache verification reports zero failures.
- GPU monitors contain only the corresponding `llama-moe-bench` process.

The process-wide NUMA-local ratios were 94.623% and 94.733%. The container
does not permit strict `set_mempolicy`, so this is retained as metadata rather
than used to alter or normalize the timings. CPU affinity and page-cache warmup
were still pinned to node1.

## Files

- `results.tsv`: compact comparison table
- `benchfix.patch`: exact `tools/moe-bench/main.cpp` working-tree diff
- `default-prompt/`, `normal-prompt/`: ubatch 512 summary, profile, token
  trace, logs, command metadata, GPU monitor, and memory samples
- `ub1024/default-prompt/`, `ub1024/normal-prompt/`: corresponding ubatch 1024
  artifacts
- `smoke/`: pp=8/tg=4 correctness run with binary logits
- `invalid-run/`: rejected/interrupted attempts retained for audit only
- `run_fixed_tests.sh`: host gate plus container benchmark runner

Formal performance runs intentionally omit `--logits-bin`; writing a complete
vocabulary row per token is a correctness diagnostic and would perturb TPOT.
Only one formal run per workload was collected, as requested, so the table has
no variance or confidence interval.
