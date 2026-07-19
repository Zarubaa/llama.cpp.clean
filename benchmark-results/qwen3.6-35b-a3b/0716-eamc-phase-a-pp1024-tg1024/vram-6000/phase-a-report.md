# 0716_eamc Phase A Report

Date: 2026-07-16

## Configuration

- Branch: `0716_eamc`, base HEAD `539f4cbf4`
- Container: `llama-cpp-clean-dev`
- GPU: physical GPU 1, NVIDIA L20; 3 MiB used before and after formal runs
- Model: `Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf`
- Cache: 6000 MiB, 60 persistent slots/layer, 256 active slots/layer
- Predictor: LRU
- Workload: pp1024, tg1024, ub1024, greedy decode
- Formal timing: no profile CSV; 3 repeats with cache reset between repeats
- Early-decode analysis: profile CSV enabled; two alternating old/new runs

The old binary was built from the unmodified base HEAD in an isolated `/tmp`
source tree. Old and new runs used the same container, model, GPU, page cache,
and command-line parameters. Completed benchmark processes were explicitly
terminated after `[moe-bench] done.` because the existing binary does not exit
after writing its summary.

## Implementation

- Count raw top-k occurrences per layer and rank unique experts by
  `count desc, first occurrence asc, expert id asc`.
- Admit high-frequency misses to EMPTY persistent slots first.
- Preserve raw route multiplicity in predictor observations.
- Touch admitted/current persistent experts in reverse rank order so the
  highest-frequency expert ends at the MRU side.
- Scan the actual LRU list from its tail when predictor scores tie, instead of
  depending on `unordered_map` iteration order.
- Add route-level profiler fields without changing the existing unique-expert
  `k_required/k_hit/k_miss` semantics.

## Timing Result

Primary result: no CSV, `repeat=3`, cache reset before every repeat.

| Metric | Old | Phase A | Change |
|---|---:|---:|---:|
| TTFT | 4100.9 ms | 3993.7 ms | -2.61% |
| Decode total | 27369.1 ms | 27198.9 ms | -0.62% |
| TPOT | 26.73 ms | 26.56 ms | -0.64% |
| Total request | 31470.0 ms | 31192.6 ms | -0.88% |
| Decode SSD / token | 43.11 MiB | 42.22 MiB | -2.06% |

A separate pair of two-run alternating tests produced a smaller decode gain
(`-0.47%`) and effectively unchanged total time (`-0.10%`). The wall-time
effect is therefore modest; the route/miss reduction below is deterministic.

## Cache Result

Single traced request, identical old/new route-shape trace:

| Metric | Old | Phase A | Change |
|---|---:|---:|---:|
| Decode unique expert misses | 24219 | 23738 | -481 (-1.99%) |
| Decode hit rate | 92.6089% | 92.7557% | +0.1468 pp |
| Decode SSD bytes | 43.08 GiB | 42.22 GiB | -0.86 GiB (-2.00%) |
| Prefill persistent route coverage | unavailable | 91.059% | new metric |
| Route ranking CPU time | unavailable | 10.56 ms / 1024 decode tokens | new metric |

The first 8 decode tokens lose 418 misses, from 1112 to 694 (`-37.6%`). The
first 128 tokens lose 482 misses, from 8447 to 7965 (`-5.7%`). Almost all
request-wide savings therefore come from early decode; token 128 onward is
unchanged.

| Decode range | Old hit | Phase A hit | Old miss/token | Phase A miss/token |
|---|---:|---:|---:|---:|
| 0 | 7.50% | 89.69% | 296.00 | 33.00 |
| 1-7 | 63.57% | 70.49% | 116.57 | 94.43 |
| 8-31 | 77.58% | 78.28% | 71.75 | 69.50 |
| 32-127 | 81.73% | 81.76% | 58.47 | 58.36 |
| 128-1023 | 94.50% | 94.50% | 17.60 | 17.60 |

## Validation

- Release CUDA build completed in `llama-cpp-clean-dev`.
- `test-moe-admission` passed.
- Diagnostic pp1024 run emitted 40 admission rows; all 40 layers had their
  60 persistent residents exactly equal to the top-60 frequency ranking.
- All 41000 formal profile layer rows passed:
  `hit + miss == required`, admission partition, route coverage bounds, and
  41-column CSV consistency.
- Repeated old runs produced exactly 24219 misses and repeated new runs exactly
  23738 misses, confirming deterministic policy behavior.

## Conclusion

Phase A fixes the initial EMPTY-slot selection problem and materially reduces
the first-token cold miss burst. At this workload it reduces SSD traffic by
about 2% and gives a small end-to-end speedup; it does not affect the stable
decode region. Further decode gains require prefetch or a prefill-to-decode
reshape rather than additional work on this admission rule alone.

Raw artifacts are under `ab/`, `ab-no-csv/`, `repeat3-no-csv/`, and
`diagnostic/`. `early-decode-comparison.csv` contains the machine-readable
window summary.
