# 0730 DRAM Cache Comparison Report

Status: complete (36/36 validated runs).

All averages below are arithmetic means of three independent processes. Prefill and decode speeds are shown explicitly in tok/s.

## Findings

- Explicit pagecache-hot changed prefill latency by -1.72% to +0.58% and decode TPOT by -3.35% to +3.18% versus the naturally hot control. This is no stable performance gain because both paths began at 100% residency and retained the same timed `fread -> pinned staging -> H2D` path.
- Pinned-memory-hot reduced prefill latency by 53.85% to 59.97% and decode TPOT by 28.30% to 38.63% versus control. It is a full-cache upper bound, not a free optimization.
- Pinned-memory-hot used about 19.56 GiB peak RSS and about 16-17 seconds of model-init/preload time, versus about 1.3 GiB RSS and 1.4-3.6 seconds model init for the other modes.
- The 8000/1024 control and pagecache-hot TPOT standard deviations are about 6.3 ms. Their small mean difference should be treated as run-to-run noise; three repeats do not support a significance claim.

## Performance

| Mode | VRAM MiB | ubatch | n | Page resident % | Prepare ms | Model init ms | Prefill ms mean ± sd [range] | Prefill tok/s | TPOT ms mean ± sd [range] | Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control-original | 6000 | 512 | 3 | 100.0000 | 0.00 | 1559.02 | 7052.97 ± 824.49 [6445.30, 7991.50] | 146.45 | 37.76 ± 2.01 [36.53, 40.08] | 26.53 |
| pagecache-hot | 6000 | 512 | 3 | 100.0000 | 3440.23 | 1435.08 | 7012.13 ± 1076.47 [6167.10, 8224.10] | 148.22 | 39.02 ± 2.45 [36.59, 41.48] | 25.69 |
| pinned-memory-hot | 6000 | 512 | 3 | 100.0000 | 3116.04 | 17263.26 | 3047.90 ± 391.63 [2750.40, 3491.60] | 339.49 | 23.17 ± 1.72 [21.20, 24.32] | 43.32 |
| control-original | 6000 | 1024 | 3 | 100.0000 | 0.00 | 1456.97 | 4378.67 ± 113.21 [4263.40, 4489.70] | 233.97 | 39.82 ± 1.88 [37.65, 41.00] | 25.15 |
| pagecache-hot | 6000 | 1024 | 3 | 100.0000 | 3225.03 | 1517.17 | 4416.33 ± 271.73 [4232.80, 4728.50] | 232.43 | 39.41 ± 2.31 [36.98, 41.57] | 25.43 |
| pinned-memory-hot | 6000 | 1024 | 3 | 100.0000 | 3497.87 | 16406.47 | 2006.03 ± 228.97 [1784.40, 2241.70] | 514.90 | 25.00 ± 0.13 [24.85, 25.09] | 40.00 |
| control-original | 8000 | 512 | 3 | 100.0000 | 0.00 | 1465.72 | 6821.53 ± 197.27 [6678.10, 7046.50] | 150.20 | 31.78 ± 0.65 [31.25, 32.51] | 31.47 |
| pagecache-hot | 8000 | 512 | 3 | 100.0000 | 3562.77 | 1585.48 | 6913.43 ± 194.69 [6690.30, 7048.70] | 148.20 | 31.19 ± 1.95 [29.72, 33.40] | 32.15 |
| pinned-memory-hot | 8000 | 512 | 3 | 100.0000 | 3098.43 | 16015.54 | 2730.87 ± 440.92 [2459.20, 3239.60] | 381.03 | 22.79 ± 1.32 [21.51, 24.15] | 43.98 |
| control-original | 8000 | 1024 | 3 | 100.0000 | 0.00 | 3637.25 | 4814.97 ± 237.16 [4671.10, 5088.70] | 213.01 | 37.75 ± 6.30 [30.75, 42.95] | 27.03 |
| pagecache-hot | 8000 | 1024 | 3 | 100.0000 | 3073.38 | 3553.42 | 4897.97 ± 196.69 [4694.90, 5087.60] | 209.29 | 36.55 ± 6.25 [32.72, 43.76] | 27.85 |
| pinned-memory-hot | 8000 | 1024 | 3 | 100.0000 | 3269.95 | 16864.01 | 2221.87 ± 58.56 [2166.40, 2283.10] | 461.09 | 23.40 ± 0.67 [22.68, 24.00] | 42.75 |

## Relative prefill performance

| VRAM MiB | ubatch | pagecache-hot vs control | pinned vs control | pinned vs pagecache-hot |
|---:|---:|---:|---:|---:|
| 6000 | 512 | +0.58% | +56.79% | +56.53% |
| 6000 | 1024 | -0.86% | +54.19% | +54.58% |
| 8000 | 512 | -1.35% | +59.97% | +60.50% |
| 8000 | 1024 | -1.72% | +53.85% | +54.64% |

## Relative decode performance

| VRAM MiB | ubatch | pagecache-hot vs control | pinned vs control | pinned vs pagecache-hot |
|---:|---:|---:|---:|---:|
| 6000 | 512 | -3.35% | +38.63% | +40.62% |
| 6000 | 1024 | +1.02% | +37.21% | +36.56% |
| 8000 | 512 | +1.87% | +28.30% | +26.93% |
| 8000 | 1024 | +3.18% | +38.00% | +35.97% |

Positive values mean lower TPOT (faster decode).

## Data path

| Mode | VRAM MiB | ubatch | Physical post-prepare GiB | Physical inference GiB | Logical source GiB | Host memcpy GiB | H2D GiB | H2D ms/token | Stall ms/token | RSS GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control-original | 6000 | 512 | 0.00 | 0.00 | 149.98 | 0.00 | 149.98 | 9.75 | 4.48 | 1.30 |
| pagecache-hot | 6000 | 512 | 0.00 | 0.00 | 149.98 | 0.00 | 149.98 | 10.21 | 4.81 | 1.31 |
| pinned-memory-hot | 6000 | 512 | 0.38 | 0.00 | 0.00 | 0.00 | 149.98 | 6.78 | 2.26 | 19.56 |
| control-original | 6000 | 1024 | 0.00 | 0.00 | 146.20 | 0.00 | 146.20 | 11.35 | 4.02 | 1.31 |
| pagecache-hot | 6000 | 1024 | 0.00 | 0.00 | 146.20 | 0.00 | 146.20 | 11.43 | 4.63 | 1.31 |
| pinned-memory-hot | 6000 | 1024 | 0.00 | 0.00 | 0.00 | 0.00 | 146.20 | 7.54 | 2.77 | 19.56 |
| control-original | 8000 | 512 | 0.00 | 0.00 | 111.25 | 0.00 | 111.25 | 8.67 | 3.42 | 1.29 |
| pagecache-hot | 8000 | 512 | 0.00 | 0.00 | 111.25 | 0.00 | 111.25 | 7.86 | 3.11 | 1.30 |
| pinned-memory-hot | 8000 | 512 | 0.00 | 0.00 | 0.00 | 0.00 | 111.25 | 5.26 | 1.94 | 19.56 |
| control-original | 8000 | 1024 | 0.00 | 0.00 | 107.99 | 0.00 | 107.99 | 11.49 | 6.57 | 1.30 |
| pagecache-hot | 8000 | 1024 | 0.00 | 0.00 | 107.99 | 0.00 | 107.99 | 12.52 | 5.00 | 1.31 |
| pinned-memory-hot | 8000 | 1024 | 0.00 | 0.00 | 0.00 | 0.00 | 107.99 | 5.37 | 1.97 | 19.56 |

## Validation

- All 36 runs used exactly 1024 prompt tokens and generated 1024 decode steps from the fixed realistic prompt.
- Token traces, ordered route hashes, VRAM misses, and H2D bytes matched within every VRAM/ubatch cell.
- Every explicit hot run began at >=99% Linux page-cache residency. Pagecache-hot stayed below 1% physical reads after prepare; pinned stayed below 1% during prefill/decode.
- Every pinned run reached 30,720 READY blobs with zero timed source reads and zero staging memcpy bytes.

## Interpretation

- `control-original` leaves Linux page cache untouched. Its recorded residency must be used when interpreting it; it is not a cold control.
- `pagecache-hot` and `control-original` both retain the original `fread -> pinned staging -> H2D` timed path when the control is naturally hot.
- `pinned-memory-hot` is the upper bound that removes timed source reads and staging memcpy, but includes full pinned-cache allocation/preload in model-init/startup metrics.
- Pinned model-init physical reads are retained in the post-prepare column; they are preload cost, while the separate inference column covers only prefill and decode.
- Profiler `ssd_reads` are logical source reads. `/proc/self/io read_bytes` is the primary process-level physical-I/O evidence.
