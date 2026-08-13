# 0730 DRAM Cache Comparison Report

Status: complete (36/36 validated runs).

All averages below are arithmetic means of three independent processes. Prefill and decode speeds are shown explicitly in tok/s.

## Findings

- Explicit pagecache-hot changed prefill latency by -4.69% to +6.72% and decode TPOT by -0.83% to +4.94% versus the naturally hot control. This is no stable performance gain because both paths began at 100% residency and retained the same timed `fread -> pinned staging -> H2D` path.
- Pinned-memory-hot reduced prefill latency by 22.42% to 66.98% and decode TPOT by 54.42% to 67.50% versus control. It is a full-cache upper bound, not a free optimization.
- Pinned-memory-hot used about 19.56 GiB peak RSS and about 16-17 seconds of model-init/preload time, versus about 1.3 GiB RSS and 1.4-3.6 seconds model init for the other modes.
- The 8000/1024 control and pagecache-hot TPOT standard deviations are about 6.3 ms. Their small mean difference should be treated as run-to-run noise; three repeats do not support a significance claim.

## Performance

| Mode | VRAM MiB | ubatch | n | Page resident % | Prepare ms | Model init ms | Prefill ms mean ± sd [range] | Prefill tok/s | TPOT ms mean ± sd [range] | Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control-original | 6000 | 512 | 3 | 100.0000 | 0.00 | 9836.21 | 8314.43 ± 136.98 [8214.60, 8470.60] | 123.18 | 159.91 ± 0.37 [159.48, 160.15] | 6.25 |
| pagecache-hot | 6000 | 512 | 3 | 100.0000 | 3500.83 | 5406.31 | 8311.90 ± 42.37 [8263.10, 8339.30] | 123.20 | 161.23 ± 3.27 [157.46, 163.28] | 6.20 |
| pinned-memory-hot | 6000 | 512 | 3 | 100.0000 | 3523.44 | 36454.44 | 6450.10 ± 1818.64 [4757.40, 8372.80] | 167.39 | 61.25 ± 9.69 [51.53, 70.91] | 16.61 |
| control-original | 6000 | 1024 | 3 | 100.0000 | 0.00 | 10689.12 | 4892.70 ± 18.92 [4880.60, 4914.50] | 209.29 | 160.79 ± 2.30 [159.26, 163.44] | 6.22 |
| pagecache-hot | 6000 | 1024 | 3 | 100.0000 | 3499.65 | 6416.64 | 5121.97 ± 395.21 [4844.70, 5574.50] | 200.69 | 160.84 ± 4.09 [157.95, 165.52] | 6.22 |
| pinned-memory-hot | 6000 | 1024 | 3 | 100.0000 | 3498.48 | 32537.51 | 2447.07 ± 1679.86 [1476.80, 4386.80] | 539.94 | 52.25 ± 0.35 [51.85, 52.49] | 19.14 |
| control-original | 8000 | 512 | 3 | 100.0000 | 0.00 | 9697.41 | 8210.77 ± 59.50 [8146.00, 8263.00] | 124.72 | 100.28 ± 35.79 [58.97, 121.93] | 11.17 |
| pagecache-hot | 8000 | 512 | 3 | 100.0000 | 3493.38 | 6164.25 | 7659.37 ± 977.88 [6530.30, 8236.40] | 135.28 | 95.33 ± 44.03 [44.48, 120.83] | 13.02 |
| pinned-memory-hot | 8000 | 512 | 3 | 100.0000 | 3534.29 | 32391.17 | 4653.50 ± 396.12 [4196.20, 4890.60] | 221.17 | 45.71 ± 2.18 [44.30, 48.22] | 21.91 |
| control-original | 8000 | 1024 | 3 | 100.0000 | 0.00 | 7839.37 | 4787.27 ± 60.22 [4718.40, 4830.00] | 213.92 | 45.22 ± 0.26 [44.96, 45.48] | 22.11 |
| pagecache-hot | 8000 | 1024 | 3 | 100.0000 | 3495.27 | 7724.59 | 4725.13 ± 60.19 [4660.70, 4779.90] | 216.74 | 45.37 ± 0.28 [45.15, 45.68] | 22.04 |
| pinned-memory-hot | 8000 | 1024 | 3 | 100.0000 | 3493.34 | 17843.58 | 1580.53 ± 96.11 [1524.00, 1691.50] | 649.43 | 15.69 ± 0.02 [15.67, 15.70] | 63.75 |

## Relative prefill performance

| VRAM MiB | ubatch | pagecache-hot vs control | pinned vs control | pinned vs pagecache-hot |
|---:|---:|---:|---:|---:|
| 6000 | 512 | +0.03% | +22.42% | +22.40% |
| 6000 | 1024 | -4.69% | +49.99% | +52.22% |
| 8000 | 512 | +6.72% | +43.32% | +39.24% |
| 8000 | 1024 | +1.30% | +66.98% | +66.55% |

## Relative decode performance

| VRAM MiB | ubatch | pagecache-hot vs control | pinned vs control | pinned vs pagecache-hot |
|---:|---:|---:|---:|---:|
| 6000 | 512 | -0.83% | +61.70% | +62.01% |
| 6000 | 1024 | -0.03% | +67.50% | +67.51% |
| 8000 | 512 | +4.94% | +54.42% | +52.05% |
| 8000 | 1024 | -0.34% | +65.31% | +65.43% |

Positive values mean lower TPOT (faster decode).

## Data path

| Mode | VRAM MiB | ubatch | Physical post-prepare GiB | Physical inference GiB | Logical source GiB | Host memcpy GiB | H2D GiB | H2D ms/token | Stall ms/token | RSS GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control-original | 6000 | 512 | 0.00 | 0.00 | 149.98 | 0.00 | 149.98 | 25.45 | 10.82 | 1.30 |
| pagecache-hot | 6000 | 512 | 0.00 | 0.00 | 149.98 | 0.00 | 149.98 | 25.93 | 13.45 | 1.31 |
| pinned-memory-hot | 6000 | 512 | 0.00 | 0.00 | 0.00 | 0.00 | 149.98 | 6.91 | 2.29 | 19.79 |
| control-original | 6000 | 1024 | 0.00 | 0.00 | 146.20 | 0.00 | 146.20 | 26.84 | 13.43 | 1.31 |
| pagecache-hot | 6000 | 1024 | 0.00 | 0.00 | 146.20 | 0.00 | 146.20 | 26.56 | 13.27 | 1.31 |
| pinned-memory-hot | 6000 | 1024 | 0.00 | 0.00 | 0.00 | 0.00 | 146.20 | 6.15 | 1.19 | 19.79 |
| control-original | 8000 | 512 | 0.00 | 0.00 | 111.25 | 0.00 | 111.25 | 18.22 | 9.78 | 1.30 |
| pagecache-hot | 8000 | 512 | 0.00 | 0.00 | 111.25 | 0.00 | 111.25 | 18.22 | 7.80 | 1.31 |
| pinned-memory-hot | 8000 | 512 | 0.00 | 0.00 | 0.00 | 0.00 | 111.25 | 4.19 | 0.79 | 19.79 |
| control-original | 8000 | 1024 | 0.00 | 0.00 | 107.99 | 0.00 | 107.99 | 18.99 | 9.59 | 1.31 |
| pagecache-hot | 8000 | 1024 | 0.00 | 0.00 | 107.99 | 0.00 | 107.99 | 18.74 | 10.41 | 1.31 |
| pinned-memory-hot | 8000 | 1024 | 0.00 | 0.00 | 0.00 | 0.00 | 107.99 | 4.37 | 0.85 | 19.79 |

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
