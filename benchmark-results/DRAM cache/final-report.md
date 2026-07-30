# DRAM Expert Cache Report

Status: complete primary matrix (128 base formal runs + 7 CV-extension runs + 16 diagnostics; 135 formal total).
The real-prompt robustness matrix contains 32 base runs + 14 CV-extension runs (46 total).

## Main result

| Mode | VRAM MiB | ubatch | n | Prefill ms | Prefill tok/s | TPOT ms | Decode tok/s | 95% median CI | CV % | vs control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control-0716 | 6000 | 512 | 8 | 5512.10 | 186.00 | 29.88 | 33.00 | [29.55, 30.51] | 3.82 | baseline |
| control-0716 | 6000 | 1024 | 8 | 4103.30 | 250.00 | 29.66 | 34.00 | [29.58, 29.99] | 0.71 | baseline |
| control-0716 | 8000 | 512 | 8 | 5259.45 | 195.00 | 21.99 | 45.00 | [21.81, 22.21] | 1.00 | baseline |
| control-0716 | 8000 | 1024 | 8 | 4402.45 | 232.50 | 22.24 | 45.00 | [22.01, 22.37] | 1.29 | baseline |
| pageable-cold | 6000 | 512 | 8 | 13532.00 | 75.50 | 38.69 | 26.00 | [38.27, 39.00] | 3.68 | -29.24% |
| pageable-cold | 6000 | 1024 | 8 | 11932.50 | 86.00 | 38.46 | 26.00 | [38.25, 38.71] | 0.58 | -29.79% |
| pageable-cold | 8000 | 512 | 8 | 13366.75 | 77.00 | 30.99 | 32.00 | [30.86, 31.22] | 0.73 | -41.31% |
| pageable-cold | 8000 | 1024 | 8 | 12082.75 | 84.50 | 30.87 | 32.00 | [30.72, 31.29] | 0.87 | -39.03% |
| pageable-hot | 6000 | 512 | 15 | 8529.30 | 120.00 | 37.91 | 26.00 | [37.79, 38.11] | 4.15 | -27.52% |
| pageable-hot | 6000 | 1024 | 8 | 6474.50 | 158.00 | 37.72 | 26.50 | [35.69, 38.29] | 3.91 | -26.50% |
| pageable-hot | 8000 | 512 | 8 | 8443.50 | 121.00 | 28.12 | 36.00 | [28.00, 28.43] | 1.00 | -27.94% |
| pageable-hot | 8000 | 1024 | 8 | 6595.30 | 155.50 | 28.88 | 35.00 | [28.54, 29.21] | 1.46 | -29.83% |
| pinned-hot | 6000 | 512 | 8 | 3136.20 | 326.50 | 20.29 | 49.00 | [20.15, 20.71] | 2.84 | +32.23% |
| pinned-hot | 6000 | 1024 | 8 | 2392.20 | 428.00 | 20.08 | 50.00 | [19.88, 20.21] | 1.58 | +32.55% |
| pinned-hot | 8000 | 512 | 8 | 2870.45 | 356.50 | 17.98 | 56.00 | [17.71, 18.57] | 1.89 | +18.56% |
| pinned-hot | 8000 | 1024 | 8 | 2310.40 | 443.00 | 17.91 | 56.00 | [17.75, 18.20] | 2.66 | +19.84% |

Positive `vs control` is a speedup; negative is a slowdown.

## Data-path breakdown

Selected cell: 8000 MiB VRAM cache, ubatch 512.

| Mode | Host hit % | Source GiB | Host memcpy ms/tok | H2D ms/tok | Stall ms/tok | RSS GiB | Preload ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| control-0716 | 0.00 | 44.41 | 0.00 | 5.70 | 2.90 | 1.28 | 0.00 |
| pageable-cold | 64.56 | 15.76 | 13.31 | 5.59 | 3.05 | 17.04 | 0.44 |
| pageable-hot | 100.00 | 0.00 | 14.77 | 5.66 | 2.71 | 19.51 | 10461.38 |
| pinned-hot | 100.00 | 0.00 | 0.00 | 2.24 | 1.03 | 19.53 | 9889.08 |

## Pageable-cold warmup

Diagnostic profile for 8000 MiB / ubatch 512:

| Decode window | TPOT ms | Decode tok/s | READY blobs at end | READY GiB | Host hit % | Source reads | Source GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 43.86 | 22.80 | 21105 | 12.50 | 95.24 | 3 | 0.00 |
| 1-7 | 113.58 | 8.80 | 21573 | 12.78 | 70.40 | 468 | 0.28 |
| 8-31 | 90.19 | 11.09 | 22884 | 13.57 | 68.22 | 1311 | 0.78 |
| 32-127 | 56.02 | 17.85 | 24093 | 14.28 | 87.41 | 1209 | 0.72 |
| 128-1023 | 23.04 | 43.41 | 26577 | 15.76 | 92.47 | 2484 | 1.47 |

## Real-prompt robustness

The fixed Chinese-English prompt uses 8000 MiB / ubatch 512.

| Mode | n | Prefill ms | Prefill tok/s | TPOT median ms | Decode tok/s | TPOT p95 ms | CV % |
|---|---:|---:|---:|---:|---:|---:|---:|
| control-0716 | 8 | 5411.50 | 189.00 | 22.43 | 44.50 | 24.42 | 4.67 |
| pageable-cold | 8 | 13726.15 | 74.50 | 31.46 | 31.50 | 33.88 | 4.61 |
| pageable-hot | 15 | 8232.00 | 124.00 | 28.37 | 35.00 | 32.52 | 6.69 |
| pinned-hot | 15 | 1558.20 | 657.00 | 13.75 | 73.00 | 18.62 | 18.41 |

## Validation and controls

- Four-mode token traces matched and max absolute logits difference was 0 (threshold 1e-5).
- The manifest contains 30,720 blobs; hot modes reached 30,720 READY blobs and reported zero timed source reads.
- Up to 256 READY blobs per run were checked against source bytes with zero verification failures.
- The primary pageable-hot 6000/512 cell was extended to 15 runs; its final TPOT CV is 4.15%.
- Real-prompt pageable-hot and pinned-hot were extended to 15 runs. Their final CVs remain 6.69% and 18.41%, respectively; all token traces and transfer volumes match, so the residual variation is retained rather than filtered.
- Minimum recorded Linux page-cache residency before a formal run was 100.0000%.
- Minimum recorded NUMA-node-0 share at peak sampled RSS was 95.32%.
- CUDA pinned allocations are not charged to VmLck by this NVIDIA driver. Pinned validity is based on successful cudaHostAlloc, zero staging memcpy bytes, and direct async H2D.
- Archived 0716 baseline TPOT median was 35.30 ms versus 32.71 ms for experiment-off (-7.34%). This exceeds the 3% cross-check threshold, so cross-version comparisons remain qualified; the A/B/C/D table uses one experiment binary.

## Microbenchmark

| Topology | Operation | Mean us | P95 us | Aggregate GiB/s |
|---|---|---:|---:|---:|
| local | fread-pinned | 89.40 | 119.00 | 6.64 |
| local | pageable-pinned | 53.02 | 77.00 | 11.19 |
| local | pageable-staging-h2d | 83.87 | 111.00 | 7.08 |
| remote | fread-pinned | 89.79 | 120.00 | 6.61 |
| remote | pageable-pinned | 62.43 | 115.90 | 9.51 |
| remote | pageable-staging-h2d | 94.83 | 158.25 | 6.26 |
| local | pinned-h2d | 30.85 | 35.00 | 19.24 |
| remote | pinned-h2d | 32.40 | 37.00 | 18.32 |

## Existing benchmark comparison

Existing `benchmark-results/new-result` values are single historical runs and are shown only as a sanity check.

| VRAM MiB | ubatch | Historical TPOT ms | Current control median ms |
|---:|---:|---:|---:|
| 6000 | 512 | 25.76 | 29.88 |
| 6000 | 1024 | 30.19 | 29.66 |
| 8000 | 512 | 18.57 | 21.99 |
| 8000 | 1024 | 19.81 | 22.24 |

## Decision

- Pageable-hot versus control: -27.94% paired TPOT change, bootstrap 95% CI [-29.67%, -26.07%]. The required +10% gain is not met.
- Pinned-hot versus pageable-hot: +36.00%, bootstrap 95% CI [+34.91%, +36.86%].
- Pinned-hot versus control: +18.56% at the selected cell.
- Pinned H2D plus measured stall is 18.19% of TPOT. This exceeds the 15% Fate threshold, so hidden-state prefetch remains worth testing.
- Do not deploy the current full pageable cache as a synchronous replacement for fread. Its host memcpy and staging path is slower than the hot Linux page-cache control.
- Use the existing pinned staging pool as the Fate destination path, prefetch asynchronously ahead of demand, and evaluate a bounded pinned hot-expert cache instead of pinning all 18.22 GiB by default.

Machine-readable companion files: `summary-table.tsv`, `real-prompt-summary.tsv`, `pageable-cold-warmup.tsv`, and per-cell `case-summary.tsv`.
Raw summaries, profiles, logs, metadata, memory samples, and token traces remain below each mode/cell directory.
