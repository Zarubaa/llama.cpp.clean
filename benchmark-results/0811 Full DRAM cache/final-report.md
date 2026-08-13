# 0811 Full DRAM cache Report

Validated runs: 48

Each prompt_id is a distinct workload sample. A repeat is a fresh-process rerun of the same prompt.
All cache modes must use the same prompt_id before their paired delta is interpreted.

## Protocol

- GPU2, CPU affinity `32-63,96-127`, `pp=1024`, `tg=1024`, prompts 01-04.
- Each table cell contains four different prompt samples, one fresh process per prompt; these are not four repeats of one prompt.
- `control-original` uses natural Linux page cache, `pagecache-hot` explicitly reads the GGUF before inference, and `pinned-memory-hot` preloads all expert weights into pinned Host DRAM.
- The initial control run began with the GPU at 210 MHz. Its complete original artifacts are preserved under `diagnostics/original-cold-clock-run`; the table uses the verified 2520 MHz rerun. Both token and route traces match the other modes.

## Performance

The table reports four-prompt workload means and medians. Speeds are explicit in token/s; ranges are prompt-to-prompt variation, not process-repeat noise.

| Mode | VRAM MiB | ubatch | prompts | Prefill tok/s mean | median | range | Decode tok/s mean | median | range | TPOT ms mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control-original | 6000 | 512 | 4 | 192.89 | 192.87 | 187.83-197.99 | 16.74 | 16.98 | 14.98-18.02 | 60.07 |
| control-original | 6000 | 1024 | 4 | 262.80 | 261.52 | 260.36-267.78 | 16.86 | 16.84 | 15.15-18.60 | 59.69 |
| control-original | 8000 | 512 | 4 | 197.76 | 198.37 | 195.32-198.98 | 22.05 | 22.43 | 19.59-23.75 | 45.66 |
| control-original | 8000 | 1024 | 4 | 248.60 | 250.56 | 239.75-253.52 | 22.03 | 22.12 | 19.84-24.07 | 45.63 |
| pagecache-hot | 6000 | 512 | 4 | 186.16 | 187.21 | 170.06-200.17 | 18.10 | 17.78 | 15.47-21.35 | 55.99 |
| pagecache-hot | 6000 | 1024 | 4 | 262.64 | 262.72 | 257.17-267.94 | 16.63 | 16.67 | 14.85-18.33 | 60.52 |
| pagecache-hot | 8000 | 512 | 4 | 200.09 | 199.74 | 195.72-205.15 | 22.15 | 22.33 | 19.98-23.97 | 45.38 |
| pagecache-hot | 8000 | 1024 | 4 | 254.86 | 255.58 | 251.78-256.51 | 22.11 | 22.17 | 20.16-23.97 | 45.45 |
| pinned-memory-hot | 6000 | 512 | 4 | 467.31 | 444.74 | 335.09-644.67 | 39.61 | 36.97 | 33.70-50.81 | 25.87 |
| pinned-memory-hot | 6000 | 1024 | 4 | 457.80 | 452.69 | 443.12-482.72 | 35.53 | 35.69 | 33.17-37.59 | 28.20 |
| pinned-memory-hot | 8000 | 512 | 4 | 350.58 | 349.19 | 345.57-358.37 | 38.96 | 39.61 | 35.35-41.28 | 25.77 |
| pinned-memory-hot | 8000 | 1024 | 4 | 438.23 | 438.99 | 429.06-445.88 | 39.84 | 40.33 | 37.59-41.09 | 25.14 |

## Paired Change Versus Control

Positive speed deltas mean faster. Positive TPOT improvement means lower latency. Every delta uses the same prompt_id.

| Mode | VRAM MiB | ubatch | prompts | Prefill speed mean | median | Decode speed mean | median | TPOT improvement mean | median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pagecache-hot | 6000 | 512 | 4 | -3.56% | -2.86% | +9.21% | -0.60% | +5.98% | -0.60% |
| pagecache-hot | 6000 | 1024 | 4 | -0.04% | -0.06% | -1.36% | -1.25% | -1.37% | -1.27% |
| pagecache-hot | 8000 | 512 | 4 | +1.17% | +0.85% | +0.56% | +0.99% | +0.54% | +0.99% |
| pagecache-hot | 8000 | 1024 | 4 | +2.57% | +1.22% | +0.39% | +0.46% | +0.38% | +0.45% |
| pinned-memory-hot | 6000 | 512 | 4 | +144.01% | +131.95% | +138.00% | +119.12% | +56.71% | +54.33% |
| pinned-memory-hot | 6000 | 1024 | 4 | +74.24% | +72.43% | +111.81% | +111.21% | +52.53% | +52.59% |
| pinned-memory-hot | 8000 | 512 | 4 | +77.27% | +77.28% | +77.01% | +77.13% | +43.47% | +43.52% |
| pinned-memory-hot | 8000 | 1024 | 4 | +76.32% | +76.51% | +81.35% | +82.59% | +44.75% | +45.18% |

## Statistics

`summary-table.tsv` has one row per process. `prompt-means.tsv` averages process repeats within each prompt. `paired-deltas.tsv` subtracts control for the same prompt, VRAM budget, and ubatch. Prompt-to-prompt standard deviation is workload variation, not scheduler noise.

## Coverage

Prompt IDs observed: 01, 02, 03, 04
Modes observed: control-original, pagecache-hot, pinned-memory-hot

## Validation

All prompt groups contain three modes with matching prompt SHA, token trace SHA, and route trace SHA.

## Interpretation

- Explicit page-cache preheating has no consistent steady-state benefit once the control is already 100% resident. Excluding the prompt-01 outlier, paired decode changes are approximately -1.9% to +1.6%; the 6000/512 mean is raised by one prompt-01 result, while its median is -0.60%.
- Full pinned Host DRAM is a clear upper bound: mean decode reaches 35.53-39.84 token/s and mean prefill reaches 350.58-467.31 token/s, substantially above the file/page-cache path.
- The large prompt-to-prompt range, especially pinned prefill at 6000/512, is workload variation. With one process per prompt it must not be reported as repeat-run variance.
