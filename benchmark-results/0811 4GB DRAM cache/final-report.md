# 0811 4GB DRAM cache Report

Validated runs: 48

Each prompt_id is a distinct workload sample. A repeat is a fresh-process rerun of the same prompt.
All cache modes must use the same prompt_id before their paired delta is interpreted.

## Protocol

- GPU1, CPU affinity `0-31,64-95`, `pp=1024`, `tg=1024`, prompts 01-04.
- Each table cell contains four different prompt samples, one fresh process per prompt; these are not four repeats of one prompt.
- `control` is the frozen baseline binary. `host4gb-cold` starts with an empty 4096 MiB pinned cache; `host4gb-hot` preloads a prompt-specific 2240-expert hotset.
- Sixteen prompt/cell calibration processes generated the hotsets and are stored under `calibration`; they are excluded from all performance means.
- The initial control run began with the GPU at 210 MHz. Its complete original artifacts are preserved under `diagnostics/original-cold-clock-run`; the table uses the verified 2520 MHz rerun.

## Performance

The table reports four-prompt workload means and medians. Speeds are explicit in token/s; ranges are prompt-to-prompt variation, not process-repeat noise.

| Mode | VRAM MiB | ubatch | prompts | Prefill tok/s mean | median | range | Decode tok/s mean | median | range | TPOT ms mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 6000 | 512 | 4 | 185.34 | 188.22 | 173.59-191.33 | 16.55 | 16.48 | 15.10-18.16 | 60.73 |
| control | 6000 | 1024 | 4 | 259.66 | 259.25 | 256.69-263.44 | 16.51 | 16.48 | 14.92-18.17 | 60.91 |
| control | 8000 | 512 | 4 | 198.16 | 197.64 | 194.86-202.50 | 21.91 | 22.18 | 19.76-23.54 | 45.87 |
| control | 8000 | 1024 | 4 | 248.42 | 248.53 | 243.54-253.08 | 21.96 | 21.97 | 19.86-24.02 | 45.80 |
| host4gb-cold | 6000 | 512 | 4 | 172.90 | 172.70 | 146.84-199.37 | 22.80 | 24.59 | 13.75-28.26 | 47.39 |
| host4gb-cold | 6000 | 1024 | 4 | 246.35 | 245.39 | 239.99-254.64 | 26.77 | 27.06 | 24.94-28.03 | 37.42 |
| host4gb-cold | 8000 | 512 | 4 | 203.01 | 203.56 | 198.25-206.66 | 33.55 | 33.78 | 31.30-35.36 | 29.87 |
| host4gb-cold | 8000 | 1024 | 4 | 237.75 | 237.61 | 234.82-240.97 | 33.29 | 33.59 | 31.20-34.79 | 30.09 |
| host4gb-hot | 6000 | 512 | 4 | 208.14 | 208.44 | 200.84-214.85 | 24.70 | 24.87 | 20.93-28.12 | 40.95 |
| host4gb-hot | 6000 | 1024 | 4 | 295.74 | 295.37 | 293.33-298.89 | 26.65 | 26.96 | 24.83-27.86 | 37.59 |
| host4gb-hot | 8000 | 512 | 4 | 223.22 | 222.05 | 218.22-230.57 | 33.71 | 34.07 | 31.57-35.12 | 29.72 |
| host4gb-hot | 8000 | 1024 | 4 | 283.88 | 282.79 | 280.37-289.58 | 33.33 | 33.59 | 31.20-34.93 | 30.06 |

## Paired Change Versus Control

Positive speed deltas mean faster. Positive TPOT improvement means lower latency. Every delta uses the same prompt_id.

| Mode | VRAM MiB | ubatch | prompts | Prefill speed mean | median | Decode speed mean | median | TPOT improvement mean | median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| host4gb-cold | 6000 | 512 | 4 | -6.58% | -3.76% | +38.60% | +56.28% | +21.27% | +36.01% |
| host4gb-cold | 6000 | 1024 | 4 | -5.13% | -5.05% | +62.57% | +63.58% | +38.41% | +38.84% |
| host4gb-cold | 8000 | 512 | 4 | +2.45% | +2.68% | +53.39% | +53.02% | +34.77% | +34.64% |
| host4gb-cold | 8000 | 1024 | 4 | -4.27% | -4.85% | +52.01% | +52.92% | +34.13% | +34.56% |
| host4gb-hot | 6000 | 512 | 4 | +12.50% | +11.45% | +49.77% | +57.56% | +32.30% | +36.51% |
| host4gb-hot | 6000 | 1024 | 4 | +13.90% | +14.21% | +61.85% | +62.79% | +38.13% | +38.53% |
| host4gb-hot | 8000 | 512 | 4 | +12.66% | +11.82% | +54.15% | +53.80% | +35.07% | +34.94% |
| host4gb-hot | 8000 | 1024 | 4 | +14.28% | +14.30% | +52.17% | +52.79% | +34.21% | +34.50% |

## Statistics

`summary-table.tsv` has one row per process. `prompt-means.tsv` averages process repeats within each prompt. `paired-deltas.tsv` subtracts control for the same prompt, VRAM budget, and ubatch. Prompt-to-prompt standard deviation is workload variation, not scheduler noise.

## Coverage

Prompt IDs observed: 01, 02, 03, 04
Modes observed: control, host4gb-cold, host4gb-hot

## Validation

All prompt groups contain three modes with matching prompt SHA, token trace SHA, and route trace SHA.

## Interpretation

- Hot 4 GiB cache improves prefill speed by 12.50%-14.28% on average versus control. Cold cache changes prefill by -6.58% to +2.45%, because the first request must populate the Host tier.
- Hot 4 GiB cache improves mean decode speed by 49.77%-61.85% versus control. Cold cache also improves mean decode by 38.60%-62.57%, although 6000/512 has a prompt-03 cold-path outlier with elevated source read, host fill, and stall time.
- Hot versus cold primarily changes prefill: paired mean prefill gains are 9.96%-21.48%. Decode is nearly identical in three cells (-0.45% to +0.48%); 6000/512 gains 13.53% because the calibrated hotset avoids the prompt-03 cold admission failure mode.
- Mean hot-cache decode speeds are 24.70, 26.65, 33.71, and 33.33 token/s for 6000/512, 6000/1024, 8000/512, and 8000/1024 respectively. The full-pinned upper bound remains 35.53-39.84 token/s, leaving additional room for better admission/prefetch.
