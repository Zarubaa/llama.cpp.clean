# control-p01-clock-check Report

Validated runs: 2

Each prompt_id is a distinct workload sample. A repeat is a fresh-process rerun of the same prompt.
All cache modes must use the same prompt_id before their paired delta is interpreted.

## Performance

The table reports four-prompt workload means and medians. Speeds are explicit in token/s; ranges are prompt-to-prompt variation, not process-repeat noise.

| Mode | VRAM MiB | ubatch | prompts | Prefill tok/s mean | median | range | Decode tok/s mean | median | range | TPOT ms mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 6000 | 512 | 1 | 186.20 | 186.20 | 186.20-186.20 | 14.93 | 14.93 | 14.93-14.93 | 67.00 |

## Paired Change Versus Control

Positive speed deltas mean faster. Positive TPOT improvement means lower latency. Every delta uses the same prompt_id.

| Mode | VRAM MiB | ubatch | prompts | Prefill speed mean | median | Decode speed mean | median | TPOT improvement mean | median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Statistics

`summary-table.tsv` has one row per process. `prompt-means.tsv` averages process repeats within each prompt. `paired-deltas.tsv` subtracts control for the same prompt, VRAM budget, and ubatch. Prompt-to-prompt standard deviation is workload variation, not scheduler noise.

## Coverage

Prompt IDs observed: 01
Modes observed: control

## Validation

Matrix consistency failures: 2
- (6000, 512, '01', 1): expected 3 modes, found 1
- (6000, 512, '01', 2): expected 3 modes, found 1
