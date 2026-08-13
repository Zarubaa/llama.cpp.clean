# Prompt-Aware Test Protocol

## What A Repeat Means

There are two independent sources of variation:

1. `prompt_id` changes the language, topic, document structure, and MoE
   routing workload.
2. A fresh process rerun with the same `prompt_id` measures runtime noise from
   clocks, NUMA placement, background load, and scheduling.

They must not be collapsed into one unnamed `run` dimension.

## Recommended Matrix

For each `(mode, vram_cache_mb, ubatch)` cell, run all eight prompts once:

```text
prompt-01 ... prompt-08
```

With three modes and four hardware cells this is:

```text
3 modes x 4 cells x 8 prompt ids = 96 fresh processes
```

Every prompt id must be run under every mode in the same hardware cell. A
control run using prompts 1--3 cannot be compared with a hot-cache run using
prompts 4--6; that would confound cache mode with prompt routing.

The executable implementation is `scripts/run-prompt-matrix.sh`. It writes a
new result tree (by default `benchmark-results/prompt-aware-8x1024`) and never
modifies the historical 0730/0805/0806 directories. A no-GPU command-plan
check is:

```bash
PLAN_ONLY=1 DRY_RUN=1 \
  PROMPT_IDS='01 02' VRAMS='6000' UBATCHES='512' PROCESS_REPEATS=1 \
  benchmark-results/prompt-dataset-8x1024/scripts/run-prompt-matrix.sh
```

The full default matrix is 3 x 4 x 8 = 96 fresh processes:

```bash
GPU_INDEX=0 CPU_LIST='0-31,64-95' PROCESS_REPEATS=1 \
  benchmark-results/prompt-dataset-8x1024/scripts/run-prompt-matrix.sh
```

The default modes are `control-original`, `pagecache-hot`, and
`pinned-memory-hot`. For the 4 GiB tiered implementation, set
`MODES='control host4gb-cold host4gb-hot'`; the runner then requires the
existing per-cell hotset files for `host4gb-hot`.

If process noise must also be estimated, select three representative prompt
ids (recommended `01`, `04`, and `07`) and run each of them three times per
mode/cell. That adds 27 processes per cell and gives a genuine within-prompt
standard deviation.

If the budget allows only three runs per mode/cell, use three different prompt
ids, but label the rows as `prompt_sample`, not `repeat`. Report the individual
prompt values and a prompt-level interval; do not interpret their standard
deviation as process noise.

For staged batches, batch A can use prompt IDs `01 02 03` and batch B can use
`04 05 06`, but every mode in a batch must use exactly the same IDs. This
measures workload diversity, not same-workload scheduler noise. For noise
estimation, use for example `PROMPT_IDS='01 04 07' PROCESS_REPEATS=3`; average
the three processes within each prompt before computing prompt-level statistics.

## Balanced Order

For any three configured modes, rotate the mode order by prompt id. For the
4 GiB names this is:

```text
prompt 01: control -> cold -> hot
prompt 02: cold -> hot -> control
prompt 03: hot -> control -> cold
prompt 04: control -> cold -> hot
```

Continue the three-row cycle for prompts 05--08. This gives each mode the
same number of first, second, and third positions. Start each row in a fresh
process and verify GPU memory returns to baseline before the next row.

## Statistics

For each mode/cell, report mean, median, standard deviation, IQR, min/max, and
the eight prompt-level values for prefill tok/s, decode tok/s, TPOT, source
bytes, H2D ms/token, host-cache hit rate, GPU-cache hit rate, and stall
ms/token. Compute paired deltas against control for the same `prompt_id`.

Use a bootstrap confidence interval that resamples `prompt_id` as the unit,
not individual profile rows or tokens. If three process repeats are present,
first average within each prompt id, then bootstrap the prompt-level means.
Keep `prompt_id`, `prompt_sha256`, `token_sha256`, and `route_trace_sha256` in
the machine-readable summary.

## Token Boundary

The Qwen3.6 tokenizer measured the complete files as follows:

```text
01 1222    02 1173    03 1154    04 1102
05 1083    06 1094    07 1108    08 1095
```

The production benchmark uses `--pp 1024`, so it measures the first 1024
tokens of every file. A run is valid only when its summary records
`n_prompt: 1024`; do not infer this from byte count.

## Generated Artifacts

Each run is stored as
`<mode>/vram-<VRAM>-ub<UBATCH>/prompt-<ID>/repeat-<N>-*`. Metadata includes
`prompt_id`, prompt path, byte size, SHA256, model and binary identity. The
runner validates the token trace before moving files out of its partial
prefix. `summary-table.tsv` retains one row per fresh process,
`prompt-means.tsv` averages repeats within each prompt, and
`paired-deltas.tsv` subtracts control for the same prompt and hardware cell.
