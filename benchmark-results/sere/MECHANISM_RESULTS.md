# Mechanism smoke result: final-20260807

Classification: **mechanism-only, synthetic similarity matrix**.

This run validates the integration plumbing and offload behavior. It is not a
quality result and must not be included in a formal SERE performance aggregate.

## Configuration

```text
branch:            0805_dram-bench
base commit:       b559e7a52fcb428bfb51a5c5cac2a1962851ad8e
model:             Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf
GPU:               NVIDIA L20, CUDA_VISIBLE_DEVICES=0
prompt source:     normal-long-prompt.zh-en.txt
prompt used:       128 / 16961 tokens
prompt token hash: f1f1841cdd410d1d
generation:        16 greedy tokens
ubatch:            128
expert cache:      6000 MiB, exact LRU
host cache:        off
page cache:        cold, 0% resident after prepare
repeat:            1, one fresh process per case
matrix:            uniform synthetic off-diagonal similarity 0.95
sidecar metric:    1 (historical label only; no Frobenius calibration was run)
```

The smoke was run before the synthetic fixture metadata was corrected, so its
logs accurately report metric ID 1. The matrix values were artificial then and
remain artificial; the canonical regenerated fixture now uses metric ID 0
(`unspecified`). This metadata correction does not turn the historical run into
a calibrated result.

## Results

| Case | TPOT ms | Profiled source GiB | Effective cache hit | Original route hit | Reroutes/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact LRU, SERE off | 175.87 | 2.19 | 76.0% | 76.0% | 0.00 |
| paper S=8, rho=0 | 172.93 | 2.19 | 76.0% | 76.0% | 0.00 |
| paper S=4, rho=0 | 125.49 | 0.96 | 79.0% | 71.6% | 160.00 |
| miss S=4, rho=0 | 112.62 | 0.91 | 88.0% | 72.5% | 56.44 |

For SERE cases, `Original route hit` tests the original router IDs against the
cache state already mutated by SERE's earlier effective routes. It is not a
counterfactual exact-LRU trajectory, so the S=4 values must not be compared
directly with the baseline 76.0% as an isolated policy effect.

The S=4 cases reduced profiled source-read requests by approximately 56-58% and
wall TPOT by approximately 29-36% in this single short run. Profiler source
bytes are logical pread requests, not physical block-device bytes: after prefill,
the model file was about 81% page-cache resident, and the baseline process-I/O
decode delta was about 0.45 GiB versus 2.19 GiB of logical source requests.

This is a mechanism signal only. The matrix is artificial, there was no
repetition, only 16 decode tokens were measured, and the whole-device VRAM
baseline was already 12.72 GiB. The speed values are not suitable for a formal
performance claim.

## Hard checks

All checks in [validation.json](mechanism-only/runs/final-20260807/validation.json)
passed:

- every profile CSV has 62 columns, 40 prefill rows, and 640 decode rows
- all prefill SERE counters are zero
- all cases use the same prompt tokens and have row-identical prefill
  route/cache/I/O fields and byte-identical prefill logits
- baseline and S=8 logits are byte-identical
- baseline and S=8 route/cache/I/O fields are row-identical
- S=8 reroutes zero routes
- `paper` reroutes every synthetic secondary route
- `miss` reroutes only routes whose original expert missed
- page-cache residency after cold preparation is 0% for every case
- both S=4 cases reduce effective misses and profiled source bytes
- original/effective reroute accounting and rerouted-miss bounds are consistent
- each decode miss has three profiled source reads and profiled H2D bytes equal
  profiled source bytes

All four 16-token greedy traces happened to remain equal, but S=4 logits are
not equal to the baseline. S=4 token equality is only an observation and is not
a correctness requirement.

## Next gate

Do not run or interpret a formal parameter sweep until a similarity matrix has
been calibrated from the current non-repacked Q4 GGUF using its actual quantized
expert functions, then passed the teacher-forced and downstream quality gates in
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).
