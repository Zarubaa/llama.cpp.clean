# Q4 MTP n_max sweep: pass-A report

## Status

This is the only complete true-normal-request pass in this result set. It is
valid as an observed end-to-end engineering run, but not yet as a claim of
lossless speculative speedup:

- All five cases used the same first 1024 tokens from a 10200-token normal
  prompt. Every log contains `raw=10200 used=1024 requested=1024`.
- Physical GPU 1 was exclusive: every case began at 3 MiB with no compute
  process, and the continuous audit contains only that case's
  `llama-moe-bench` PID.
- Every repeat emitted exactly 1024 tokens and all accounting invariants close.
- Greedy output is deterministic within a case, but differs across cases. See
  "Output equivalence" below.
- The reverse pass is excluded because unrelated system-wide compilation and
  evaluation work started during it. See `reverse-pass/INVALID.md`.

## Configuration

| Parameter | Value |
|---|---|
| Model | `Qwen3.6-35BA3B-MTP-UD-Q4_K_M.moe.gguf` |
| Model SHA-256 | `d9e4b0e8cbe3f94d057f5aecde28c38e08b42c3ec8840da29f5abd6b716e6900` |
| Prompt SHA-256 | `e690343f2fffd2560484f93def97aa39e943419eb62b7d8f3b6d7509d2dc8eb2` |
| Input / output | 1024 / 1024 tokens |
| Context / ubatch | 4096 / 1024 |
| Repeats | 3 per process |
| Expert cache / predictor | 6000 MiB / LRU |
| Cache calibration | OS page cache warm; GPU expert slots reset before every measured repeat |
| GPU | physical GPU 1, NVIDIA L20, UUID `GPU-76684970-c5fc-f3d7-8a03-cc1f3f58975e` |
| Case order | no-MTP, n_max=4, n_max=1, n_max=3, n_max=2 |
| Git commit | `28b84e3e22c31eac9df374c1135bc35b51713b35` plus saved working-tree diff |
| Benchmark binary SHA-256 | `e16d5af1e122e7e5e57f8a96e3e6bebc3716cf7071253251de84374d0ebb4436` |

`--moe-cache-vram-mb 6000` limits only the expert slot pool. The full process
uses additional VRAM for non-expert weights, contexts, KV state, and compute
buffers.

## End-to-end results

All times are the mean reported by `llama-moe-bench` over three repeats.
`Relative decode` is `baseline TPOT / case TPOT`; values below 1 are slower
than no-MTP.

| Case | Accept | Prefill target ms | MTP process ms | Prefill total ms | Decode ms | TPOT ms | tok/s | Relative decode | E2E ms | Decode hit | Decode read GB | Process VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no-MTP | - | 4113.9 | 0.0 | 4113.9 | 53010.9 | 51.77 | 19.32 | 1.000x | 57124.8 | 79.2% | 364.27 | 9.87 |
| n_max=1 | 81.0% | 4176.1 | 147.7 | 4323.8 | 54878.4 | 53.59 | 18.66 | 0.966x | 59202.2 | 76.5% | 390.78 | 11.88 |
| n_max=2 | 70.0% | 4023.7 | 146.4 | 4170.1 | 55186.6 | 53.89 | 18.56 | 0.961x | 59356.6 | 75.2% | 418.47 | 11.95 |
| n_max=3 | 51.7% | 4000.5 | 145.2 | 4145.7 | 55711.1 | 54.41 | 18.38 | 0.951x | 59856.9 | 76.9% | 436.19 | 12.01 |
| n_max=4 | 45.8% | 4067.2 | 149.0 | 4216.1 | 59217.1 | 57.83 | 17.29 | 0.895x | 63433.2 | 76.4% | 474.29 | 12.07 |

Observed decode slowdown versus no-MTP is 3.5%, 4.1%, 5.1%, and 11.7% for
`n_max=1..4`, respectively. No MTP setting accelerated this workload.

## Speculative work

Counts below aggregate all three repeats. Verification steps are target
requests divided by three; the no-MTP reference is 1024 target requests per
repeat.

| Case | Drafted | Accepted | Accept rate | Target requests | Verify steps / repeat | Target request reduction |
|---|---:|---:|---:|---:|---:|---:|
| n_max=1 | 1696 | 1373 | 81.0% | 1699 | 566.3 | 44.7% |
| n_max=2 | 2558 | 1790 | 70.0% | 1282 | 427.3 | 58.3% |
| n_max=3 | 3612 | 1866 | 51.7% | 1206 | 402.0 | 60.7% |
| n_max=4 | 4338 | 1986 | 45.8% | 1086 | 362.0 | 64.6% |

Longer drafts reduce target requests, but also lower acceptance, reduce the
aggregate expert-cache hit rate, increase expert data movement, and add about
2.0-2.2 GiB of non-expert/context VRAM. In this 6 GiB expert-cache regime,
those costs exceed the saved verification steps.

## Output equivalence

Each hash is stable across all three repeats of that case:

| Case | Emitted-token SHA-256 | First difference from no-MTP (1-based) |
|---|---|---:|
| no-MTP | `49d4139bb432e436bd9113fe8f1a8a510fd1ff56ae1ed6c15f44dc00d4c9b39a` | - |
| n_max=1 | `3e23d62e1cdd49b25ddf1ea9258f3389f61d133541407eeeabfdc3affbce9344` | 29 |
| n_max=2 | `3e23d62e1cdd49b25ddf1ea9258f3389f61d133541407eeeabfdc3affbce9344` | 29 |
| n_max=3 | `3f7cf2fda3a291ba7892b9dc7711e889ff7cb36b6fb8cd29372114d1d16d9a27` | 29 |
| n_max=4 | `9ae2074148006231f30edee7de6c542201a9e55c255c781ae6fd6155f50265ba` | 2 |

The trace invariants and rollback positions are internally consistent with the
server algorithm, so the traces alone do not prove a rollback bug. The target
execution path nevertheless changes under MTP: `n_rs_seq=N`, verification
uses batches of `1 + draft`, and Qwen's hybrid Gated Delta Net can switch from
the autoregressive fused path to the chunked fused path. A target top-1 flip is
therefore possible through batch-path numerical differences. This must be
isolated before calling MTP output-equivalent or lossless.

Recommended next diagnostic:

1. Add an MTP no-draft control that preserves the MTP target context and
   `n_rs_seq` but returns an empty draft.
2. Teacher-force the common prefix at the first mismatch and log target top-2
   IDs, logits, and margin for scalar versus batched verification.
3. Assert target/draft `pos_max == n_past - 1` after every rollback.
4. Repeat with MoE offload disabled to separate hybrid-model batch numerics
   from the offload multi-token path.

Until that diagnostic is complete, this pass is suitable for engineering
profiling, not a paper claim of equivalent-output speculative acceleration.

## Variability and audit

The per-repeat target-profile CV was 1.0%, 0.7%, 1.2%, and 2.6% for
`n_max=1..4`. The no-MTP target profile had an 8.5% CV because repeat 0 was
faster (`45.6 s`) than repeats 1 and 2 (`53.2 s`, `52.8 s`). This is another
reason to collect more independent clean-system runs for paper statistics.

The continuous files `gpu-audit.csv`, `compute-audit.csv`, and
`pmon-audit.csv` cover the full pass. No Python process or second benchmark
entered physical GPU 1. All logs end with `events_in_use=0` and contain no
OOM, abort, short-read, or model-load errors.
