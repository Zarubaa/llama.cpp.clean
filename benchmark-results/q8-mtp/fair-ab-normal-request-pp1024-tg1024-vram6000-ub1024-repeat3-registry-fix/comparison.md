# Q8 MTP/offload corrected comparison

## Configuration

- Model: `Qwen3.6-35BA3B-MTP.moe.gguf` (`Q8_0`)
- Prompt/generation: `pp=1024`, `tg=1024`, normal request prompt
- Expert cache: `6000 MB`, shared LRU pool, 45 persistent slots per layer
- Batch: `ubatch=1024`
- Repeats: 3
- MTP: `draft-mtp`, `n_max=1`, Q8_0 draft K/V

## Main results

| Metric | No MTP | MTP n_max=1 | Change |
|---|---:|---:|---:|
| Prefill total | 5313.3 ms | 5193.1 ms | 1.023x |
| Decode total | 70811.0 ms | 62868.1 ms | 1.126x |
| TPOT | 69.15 ms | 61.39 ms | 1.126x |
| End-to-end total | 76124.3 ms | 68061.2 ms | 1.118x |
| Decode SSD reads | 488.55 GB | 479.43 GB | -1.87% |
| Process VRAM peak | 9.61 GB | 11.62 GB | +2.01 GB |
| Target verification requests | 3072 | 1623 | -47.2% |
| Target profiler rows | 122880 | 64920 | -47.2% |

MTP proposed 1623 draft tokens and accepted 1449 (`89.3%`). Accounting closes exactly:

```text
1623 target verification rounds + 1449 accepted drafts = 3072 generated tokens
64920 target rows / 40 target MoE layers = 1623 target verification rounds
```

The small prefill difference is run-to-run cache/I/O variation and is not an MTP speedup claim. MTP adds a measured 208.8 ms prefill-process stage.

## Correctness checks

- Every repeat produced exactly 1024 output tokens.
- All three no-MTP repeats have token SHA256 `21f93bf4...a513e`.
- All three MTP repeats have token SHA256 `775619af...218b4`.
- MTP target, draft, and process rows all cover the expected 40/1/1 MoE layers.
- A final lifecycle smoke test returned exit code 0 with `events_in_use=0`; its token trace is byte-identical before and after the I/O shutdown fix.

No-MTP and parallel MTP verification are deterministic within each mode but are not bitwise identical to each other. Diagnostic `n_max=0` matches no-MTP; `n_max=1` changes Qwen3.6 hybrid/GDN execution from a one-token to a two-token target batch, whose GPU floating-point path is not bitwise invariant. Quality evaluation is still required before making a lossless-quality claim.

## Invalidated results

Results produced before the per-context graph registry fix, including the earlier approximately `7.24x` speedup and approximately `86%` acceptance result, are invalid. In those runs, target and MTP contexts overwrote a shared graph tensor registry, so reused target graphs skipped correct expert loading and profiling. Only the files in this `registry-fix` directory should be used as the corrected full-length baseline.
