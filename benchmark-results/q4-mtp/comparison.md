# Q4_K_M MTP/offload fair normal-request A/B

## Configuration

- Source model: `/data/hf_models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- Repacked model: `/data/hf_models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf`
- Repacked model size: 22,663,891,712 bytes; 41 MoE layers; 256 experts/layer
- Prompt: corrected Q8 normal-request prompt, SHA256 `e690343f...dc8eb2`
- Common parameters: `pp=1024`, `tg=1024`, `repeat=3`, `ubatch=1024`, `ctx=4096`
- Offload: shared LRU pool, `6000 MB`, 59 persistent slots/layer, all non-expert layers on GPU
- MTP case: `draft-mtp`, `n_max=1`, Q8_0 draft K/V
- Both runs used GPU 0 and started after `POSIX_FADV_DONTNEED` on the same repacked file.

## Results

| Metric | Without MTP | With MTP n_max=1 | Difference |
|---|---:|---:|---:|
| Warm TTFT | 3250.6 ms | 3578.1 ms | MTP +10.1% |
| Decode / TPOT | 32292.3 ms / 31.54 ms | 38798.9 ms / 37.89 ms | MTP +20.2% |
| Decode throughput | 31.7 tok/s | 26.4 tok/s | MTP -16.8% |
| Warm end-to-end estimate | 35542.9 ms | 42377.0 ms | MTP +19.2% |
| Decode cache hit | 90.8% | 87.3% | -3.5 points |
| Decode SSD bytes | 160.14 GB | 205.87 GB | MTP +28.6% |
| Process VRAM peak | 9.83 GB | 11.85 GB | MTP +2.02 GB |
| Target verification requests | 3072 | 1569 | -48.9% |

MTP proposed 1569 draft tokens and accepted 1503 (`95.8%`). Token accounting and profiler coverage close exactly:

```text
1569 target verification rounds + 1503 accepted drafts = 3072 generated tokens
62760 target rows / 40 target MoE layers = 1569 target verification rounds
1569 draft rows / 1 MTP layer = 1569 draft requests
1569 process rows / 1 MTP layer = 1569 process requests
```

Despite the high acceptance rate, MTP is slower in this configuration. Two-token target verification requires about 14 unique experts per target layer/request instead of 8 for one-token decoding. The target hit rate falls from 90.8% to 87.0%, target misses rise from about 30.1k to 38.2k per repeat, and draft/process add more work. Q4 also makes the no-MTP baseline fast enough that the extra MTP stages are not amortized at `n_max=1`.

## Cold-start caveat

The summaries report mixed one-cold-plus-two-warm averages. Cold TTFT was 138470.1 ms without MTP and 76836.3 ms with MTP even though the file was nearly uncached before both runs (17.5 MiB versus 38.9 MiB resident out of 21.1 GiB). This storage variance makes the raw reported totals, 80616.1 ms versus 66796.4 ms, unsuitable as an MTP speedup claim. The table therefore uses warm TTFT plus the measured decode average for the end-to-end comparison.

The Q4 model is under container `/data`, while the corrected Q8 model is under `/workspace` on a different filesystem. Q4-without-MTP versus Q4-with-MTP is an internal A/B on the same file; Q4 and Q8 absolute I/O latency should not be attributed to quantization until both are placed on the same storage device.

## Correctness

- Every mode and repeat emitted exactly 1024 tokens.
- All no-MTP repeats have token SHA256 `a2c82888...3b600f`.
- All MTP repeats have token SHA256 `87a95f9a...6ab02f`.
- Logs contain no model-load, OOM, decode, or I/O errors.
- The I/O worker exited with `events_in_use=0`; no benchmark process remains.

Raw artifacts are in `without-mtp/` and `with-mtp/`.
