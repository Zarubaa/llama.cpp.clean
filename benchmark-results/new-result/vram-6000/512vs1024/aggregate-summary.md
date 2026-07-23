# 512 vs 1024 fair retest

Fixed: normal prompt, pp=1024, tg=1024, ctx=4096, cache=6000MB, predictor=lru, CUDA_VISIBLE_DEVICES=0.

Sequence: 512, 1024, 1024, 512, 512, 1024, 1024, 512.

## Per-ub aggregate

### ub=512
- prefill_ms_per_token: mean=5.505, stdev=0.042, min=5.450, max=5.550
- decode_ms_per_token: mean=28.758, stdev=2.908, min=24.400, max=30.390
- prefill_hit_pct: mean=17.900, stdev=0.000, min=17.900, max=17.900
- decode_hit_pct: mean=91.600, stdev=0.000, min=91.600, max=91.600
- decode_ssd_read_gb: mean=49.030, stdev=0.000, min=49.030, max=49.030
- vram_peak_gb: mean=9.250, stdev=0.000, min=9.250, max=9.250

### ub=1024
- prefill_ms_per_token: mean=4.287, stdev=0.005, min=4.280, max=4.290
- decode_ms_per_token: mean=27.462, stdev=0.163, min=27.270, max=27.630
- prefill_hit_pct: mean=0.000, stdev=0.000, min=0.000, max=0.000
- decode_hit_pct: mean=92.800, stdev=0.000, min=92.800, max=92.800
- decode_ssd_read_gb: mean=42.220, stdev=0.000, min=42.220, max=42.220
- vram_peak_gb: mean=9.740, stdev=0.000, min=9.740, max=9.740

## Raw cases
- run01 ub=512: prefill=5.52 ms/tok, decode=24.40 ms/tok, prefill_hit=17.9%, decode_hit=91.6%, vram=9.25 GB
- run02 ub=1024: prefill=4.29 ms/tok, decode=27.27 ms/tok, prefill_hit=0.0%, decode_hit=92.8%, vram=9.74 GB
- run03 ub=1024: prefill=4.29 ms/tok, decode=27.39 ms/tok, prefill_hit=0.0%, decode_hit=92.8%, vram=9.74 GB
- run04 ub=512: prefill=5.55 ms/tok, decode=30.39 ms/tok, prefill_hit=17.9%, decode_hit=91.6%, vram=9.25 GB
- run05 ub=512: prefill=5.50 ms/tok, decode=30.13 ms/tok, prefill_hit=17.9%, decode_hit=91.6%, vram=9.25 GB
- run06 ub=1024: prefill=4.28 ms/tok, decode=27.63 ms/tok, prefill_hit=0.0%, decode_hit=92.8%, vram=9.74 GB
- run07 ub=1024: prefill=4.29 ms/tok, decode=27.56 ms/tok, prefill_hit=0.0%, decode_hit=92.8%, vram=9.74 GB
- run08 ub=512: prefill=5.45 ms/tok, decode=30.11 ms/tok, prefill_hit=17.9%, decode_hit=91.6%, vram=9.25 GB
