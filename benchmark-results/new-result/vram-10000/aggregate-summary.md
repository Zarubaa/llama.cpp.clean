# vram-10000 benchmark

Fixed: normal prompt, pp=1024, tg=1024, ctx=4096, cache=10000MB, predictor=lru, CUDA_VISIBLE_DEVICES=0.

| ub | prefill ms/tok | decode ms/tok | prefill hit | decode hit | decode SSD GB | VRAM peak GB |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 9.10 | 17.74 | 88.6% | 96.9% | 17.81 | 12.91 |
| 128 | 6.88 | 17.70 | 62.2% | 96.8% | 18.76 | 13.03 |
| 256 | 6.40 | 18.80 | 47.9% | 96.7% | 19.16 | 13.15 |
| 512 | 5.40 | 18.67 | 27.5% | 96.8% | 18.55 | 13.36 |
| 1024 | 4.71 | 19.56 | 0.0% | 96.5% | 20.68 | 13.86 |
