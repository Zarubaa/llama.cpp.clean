# Formal-sized SERE performance run

Run root: `/home/chaoyang.zhang/docker_workspace/projects/llama.cpp.clean/benchmark-results/sere/performance/formal-synthetic-8g-20260808T151500Z`

The workload follows the corrected single-batch bench protocol: `pp=1024`, `tg=1024`, `repeat=1`, the normal Chinese-English prompt truncated to 1024 used tokens, fresh process per case, LRU, host cache off, and a page-cache warm read outside the measured interval. `TTFT` includes final prefill logits synchronization; `TPOT` is calculated only after all 1024 decode tokens finish. No `logits-bin` output is enabled.

Cases: exact LRU, paper S=4, cache-aware miss S=4, and paper S=8 no-op. Raw artifacts are kept below each `vram-*/` directory. See `results.tsv` and `validation.txt` for the compact table and gates.

The sidecar is the deterministic synthetic fixture currently available on this branch. These numbers are formal-sized transfer/timing evidence for the integration mechanism only; they are not a real Q4 similarity, quality, or publishable SERE speedup claim.
