# Formal SERE performance run: 6 GiB sensitivity

This run is a completed sensitivity point for the corrected single-batch
`llama-moe-bench` protocol. It was executed in `llama-cpp-clean-dev` on GPU3
with a fresh bench process for every case.

Configuration:

- branch `0808_SERE`, commit `13aedd344c51fb236db98452a42e751b9c2e89e1`
- model `Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf`, SHA256
  `4674317e422518acafef023522768e25a42b58e22f960e5a83133bedd0cd14b4`
- prompt file `benchmark-results/new-result/prompts/normal-long-prompt.zh-en.txt`
- prompt used tokens `1024`, token hash `318c1480f039b981`
- `pp=1024`, `tg=1024`, `ctx=4096`, bench `repeat=1`
- VRAM expert cache `6000 MiB`, predictor `lru`, host cache `off`
- `page-cache-hot` prepared the model before each measured process
- ubatches `512` and `1024`; cases `lru`, paper S=4, cache-aware miss S=4
- no logits-bin output; CPU affinity `32-63,96-127`; target GPU3

The deterministic synthetic SERE sidecar is the mechanism fixture currently
available on this branch. These numbers measure transfer/timing behavior of
the integration; they are not a real Q4 similarity or model-quality result.

| ubatch | case | TTFT (ms) | TPOT (ms) | total (ms) | decode hit | decode SSD read (GiB) |
|---:|---|---:|---:|---:|---:|---:|
| 512 | LRU | 6365.5 | 62.59 | 70459.7 | 75.6% | 142.27 |
| 512 | paper S=4 | 6328.9 | 41.12 | 48439.0 | 72.9% | 78.91 |
| 512 | miss S=4 | 6466.5 | 34.54 | 41838.7 | 85.8% | 67.02 |
| 1024 | LRU | 4280.5 | 67.50 | 73402.3 | 73.5% | 154.32 |
| 1024 | paper S=4 | 4299.0 | 34.94 | 40081.1 | 76.6% | 68.24 |
| 1024 | miss S=4 | 4351.3 | 36.79 | 42028.9 | 84.1% | 73.53 |

All six cases have exit code 0, complete status, 1025 token-trace data rows,
and the expected layer rows (`41040` for ubatch 512 and `41000` for ubatch
1024). Page-cache samples are 100% at every checkpoint and host-cache
verification reports zero failures. GPU3 had no external compute process and
the before/after GPU memory gate passed for every case.

The NUMA gate is intentionally `warn`: sampled local pages were 93.331--93.410%
against the 95% advisory threshold. This is an environment placement warning,
not a bench failure; it is recorded per case in `metadata.txt`.

Raw artifacts are under `page-cache-hot/vram-6000/ub{512,1024}/`. The compact
table is in `page-cache-hot/results.tsv`, and independent checks are in
`page-cache-hot/validation.txt`.
