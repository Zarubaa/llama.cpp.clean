# 0730 DRAM Cache Comparison

This experiment compares three steady-state Host data paths with a realistic
Chinese-English technical prompt:

| Mode | Linux page-cache policy | Application Host cache | Expert transfer path |
|---|---|---|---|
| `control-original` | `natural` | `off/none` | page cache `fread` -> pinned staging -> H2D |
| `pagecache-hot` | explicit `hot` (>=99%) | `off/none` | page cache `fread` -> pinned staging -> H2D |
| `pinned-memory-hot` | explicit `hot` (>=99%) | `pinned/all` | pinned expert cache -> direct async H2D |

The control does not modify Linux page cache. Because the model is already
fully resident, control and pagecache-hot are expected to have the same timed
inference path; pagecache-hot additionally verifies/prepares residency before
model initialization, outside inference TTFT.

Fixed settings:

- Model: `/home/haozhe.lou/models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf`
- Prompt: `benchmark-results/new-result/prompts/normal-long-prompt.zh-en.txt`
- `pp=1024`, `tg=1024`, greedy decode, LRU predictor, one repeat per process
- GPU2; CPU affinity `32-63,96-127` (NUMA node 1)
- VRAM expert cache: 6000 and 8000 MiB
- ubatch: 512 and 1024
- Three independent processes per mode/cell; 36 formal runs total

Run with:

```bash
scripts/run-experiments.sh
```

Every run is validated before its partial files are atomically published.
`summary-table.tsv` contains per-run data. `aggregate-table.tsv` and
`final-report.md` report arithmetic means, standard deviations, ranges, and
prefill/decode speeds.

The first GPU0 attempt was stopped when an unrelated `llama-cli` process
started on that device during a run. Those non-formal artifacts are retained
under `aborted-gpu0-interference/`; all formal runs are restarted on GPU2 so
the reported matrix uses one device and one NUMA-local CPU set throughout.
One later GPU2 run was similarly rejected when an unrelated multi-GPU
`autoround` job started. Its artifacts are retained under
`aborted-external-gpu-interference/`; the affected run was repeated only after
all four GPUs returned to zero compute processes.
