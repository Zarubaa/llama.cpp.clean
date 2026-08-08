# SERE integration experiments

This directory is the canonical output root for the Qwen3.6 Q4 SERE
integration on branch `0808_SERE`.

## Scope of the current implementation

- SERE is disabled when `--moe-sere-top-k=0` (the default).
- Rerouting is applied only to callbacks belonging to explicitly identified
  single-token decode requests. Prefill remains exact.
- `paper` reroutes every non-primary secondary route accepted by the threshold.
- `miss` is a cache-aware extension that only reroutes a secondary route when
  its original expert is absent from the persistent cache.
- The router weight and route rank are preserved. Only the physical expert slot
  used for that rank is changed.
- The implementation reduces expert source reads, H2D traffic, and cache churn.
  It does not reduce top-k from eight routes to `S` computations.
- With one decode token per request, `paper` is the batch-size-one degeneration
  of the paper algorithm. It is not the paper's batch-union execution scheme.
- `--moe-sere-shadow` computes the SERE mapping and counters but executes the
  raw router IDs. Predictor observation, LRU/cache state, transfers, tokens and
  logits therefore remain on the exact baseline path.
- Shadow counters are a one-step counterfactual evaluated against the current
  baseline cache state at each callback. They are not an independent shadow LRU
  replay whose rerouted cache state is carried across decode tokens.

The current target is:

```text
repacked model: models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf
source model:   models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
architecture:   qwen3_5_moe
MoE shape:      40 layers, 256 experts/layer, top-8
```

## Directory classes

```text
mechanism-only/       Synthetic matrices and integration smoke results.
calibration/          Q4-matched matrix generation artifacts.
quality/              Teacher-forced and downstream quality results.
performance/          Formal SSD/DRAM performance sweeps.
```

Never aggregate `mechanism-only` measurements into a quality or formal
performance table. A synthetic matrix verifies the sidecar, routing, profiler,
and I/O mechanisms only; it says nothing about real expert interchangeability.

See [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) for calibration, correctness gates,
the parameter matrix, quality metrics, and fair-run requirements.

The historical synthetic integration run is summarized in
[MECHANISM_RESULTS.md](MECHANISM_RESULTS.md). Keep it as mechanism evidence from
the earlier implementation; new runs on `0808_SERE` must use a new run ID and
produce their own `validation.json`.

## Mechanism smoke

Generate the deterministic synthetic input and export it through the same
converter used for a real matrix:

```bash
repo=/home/chaoyang.zhang/docker_workspace/projects/llama.cpp.clean
cd "$repo"
mkdir -p benchmark-results/sere/mechanism-only/synthetic
/opt/miniconda3/bin/python \
  benchmark-results/sere/mechanism-only/generate-synthetic-matrix.py \
  benchmark-results/sere/mechanism-only/synthetic/similarity.npz \
  --force
/opt/miniconda3/bin/python scripts/export-sere-similarity.py \
  benchmark-results/sere/mechanism-only/synthetic/similarity.npz \
  benchmark-results/sere/mechanism-only/synthetic/similarity.sere \
  --model models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf \
  --metric unspecified --n-layers 40 --n-experts 256 --force
```

Run these two commands on the host: it has NumPy in `/opt/miniconda3`, while the
container's default Python does not. `unspecified` is intentional because this
uniform matrix is artificial and was not produced by any calibration metric.
The exported V2 sidecar is bound to the repacked model's architecture, router
top-k, quantization and expert layout. It is consumed by C++ and has no Python
runtime dependency. Both commands require `--force` when replacing an existing
fixture. See [synthetic/README.md](mechanism-only/synthetic/README.md) for the
artifact boundary.

First confirm that the selected GPU has no competing compute process. Then run
all cases inside `llama-cpp-clean-dev` from the host with exactly one explicit
CUDA selector:

```bash
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader

docker exec \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=1 \
  llama-cpp-clean-dev bash -lc \
  'cd /workspace && benchmark-results/sere/run-mechanism-smoke.sh RUN_ID'
```

The runner rejects an unset, empty, or multi-GPU `CUDA_VISIBLE_DEVICES`. It
records the selected GPU, all GPU memory/utilization values, and compute
processes in `gpu-snapshot-start.txt` and `gpu-snapshot-end.txt`.

The script runs five independent cold processes with `repeat=1` for exact LRU,
S=8 no-op, active `paper` S=4, shadow `paper` S=4, and cache-aware active
`miss` S=4. It writes a timestamped directory under `mechanism-only/runs/`,
refuses to reuse a run directory, and writes `validation.json` with every
pass/fail result. It exits non-zero when an assertion fails.

This is deliberately a mechanism test, not a formal performance run. It uses a
uniform synthetic matrix, only 16 decode tokens, and writes full logits inside
the measured decode loop. `--page-cache-policy cold` verifies the model file is
at most 1% resident before each process loads it, but prefill can warm the file
before decode. Do not use its TTFT or TPOT in a quality or performance claim.

Required checks after the smoke run:

1. All prefill SERE counters are zero.
2. S=8 has zero rerouted routes and matches the baseline token trace.
3. `paper` S=4 has non-zero decode reroutes.
4. Shadow `paper` S=4 reports non-zero counters and `mode=shadow`, while its
   token trace, complete logits, raw route hashes, and deterministic cache/LRU/
   transfer accounting exactly match `lru-off`.
5. `miss` reroutes only original misses.
6. Every one of the five CSVs has 62 columns, 17 request rows, one prefill
   request, 16 decode requests, and exactly layers 0-39 once per request.
7. Effective unique misses and profiled source-read requests decrease for the
   two active S=4 cases; shadow is excluded from this reduction assertion.
8. All cases use the same prompt tokens and have identical prefill routing,
   cache/I/O accounting, and prefill logits.
9. Original/effective reroute accounting, source-read counts, and H2D bytes are
   internally consistent.
10. Every token trace and logits file contains repeat 0 with steps 0-16.
11. The baseline has zero SERE counters, and both GPU snapshots are present.
12. `validation.json` records SHA256 for the model, prompt, sidecar, benchmark
    binary, runner, validator, GPU snapshots, tracked Git diff, and Git status.

`route_hash` is computed from original router IDs before replacement in the C++
implementation. Active S=4 hashes cannot be compared directly with the baseline
after logits and hidden states diverge, so that remains a code invariant. The
shadow S=4 path does not diverge numerically, and `validation.json` therefore
requires every one of its raw route hashes to match the baseline.

An existing run can be checked again without CUDA:

```bash
python3 benchmark-results/sere/mechanism-only/validate-smoke.py \
  benchmark-results/sere/mechanism-only/runs/RUN_ID
```
