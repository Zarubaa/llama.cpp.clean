# Q4 SERE experiment plan

## 1. What is being tested

The hypothesis is narrower than the original serving paper:

> During single-token decode with expert-weight offload, replacing selected
> secondary expert routes with functionally similar primary experts can reduce
> cache misses and expert transfer traffic enough to improve wall-clock TPOT,
> while keeping task and token-distribution degradation within a fixed budget.

The current implementation exposes two policies:

| Policy | Secondary hit | Secondary miss | Purpose |
| --- | --- | --- | --- |
| `paper` | May reroute | May reroute | Batch-size-one paper semantics |
| `miss` | Preserved | May reroute | Isolate replacement to offload misses |

In both policies, a route is preserved when its best primary similarity is
below `--moe-sere-threshold`.

## 2. Q4-matched similarity calibration

Do not reuse a matrix from another branch, MTP model, Qwen3 model, BF16 model,
or another quantization. The cloned SERE calibrator supports `qwen3_moe`, while
this target is `qwen3_5_moe`; its script also runs only one sliced batch and
materializes every expert output at once.

The primary matrix must be computed from the source Q4 GGUF with llama.cpp's
actual quantized expert kernels:

1. Run the non-repacked Q4 model exactly, with all weights resident on GPU.
2. Capture each MoE layer's actual FFN input (`attn_post_norm`) from held-out
   calibration requests.
3. Execute all 256 routed experts in bounded expert/token tiles using the same
   GGML quantized kernels as inference. Exclude the shared expert and routing
   weights from the expert function output.
4. Accumulate squared Frobenius distances across all tokens before taking the
   square root:

   ```text
   D2[i,j] += ||E_i(H_batch) - E_j(H_batch)||_F^2
   D[i,j] = sqrt(D2[i,j])
   similarity[i,j] = 1 - D[i,j] / max(D)
   ```

5. Use 8K-16K decode hidden states from Chinese, English, code, math, and
   general dialogue. Keep this corpus disjoint from all tuning and test data.
6. Save `[40,256,256]` float32 NPZ and export it with
   `scripts/export-sere-similarity.py`.

The main artifact is `q4-frob-decode-v1`. Required calibration ablations are:

- decode-output Frobenius
- prefill-output Frobenius
- mixed-output Frobenius
- decode-output cosine
- expert-weight similarity
- an HF/BF16 matrix, labelled only as a cross-quantization transfer ablation

Each calibration directory must include `manifest.json` with source/repacked
GGUF SHA256, corpus SHA256, NPZ/sidecar SHA256, branch, commit, dirty-diff hash,
metric, normalization, sample count, split, and exact command.

## 3. Correctness gates

Do not start a formal sweep until all gates pass:

1. Full-GPU source Q4 and repacked exact offload produce matching reference
   logits/tokens within a documented floating-point tolerance.
2. The new build with SERE disabled matches the existing exact-LRU route hashes,
   token trace, logits, unique misses, and source bytes.
3. S=8 is a no-op: no route is replaced and baseline/S=8 traces match.
4. Every prefill SERE counter is zero, including a prompt whose final microbatch
   contains one token.
5. `paper` and `miss` unit fixtures match the cloned SERE CUDA kernel's
   batch-size-one behavior.
6. Truncated, non-finite, wrong-shape, and wrong-model sidecars are rejected.
7. SERE is rejected when the expert cache is full-residency.

The generic llama API still needs an explicit phase signal for unusual clients:
a one-token first prompt can otherwise look like decode, while multi-token
speculative decode can look like prefill. `llama-moe-bench` sets phase explicitly,
so this does not affect the planned benchmark.

## 4. Parameter sweep

Exact controls:

- SERE off, exact LRU
- S=8, no-op sidecar path
- random replacement matched to each configuration's reroute count

Primary sweep:

- policy: `paper`, `miss`
- primary S: 7, 6, 4, 2; include 8 and aggressive S=1 as controls
- cache budget: 6 GiB primary; 4, 8, and 10 GiB sensitivity
- prompt/decode: 1024/1024 primary; short and longer contexts as sensitivity

Do not select thresholds on final benchmark data. On a separate tuning split,
choose thresholds that yield approximately 25%, 50%, and 75% reroute acceptance,
then freeze them. Always include rho=0 as the unconstrained performance bound.

Matrix ablations must compare Frobenius, cosine, weight similarity, random
replacement, decode/prefill/mixed calibration, and 512/2K/8K calibration-token
budgets. Report nearest-neighbor agreement, top-N overlap, layer-wise Spearman
correlation, and the distribution of best-primary similarity.

## 5. Quality evaluation

Free greedy generation is not enough: after the first differing token, the two
runs no longer have the same context. Add teacher-forced replay:

1. Exact Q4 generates the fixed continuation token stream.
2. Every SERE case consumes the same tokens at every position.
3. Record exact-vs-SERE KL, JS divergence, top-1 agreement, top-5 overlap,
   reference-token NLL/perplexity delta, and first-divergence position.
4. Plot error versus position to expose recurrent-state accumulation in the
   target architecture.

Task suites:

- Chinese: CMMLU, C-Eval
- reasoning: BoolQ, BBH
- math: GSM8K, MATH
- code: HumanEval, MBPP
- language modelling: held-out perplexity
- free generation: common-prefix length and token edit distance

Full logits are roughly 1 MiB/token for this vocabulary. Save them only for
shortlisted teacher-forced cases; performance sweeps should emit small metrics
instead of multi-gigabyte logits files.

## 6. Fair performance protocol

Run SSD-cold, page-cache-hot, pageable-DRAM, and pinned-DRAM as separate tables.
For the primary SSD-cold comparison:

- one fresh process per case
- `--repeat 1`
- fixed prompt, used-token count, and prompt token hash
- `--page-cache-policy cold`, with post-prepare residency <= 1%
- `--moe-reset-cache-between-repeats`
- no warm/hot cache flags and no EAMC sidecar
- fixed visible GPU and no competing process
- interleaved control order, for example LRU/SERE/LRU

For DRAM, use `--moe-host-cache pageable --moe-host-cache-preload all`, then
repeat with `pinned`. Require all blobs ready and zero verification failures.
Preload time is reported separately from steady-state TTFT/TPOT.

Wall-clock TTFT and TPOT are the primary performance metrics. Profiler I/O,
H2D, compute, and stall intervals overlap and must not be summed as a wall time.

Required per-case outputs:

- summary, profile CSV, run log, token trace, metadata JSON
- TTFT, TPOT, tok/s, total time
- source bytes/reads and H2D bytes/time
- victim admissions and effective unique experts/misses
- original and effective unique/route hit rates
- secondary, rerouted, rerouted-miss, rejected routes and average similarity
- VRAM/DRAM peaks, host preload time, and page-cache residency

## 7. Interpretation limits

Synthetic matrices can prove parsing, rerouting, counters, and transfer changes.
They cannot prove expert similarity, acceptable quality, real-world speedup,
superiority over random replacement, cross-prompt generality, recurrent-state
stability, or replication of the paper's batch decoding speedup.

A publishable claim requires all three: a Q4-output-matched matrix,
teacher-forced/downstream quality results, and fair multi-workload performance.

