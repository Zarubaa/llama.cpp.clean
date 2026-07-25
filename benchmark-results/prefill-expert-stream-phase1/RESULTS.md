# Prefill expert streaming: phase 1

## Status

Phase 1 is implemented and validated. It overlaps gate-weight H2D with gate
MMQ at expert granularity during prefill. It does not yet implement a complete
per-expert `gate/up -> activation -> down` executor.

The fast path is active only when all of the following hold:

- `LLAMA_MOE_EXPERT_SOURCE=dram`
- `LLAMA_MOE_PREFILL_EXPERT_STREAM` is not `0`
- the microbatch contains more than 8 tokens
- the layer has more than one expert miss
- a CUDA compute backend is available

Decode and the default file/SSD source keep the original batched behavior.
`LLAMA_MOE_PREFILL_EXPERT_GROUP` controls the gate MMQ group limit and defaults
to 8.

## Implementation

1. The DRAM source is mapped read-only and prefaulted once at worker startup.
2. The eval callback sorts missed expert blobs by weight kind and routed-token
   rank, then submits expert-sized gate, up, and down transfers without waiting
   for the whole layer.
3. The worker copies each blob into pinned staging memory, issues H2D on a
   dedicated CUDA stream, and publishes the completion event for that slot and
   weight kind.
4. Gate MMQ reads expert bounds, prioritizes resident experts and experts with
   more routed tokens, waits only for the next group, and launches groups of up
   to 8 contiguous slots.
5. Up and down MMQ wait for their complete weight-kind transfer set and retain
   the original batched kernel.

The model destructor now stops the I/O worker before model buffers are released.
This fixes the previous process hang after `[moe-bench] done`; verified teardown
reports zero CUDA events in use and exits with status 0.

## Method

- GPU: one NVIDIA L20, selected with `CUDA_VISIBLE_DEVICES=0`
- CPU affinity: NUMA node 0, `0-31,64-95`
- Model: `Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf`
- Input: the natural English/project text in the repository `README.md`
- Prompt sizes: 64, 128, 256, 512, and 1024 tokens
- Effective ubatch: equal to prompt size
- Expert cache: 64 MiB; explicitly reset before every repeat
- Source: prefaulted DRAM mapping, not SSD
- Repeats: 3 cold expert-cache runs per point
- Comparison: the same final binary with streaming disabled versus group=8

The detailed summaries, per-layer CSV files, and logs are under `final-ab/`.
The complete numerical table is in `final-ab/summary.csv`.

## Results

| Tokens | Batched TTFT | Stream TTFT | Stream delta |
| ---: | ---: | ---: | ---: |
| 64 | 1360.2 ms | 1613.5 ms | +18.62% |
| 128 | 1733.9 ms | 2055.2 ms | +18.53% |
| 256 | 2220.3 ms | 2650.5 ms | +19.38% |
| 512 | 2733.4 ms | 3164.9 ms | +15.79% |
| 1024 | 3351.8 ms | 3695.5 ms | +10.25% |

There is no crossover through 1024 tokens. Streaming removes almost all host
callback blocking: at 1024 tokens callback wall time falls from 2.64 to 0.10
ms/token. However, the CUDA-timed interval rises from 0.19 to 2.95 ms/token.
That interval includes the H2D waits inserted into the compute stream and the
additional gate MMQ launches, so it must not be interpreted as arithmetic-only
kernel time. The cost of copying expert bounds to the host, synchronizing the
compute stream, sorting, waiting per group, and launching smaller MMQs exceeds
the overlap gained in this version. The penalty narrows at 1024 tokens, but the
data does not establish a profitable operating region.

The profile labels `ssd_read`, `SSD bytes`, and `SSD reads` are legacy names. In
these runs they measure copying from the prefaulted DRAM mapping into pinned
staging buffers, not physical SSD I/O.

## Correctness

The final binary was run with a 128-token natural-text prompt and 8 decode
steps. Disabled and group=8 streaming produced the same 9 greedy token IDs:

```text
0,1715,36879,36022,32814,3359,58659,14784,7025
```

The enabled debug trace shows `kind=0 action=split` and `kind=1/2
action=batched`, confirming that the measured path is the intended gate-only
pipeline rather than a silent fallback. Raw data is under `final-correctness/`.

## Conclusion

Phase 1 validates the asynchronous readiness mechanism, expert-granularity
ordering, correctness, and lifecycle, but it does not improve TTFT. The next
optimization should eliminate the synchronous expert-bounds D2H round trip and
reduce gate launch count, ideally by passing callback-side route counts/order
directly to the CUDA path or by adding an indirection-aware grouped MMQ. Only
after removing that control overhead is extending the pipeline through up,
activation, and down justified.

## Reproduction

Run inside `llama-cpp-clean-dev`:

```bash
FORCE=1 bash /workspace/benchmark-results/prefill-expert-stream-phase1/run_ab.sh
bash /workspace/benchmark-results/prefill-expert-stream-phase1/run_correctness.sh
```
