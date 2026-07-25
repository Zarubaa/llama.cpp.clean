# Prefill expert streaming: P0 control-path cleanup

## Status

P0 is implemented and validated. The implementation removes the three
identified control-path costs from gate MMQ:

1. The eval callback publishes its existing routed-token counts, slot mapping,
   and hit/miss classification as an execution plan.
2. Gate MMQ no longer copies `expert_bounds` from device to host.
3. Gate MMQ no longer calls `cudaStreamSynchronize()` or sorts expert work.

The callback also replaces its comparison sort of miss blobs with a linear
three-kind bucket pass. Decode, non-DRAM sources, and the disabled path keep the
existing behavior.

Set `LLAMA_MOE_DEBUG_PREFILL_PLAN=1` to report host plan construction and gate
grouping/wait diagnostics. Diagnostic timing is not collected when the switch
is absent, so normal benchmark runs do not pay for `steady_clock` calls.

## Method

- GPU: one NVIDIA L20, selected with `CUDA_VISIBLE_DEVICES=0`
- CPU affinity: NUMA node 0, `0-31,64-95`
- Model: `Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf`
- Input: natural project text from `README.md`
- Prompt sizes: 512 and 1024 tokens
- Effective ubatch: equal to prompt size
- Expert cache: 64 MiB, reset before every repeat
- Source: prefaulted DRAM mapping
- Repeats: 3 cold expert-cache runs per point
- Comparison: the same final binary with streaming disabled and enabled

Only 512 and 1024 were run because both exceed the predefined 3% regression
stop threshold. The smaller and larger prompt sweep was intentionally stopped.

## Correctness

Disabled and enabled modes produced the same 9 greedy token IDs for a
128-token natural-text prompt followed by 8 decode steps:

```text
0,1715,36879,36022,32814,3359,58659,14784,7025
```

The enabled trace entered `kind=0 action=split`, retained batched processing for
up/down weights, completed all 40 MoE layers, and shut down with zero CUDA
events in use.

## Diagnostics

The 128-token cold-cache debug run published and consumed one plan for every
MoE layer:

- 40 plans and 40 split gate MMQs
- 4,633 active-expert entries in total
- 596 gate groups, only 3 singleton groups
- mean group size 7.773 with group limit 8
- 6.953 ms total callback plan-build time
- 443.711 ms total host wait for gate-weight readiness

The plan is therefore not fragmented on this cold-cache workload. The
remaining cost is waiting for the serialized expert transfers and launching
many smaller gate MMQs, rather than plan sorting or group formation.

## Results

| Tokens | Batched TTFT | P0 stream TTFT | Delta |
| ---: | ---: | ---: | ---: |
| 512 | 2682.2 ms | 3131.3 ms | +16.74% |
| 1024 | 3322.5 ms | 3715.5 ms | +11.83% |

At 512 tokens, callback wall time falls from 4.15 to 0.16 ms/token, while the
CUDA-timed interval rises from 0.24 to 4.90 ms/token. At 1024 tokens, callback
wall time falls from 2.62 to 0.10 ms/token, while the CUDA-timed interval rises
from 0.19 to 2.99 ms/token. That interval includes stream waits and therefore
measures the exposed pipeline cost, not arithmetic alone.

P0 performance is effectively unchanged from phase 1: phase 1 measured
3164.9 ms at 512 and 3695.5 ms at 1024. The removed D2H, stream synchronize,
and duplicate sorting were not the dominant costs.

## Decision

The gate-only path still misses the 3% stop criterion by a wide margin. Further
micro-optimization of this path is not justified. The next implementation step
should be a complete expert executor that overlaps full expert transfer with
`gate/up -> activation -> down`, while preserving batched kernels within a
small ready set. P0 remains useful as the execution-plan and readiness
foundation for that work.

## Reproduction

Run inside `llama-cpp-clean-dev`:

```bash
bash /workspace/benchmark-results/prefill-expert-stream-p0/run_correctness.sh
bash /workspace/benchmark-results/prefill-expert-stream-p0/run_ab.sh
```
