# MoE offload shared-scratch combined-axis prototype

## Objective

This clean-repo prototype implements the single-GPU, VRAM-limited MoE offload
idea with a combined expert tensor axis:

- each MoE layer keeps a fixed number of persistent expert slots controlled by
  `--moe-cache-vram-mb`;
- streaming mode adds one shared scratch segment for current-layer misses;
- routed expert IDs are remapped to unified slot IDs before `ggml_mul_mat_id`;
- the upstream expert matmul still reads one 3D tensor, so no two-source CUDA
  kernel is required for this prototype.

The default upstream path is unchanged unless CMake is configured with
`-DLLAMA_MOE_OFFLOAD=ON` and runtime is started with `--moe-offload`.

## Runtime layout

In streaming mode, each expert kind (`gate`, `up`, `down`) gets one global
combined tensor:

```text
[L0 persistent][L1 persistent] ... [Ln persistent][shared scratch]
```

For a model with 64 MoE layers, 256 experts/layer, and 81 persistent slots/layer:

```text
persistent axis = 64 * 81
scratch axis    = 256 - 81 = 175
global axis     = 64 * 81 + 175
active/layer    = 81 + 175 = 256
```

The persistent segment is per-layer and retained across callbacks. The scratch
segment is global and transient; it is reused by every layer because only one
layer's MoE block is evaluated at a time in this single-GPU offload scenario.

## Callback behavior

After `ffn_moe_topk` is computed, `moe_eval_callback` reads the routed expert
IDs and builds a temporary `expert -> unified_slot` map for the current layer:

1. Persistent hits map to `logical_layer * persistent_slots + local_slot`.
2. Misses first fill free persistent slots while the cache warms up.
3. Additional misses use the global scratch segment.
4. If scratch is manually limited and fills up, remaining misses evict only
   non-current persistent experts.
5. Scratch assignments are not inserted into the LRU/EAMC cache state.
6. The callback writes the remapped slot IDs consumed by downstream
   `ggml_mul_mat_id`.

This changes the old hard bound from:

```text
unique experts in current layer <= persistent slots
```

to:

```text
unique experts in current layer <= persistent slots + shared scratch slots
```

## Controls

- `--moe-cache-vram-mb N`: persistent cache budget in MiB.
- `--moe-predictor lru|eamc`: persistent-cache eviction predictor.
- `LLAMA_MOE_SHARED_SCRATCH_SLOTS=N`: override scratch slots. By default,
  scratch fills the remaining per-layer expert axis, `n_experts - persistent`.
- `LLAMA_MOE_DISABLE_SHARED_SCRATCH=1`: restore persistent-only behavior for
  A/B comparison.
- `LLAMA_MOE_STREAMING_UBATCH=N|auto|0`: force, auto-size, or disable MoE
  streaming ubatch adjustment.

Benchmark summaries print slots as:

```text
slots=persistent/active_per_layer/experts
```

For the 81-slot example above, this should read `slots=81/256/256`.

## Container test flow

From the host:

```bash
cd /home/chaoyang.zhang/docker_workspace/projects/llama.cpp.clean

sudo docker run --rm -it --gpus all \
  --name llama-cpp-clean-moe \
  -v /home/chaoyang.zhang/docker_workspace/projects/llama.cpp.clean:/workspace \
  -v /home/chaoyang.zhang/docker_workspace/projects/llama.cpp.offload/models/qwen3.6-35b-a3b:/models/qwen3.6-35b-a3b:ro \
  -w /workspace \
  llama-cpp-offload:dev-cuda12.4 \
  bash
```

Inside the container:

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export CUDAToolkit_ROOT=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH

rm -rf build-moe
cmake -S . -B build-moe \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DLLAMA_MOE_OFFLOAD=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_BUILD_TESTS=OFF

cmake --build build-moe --config Release \
  --target llama-cli llama-moe-repack llama-moe-bench \
  -j$(nproc)
```

Repack once:

```bash
mkdir -p /workspace/models/qwen3.6-35b-a3b

build-moe/bin/llama-moe-repack \
  /models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  /workspace/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf
```

Smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/workspace/build-moe/bin:${LD_LIBRARY_PATH:-} \
build-moe/bin/llama-cli \
  --model /workspace/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf \
  --moe-offload \
  --moe-cache-vram-mb 8000 \
  --moe-predictor lru \
  -p "用一句话介绍你自己。" \
  -n 32 \
  -ub 16
```

Benchmark smoke:

```bash
mkdir -p /workspace/benchmark-results/qwen3.6-35b-a3b/combined-axis-smoke

CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/workspace/build-moe/bin:${LD_LIBRARY_PATH:-} \
build-moe/bin/llama-moe-bench \
  --model /workspace/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf \
  --pp 256 \
  --tg 32 \
  --repeat 1 \
  --moe-cache-vram-mb 8000 \
  --moe-predictor lru \
  -ub 16 \
  --moe-profile-csv /workspace/benchmark-results/qwen3.6-35b-a3b/combined-axis-smoke/profile.csv \
  --moe-profile-summary /workspace/benchmark-results/qwen3.6-35b-a3b/combined-axis-smoke/summary.txt
```

Expected log signals:

```text
slot pool configured: ... persistent slots/layer + ... shared-scratch slots (active/layer=..., global_axis=...)
MoE streaming ubatch auto-sized: ... active_slots=...
```

Ubatch sizing note: the automatic MoE cap only applies when the active expert
axis is smaller than the model's per-layer expert count. With the default shared
scratch plan, `active_slots == n_experts` for Qwen3.6-35B-A3B, so a layer can
never need more than 256 active experts no matter how many tokens are in the
ubatch. In that full-axis case the requested `-ub` is preserved; remaining
limits come from normal llama.cpp context, graph, CUDA workspace, and memory
constraints.

For an A/B check against old behavior:

```bash
LLAMA_MOE_DISABLE_SHARED_SCRATCH=1 CUDA_VISIBLE_DEVICES=0 ...same command...
```

That run should show `active/layer == persistent slots` and may reduce effective
ubatch or abort if per-layer unique experts exceed the persistent cache.
