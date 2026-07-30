#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESULT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd "$RESULT_ROOT/../.." && pwd)
BASELINE_COMMIT=${BASELINE_COMMIT:-f2344e9cdb7729cdc5c17bc196f0a2f51dc34909}
BASELINE_ROOT=$(mktemp -d /tmp/llama-dram-cache-baseline-XXXXXX)
CUDA_COMPILER=${CMAKE_CUDA_COMPILER:-${CUDACXX:-/usr/local/cuda-12.4/bin/nvcc}}

if [[ ! -x "$CUDA_COMPILER" ]]; then
    printf 'CUDA compiler is not executable: %s\n' "$CUDA_COMPILER" >&2
    exit 44
fi

git -C "$REPO_ROOT" archive "$BASELINE_COMMIT" | tar -x -C "$BASELINE_ROOT"
cmake -S "$BASELINE_ROOT" -B "$BASELINE_ROOT/build-moe" \
    -DGGML_CUDA=ON -DLLAMA_MOE_OFFLOAD=ON -DLLAMA_BUILD_TESTS=OFF \
    -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER"
cmake --build "$BASELINE_ROOT/build-moe" --target llama-moe-bench -j 16

{
    printf 'commit=%s\n' "$BASELINE_COMMIT"
    printf 'source=%s\n' "$BASELINE_ROOT"
    printf 'binary=%s\n' "$BASELINE_ROOT/build-moe/bin/llama-moe-bench"
    printf 'built=%s\n' "$(date -Is)"
} > "$RESULT_ROOT/baseline-binary.txt"
printf 'baseline binary: %s\n' "$BASELINE_ROOT/build-moe/bin/llama-moe-bench"
