#!/usr/bin/env bash
set -euo pipefail

repo=${SERE_REPO:-/workspace}
run_id=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
gpu=${CUDA_VISIBLE_DEVICES:-0}
model="$repo/models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf"
prompt="$repo/benchmark-results/new-result/prompts/normal-long-prompt.zh-en.txt"
sidecar="$repo/benchmark-results/sere/mechanism-only/synthetic/similarity.sere"
bench="$repo/build-sere-cuda/bin/llama-moe-bench"
run_root="$repo/benchmark-results/sere/mechanism-only/runs/$run_id"

for required in "$model" "$prompt" "$sidecar" "$bench"; do
    if [[ ! -f "$required" ]]; then
        echo "missing required file: $required" >&2
        exit 1
    fi
done
if [[ -e "$run_root" ]]; then
    echo "run directory already exists: $run_root" >&2
    exit 1
fi
mkdir -p "$run_root"

common=(
    --model "$model"
    --prompt-file "$prompt"
    --pp 128
    --tg 16
    --repeat 1
    -ub 128
    --moe-cache-vram-mb 6000
    --moe-predictor lru
    --moe-host-cache off
    --moe-host-cache-preload none
    --page-cache-policy cold
    --moe-reset-cache-between-repeats
)

run_case() {
    local name=$1
    shift
    local output="$run_root/$name"
    mkdir -p "$output"
    echo "running $name on CUDA_VISIBLE_DEVICES=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu "$bench" \
        "${common[@]}" \
        --moe-profile-csv "$output/profile.csv" \
        --moe-profile-summary "$output/summary.txt" \
        --token-trace "$output/tokens.csv" \
        --logits-bin "$output/logits.bin" \
        "$@" \
        >"$output/run.log" 2>&1
}

run_case lru-off
run_case paper-s8-rho0 \
    --moe-sere-path "$sidecar" --moe-sere-top-k 8 \
    --moe-sere-threshold 0 --moe-sere-policy paper
run_case paper-s4-rho0 \
    --moe-sere-path "$sidecar" --moe-sere-top-k 4 \
    --moe-sere-threshold 0 --moe-sere-policy paper
run_case miss-s4-rho0 \
    --moe-sere-path "$sidecar" --moe-sere-top-k 4 \
    --moe-sere-threshold 0 --moe-sere-policy miss

python3 "$repo/benchmark-results/sere/mechanism-only/validate-smoke.py" "$run_root"
echo "$run_root"

