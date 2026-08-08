#!/usr/bin/env bash
set -euo pipefail

repo=${SERE_REPO:-/workspace}
run_id=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
model="$repo/models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf"
prompt="$repo/benchmark-results/new-result/prompts/normal-long-prompt.zh-en.txt"
sidecar="$repo/benchmark-results/sere/mechanism-only/synthetic/similarity.sere"
bench="$repo/build-sere-cuda/bin/llama-moe-bench"
run_root="$repo/benchmark-results/sere/mechanism-only/runs/$run_id"

if [[ -z ${CUDA_VISIBLE_DEVICES+x} || -z $CUDA_VISIBLE_DEVICES ]]; then
    echo "CUDA_VISIBLE_DEVICES must explicitly select exactly one GPU" >&2
    exit 1
fi
if [[ $CUDA_VISIBLE_DEVICES == *,* || $CUDA_VISIBLE_DEVICES =~ [[:space:]] ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain one GPU index or UUID, not: $CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi
if [[ ! $run_id =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "invalid run ID: $run_id" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required to record the selected GPU state" >&2
    exit 1
fi

gpu=$CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
if ! nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader >/dev/null 2>&1; then
    echo "CUDA_VISIBLE_DEVICES does not identify a GPU visible to nvidia-smi: $gpu" >&2
    exit 1
fi

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

write_gpu_snapshot() {
    local label=$1
    local output="$run_root/gpu-snapshot-$label.txt"
    {
        echo "recorded_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"
        echo "CUDA_VISIBLE_DEVICES=$gpu"
        echo
        echo "selected GPU"
        nvidia-smi --id="$gpu" \
            --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu \
            --format=csv,noheader
        echo
        echo "all GPUs"
        nvidia-smi \
            --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu \
            --format=csv,noheader
        echo
        echo "compute processes"
        nvidia-smi \
            --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader
    } >"$output" 2>&1
}

write_gpu_snapshot start
trap 'write_gpu_snapshot end || true' EXIT

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
run_case paper-s4-rho0-shadow \
    --moe-sere-path "$sidecar" --moe-sere-top-k 4 \
    --moe-sere-threshold 0 --moe-sere-policy paper --moe-sere-shadow
run_case miss-s4-rho0 \
    --moe-sere-path "$sidecar" --moe-sere-top-k 4 \
    --moe-sere-threshold 0 --moe-sere-policy miss

write_gpu_snapshot end
trap - EXIT

python3 "$repo/benchmark-results/sere/mechanism-only/validate-smoke.py" \
    "$run_root" \
    --model "$model" \
    --prompt "$prompt" \
    --sidecar "$sidecar" \
    --binary "$bench"
echo "$run_root"
