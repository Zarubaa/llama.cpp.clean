#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace
OUT="$ROOT/benchmark-results/prefill-expert-stream-p0/final-ab"
MODEL="$ROOT/models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf"
BENCH="$ROOT/build-moe/bin/llama-moe-bench"
PROMPT="$(sed -n '1,1200p' "$ROOT/README.md")"
CPUS=0-31,64-95

export CUDA_VISIBLE_DEVICES=0
export LLAMA_MOE_EXPERT_SOURCE=dram
export LLAMA_MOE_PREFILL_EXPERT_GROUP=8
unset LLAMA_MOE_DEBUG_PREFILL_STREAM LLAMA_MOE_DEBUG_PREFILL_PLAN LLAMA_MOE_DEBUG_TOKENS || true

mkdir -p "$OUT"

for pp in 512 1024; do
    for mode in disabled enabled; do
        dir="$OUT/pp${pp}/${mode}"
        mkdir -p "$dir"
        if [[ "$mode" == disabled ]]; then
            export LLAMA_MOE_PREFILL_EXPERT_STREAM=0
        else
            export LLAMA_MOE_PREFILL_EXPERT_STREAM=1
        fi

        echo "[p0-ab] pp=$pp mode=$mode"
        timeout --signal=TERM 180s taskset -c "$CPUS" "$BENCH" \
            --model "$MODEL" \
            --pp "$pp" \
            --tg 1 \
            --repeat 3 \
            --moe-cache-vram-mb 64 \
            --moe-predictor lru \
            --moe-reset-cache-between-repeats \
            --moe-profile-csv "$dir/profile.csv" \
            --moe-profile-summary "$dir/summary.txt" \
            -ub "$pp" \
            -p "$PROMPT" \
            > "$dir/run.log" 2>&1
        grep '^TTFT cold:' "$dir/summary.txt"
    done
done
