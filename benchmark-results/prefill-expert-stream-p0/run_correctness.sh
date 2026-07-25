#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace
OUT="$ROOT/benchmark-results/prefill-expert-stream-p0/correctness"
MODEL="$ROOT/models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf"
BENCH="$ROOT/build-moe/bin/llama-moe-bench"
PROMPT="$(sed -n '1,1200p' "$ROOT/README.md")"
CPUS=0-31,64-95

export CUDA_VISIBLE_DEVICES=0
export LLAMA_MOE_EXPERT_SOURCE=dram
export LLAMA_MOE_PREFILL_EXPERT_GROUP=8
export LLAMA_MOE_DEBUG_TOKENS=1

mkdir -p "$OUT"

for mode in disabled enabled; do
    dir="$OUT/$mode"
    mkdir -p "$dir"
    if [[ "$mode" == disabled ]]; then
        export LLAMA_MOE_PREFILL_EXPERT_STREAM=0
        unset LLAMA_MOE_DEBUG_PREFILL_STREAM LLAMA_MOE_DEBUG_PREFILL_PLAN || true
    else
        export LLAMA_MOE_PREFILL_EXPERT_STREAM=1
        export LLAMA_MOE_DEBUG_PREFILL_STREAM=1
        export LLAMA_MOE_DEBUG_PREFILL_PLAN=1
    fi

    echo "[p0-correctness] mode=$mode"
    timeout --signal=TERM 180s taskset -c "$CPUS" "$BENCH" \
        --model "$MODEL" \
        --pp 128 \
        --tg 8 \
        --repeat 1 \
        --moe-cache-vram-mb 64 \
        --moe-predictor lru \
        --moe-reset-cache-between-repeats \
        --moe-profile-csv "$dir/profile.csv" \
        --moe-profile-summary "$dir/summary.txt" \
        -ub 128 \
        -p "$PROMPT" \
        > "$dir/run.log" 2>&1
    sed -n 's/.*\[moe-bench-token\].*token=\([-0-9]*\).*/\1/p' \
        "$dir/run.log" > "$dir/tokens.txt"
done

if cmp -s "$OUT/disabled/tokens.txt" "$OUT/enabled/tokens.txt"; then
    {
        echo 'match: yes'
        printf 'tokens: '
        paste -sd, "$OUT/disabled/tokens.txt"
    } | tee "$OUT/comparison.txt"
else
    {
        echo 'match: no'
        diff -u "$OUT/disabled/tokens.txt" "$OUT/enabled/tokens.txt" || true
    } | tee "$OUT/comparison.txt"
    exit 1
fi
