#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace
OUT="$ROOT/benchmark-results/prefill-expert-stream-phase1/final-correctness"
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
        unset LLAMA_MOE_DEBUG_PREFILL_STREAM
    else
        export LLAMA_MOE_PREFILL_EXPERT_STREAM=1
        export LLAMA_MOE_DEBUG_PREFILL_STREAM=1
    fi

    echo "[correctness] mode=$mode"
    set +e
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
        > "$dir/run.log" 2>&1 &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        if grep -q '\[moe-bench\] done' "$dir/run.log"; then
            sleep 2
            kill -TERM "$pid" 2>/dev/null || true
            break
        fi
        sleep 1
    done
    wait "$pid"
    status=$?
    set -e

    if [[ $status -ne 0 ]] && ! { [[ -s "$dir/summary.txt" ]] && grep -q '\[moe-bench\] done' "$dir/run.log"; }; then
        echo "[correctness] failed: mode=$mode status=$status" >&2
        exit "$status"
    fi
    sed -n 's/.*\[moe-bench-token\].*token=\([-0-9]*\).*/\1/p' "$dir/run.log" > "$dir/tokens.txt"
done

if cmp -s "$OUT/disabled/tokens.txt" "$OUT/enabled/tokens.txt"; then
    {
        echo 'match: yes'
        printf 'tokens: '
        paste -sd, "$OUT/disabled/tokens.txt"
    } > "$OUT/comparison.txt"
else
    {
        echo 'match: no'
        diff -u "$OUT/disabled/tokens.txt" "$OUT/enabled/tokens.txt" || true
    } > "$OUT/comparison.txt"
    exit 1
fi

cat "$OUT/comparison.txt"
