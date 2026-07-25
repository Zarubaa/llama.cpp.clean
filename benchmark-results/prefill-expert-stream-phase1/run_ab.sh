#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace
OUT="$ROOT/benchmark-results/prefill-expert-stream-phase1/final-ab"
MODEL="$ROOT/models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf"
BENCH="$ROOT/build-moe/bin/llama-moe-bench"
PROMPT="$(sed -n '1,1200p' "$ROOT/README.md")"
CPUS=0-31,64-95

export CUDA_VISIBLE_DEVICES=0
export LLAMA_MOE_EXPERT_SOURCE=dram
export LLAMA_MOE_PREFILL_EXPERT_GROUP=8

mkdir -p "$OUT"

for pp in 64 128 256 512 1024; do
    for mode in disabled enabled; do
        dir="$OUT/pp${pp}/${mode}"
        mkdir -p "$dir"

        if [[ "$mode" == disabled ]]; then
            export LLAMA_MOE_PREFILL_EXPERT_STREAM=0
        else
            export LLAMA_MOE_PREFILL_EXPERT_STREAM=1
        fi

        echo "[final-ab] pp=$pp mode=$mode"
        if [[ "${FORCE:-0}" != 1 ]] && [[ -s "$dir/summary.txt" ]] && [[ -s "$dir/profile.csv" ]] && grep -q '\[moe-bench\] done' "$dir/run.log"; then
            echo "[final-ab] reusing complete result"
            grep '^TTFT cold:' "$dir/summary.txt"
            continue
        fi

        set +e
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

        if [[ $status -ne 0 ]]; then
            if [[ -s "$dir/summary.txt" ]] && [[ -s "$dir/profile.csv" ]] && grep -q '\[moe-bench\] done' "$dir/run.log"; then
                echo "[final-ab] complete output produced before teardown; accepting result"
            else
                echo "[final-ab] failed: pp=$pp mode=$mode status=$status" >&2
                exit "$status"
            fi
        fi
        grep '^TTFT cold:' "$dir/summary.txt"
    done
done
