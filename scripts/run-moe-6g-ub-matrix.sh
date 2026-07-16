#!/usr/bin/env bash
set -euo pipefail

ROOT="${LLAMA_CPP_CLEAN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

MODEL="${MODEL:-$ROOT/models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf}"
OUT_DIR="${OUT_DIR:-$ROOT/benchmark-results/qwen3.6-35b-a3b/combined-axis-pp1024-tg1024/vram-6000}"
BIN="${BIN:-$ROOT/build-moe/bin/llama-moe-bench}"

PP="${PP:-1024}"
TG="${TG:-1024}"
REPEAT="${REPEAT:-1}"
CACHE_MB="${CACHE_MB:-6000}"
PREDICTOR="${PREDICTOR:-lru}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
FORCE="${FORCE:-0}"
EXIT_GRACE_SECONDS="${EXIT_GRACE_SECONDS:-5}"

export CUDA_VISIBLE_DEVICES
export LD_LIBRARY_PATH="$ROOT/build-moe/bin:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN" ]]; then
    echo "missing executable: $BIN" >&2
    echo "build it first: cmake --build build-moe --target llama-moe-bench -j\$(nproc)" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "missing model: $MODEL" >&2
    exit 1
fi

echo "root       : $ROOT"
echo "model      : $MODEL"
echo "out_dir    : $OUT_DIR"
echo "cache_mb   : $CACHE_MB"
echo "pp/tg      : $PP/$TG"
echo "repeat     : $REPEAT"
echo "predictor  : $PREDICTOR"
echo "cuda device: $CUDA_VISIBLE_DEVICES"
echo "force rerun: $FORCE"
echo

run_case() {
    local ub="$1"
    local prefix="${PREDICTOR}-6g-pp${PP}-tg${TG}-ub${ub}"
    local csv="$OUT_DIR/${prefix}-profile.csv"
    local summary="$OUT_DIR/${prefix}-summary.txt"
    local log="$OUT_DIR/${prefix}-run.log"

    if [[ "$FORCE" != "1" && -s "$summary" ]] && grep -q "profile rows:" "$summary"; then
        echo "===== skipping completed $prefix ====="
        echo "summary: $summary"
        echo
        return 0
    fi

    if [[ "$FORCE" == "1" ]]; then
        rm -f "$csv" "$summary" "$log"
    fi

    echo "===== running $prefix ====="
    "$BIN" \
        --model "$MODEL" \
        --pp "$PP" \
        --tg "$TG" \
        --repeat "$REPEAT" \
        --moe-cache-vram-mb "$CACHE_MB" \
        --moe-predictor "$PREDICTOR" \
        -ub "$ub" \
        --moe-profile-csv "$csv" \
        --moe-profile-summary "$summary" \
        > "$log" 2>&1 &

    local pid=$!
    echo "pid: $pid"
    echo "log: $log"

    local saw_done=0
    while kill -0 "$pid" 2>/dev/null; do
        if grep -q "\\[moe-bench\\] done\\." "$log" 2>/dev/null; then
            saw_done=1
            sleep "$EXIT_GRACE_SECONDS"
            if kill -0 "$pid" 2>/dev/null; then
                echo "process $pid still alive after summary; terminating to continue matrix"
                kill "$pid" 2>/dev/null || true
                sleep 2
                if kill -0 "$pid" 2>/dev/null; then
                    kill -9 "$pid" 2>/dev/null || true
                fi
            fi
            break
        fi
        sleep 5
    done

    wait "$pid" 2>/dev/null || true

    if [[ "$saw_done" != "1" ]]; then
        if grep -q "\\[moe-bench\\] done\\." "$log" 2>/dev/null; then
            saw_done=1
        fi
    fi

    if [[ "$saw_done" != "1" || ! -s "$summary" ]]; then
        echo "case failed or summary missing: $prefix" >&2
        echo "see log: $log" >&2
        return 1
    fi

    echo "===== done $prefix ====="
    echo "summary: $summary"
    echo
}

for UB in 128 256 512 1024; do
    run_case "$UB"
done

echo "all 6GB ubatch runs completed: $OUT_DIR"
