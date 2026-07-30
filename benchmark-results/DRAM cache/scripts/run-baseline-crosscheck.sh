#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESULT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd "$RESULT_ROOT/../.." && pwd)
MODEL=${MODEL:-/home/haozhe.lou/models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf}
EXPERIMENT_BIN=${BENCH_BIN:-$REPO_ROOT/build-moe/bin/llama-moe-bench}
GPU_INDEX=${GPU_INDEX:-0}
CPU_LIST=${CPU_LIST:-0-31,64-95}

if [[ ! -s "$RESULT_ROOT/baseline-binary.txt" ]]; then
    "$SCRIPT_DIR/build-baseline.sh"
fi
BASELINE_BIN=$(awk -F= '$1 == "binary" {print $2}' "$RESULT_ROOT/baseline-binary.txt")
[[ -x "$BASELINE_BIN" ]]

apps=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true)
if [[ -n "${apps//[[:space:]]/}" ]]; then
    printf 'GPU %s has active compute processes:\n%s\n' "$GPU_INDEX" "$apps" >&2
    exit 42
fi
dd if="$MODEL" of=/dev/null bs=16M status=none

OUT="$RESULT_ROOT/baseline-crosscheck"
mkdir -p "$OUT/baseline" "$OUT/experiment-off"

run_case() {
    local binary=$1
    local summary=$2
    local stdout=$3
    local stderr=$4
    shift 4

    rm -f "$summary" "$stdout" "$stderr"
    taskset -c "$CPU_LIST" env "CUDA_VISIBLE_DEVICES=$GPU_INDEX" "$binary" "$@" \
        > "$stdout" 2> "$stderr" &
    local pid=$!
    local deadline=$((SECONDS + ${BASELINE_TIMEOUT_S:-300}))
    while kill -0 "$pid" 2>/dev/null; do
        if [[ -s "$summary" ]] && rg -q '^TPOT: [1-9]' "$summary"; then
            # The archived 0716 binary can leave its worker threads waiting
            # after writing the complete summary. The measured run is done.
            printf 'terminated_after_summary=1\n' >> "$stderr"
            kill -TERM "$pid" 2>/dev/null || true
            for _ in {1..20}; do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.5
            done
            kill -KILL "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        if (( SECONDS >= deadline )); then
            printf 'summary timeout after %ss\n' "${BASELINE_TIMEOUT_S:-300}" >&2
            kill -KILL "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
            return 124
        fi
        sleep 1
    done
    wait "$pid"
}

for round in 1 2 3; do
    if (( round % 2 == 1 )); then
        order=(baseline experiment-off)
    else
        order=(experiment-off baseline)
    fi
    for kind in "${order[@]}"; do
        printf -v run 'run-%02d' "$round"
        if [[ "$kind" == baseline ]]; then
            binary=$BASELINE_BIN
            extra=()
        else
            binary=$EXPERIMENT_BIN
            extra=(--moe-host-cache off --moe-host-cache-preload none)
        fi
        run_case "$binary" "$OUT/$kind/$run-summary.txt" \
            "$OUT/$kind/$run-stdout.log" "$OUT/$kind/$run-stderr.log" \
            --model "$MODEL" -ngl 99 -c 4096 --pp 1024 --tg 1024 --repeat 1 -ub 1024 \
            --moe-cache-vram-mb 6000 --moe-predictor lru \
            --moe-profile-summary "$OUT/$kind/$run-summary.txt" "${extra[@]}"
    done
done

"$SCRIPT_DIR/compare_baseline.py"
printf 'baseline cross-check passed: %s\n' "$OUT/report.tsv"
