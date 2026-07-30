#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESULT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd "$RESULT_ROOT/../.." && pwd)
MODEL=${MODEL:-/home/haozhe.lou/models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf}
BENCH_BIN=${HOST_BENCH_BIN:-$REPO_ROOT/build-moe/bin/llama-moe-host-cache-bench}
GPU_INDEX=${GPU_INDEX:-0}
LOCAL_CPUS=${LOCAL_CPUS:-0-31,64-95}
REMOTE_CPUS=${REMOTE_CPUS:-32-63,96-127}
SAMPLES=${SAMPLES:-256}
ITERATIONS=${ITERATIONS:-20}
WARMUPS=${WARMUPS:-3}
OUT="$RESULT_ROOT/microbench"

apps=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true)
if [[ -n "${apps//[[:space:]]/}" ]]; then
    printf 'GPU %s has active compute processes:\n%s\n' "$GPU_INDEX" "$apps" >&2
    exit 42
fi

dd if="$MODEL" of=/dev/null bs=16M status=none
read -r resident size < <(fincore --bytes --noheadings -o RES,SIZE "$MODEL" | awk '{print $1, $2}')
if (( resident * 100 < size * 99 )); then
    printf 'page cache residency below 99%%\n' >&2
    exit 43
fi

mkdir -p "$OUT/pageable-copy" "$OUT/pinned-h2d"
for topology in local remote; do
    if [[ "$topology" == local ]]; then
        cpus=$LOCAL_CPUS
    else
        cpus=$REMOTE_CPUS
    fi
    raw="$OUT/raw-$topology.csv"
    taskset -c "$cpus" env "CUDA_VISIBLE_DEVICES=$GPU_INDEX" "$BENCH_BIN" \
        --model "$MODEL" --output "$raw" --samples "$SAMPLES" \
        --warmups "$WARMUPS" --iterations "$ITERATIONS" \
        > "$OUT/$topology-stdout.log" 2> "$OUT/$topology-stderr.log"
    awk -F, 'NR == 1 || $1 == "fread-pinned" || $1 == "pageable-pinned" || $1 == "pageable-staging-h2d"' \
        "$raw" > "$OUT/pageable-copy/$topology.csv"
    awk -F, 'NR == 1 || $1 == "pinned-h2d"' "$raw" > "$OUT/pinned-h2d/$topology.csv"
done

"$SCRIPT_DIR/summarize_microbench.py"
printf 'microbenchmark complete: %s\n' "$OUT/summary.tsv"
