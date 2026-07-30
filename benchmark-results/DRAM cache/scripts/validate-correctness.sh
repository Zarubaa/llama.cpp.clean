#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESULT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd "$RESULT_ROOT/../.." && pwd)
MODEL=${MODEL:-/home/haozhe.lou/models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf}
BENCH_BIN=${BENCH_BIN:-$REPO_ROOT/build-moe/bin/llama-moe-bench}
GPU_INDEX=${GPU_INDEX:-0}
CPU_LIST=${CPU_LIST:-0-31,64-95}
OUT="$RESULT_ROOT/correctness"

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

mkdir -p "$OUT"
for mode in control-0716 pageable-cold pageable-hot pinned-hot; do
    case "$mode" in
        control-0716) cache=off; preload=none ;;
        pageable-cold) cache=pageable; preload=none ;;
        pageable-hot) cache=pageable; preload=all ;;
        pinned-hot) cache=pinned; preload=all ;;
    esac
    mode_dir="$OUT/$mode"
    mkdir -p "$mode_dir"
    taskset -c "$CPU_LIST" env "CUDA_VISIBLE_DEVICES=$GPU_INDEX" "$BENCH_BIN" \
        --model "$MODEL" -ngl 99 -c 64 --pp 8 --tg 4 --repeat 1 -ub 8 \
        --moe-cache-vram-mb 6000 --moe-predictor lru \
        --moe-host-cache "$cache" --moe-host-cache-preload "$preload" \
        --moe-profile-csv "$mode_dir/profile.csv" \
        --moe-profile-summary "$mode_dir/summary.txt" \
        --token-trace "$mode_dir/tokens.csv" \
        --logits-bin "$mode_dir/logits.bin" \
        > "$mode_dir/stdout.log" 2> "$mode_dir/stderr.log"
    awk -F, 'NR == 1 { n = NF } NF != n { bad = 1 } END { exit bad }' "$mode_dir/profile.csv"
    rg -q 'Host cache verification: verified=[0-9]+ failures=0' "$mode_dir/summary.txt"
done

"$SCRIPT_DIR/compare_logits.py" --root "$OUT" --threshold 1e-5
nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,utilization.gpu --format=csv,noheader > "$OUT/gpu-after.csv"
printf 'correctness validation passed; report=%s\n' "$OUT/report.tsv"
