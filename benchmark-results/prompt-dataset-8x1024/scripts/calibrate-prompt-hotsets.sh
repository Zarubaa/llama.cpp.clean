#!/usr/bin/env bash
set -euo pipefail

# Generate one bounded 4 GiB hotset per prompt and hardware cell. Calibration
# uses the same routing workload as the eventual prompt run, but is not part
# of timed performance statistics.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATASET_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd "$DATASET_ROOT/../.." && pwd)
RESULT_ROOT=${RESULT_ROOT:-$REPO_ROOT/benchmark-results/0811\ 4GB\ DRAM\ cache}
MODEL=${MODEL:-/home/haozhe.lou/models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf}
EXPERIMENT_BIN=${EXPERIMENT_BIN:-$REPO_ROOT/build-moe/bin/llama-moe-bench}
GPU_INDEX=${GPU_INDEX:-1}; CPU_LIST=${CPU_LIST:-0-31,64-95}
PP=${PP:-1024}; TG=${TG:-1024}
PROMPT_IDS=${PROMPT_IDS:-"01 02 03 04"}; VRAMS=${VRAMS:-"6000 8000"}; UBATCHES=${UBATCHES:-"512 1024"}
PLAN_ONLY=${PLAN_ONLY:-0}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
manifest_value() {
    awk -F '\t' -v id="$1" -v column="$2" '
      NR==1 {for(i=1;i<=NF;i++) col[$i]=i; next} $1==id {print $(col[column]); exit}' \
      "$DATASET_ROOT/manifest.tsv"
}
prompt_path() { printf '%s/%s\n' "$DATASET_ROOT" "$(manifest_value "$1" file)"; }
assert_ready() {
    [[ -r "$MODEL" && -x "$EXPERIMENT_BIN" ]] || die "model/binary unavailable"
    command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
    command -v taskset >/dev/null || die "taskset is required"
    local apps
    apps=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
    [[ -z "${apps//[[:space:]]/}" ]] || die "GPU $GPU_INDEX has active compute processes: $apps"
}
calibrate_one() {
    local prompt_id=$1 vram=$2 ubatch=$3
    local prompt_file dir heat hotset
    prompt_file=$(prompt_path "$prompt_id")
    dir="$RESULT_ROOT/calibration/prompt-${prompt_id}/vram-${vram}-ub${ubatch}"
    heat="$dir/heat-all.tsv"; hotset="$RESULT_ROOT/hotsets/prompt-${prompt_id}/vram-${vram}-ub${ubatch}.tsv"
    mkdir -p "$dir" "$(dirname "$hotset")"
    if [[ -s "$hotset" && $(wc -l < "$hotset") -eq 2240 ]]; then
        printf '[skip calibration] prompt=%s vram=%s ubatch=%s\n' "$prompt_id" "$vram" "$ubatch"; return
    fi
    local -a command=("$EXPERIMENT_BIN" --model "$MODEL" --prompt-file "$prompt_file" -ngl 99 -c 4096
      --pp "$PP" --tg "$TG" --repeat 1 -ub "$ubatch" --moe-cache-vram-mb "$vram" --moe-predictor lru
      --moe-host-cache off --moe-host-cache-preload none --moe-tier-policy legacy --moe-decode-global-cache off
      --page-cache-policy hot --moe-heat-output "$heat" --moe-profile-csv "$dir/profile.csv"
      --moe-profile-summary "$dir/summary.txt" --token-trace "$dir/tokens.csv")
    printf 'prompt_id=%s\nvram_cache_mb=%s\nubatch=%s\nprompt_file=%s\nprompt_sha256=%s\n' \
      "$prompt_id" "$vram" "$ubatch" "$prompt_file" "$(sha256sum "$prompt_file" | awk '{print $1}')" > "$dir/metadata.txt"
    printf 'command=' >> "$dir/metadata.txt"; printf '%q ' taskset -c "$CPU_LIST" env "CUDA_VISIBLE_DEVICES=$GPU_INDEX" "${command[@]}" >> "$dir/metadata.txt"; printf '\n' >> "$dir/metadata.txt"
    if [[ "$PLAN_ONLY" == 1 ]]; then
        printf '%q ' taskset -c "$CPU_LIST" env "CUDA_VISIBLE_DEVICES=$GPU_INDEX" "${command[@]}"; printf '\n'; return
    fi
    printf '[calibration] prompt=%s vram=%s ubatch=%s\n' "$prompt_id" "$vram" "$ubatch"
    taskset -c "$CPU_LIST" env "CUDA_VISIBLE_DEVICES=$GPU_INDEX" "${command[@]}" > "$dir/stdout.log" 2> "$dir/stderr.log"
    local skip=60
    [[ "$vram" == 8000 ]] && skip=81
    awk -v begin="$skip" -v end="$((skip + 56))" '!/^#/ && $4 >= begin && $4 < end {print $1, $2, $3}' "$heat" > "$hotset"
    [[ $(wc -l < "$hotset") -eq 2240 ]] || die "hotset does not contain 2240 entries: $hotset"
}

"$DATASET_ROOT/validate-dataset.py" >/dev/null
if [[ "$PLAN_ONLY" != 1 ]]; then assert_ready; fi
mkdir -p "$RESULT_ROOT"
for prompt_id in $PROMPT_IDS; do
  [[ -n "$(manifest_value "$prompt_id" file)" ]] || die "unknown prompt id: $prompt_id"
  for vram in $VRAMS; do for ubatch in $UBATCHES; do calibrate_one "$prompt_id" "$vram" "$ubatch"; done; done
done
