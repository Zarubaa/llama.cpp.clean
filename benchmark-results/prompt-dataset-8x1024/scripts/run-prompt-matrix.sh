#!/usr/bin/env bash
set -euo pipefail

# Prompt-aware runner: prompt_id is workload variation; repeat is process noise.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATASET_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd "$DATASET_ROOT/../.." && pwd)
RESULT_ROOT=${RESULT_ROOT:-$REPO_ROOT/benchmark-results/prompt-aware-8x1024}
MODEL=${MODEL:-/home/haozhe.lou/models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf}
EXPERIMENT_BIN=${EXPERIMENT_BIN:-$REPO_ROOT/build-moe/bin/llama-moe-bench}
BASELINE_BIN=${BASELINE_BIN:-/home/haozhe.lou/llama.cpp.baseline-9f0199dbb/build-moe/bin/llama-moe-bench}
GPU_INDEX=${GPU_INDEX:-0}; CPU_LIST=${CPU_LIST:-0-31,64-95}
PP=${PP:-1024}; TG=${TG:-1024}; PROCESS_REPEATS=${PROCESS_REPEATS:-1}
PROMPT_IDS=${PROMPT_IDS:-"01 02 03 04 05 06 07 08"}
MODES=${MODES:-"control-original pagecache-hot pinned-memory-hot"}
VRAMS=${VRAMS:-"6000 8000"}; UBATCHES=${UBATCHES:-"512 1024"}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}; DRY_RUN=${DRY_RUN:-0}
PLAN_ONLY=${PLAN_ONLY:-0}
HOTSET_BY_PROMPT=${HOTSET_BY_PROMPT:-0}
DM_STAT_PATH=${DM_STAT_PATH:-/sys/dev/block/253:0/stat}
BLOCK_STAT_PATH=${BLOCK_STAT_PATH:-/sys/class/block/sdb3/stat}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
manifest_value() {
    awk -F '\t' -v id="$1" -v column="$2" '
      NR==1 {for(i=1;i<=NF;i++) col[$i]=i; next} $1==id {print $(col[column]); exit}' \
      "$DATASET_ROOT/manifest.tsv"
}
prompt_path() { printf '%s/%s\n' "$DATASET_ROOT" "$(manifest_value "$1" file)"; }
assert_gpu_idle() {
    local apps
    apps=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>/dev/null || true)
    [[ -z "${apps//[[:space:]]/}" ]] || die "GPU $GPU_INDEX has active compute processes: $apps"
}
assert_ready() {
    [[ -r "$MODEL" ]] || die "model is not readable: $MODEL"
    [[ -x "$EXPERIMENT_BIN" ]] || die "experiment binary is not executable: $EXPERIMENT_BIN"
    [[ -r "$DATASET_ROOT/manifest.tsv" ]] || die "manifest is not readable"
    [[ -r "$DM_STAT_PATH" && -r "$BLOCK_STAT_PATH" ]] || die "block statistics unavailable"
    for cmd in nvidia-smi taskset python3 sha256sum awk; do command -v "$cmd" >/dev/null || die "missing command: $cmd"; done
    assert_gpu_idle
}
mode_config() {
    case "$1" in
      control-original) printf '%s\t%s\t%s\t%s\n' "$EXPERIMENT_BIN" natural off none ;;
      pagecache-hot) printf '%s\t%s\t%s\t%s\n' "$EXPERIMENT_BIN" hot off none ;;
      pinned-memory-hot) printf '%s\t%s\t%s\t%s\n' "$EXPERIMENT_BIN" hot pinned all ;;
      control) [[ -x "$BASELINE_BIN" ]] || die "baseline binary is not executable: $BASELINE_BIN"; printf '%s\t%s\t%s\t%s\n' "$BASELINE_BIN" hot off none ;;
      host4gb-cold) printf '%s\t%s\t%s\t%s\n' "$EXPERIMENT_BIN" hot pinned none ;;
      host4gb-hot) printf '%s\t%s\t%s\t%s\n' "$EXPERIMENT_BIN" hot pinned hotset ;;
      *) die "unknown mode: $1" ;;
    esac
}
write_block_stats() {
    local out=$1 dm_before=$2 block_before=$3; local -a d0 b0 d1 b1
    read -ra d0 <<< "$dm_before"; read -ra b0 <<< "$block_before"
    read -ra d1 < "$DM_STAT_PATH"; read -ra b1 < "$BLOCK_STAT_PATH"
    {
      printf 'device\treads_completed_before\tsectors_read_before\treads_completed_after\tsectors_read_after\treads_completed_delta\tsectors_read_delta\n'
      printf '253:0\t%s\t%s\t%s\t%s\t%s\t%s\n' "${d0[0]}" "${d0[2]}" "${d1[0]}" "${d1[2]}" "$((d1[0]-d0[0]))" "$((d1[2]-d0[2]))"
      printf 'sdb3\t%s\t%s\t%s\t%s\t%s\t%s\n' "${b0[0]}" "${b0[2]}" "${b1[0]}" "${b1[2]}" "$((b1[0]-b0[0]))" "$((b1[2]-b0[2]))"
    } > "$out"
}
run_one() {
    local mode=$1 vram=$2 ubatch=$3 prompt_id=$4 repeat=$5
    local prompt_file=$(prompt_path "$prompt_id") prompt_hash binary policy host_cache preload
    [[ -r "$prompt_file" ]] || die "prompt $prompt_id is not readable: $prompt_file"
    prompt_hash=$(sha256sum "$prompt_file" | awk '{print $1}')
    IFS=$'\t' read -r binary policy host_cache preload < <(mode_config "$mode")
    local cell="$RESULT_ROOT/$mode/vram-${vram}-ub${ubatch}/prompt-${prompt_id}"
    local prefix="$cell/repeat-${repeat}"
    local partial="${prefix}-partial-$$"
    mkdir -p "$cell"
    [[ ! -e "${prefix}-metadata.txt" ]] || { printf '[skip] %s\n' "$prefix"; return; }
    local -a command=("$binary" --model "$MODEL" --prompt-file "$prompt_file" -ngl 99 -c 4096
      --pp "$PP" --tg "$TG" --repeat 1 -ub "$ubatch" --moe-cache-vram-mb "$vram" --moe-predictor lru
      --moe-host-cache "$host_cache" --moe-host-cache-preload "$preload" --page-cache-policy "$policy"
      --moe-profile-csv "${partial}-profile.csv" --moe-profile-summary "${partial}-summary.txt" --token-trace "${partial}-tokens.csv")
    if [[ "$mode" == host4gb-cold || "$mode" == host4gb-hot ]]; then
      command+=(--moe-host-cache-capacity-mb 4096 --moe-tier-policy aged-lfu --moe-tier-half-life 128 --moe-decode-global-cache on)
      if [[ "$mode" == host4gb-hot ]]; then
        local hotset
        if [[ "$HOTSET_BY_PROMPT" == 1 ]]; then
          hotset="$RESULT_ROOT/hotsets/prompt-${prompt_id}/vram-${vram}-ub${ubatch}.tsv"
        else
          hotset="$RESULT_ROOT/hotsets/vram-${vram}-ub${ubatch}.tsv"
        fi
        [[ -s "$hotset" ]] || die "missing hotset: $hotset"
        command+=(--moe-host-cache-hotset "$hotset")
      fi
    fi
    {
      printf 'mode=%s\nprompt_id=%s\nrepeat=%s\n' "$mode" "$prompt_id" "$repeat"
      printf 'page_cache_policy=%s\nhost_cache=%s\nhost_cache_preload=%s\n' "$policy" "$host_cache" "$preload"
      printf 'vram_cache_mb=%s\nubatch=%s\npp=%s\ntg=%s\n' "$vram" "$ubatch" "$PP" "$TG"
      printf 'model=%s\nmodel_size=%s\nprompt_file=%s\nprompt_bytes=%s\nprompt_sha256=%s\n' "$MODEL" "$(stat -c %s "$MODEL")" "$prompt_file" "$(stat -c %s "$prompt_file")" "$prompt_hash"
      printf 'binary=%s\nbinary_sha256=%s\ngpu_index=%s\ncpu_list=%s\nstarted=%s\n' "$binary" "$(sha256sum "$binary" | awk '{print $1}')" "$GPU_INDEX" "$CPU_LIST" "$(date -Is)"
      printf 'command='; printf '%q ' taskset -c "$CPU_LIST" env "CUDA_VISIBLE_DEVICES=$GPU_INDEX" "${command[@]}"; printf '\n'
    } > "${partial}-metadata.txt"
    if [[ "$DRY_RUN" == 1 ]]; then rm -f "${partial}-metadata.txt"; printf '%q ' taskset -c "$CPU_LIST" env "CUDA_VISIBLE_DEVICES=$GPU_INDEX" "${command[@]}"; printf '\n'; return; fi
    local dm_before block_before gpu_before gpu_after; dm_before=$(<"$DM_STAT_PATH"); block_before=$(<"$BLOCK_STAT_PATH")
    gpu_before=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,clocks.current.sm,power.draw --format=csv,noheader)
    printf '[run] mode=%s vram=%s ubatch=%s prompt=%s repeat=%s\n' "$mode" "$vram" "$ubatch" "$prompt_id" "$repeat"
    local begin_ns=$(date +%s%N) pid rc; set +e
    taskset -c "$CPU_LIST" env "CUDA_VISIBLE_DEVICES=$GPU_INDEX" "${command[@]}" > "${partial}-stdout.log" 2> "${partial}-stderr.log" & pid=$!
    printf 'elapsed_ms,rss_kib,hwm_kib,locked_kib\n' > "${partial}-memory.csv"
    while kill -0 "$pid" 2>/dev/null; do
      local now_ns=$(date +%s%N) rss=$(awk '/^VmRSS:/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
      local hwm=$(awk '/^VmHWM:/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo 0) lck=$(awk '/^VmLck:/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
      printf '%s,%s,%s,%s\n' "$(((now_ns-begin_ns)/1000000))" "$rss" "$hwm" "$lck" >> "${partial}-memory.csv"; sleep 1
    done
    wait "$pid"; rc=$?; set -e; gpu_after=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,clocks.current.sm,power.draw --format=csv,noheader)
    write_block_stats "${partial}-block-io.tsv" "$dm_before" "$block_before"
    printf 'finished=%s\nexit_code=%s\ngpu_before=%s\ngpu_after=%s\n' "$(date -Is)" "$rc" "$gpu_before" "$gpu_after" >> "${partial}-metadata.txt"
    (( rc == 0 )) || die "$prefix failed with exit=$rc; partial files retained"
    python3 "$SCRIPT_DIR/validate-prompt-run.py" --prefix "$partial" --prompt-id "$prompt_id" --prompt-file "$prompt_file" --pp "$PP" --tg "$TG"
    for file in "${partial}"-*; do mv "$file" "${prefix}${file#"$partial"}"; done
    printf '[done] %s\n' "$prefix"
}
write_metadata() {
    mkdir -p "$RESULT_ROOT"
    { printf 'started=%s\nrepo=%s\nbranch=%s\ncommit=%s\n' "$(date -Is)" "$REPO_ROOT" "$(git -C "$REPO_ROOT" branch --show-current)" "$(git -C "$REPO_ROOT" rev-parse HEAD)"
      printf 'dataset=%s\nmodel=%s\nmodel_size=%s\nprompt_ids=%s\nprocess_repeats=%s\nmodes=%s\nvrams=%s\nubatches=%s\npp=%s\ntg=%s\ngpu_index=%s\ncpu_list=%s\nhotset_by_prompt=%s\n' "$DATASET_ROOT" "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT_IDS" "$PROCESS_REPEATS" "$MODES" "$VRAMS" "$UBATCHES" "$PP" "$TG" "$GPU_INDEX" "$CPU_LIST" "$HOTSET_BY_PROMPT"; } > "$RESULT_ROOT/run-metadata.txt"
}
"$DATASET_ROOT/validate-dataset.py" >/dev/null
if [[ "$PLAN_ONLY" != 1 ]]; then assert_ready; fi
write_metadata
[[ "$PREFLIGHT_ONLY" == 1 ]] && exit 0
prompt_index=0
for vram in $VRAMS; do for ubatch in $UBATCHES; do for prompt_id in $PROMPT_IDS; do
  [[ -n "$(manifest_value "$prompt_id" file)" ]] || die "unknown prompt id: $prompt_id"
  read -ra mode_list <<< "$MODES"; offset=$((prompt_index % ${#mode_list[@]}))
  for repeat in $(seq -w 1 "$PROCESS_REPEATS"); do for ((j=0;j<${#mode_list[@]};j++)); do
    mode=${mode_list[$(((j+offset)%${#mode_list[@]}))]}; run_one "$mode" "$vram" "$ubatch" "$prompt_id" "$repeat"
  done; done
  prompt_index=$((prompt_index+1))
done; done; done
python3 "$SCRIPT_DIR/summarize-prompt-matrix.py" --result-root "$RESULT_ROOT"; printf 'completed=%s\n' "$(date -Is)" >> "$RESULT_ROOT/run-metadata.txt"
