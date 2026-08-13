#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESULT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd "$RESULT_ROOT/../.." && pwd)

MODEL=${MODEL:-/home/haozhe.lou/models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf}
PROMPT_FILE=${PROMPT_FILE:-$REPO_ROOT/benchmark-results/new-result/prompts/normal-long-prompt.zh-en.txt}
EXPERIMENT_BIN=${EXPERIMENT_BIN:-$REPO_ROOT/build-moe/bin/llama-moe-bench}
BASELINE_BIN=${BASELINE_BIN:-/home/haozhe.lou/llama.cpp.baseline-9f0199dbb/build-moe/bin/llama-moe-bench}
GPU_INDEX=${GPU_INDEX:-1}
CPU_LIST=${CPU_LIST:-0-31,64-95}
PP=${PP:-1024}
TG=${TG:-1024}
DM_STAT_PATH=${DM_STAT_PATH:-/sys/dev/block/253:0/stat}
BLOCK_STAT_PATH=${BLOCK_STAT_PATH:-/sys/class/block/sdb3/stat}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
SKIP_CALIBRATION=${SKIP_CALIBRATION:-0}

VRAMS=(6000 8000)
UBATCHES=(512 1024)
MODES=(control host4gb-cold host4gb-hot)

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

assert_gpu_idle() {
    local apps
    apps=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
    if [[ -n "${apps//[[:space:]]/}" ]]; then
        printf 'GPU %s has active compute processes:\n%s\n' "$GPU_INDEX" "$apps" >&2
        return 42
    fi
}

assert_no_benchmark() {
    local pids
    pids=$(pgrep -x llama-moe-bench || true)
    if [[ -n "${pids//[[:space:]]/}" ]]; then
        printf 'Another llama-moe-bench process is active: %s\n' "$pids" >&2
        return 43
    fi
}

assert_ready() {
    [[ -r "$MODEL" ]] || die "model is not readable: $MODEL"
    [[ -r "$PROMPT_FILE" ]] || die "prompt is not readable: $PROMPT_FILE"
    [[ -x "$EXPERIMENT_BIN" ]] || die "experiment binary is not executable: $EXPERIMENT_BIN"
    [[ -x "$BASELINE_BIN" ]] || die "baseline binary is not executable: $BASELINE_BIN"
    [[ -r "$DM_STAT_PATH" && -r "$BLOCK_STAT_PATH" ]] || die "block statistics are unavailable"
    assert_gpu_idle
    assert_no_benchmark
}

sample_memory() {
    local pid=$1 output=$2 begin_ns=$3
    printf 'elapsed_ms,rss_kib,hwm_kib,locked_kib,n0_pages,n1_pages\n' > "$output"
    while kill -0 "$pid" 2>/dev/null; do
        local now_ns rss hwm locked nodes n0 n1
        now_ns=$(date +%s%N)
        rss=$(awk '/^VmRSS:/ {print $2}' "/proc/$pid/status" 2>/dev/null || true)
        hwm=$(awk '/^VmHWM:/ {print $2}' "/proc/$pid/status" 2>/dev/null || true)
        locked=$(awk '/^VmLck:/ {print $2}' "/proc/$pid/status" 2>/dev/null || true)
        nodes=$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^N[0-9]+=/) print $i}' "/proc/$pid/numa_maps" 2>/dev/null || true)
        n0=$(awk -F= '$1 == "N0" {sum += $2} END {print sum + 0}' <<< "$nodes")
        n1=$(awk -F= '$1 == "N1" {sum += $2} END {print sum + 0}' <<< "$nodes")
        printf '%s,%s,%s,%s,%s,%s\n' "$(((now_ns - begin_ns) / 1000000))" \
            "${rss:-0}" "${hwm:-0}" "${locked:-0}" "$n0" "$n1" >> "$output"
        sleep 1
    done
}

write_block_stats() {
    local output=$1 dm_before_line=$2 block_before_line=$3
    local -a dm_before block_before dm_after block_after
    read -ra dm_before <<< "$dm_before_line"
    read -ra block_before <<< "$block_before_line"
    read -ra dm_after < "$DM_STAT_PATH"
    read -ra block_after < "$BLOCK_STAT_PATH"
    {
        printf 'device\treads_completed_before\tsectors_read_before\treads_completed_after\tsectors_read_after\treads_completed_delta\tsectors_read_delta\n'
        printf '253:0\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${dm_before[0]}" "${dm_before[2]}" "${dm_after[0]}" "${dm_after[2]}" \
            "$((dm_after[0] - dm_before[0]))" "$((dm_after[2] - dm_before[2]))"
        printf 'sdb3\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${block_before[0]}" "${block_before[2]}" "${block_after[0]}" "${block_after[2]}" \
            "$((block_after[0] - block_before[0]))" "$((block_after[2] - block_before[2]))"
    } > "$output"
}

hotset_path() {
    printf '%s/hotsets/vram-%s-ub%s.tsv\n' "$RESULT_ROOT" "$1" "$2"
}

calibrate_cell() {
    local vram=$1 ubatch=$2
    local dir="$RESULT_ROOT/calibration/vram-${vram}-ub${ubatch}"
    local heat="$dir/heat-all.tsv"
    local hotset
    hotset=$(hotset_path "$vram" "$ubatch")
    mkdir -p "$dir" "$RESULT_ROOT/hotsets"
    if [[ -s "$heat" && -s "$hotset" && $(wc -l < "$hotset") -eq 2240 ]]; then
        printf '[skip calibration] vram=%s ubatch=%s\n' "$vram" "$ubatch"
        return
    fi
    assert_ready
    local -a command=(
        "$EXPERIMENT_BIN" --model "$MODEL" --prompt-file "$PROMPT_FILE" -ngl 99 -c 4096
        --pp "$PP" --tg "$TG" --repeat 1 -ub "$ubatch"
        --moe-cache-vram-mb "$vram" --moe-predictor lru
        --moe-host-cache off --moe-host-cache-preload none
        --moe-tier-policy legacy --moe-decode-global-cache off
        --page-cache-policy hot --moe-heat-output "$heat"
        --moe-profile-csv "$dir/profile.csv" --moe-profile-summary "$dir/summary.txt"
        --token-trace "$dir/tokens.csv"
    )
    printf '[calibration] vram=%s ubatch=%s\n' "$vram" "$ubatch"
    taskset -c "$CPU_LIST" env "CUDA_VISIBLE_DEVICES=$GPU_INDEX" "${command[@]}" \
        > "$dir/stdout.log" 2> "$dir/stderr.log"
    local skip
    if [[ "$vram" == 6000 ]]; then skip=60; else skip=81; fi
    awk -v begin="$skip" -v end="$((skip + 56))" \
        '!/^#/ && $4 >= begin && $4 < end {print $1, $2, $3}' "$heat" > "$hotset"
    [[ $(wc -l < "$hotset") -eq 2240 ]] || die "hotset does not contain 56 experts/layer: $hotset"
    assert_gpu_idle
}

canonical_valid() {
    local prefix=$1 mode=$2
    "$SCRIPT_DIR/validate-run.py" --quiet --prefix "$prefix" --mode "$mode" \
        --policy hot --model-size "$(stat -c %s "$MODEL")"
}

canonical_absent() {
    local prefix=$1
    if compgen -G "${prefix}-*" >/dev/null; then
        printf 'Incomplete canonical files exist: %s-*\n' "$prefix" >&2
        return 1
    fi
}

run_one() {
    local mode=$1 vram=$2 ubatch=$3 run=$4
    local cell="$RESULT_ROOT/$mode/vram-${vram}-ub${ubatch}"
    local prefix="$cell/run-$run"
    mkdir -p "$cell"
    if canonical_valid "$prefix" "$mode" >/dev/null 2>&1; then
        printf '[skip valid] %s\n' "$prefix"
        return
    fi
    canonical_absent "$prefix" || die "move incomplete files aside before resuming"
    assert_ready

    local binary host_cache preload tier global hotset
    if [[ "$mode" == control ]]; then
        binary=$BASELINE_BIN
        host_cache=off
        preload=none
        tier=legacy
        global=off
        hotset=
    else
        binary=$EXPERIMENT_BIN
        host_cache=pinned
        tier=aged-lfu
        global=on
        hotset=$(hotset_path "$vram" "$ubatch")
        [[ -s "$hotset" ]] || die "missing hotset: $hotset"
        if [[ "$mode" == host4gb-hot ]]; then preload=hotset; else preload=none; fi
    fi

    local partial="${prefix}-partial-$$"
    local dm_before block_before gpu_before gpu_after gpu_used_before gpu_used_after
    dm_before=$(<"$DM_STAT_PATH")
    block_before=$(<"$BLOCK_STAT_PATH")
    gpu_before=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,clocks.current.sm,power.draw --format=csv,noheader)
    gpu_used_before=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')

    local -a command=(
        "$binary" --model "$MODEL" --prompt-file "$PROMPT_FILE" -ngl 99 -c 4096
        --pp "$PP" --tg "$TG" --repeat 1 -ub "$ubatch"
        --moe-cache-vram-mb "$vram" --moe-predictor lru
        --moe-host-cache "$host_cache" --moe-host-cache-preload "$preload"
        --page-cache-policy hot --moe-profile-csv "${partial}-profile.csv"
        --moe-profile-summary "${partial}-summary.txt" --token-trace "${partial}-tokens.csv"
    )
    if [[ "$mode" != control ]]; then
        command+=(
            --moe-host-cache-capacity-mb 4096 --moe-tier-policy "$tier"
            --moe-tier-half-life 128 --moe-decode-global-cache "$global"
        )
        if [[ "$preload" == hotset ]]; then
            command+=(--moe-host-cache-hotset "$hotset")
        fi
    fi

    {
        printf 'mode=%s\nrun=%s\npage_cache_policy=hot\n' "$mode" "$run"
        printf 'host_cache=%s\nhost_cache_preload=%s\ntier_policy=%s\ndecode_global_cache=%s\n' \
            "$host_cache" "$preload" "$tier" "$global"
        printf 'vram_cache_mb=%s\nubatch=%s\npp=%s\ntg=%s\n' "$vram" "$ubatch" "$PP" "$TG"
        printf 'model=%s\nmodel_size=%s\nprompt_file=%s\nprompt_sha256=%s\n' \
            "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT_FILE" "$(sha256sum "$PROMPT_FILE" | awk '{print $1}')"
        printf 'binary=%s\nbinary_sha256=%s\nhotset=%s\nhotset_sha256=%s\n' \
            "$binary" "$(sha256sum "$binary" | awk '{print $1}')" "$hotset" \
            "$([[ -n "$hotset" ]] && sha256sum "$hotset" | awk '{print $1}' || true)"
        printf 'gpu_index=%s\ncpu_list=%s\ngpu_before=%s\nstarted=%s\n' \
            "$GPU_INDEX" "$CPU_LIST" "$gpu_before" "$(date -Is)"
        printf 'command='; printf '%q ' taskset -c "$CPU_LIST" env "CUDA_VISIBLE_DEVICES=$GPU_INDEX" "${command[@]}"; printf '\n'
    } > "${partial}-metadata.txt"

    printf '[run] mode=%s vram=%s ubatch=%s run=%s\n' "$mode" "$vram" "$ubatch" "$run"
    local begin_ns pid rc
    begin_ns=$(date +%s%N)
    set +e
    taskset -c "$CPU_LIST" env "CUDA_VISIBLE_DEVICES=$GPU_INDEX" "${command[@]}" \
        > "${partial}-stdout.log" 2> "${partial}-stderr.log" &
    pid=$!
    sample_memory "$pid" "${partial}-memory.csv" "$begin_ns"
    wait "$pid"
    rc=$?
    set -e
    gpu_after=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,clocks.current.sm,power.draw --format=csv,noheader)
    gpu_used_after=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    write_block_stats "${partial}-block-io.tsv" "$dm_before" "$block_before"
    {
        printf 'finished=%s\nexit_code=%s\ngpu_after=%s\n' "$(date -Is)" "$rc" "$gpu_after"
        printf 'gpu_used_before_mib=%s\ngpu_used_after_mib=%s\ngpu_used_delta_mib=%s\n' \
            "$gpu_used_before" "$gpu_used_after" "$((gpu_used_after - gpu_used_before))"
        awk -F, 'NR > 1 {if ($2 > rss) rss=$2; if ($3 > hwm) hwm=$3; if ($4 > locked) locked=$4} END {printf "peak_sample_rss_kib=%d\npeak_sample_hwm_kib=%d\npeak_sample_locked_kib=%d\n",rss,hwm,locked}' "${partial}-memory.csv"
    } >> "${partial}-metadata.txt"
    (( rc == 0 )) || die "$prefix failed with exit=$rc; partial files retained"

    awk '/^Page cache (policy|prepare|resident|sample valid|timing)/' "${partial}-summary.txt" > "${partial}-page-cache.txt"
    awk '/^Process IO (read_bytes|rchar|sample valid):/' "${partial}-summary.txt" > "${partial}-process-io.txt"
    "$SCRIPT_DIR/validate-run.py" --prefix "$partial" --mode "$mode" --policy hot \
        --model-size "$(stat -c %s "$MODEL")"
    assert_gpu_idle
    if (( gpu_used_after > gpu_used_before + 64 || gpu_used_before > gpu_used_after + 64 )); then
        die "GPU memory did not return to baseline: before=$gpu_used_before after=$gpu_used_after MiB"
    fi
    local file suffix
    for file in "${partial}"-*; do
        suffix=${file#"$partial"}
        mv "$file" "${prefix}${suffix}"
    done
    printf '[done] %s\n' "$prefix"
}

write_top_metadata() {
    local model_sha256
    model_sha256=$(sha256sum "$MODEL" | awk '{print $1}')
    {
        printf 'started=%s\nrepo=%s\n' "$(date -Is)" "$REPO_ROOT"
        printf 'branch=%s\ncommit=%s\n' "$(git -C "$REPO_ROOT" branch --show-current)" "$(git -C "$REPO_ROOT" rev-parse HEAD)"
        printf 'model=%s\nmodel_size=%s\nmodel_sha256=%s\n' "$MODEL" "$(stat -c %s "$MODEL")" "$model_sha256"
        printf 'prompt_file=%s\nprompt_sha256=%s\n' "$PROMPT_FILE" "$(sha256sum "$PROMPT_FILE" | awk '{print $1}')"
        printf 'baseline_binary=%s\nbaseline_sha256=%s\n' "$BASELINE_BIN" "$(sha256sum "$BASELINE_BIN" | awk '{print $1}')"
        printf 'experiment_binary=%s\nexperiment_sha256=%s\n' "$EXPERIMENT_BIN" "$(sha256sum "$EXPERIMENT_BIN" | awk '{print $1}')"
        printf 'gpu_index=%s\ncpu_list=%s\npp=%s\ntg=%s\nruns_per_cell=3\n' "$GPU_INDEX" "$CPU_LIST" "$PP" "$TG"
        printf 'git_status_begin\n'; git -C "$REPO_ROOT" status --short; printf 'git_status_end\n'
        nvidia-smi -i "$GPU_INDEX" --query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version --format=csv,noheader
        lscpu | rg 'CPU\(s\)|Model name|NUMA node'
        free -b
        ulimit -l
    } > "$RESULT_ROOT/run-metadata.txt"
}

for command in nvidia-smi pgrep taskset python3 sha256sum rg awk; do
    command -v "$command" >/dev/null 2>&1 || die "required command not found: $command"
done
assert_ready
printf 'Preflight passed: GPU%s idle, binaries and realistic prompt ready.\n' "$GPU_INDEX"
if [[ "$PREFLIGHT_ONLY" == 1 ]]; then exit 0; fi

write_top_metadata
if [[ "$SKIP_CALIBRATION" != 1 ]]; then
    for vram in "${VRAMS[@]}"; do
        for ubatch in "${UBATCHES[@]}"; do calibrate_cell "$vram" "$ubatch"; done
    done
fi

for vram in "${VRAMS[@]}"; do
    for ubatch in "${UBATCHES[@]}"; do
        for run in 01 02 03; do
            case "$run" in
                01) order=(control host4gb-cold host4gb-hot) ;;
                02) order=(host4gb-hot host4gb-cold control) ;;
                03) order=(host4gb-cold control host4gb-hot) ;;
            esac
            for mode in "${order[@]}"; do run_one "$mode" "$vram" "$ubatch" "$run"; done
        done
    done
done

"$SCRIPT_DIR/summarize.py" --require-complete
printf 'completed=%s\n' "$(date -Is)" >> "$RESULT_ROOT/run-metadata.txt"
