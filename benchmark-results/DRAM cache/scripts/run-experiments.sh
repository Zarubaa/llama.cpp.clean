#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESULT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd "$RESULT_ROOT/../.." && pwd)

MODEL=${MODEL:-/home/haozhe.lou/models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf}
BENCH_BIN=${BENCH_BIN:-$REPO_ROOT/build-moe/bin/llama-moe-bench}
GPU_INDEX=${GPU_INDEX:-0}
CPU_LIST=${CPU_LIST:-0-31,64-95}
RUNS=${RUNS:-8}
PP=${PP:-1024}
TG=${TG:-1024}
RUN_MAIN=${RUN_MAIN:-1}
RUN_DIAGNOSTICS=${RUN_DIAGNOSTICS:-1}
RUN_REAL_PROMPT=${RUN_REAL_PROMPT:-1}
RUN_TARGET_MODES=${RUN_TARGET_MODES:-}
RUN_TARGET_VRAMS=${RUN_TARGET_VRAMS:-}
RUN_TARGET_UBATCHES=${RUN_TARGET_UBATCHES:-}
REAL_PROMPT_FILE=${REAL_PROMPT_FILE:-$REPO_ROOT/benchmark-results/new-result/prompts/normal-long-prompt.zh-en.txt}

MODES=(control-0716 pageable-cold pageable-hot pinned-hot)
VRAMS=(6000 8000)
UBATCHES=(512 1024)

mode_args() {
    case "$1" in
        control-0716) printf '%s\n' off none ;;
        pageable-cold) printf '%s\n' pageable none ;;
        pageable-hot) printf '%s\n' pageable all ;;
        pinned-hot) printf '%s\n' pinned all ;;
        *) return 1 ;;
    esac
}

mode_selected() {
    [[ -z "$RUN_TARGET_MODES" || ",$RUN_TARGET_MODES," == *",$1,"* ]]
}

value_selected() {
    [[ -z "$1" || ",$1," == *",$2,"* ]]
}

assert_gpu_idle() {
    local apps
    apps=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true)
    if [[ -n "${apps//[[:space:]]/}" ]]; then
        printf 'GPU %s has active compute processes:\n%s\n' "$GPU_INDEX" "$apps" >&2
        return 42
    fi
}

page_cache_bytes() {
    fincore --bytes --noheadings -o RES,SIZE "$MODEL" | awk '{print $1, $2}'
}

warm_page_cache() {
    dd if="$MODEL" of=/dev/null bs=16M status=none
    local resident size
    read -r resident size < <(page_cache_bytes)
    if (( resident * 100 < size * 99 )); then
        printf 'page cache residency below 99%%: %s/%s bytes\n' "$resident" "$size" >&2
        return 43
    fi
}

write_top_metadata() {
    {
        printf 'started=%s\n' "$(date -Is)"
        printf 'repo=%s\n' "$REPO_ROOT"
        printf 'branch=%s\n' "$(git -C "$REPO_ROOT" branch --show-current)"
        printf 'commit=%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
        printf 'model=%s\n' "$MODEL"
        printf 'model_sha256=%s\n' "$(sha256sum "$MODEL" | awk '{print $1}')"
        printf 'benchmark_binary=%s\n' "$BENCH_BIN"
        printf 'gpu_index=%s\n' "$GPU_INDEX"
        printf 'cpu_list=%s\n' "$CPU_LIST"
        printf 'pp=%s\ntg=%s\nruns=%s\n' "$PP" "$TG" "$RUNS"
        printf 'ulimit_locked_kib=%s\n' "$(ulimit -l)"
        printf 'git_status_begin\n'
        git -C "$REPO_ROOT" status --short
        printf 'git_status_end\n'
        nvidia-smi -i "$GPU_INDEX" --query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version --format=csv,noheader
        lscpu | rg 'CPU\(s\)|Model name|NUMA node'
        free -b
    } > "$RESULT_ROOT/run-metadata.txt"
}

validate_run() {
    local mode=$1 prefix=$2
    local summary=${prefix}-summary.txt
    local profile=${prefix}-profile.csv
    [[ -s "$summary" && -s "$profile" && -s "${prefix}-metadata.txt" && -s "${prefix}-tokens.csv" ]] || return 1
    [[ -s "${prefix}-stdout.log" && -s "${prefix}-stderr.log" && -s "${prefix}-memory.csv" ]] || return 1
    awk -F, 'NR == 1 { n = NF } NF != n { bad = 1 } END { exit bad }' "$profile"
    rg -q 'Host cache verification: verified=[0-9]+ failures=0' "$summary"
    if [[ "$mode" == pageable-hot || "$mode" == pinned-hot ]]; then
        rg -q 'SSD reads: 0 ' "$summary"
        rg -q 'Host cache ready: blobs=30720/30720' "$summary"
    fi
    if [[ "$mode" == pinned-hot ]]; then
        rg -q 'Host cache transfer: memcpy=0.00 GB' "$summary"
    fi
}

run_one() {
    local mode=$1 vram=$2 ubatch=$3 label=$4 prompt_kind=${5:-default}
    local cache preload
    mapfile -t cache_config < <(mode_args "$mode")
    cache=${cache_config[0]}
    preload=${cache_config[1]}

    local cell_dir
    if [[ "$prompt_kind" == real ]]; then
        cell_dir="$RESULT_ROOT/real-prompt/$mode/vram-${vram}-ub${ubatch}"
    else
        cell_dir="$RESULT_ROOT/$mode/vram-${vram}-ub${ubatch}"
    fi
    mkdir -p "$cell_dir"
    local prefix="$cell_dir/$label"

    if validate_run "$mode" "$prefix" 2>/dev/null; then
        printf '[skip] %s\n' "$prefix"
        return
    fi

    assert_gpu_idle
    warm_page_cache
    local cache_before cache_after
    cache_before=$(page_cache_bytes)
    local gpu_before gpu_after
    gpu_before=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader)

    local -a command=(
        "$BENCH_BIN" --model "$MODEL" -ngl 99 -c 4096
        --pp "$PP" --tg "$TG" --repeat 1 -ub "$ubatch"
        --moe-cache-vram-mb "$vram" --moe-predictor lru
        --moe-host-cache "$cache" --moe-host-cache-preload "$preload"
        --moe-profile-csv "${prefix}-profile.csv"
        --moe-profile-summary "${prefix}-summary.txt"
        --token-trace "${prefix}-tokens.csv"
    )
    if [[ "$prompt_kind" == real ]]; then
        local prompt
        prompt=$(<"$REAL_PROMPT_FILE")
        command+=(-p "$prompt")
    fi

    {
        printf 'mode=%s\nhost_cache=%s\npreload=%s\n' "$mode" "$cache" "$preload"
        printf 'vram_cache_mb=%s\nubatch=%s\npp=%s\ntg=%s\n' "$vram" "$ubatch" "$PP" "$TG"
        printf 'gpu_index=%s\ncpu_list=%s\n' "$GPU_INDEX" "$CPU_LIST"
        printf 'page_cache_before=%s\n' "$cache_before"
        printf 'gpu_before=%s\n' "$gpu_before"
        printf 'started=%s\n' "$(date -Is)"
        printf 'command='; printf '%q ' taskset -c "$CPU_LIST" env "CUDA_VISIBLE_DEVICES=$GPU_INDEX" "${command[@]}"; printf '\n'
    } > "${prefix}-metadata.txt"

    printf 'elapsed_s,rss_kib,hwm_kib,locked_kib,n0_pages,n1_pages\n' > "${prefix}-memory.csv"
    set +e
    taskset -c "$CPU_LIST" env "CUDA_VISIBLE_DEVICES=$GPU_INDEX" "${command[@]}" \
        > "${prefix}-stdout.log" 2> "${prefix}-stderr.log" &
    local bench_pid=$!
    local sample_start=$SECONDS
    while kill -0 "$bench_pid" 2>/dev/null; do
        local rss hwm locked nodes n0 n1
        rss=$(awk '/^VmRSS:/ {print $2}' "/proc/$bench_pid/status" 2>/dev/null || true)
        hwm=$(awk '/^VmHWM:/ {print $2}' "/proc/$bench_pid/status" 2>/dev/null || true)
        locked=$(awk '/^VmLck:/ {print $2}' "/proc/$bench_pid/status" 2>/dev/null || true)
        nodes=$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^N[0-9]+=/) print $i}' "/proc/$bench_pid/numa_maps" 2>/dev/null || true)
        n0=$(awk -F= '$1 == "N0" {sum += $2} END {print sum + 0}' <<< "$nodes")
        n1=$(awk -F= '$1 == "N1" {sum += $2} END {print sum + 0}' <<< "$nodes")
        printf '%s,%s,%s,%s,%s,%s\n' "$((SECONDS - sample_start))" "${rss:-0}" "${hwm:-0}" "${locked:-0}" "$n0" "$n1" >> "${prefix}-memory.csv"
        sleep 1
    done
    wait "$bench_pid"
    local rc=$?
    set -e
    cache_after=$(page_cache_bytes)
    gpu_after=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader)
    {
        printf 'finished=%s\nexit_code=%s\n' "$(date -Is)" "$rc"
        printf 'page_cache_after=%s\n' "$cache_after"
        printf 'gpu_after=%s\n' "$gpu_after"
        awk -F, 'NR > 1 && $2 > max {max=$2; hwm=$3; locked=$4; n0=$5; n1=$6} END {printf "peak_sample_rss_kib=%s\npeak_sample_hwm_kib=%s\npeak_sample_locked_kib=%s\npeak_sample_n0_pages=%s\npeak_sample_n1_pages=%s\n", max+0,hwm+0,locked+0,n0+0,n1+0}' "${prefix}-memory.csv"
    } >> "${prefix}-metadata.txt"
    if (( rc != 0 )); then
        printf '[fail] %s exit=%s\n' "$prefix" "$rc" >&2
        return "$rc"
    fi
    local peak_nodes n0_peak n1_peak local_pages remote_pages
    peak_nodes=$(awk -F, 'NR > 1 && $2 > max {max=$2; n0=$5; n1=$6} END {print n0+0, n1+0}' "${prefix}-memory.csv")
    read -r n0_peak n1_peak <<< "$peak_nodes"
    if [[ "$GPU_INDEX" == 0 || "$GPU_INDEX" == 1 ]]; then
        local_pages=$n0_peak
        remote_pages=$n1_peak
    else
        local_pages=$n1_peak
        remote_pages=$n0_peak
    fi
    if (( local_pages + remote_pages > 0 && local_pages * 100 < (local_pages + remote_pages) * 95 )); then
        printf '[fail] NUMA-local pages below 95%% for %s: local=%s remote=%s\n' "$prefix" "$local_pages" "$remote_pages" >&2
        return 44
    fi
    validate_run "$mode" "$prefix"
    assert_gpu_idle
    printf '[done] %s\n' "$prefix"
}

mkdir -p "$RESULT_ROOT/microbench/pageable-copy" "$RESULT_ROOT/microbench/pinned-h2d"
for mode in "${MODES[@]}"; do
    for vram in "${VRAMS[@]}"; do
        value_selected "$RUN_TARGET_VRAMS" "$vram" || continue
        for ubatch in "${UBATCHES[@]}"; do
            value_selected "$RUN_TARGET_UBATCHES" "$ubatch" || continue
            mkdir -p "$RESULT_ROOT/$mode/vram-${vram}-ub${ubatch}"
        done
    done
done

assert_gpu_idle
write_top_metadata

if [[ "$RUN_MAIN" == 1 ]]; then
    for vram in "${VRAMS[@]}"; do
        value_selected "$RUN_TARGET_VRAMS" "$vram" || continue
        for ubatch in "${UBATCHES[@]}"; do
            value_selected "$RUN_TARGET_UBATCHES" "$ubatch" || continue
            for (( round = 1; round <= RUNS; ++round )); do
                if (( round % 2 == 1 )); then
                    order=(control-0716 pageable-cold pageable-hot pinned-hot)
                else
                    order=(pinned-hot pageable-hot pageable-cold control-0716)
                fi
                for mode in "${order[@]}"; do
                    mode_selected "$mode" || continue
                    printf -v label 'run-%02d' "$round"
                    run_one "$mode" "$vram" "$ubatch" "$label"
                done
            done
            if [[ "$RUN_DIAGNOSTICS" == 1 ]]; then
                for mode in "${MODES[@]}"; do
                    mode_selected "$mode" || continue
                    run_one "$mode" "$vram" "$ubatch" diagnostic
                done
            fi
        done
    done
fi

if [[ "$RUN_REAL_PROMPT" == 1 ]] &&
   value_selected "$RUN_TARGET_VRAMS" 8000 &&
   value_selected "$RUN_TARGET_UBATCHES" 512; then
    for (( round = 1; round <= RUNS; ++round )); do
        if (( round % 2 == 1 )); then
            order=(control-0716 pageable-cold pageable-hot pinned-hot)
        else
            order=(pinned-hot pageable-hot pageable-cold control-0716)
        fi
        for mode in "${order[@]}"; do
            mode_selected "$mode" || continue
            printf -v label 'run-%02d' "$round"
            run_one "$mode" 8000 512 "$label" real
        done
    done
fi

"$SCRIPT_DIR/summarize.py"
printf 'completed=%s\n' "$(date -Is)" >> "$RESULT_ROOT/run-metadata.txt"
