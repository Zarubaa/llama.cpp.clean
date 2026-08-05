#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
result_root=${RESULT_ROOT:-"$repo_root/benchmark-results/0809_fix_results"}
container_name=${CONTAINER_NAME:-llama-cpp-clean-dev}
gpu_index=${GPU_INDEX:-3}
cuda_visible=${CUDA_DEVICE_INDEX:-$gpu_index}
cpu_list=${CPU_LIST:-32-63,96-127}
ubatch=${UBATCH:-512}
model=/workspace/models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf
prompt_file=/workspace/benchmark-results/new-result/prompts/normal-long-prompt.zh-en.txt
stable_idle_seconds=${STABLE_IDLE_SECONDS:-60}
numa_gate=${NUMA_GATE:-warn}
only_case=${ONLY_CASE:-both}
if [[ "$only_case" != both && "$only_case" != default-prompt && "$only_case" != normal-prompt ]]; then
    printf 'ONLY_CASE must be both, default-prompt, or normal-prompt\n' >&2
    exit 2
fi
if ! [[ "$ubatch" =~ ^[1-9][0-9]*$ ]]; then
    printf 'UBATCH must be a positive integer\n' >&2
    exit 2
fi
if [[ "$result_root" != /* ]]; then
    result_root="$repo_root/$result_root"
fi
if [[ "$result_root" != "$repo_root"/* ]]; then
    printf 'RESULT_ROOT must be inside %s\n' "$repo_root" >&2
    exit 2
fi
mkdir -p "$result_root"
result_rel=${result_root#"$repo_root"/}
container_result_root="/workspace/$result_rel"
if [[ "$numa_gate" != warn && "$numa_gate" != strict ]]; then
    printf 'NUMA_GATE must be warn or strict\n' >&2
    exit 2
fi
declare -A observed_gpu_pids=()
idle_since=0

host_gpu_apps() {
    nvidia-smi -i "$gpu_index" --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null || true
}

wait_for_host_gpu() {
    idle_since=0
    while true; do
        local apps
        apps=$(host_gpu_apps)
        if [[ -n "${apps//[[:space:]]/}" ]]; then
            local pid name memory
            while IFS=, read -r pid name memory; do
                pid=${pid//[[:space:]]/}
                if [[ "$pid" =~ ^[0-9]+$ ]]; then
                    observed_gpu_pids[$pid]=1
                fi
            done <<< "$apps"
        fi
        local -a live_pids=()
        local tracked_pid
        for tracked_pid in "${!observed_gpu_pids[@]}"; do
            if [[ -d "/proc/$tracked_pid" ]]; then
                live_pids+=("$tracked_pid")
            fi
        done
        if [[ -n "${apps//[[:space:]]/}" || ${#live_pids[@]} -gt 0 ]]; then
            idle_since=0
            printf '[wait] GPU %s apps=[%s] tracked_live=[%s]\n' \
                "$gpu_index" "${apps//$'\n'/; }" "${live_pids[*]}" >&2
        else
            local now
            now=$(date +%s)
            if (( idle_since == 0 )); then
                idle_since=$now
            fi
            if (( now - idle_since >= stable_idle_seconds )); then
                return
            fi
        printf '[wait] GPU %s stable idle %ss/%ss\n' \
                "$gpu_index" "$((now - idle_since))" "$stable_idle_seconds" >&2
        fi
        sleep 5
    done
}

warm_model_page_cache() {
    docker exec "$container_name" bash -lc \
        "python3 -c 'import os, sys; fd = os.open(sys.argv[1], os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)' '$model'; \
         read -r before_resident size < <(fincore --bytes --noheadings -o RES,SIZE '$model'); \
         taskset -c '$cpu_list' dd if='$model' of=/dev/null bs=16M status=none; \
         read -r resident size < <(fincore --bytes --noheadings -o RES,SIZE '$model'); \
         if (( resident * 100 < size * 99 )); then exit 43; fi; \
         printf 'before_warm=%s total=%s after_warm=%s\\n' \"\$before_resident\" \"\$size\" \"\$resident\""
}

write_top_metadata() {
    {
        printf 'started=%s\n' "$(date -Is)"
        printf 'container=%s\n' "$container_name"
        printf 'stable_idle_seconds=%s\n' "$stable_idle_seconds"
        printf 'numa_gate=%s\n' "$numa_gate"
        printf 'gpu_index=%s\ncuda_visible_devices=%s\ncpu_list=%s\n' "$gpu_index" "$cuda_visible" "$cpu_list"
        printf 'ubatch=%s\n' "$ubatch"
        printf 'repo=%s\nbranch=%s\ncommit=%s\n' "$repo_root" \
            "$(git -C "$repo_root" branch --show-current)" "$(git -C "$repo_root" rev-parse HEAD)"
        printf 'model=%s\nmodel_sha256=%s\n' "$model" \
            "$(docker exec "$container_name" bash -lc "sha256sum '$model' | awk '{print \$1}'")"
        printf 'prompt_file=%s\nprompt_sha256=%s\n' "$prompt_file" \
            "$(docker exec "$container_name" bash -lc "sha256sum '$prompt_file' | awk '{print \$1}'")"
        printf 'benchmark_binary=/workspace/build-moe/bin/llama-moe-bench\n'
        printf 'benchmark_binary_sha256=%s\n' \
            "$(docker exec "$container_name" bash -lc "sha256sum /workspace/build-moe/bin/llama-moe-bench | awk '{print \$1}'")"
        printf 'host_gpu_apps_at_start\n%s\nhost_gpu_apps_at_start_end\n' "$(host_gpu_apps)"
        nvidia-smi -i "$gpu_index" --query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version \
            --format=csv,noheader
        printf 'git_status_begin\n'
        git -C "$repo_root" status --short
        printf 'git_status_end\n'
    } > "$result_root/run-metadata.txt"
}

run_case() {
    local name=$1
    local case_prompt=${2:-}
    local prompt_shell=
    local -a prompt_args=()
    if [[ -n "$case_prompt" ]]; then
        prompt_args=(--prompt-file "$case_prompt")
        printf -v prompt_shell '%s %q' '--prompt-file' "$case_prompt"
    fi
    local case_dir="$result_root/$name"
    mkdir -p "$case_dir"
    wait_for_host_gpu
    local cache_before
    cache_before=$(warm_model_page_cache)
    wait_for_host_gpu
    local gpu_before
    gpu_before=$(nvidia-smi -i "$gpu_index" --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw \
        --format=csv,noheader)
    {
        printf 'case=%s\nstarted=%s\n' "$name" "$(date -Is)"
        printf 'model=%s\nprompt_mode=%s\n' "$model" "$name"
        printf 'gpu_index=%s\ncuda_visible_devices=%s\ncpu_list=%s\n' "$gpu_index" "$cuda_visible" "$cpu_list"
        printf 'ubatch=%s\n' "$ubatch"
        printf 'page_cache_before=%s\ngpu_before=%s\n' "$cache_before" "$gpu_before"
        printf 'command='; printf '%q ' taskset -c "$cpu_list" env "CUDA_VISIBLE_DEVICES=$cuda_visible" \
            /workspace/build-moe/bin/llama-moe-bench --model "$model" -ngl 99 -c 4096 \
            --pp 1024 --tg 1024 --repeat 1 -ub "$ubatch" --moe-cache-vram-mb 8000 \
            --moe-predictor lru --moe-host-cache off --moe-host-cache-preload none \
            --page-cache-policy natural --moe-profile-csv "$container_result_root/$name/profile.csv" \
            --moe-profile-summary "$container_result_root/$name/summary.txt" \
            --token-trace "$container_result_root/$name/tokens.csv" "${prompt_args[@]}"
        printf '\n'
    } > "$case_dir/metadata.txt"

    local monitor_pid
    ( while true; do
          printf '%s ' "$(date -Is)"
          host_gpu_apps
          sleep 5
      done ) > "$case_dir/gpu-monitor.log" 2>&1 &
    monitor_pid=$!

    set +e
    docker exec "$container_name" bash -lc \
        "set -e; cd /workspace; mkdir -p '$container_result_root/$name'; \
         printf 'elapsed_s,rss_kib,hwm_kib,locked_kib,n0_pages,n1_pages\\n' > '$container_result_root/$name/memory.csv'; \
         set +e; taskset -c '$cpu_list' env CUDA_VISIBLE_DEVICES='$cuda_visible' \
         build-moe/bin/llama-moe-bench --model '$model' -ngl 99 -c 4096 \
         --pp 1024 --tg 1024 --repeat 1 -ub '$ubatch' --moe-cache-vram-mb 8000 \
         --moe-predictor lru --moe-host-cache off --moe-host-cache-preload none \
         --page-cache-policy natural --moe-profile-csv '$container_result_root/$name/profile.csv' \
         --moe-profile-summary '$container_result_root/$name/summary.txt' \
         --token-trace '$container_result_root/$name/tokens.csv' $prompt_shell \
         > '$container_result_root/$name/stdout.log' 2> '$container_result_root/$name/stderr.log' & \
         bench_pid=\$!; started=\$SECONDS; \
         while kill -0 \$bench_pid 2>/dev/null; do \
             rss=\$(awk '/^VmRSS:/ {print \$2}' /proc/\$bench_pid/status 2>/dev/null || true); \
             hwm=\$(awk '/^VmHWM:/ {print \$2}' /proc/\$bench_pid/status 2>/dev/null || true); \
             locked=\$(awk '/^VmLck:/ {print \$2}' /proc/\$bench_pid/status 2>/dev/null || true); \
             nodes=\$(awk '{for (i=1;i<=NF;i++) if (\$i ~ /^N[0-9]+=/) print \$i}' /proc/\$bench_pid/numa_maps 2>/dev/null || true); \
             n0=\$(awk -F= '\$1 == \"N0\" {sum += \$2} END {print sum + 0}' <<< \"\$nodes\"); \
             n1=\$(awk -F= '\$1 == \"N1\" {sum += \$2} END {print sum + 0}' <<< \"\$nodes\"); \
             printf '%s,%s,%s,%s,%s,%s\\n' \$((SECONDS-started)) \"\${rss:-0}\" \"\${hwm:-0}\" \"\${locked:-0}\" \"\$n0\" \"\$n1\" >> '$container_result_root/$name/memory.csv'; \
             sleep 1; \
         done; wait \$bench_pid; rc=\$?; printf 'exit_code=%s\\n' \$rc; exit \$rc" 
    local rc=$?
    set -e
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    local gpu_after
    gpu_after=$(nvidia-smi -i "$gpu_index" --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw \
        --format=csv,noheader)
    {
        printf 'finished=%s\nexit_code=%s\npage_cache_after=%s\ngpu_after=%s\n' \
            "$(date -Is)" "$rc" "$(docker exec "$container_name" bash -lc "fincore --bytes --noheadings -o RES,SIZE '$model'")" "$gpu_after"
        printf 'host_gpu_apps_after\n%s\nhost_gpu_apps_after_end\n' "$(host_gpu_apps)"
    } >> "$case_dir/metadata.txt"
    if (( rc != 0 )); then
        return "$rc"
    fi
    if awk -F, 'NF >= 3 && $2 !~ /llama-moe-bench/ { bad = 1 } END { exit bad ? 0 : 1 }' \
        "$case_dir/gpu-monitor.log"; then
        printf 'foreign GPU compute process observed during case %s\n' "$name" >&2
        return 45
    fi
    if [[ -n "$(host_gpu_apps)" ]]; then
        printf 'GPU %s still has a compute process after case %s\n' "$gpu_index" "$name" >&2
        return 46
    fi
    local peak_nodes n0_peak n1_peak
    peak_nodes=$(awk -F, 'NR > 1 && $2 > max {max=$2; n0=$5; n1=$6} END {print n0+0, n1+0}' "$case_dir/memory.csv")
    read -r n0_peak n1_peak <<< "$peak_nodes"
    local numa_local_pct=0
    if (( n0_peak + n1_peak > 0 )); then
        numa_local_pct=$(awk -v n0="$n0_peak" -v n1="$n1_peak" 'BEGIN { printf "%.3f", 100*n1/(n0+n1) }')
    fi
    {
        printf 'numa_n0_pages=%s\nnuma_n1_pages=%s\nnuma_local_pct=%s\n' \
            "$n0_peak" "$n1_peak" "$numa_local_pct"
    } >> "$case_dir/metadata.txt"
    if (( n0_peak + n1_peak > 0 && n1_peak * 100 < (n0_peak + n1_peak) * 95 )); then
        printf 'NUMA-local pages below 95%%: N0=%s N1=%s (measured %.3f%%)\n' \
            "$n0_peak" "$n1_peak" "$numa_local_pct" >&2
        if [[ "$numa_gate" == strict ]]; then
            return 44
        fi
    fi
    return "$rc"
}

wait_for_host_gpu
write_top_metadata
if [[ "$only_case" == both || "$only_case" == default-prompt ]]; then
    run_case default-prompt
fi
if [[ "$only_case" == both ]]; then
    sleep 30
fi
if [[ "$only_case" == both || "$only_case" == normal-prompt ]]; then
    run_case normal-prompt "$prompt_file"
fi
printf 'completed=%s\n' "$(date -Is)" >> "$result_root/run-metadata.txt"
