#!/usr/bin/env bash
# Resumable formal SERE performance runner. Run on the host, or set
# RUN_IN_CONTAINER=1 when invoking this file from llama-cpp-clean-dev.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IN_CONTAINER=0
if [[ "${RUN_IN_CONTAINER:-0}" == 1 || -f /.dockerenv ]]; then IN_CONTAINER=1; fi
if (( IN_CONTAINER )); then
    REPO_ROOT=${REPO_ROOT:-/workspace}
    CONTAINER_ROOT=${CONTAINER_ROOT:-$REPO_ROOT}
else
    REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
    CONTAINER_ROOT=${CONTAINER_ROOT:-/workspace}
fi

CONTAINER_NAME=${CONTAINER_NAME:-llama-cpp-clean-dev}
RUN_ID=${RUN_ID:-formal-$(date -u +%Y%m%dT%H%M%SZ)}
STORAGE_MODE=${STORAGE_MODE:-ssd-cold}
RUN_ROOT=${RESULT_ROOT:-$REPO_ROOT/benchmark-results/sere/performance/$RUN_ID/$STORAGE_MODE}
if [[ "$RUN_ROOT" != /* ]]; then RUN_ROOT="$REPO_ROOT/$RUN_ROOT"; fi
if (( ! IN_CONTAINER )) && [[ "$RUN_ROOT" != "$REPO_ROOT"/* ]]; then
    die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
    die "RESULT_ROOT must be inside $REPO_ROOT so the container can see it"
fi

MODEL_HOST=${MODEL_HOST:-$REPO_ROOT/models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf}
MODEL=${MODEL:-$CONTAINER_ROOT/models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf}
PROMPT=${PROMPT:-$CONTAINER_ROOT/benchmark-results/new-result/prompts/normal-long-prompt.zh-en.txt}
SIDECAR=${SIDECAR:-$CONTAINER_ROOT/benchmark-results/sere/mechanism-only/synthetic/similarity.sere}
BENCH=${BENCH:-$CONTAINER_ROOT/build-sere-cuda/bin/llama-moe-bench}

GPU_INDEX=${GPU_INDEX:-3}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_INDEX}
LOCAL_NUMA_NODE=${LOCAL_NUMA_NODE:-}
CPU_LIST=${CPU_LIST:-}
NUMA_MIN_LOCAL_PCT=${NUMA_MIN_LOCAL_PCT:-95.0}
NUMA_GATE=${NUMA_GATE:-strict}
STABLE_IDLE_SECONDS=${STABLE_IDLE_SECONDS:-30}
GPU_IDLE_TIMEOUT_SECONDS=${GPU_IDLE_TIMEOUT_SECONDS:-900}
GPU_MONITOR_INTERVAL_SECONDS=${GPU_MONITOR_INTERVAL_SECONDS:-5}
GPU_MEMORY_DELTA_MIB=${GPU_MEMORY_DELTA_MIB:-64}
ALLOW_GPU_PROCESS_PATTERN=${ALLOW_GPU_PROCESS_PATTERN:-llama-moe-bench}

PP=${PP:-1024}
TG=${TG:-1024}
CTX=${CTX:-4096}
RUNS=${RUNS:-3}
PREDICTOR=${PREDICTOR:-lru}
SERE_THRESHOLD=${SERE_THRESHOLD:-0}

case "$STORAGE_MODE" in
    ssd-cold) PAGE_CACHE_POLICY=${PAGE_CACHE_POLICY:-cold}; HOST_CACHE=${HOST_CACHE:-off}; HOST_CACHE_PRELOAD=${HOST_CACHE_PRELOAD:-none} ;;
    page-cache-hot) PAGE_CACHE_POLICY=${PAGE_CACHE_POLICY:-hot}; HOST_CACHE=${HOST_CACHE:-off}; HOST_CACHE_PRELOAD=${HOST_CACHE_PRELOAD:-none} ;;
    pageable-dram) PAGE_CACHE_POLICY=${PAGE_CACHE_POLICY:-cold}; HOST_CACHE=${HOST_CACHE:-pageable}; HOST_CACHE_PRELOAD=${HOST_CACHE_PRELOAD:-all} ;;
    pinned-dram) PAGE_CACHE_POLICY=${PAGE_CACHE_POLICY:-cold}; HOST_CACHE=${HOST_CACHE:-pinned}; HOST_CACHE_PRELOAD=${HOST_CACHE_PRELOAD:-all} ;;
    *) printf 'invalid STORAGE_MODE: %s\n' "$STORAGE_MODE" >&2; exit 2 ;;
esac

TARGET_VRAMS=${RUN_TARGET_VRAMS:-6000,8000}
TARGET_UBATCHES=${RUN_TARGET_UBATCHES:-512,1024}
TARGET_CASES=${RUN_TARGET_CASES:-lru,paper-s4,miss-s4}
TARGET_RUNS=${RUN_TARGET_RUNS:-}
FORCE=${FORCE:-0}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "invalid RUN_ID"
[[ "$PP" =~ ^[1-9][0-9]*$ && "$TG" =~ ^[1-9][0-9]*$ && "$CTX" =~ ^[1-9][0-9]*$ ]] || die "PP/TG/CTX must be positive integers"
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || die "RUNS must be a positive integer"
[[ "$NUMA_GATE" == strict || "$NUMA_GATE" == warn ]] || die "NUMA_GATE must be strict or warn"
[[ "$NUMA_MIN_LOCAL_PCT" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid NUMA_MIN_LOCAL_PCT"
[[ "$PAGE_CACHE_POLICY" == cold || "$PAGE_CACHE_POLICY" == hot || "$PAGE_CACHE_POLICY" == natural ]] 2>/dev/null || true
[[ "$PREDICTOR" == lru ]] || die "formal SERE runner requires PREDICTOR=lru"

check_csv() {
    local list=$1 name=$2 item
    [[ -n "$list" ]] || die "$name is empty"
    IFS=',' read -r -a values <<< "$list"
    for item in "${values[@]}"; do
        [[ "$item" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid $name item: $item"
    done
}
check_csv "$TARGET_VRAMS" RUN_TARGET_VRAMS
check_csv "$TARGET_UBATCHES" RUN_TARGET_UBATCHES
check_csv "$TARGET_CASES" RUN_TARGET_CASES
[[ -z "$TARGET_RUNS" ]] || check_csv "$TARGET_RUNS" RUN_TARGET_RUNS

contains() { [[ ",$1," == *",$2,"* ]]; }
for command in awk date grep mkdir mv sed sha256sum stat sleep taskset; do
    command -v "$command" >/dev/null 2>&1 || die "missing command: $command"
done
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"
(( IN_CONTAINER )) || command -v docker >/dev/null 2>&1 || die "docker is required on host"
mkdir -p "$RUN_ROOT"

nvidia_smi() { env -u CUDA_VISIBLE_DEVICES nvidia-smi "$@"; }
gpu_apps() {
    nvidia_smi -i "$GPU_INDEX" --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null || true
}
gpu_mem() {
    nvidia_smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null |
        awk 'NR == 1 {gsub(/[[:space:]]/, ""); print $1; exit}'
}
foreign_apps() {
    local apps=$1 allowed_pid=${2:-} pid name mem
    while IFS=',' read -r pid name mem; do
        pid=$(printf '%s' "$pid" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        name=$(printf '%s' "$name" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [[ -z "$name" ]] && continue
        # Match the exact bench PID. During CUDA teardown nvidia-smi can
        # briefly report that same PID as "[No data]"; it is still ours.
        [[ -n "$allowed_pid" && "$pid" == "$allowed_pid" ]] && continue
        return 0
    done <<< "$apps"
    return 1
}
write_gpu_snapshot() {
    local label=$1
    {
        printf 'recorded_utc=%s\ngpu_index=%s\ncuda_visible_devices=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GPU_INDEX" "$CUDA_VISIBLE_DEVICES"
        printf 'selected_gpu\n'
        nvidia_smi -i "$GPU_INDEX" --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true
        printf 'compute_processes\n'; gpu_apps
    } > "$RUN_ROOT/gpu-snapshot-$label.txt" 2>&1
}
wait_gpu_idle() {
    local why=$1 idle_since=0 deadline=$((SECONDS + GPU_IDLE_TIMEOUT_SECONDS)) apps now
    while (( SECONDS < deadline )); do
        apps=$(gpu_apps)
        if [[ -n "${apps//[[:space:]]/}" ]]; then
            idle_since=0
            printf '[wait] GPU %s busy (%s): %s\n' "$GPU_INDEX" "$why" "${apps//$'\n'/; }" >&2
        else
            now=$SECONDS
            (( idle_since == 0 )) && idle_since=$now
            if (( now - idle_since >= STABLE_IDLE_SECONDS )); then return 0; fi
            printf '[wait] GPU %s idle %ss/%ss (%s)\n' "$GPU_INDEX" "$((now-idle_since))" "$STABLE_IDLE_SECONDS" "$why" >&2
        fi
        sleep 5
    done
    printf 'GPU %s did not become idle within %ss (%s)\n' "$GPU_INDEX" "$GPU_IDLE_TIMEOUT_SECONDS" "$why" >&2
    return 1
}

if ! nvidia_smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader >/dev/null 2>&1; then
    die "GPU_INDEX is not visible: $GPU_INDEX"
fi
if [[ -z "$LOCAL_NUMA_NODE" ]]; then
    bus=$(nvidia_smi -i "$GPU_INDEX" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null |
        tr -d '[:space:]' | tr '[:upper:]' '[:lower:]' || true)
    short=${bus#00000000:}
    LOCAL_NUMA_NODE=$(cat "/sys/bus/pci/devices/0000:${short}/numa_node" 2>/dev/null || true)
fi
if [[ ! "$LOCAL_NUMA_NODE" =~ ^[0-9]+$ && "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
    if (( GPU_INDEX <= 1 )); then LOCAL_NUMA_NODE=0; else LOCAL_NUMA_NODE=1; fi
fi
[[ "$LOCAL_NUMA_NODE" =~ ^[0-9]+$ ]] || die "cannot determine NUMA node; set LOCAL_NUMA_NODE"
if [[ -z "$CPU_LIST" ]]; then CPU_LIST=$(cat "/sys/devices/system/node/node${LOCAL_NUMA_NODE}/cpulist" 2>/dev/null || true); fi
[[ -n "$CPU_LIST" ]] || die "cannot determine CPU_LIST; set CPU_LIST"

container_test() {
    if (( IN_CONTAINER )); then test -f "$1"; else docker exec "$CONTAINER_NAME" test -f "$1"; fi
}
container_sha() {
    if (( IN_CONTAINER )); then sha256sum "$1" | awk '{print $1}'; else docker exec "$CONTAINER_NAME" sha256sum "$1" | awk '{print $1}'; fi
}
container_size() {
    if (( IN_CONTAINER )); then stat -c %s "$1"; else docker exec "$CONTAINER_NAME" stat -c %s "$1"; fi
}
container_test "$BENCH" || die "missing bench: $BENCH"
container_test "$MODEL" || die "missing model: $MODEL"
container_test "$PROMPT" || die "missing prompt: $PROMPT"
container_test "$SIDECAR" || die "missing SERE sidecar: $SIDECAR"
MODEL_SIZE=$(container_size "$MODEL")
MODEL_SHA256=${MODEL_SHA256:-$(container_sha "$MODEL")}
PROMPT_SHA256=${PROMPT_SHA256:-$(container_sha "$PROMPT")}
SIDECAR_SHA256=${SIDECAR_SHA256:-$(container_sha "$SIDECAR")}
BENCH_SHA256=${BENCH_SHA256:-$(container_sha "$BENCH")}

if [[ ! -s "$RUN_ROOT/run-metadata.txt" ]]; then
    {
        printf 'started_utc=%s\nrun_id=%s\nstorage_mode=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_ID" "$STORAGE_MODE"
        printf 'repo=%s\nbranch=%s\ncommit=%s\ncontainer=%s\ninside_container=%s\n' "$REPO_ROOT" \
            "$(git -C "$REPO_ROOT" branch --show-current 2>/dev/null || echo unavailable)" \
            "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unavailable)" "$CONTAINER_NAME" "$IN_CONTAINER"
        printf 'model=%s\nmodel_size=%s\nmodel_sha256=%s\nprompt=%s\nprompt_sha256=%s\nsidecar=%s\nsidecar_sha256=%s\n' \
            "$MODEL" "$MODEL_SIZE" "$MODEL_SHA256" "$PROMPT" "$PROMPT_SHA256" "$SIDECAR" "$SIDECAR_SHA256"
        printf 'benchmark=%s\nbenchmark_sha256=%s\n' "$BENCH" "$BENCH_SHA256"
        printf 'gpu_index=%s\ncuda_visible_devices=%s\nnuma_local_node=%s\ncpu_list=%s\nnuma_gate=%s\nnuma_min_local_pct=%s\n' \
            "$GPU_INDEX" "$CUDA_VISIBLE_DEVICES" "$LOCAL_NUMA_NODE" "$CPU_LIST" "$NUMA_GATE" "$NUMA_MIN_LOCAL_PCT"
        printf 'page_cache_policy=%s\nhost_cache=%s\nhost_cache_preload=%s\npp=%s\ntg=%s\nctx=%s\nruns=%s\npredictor=%s\n' \
            "$PAGE_CACHE_POLICY" "$HOST_CACHE" "$HOST_CACHE_PRELOAD" "$PP" "$TG" "$CTX" "$RUNS" "$PREDICTOR"
        printf 'target_vrams=%s\ntarget_ubatches=%s\ntarget_cases=%s\ntarget_runs=%s\n' "$TARGET_VRAMS" "$TARGET_UBATCHES" "$TARGET_CASES" "${TARGET_RUNS:-all}"
        printf 'git_status_begin\n'; git -C "$REPO_ROOT" status --short 2>/dev/null || true; printf 'git_status_end\n'
    } > "$RUN_ROOT/run-metadata.txt"
else
    printf 'resume_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_ROOT/run-metadata.txt"
fi
write_gpu_snapshot start
printf 'RUNNING\n' > "$RUN_ROOT/status"

RESULTS_TSV="$RUN_ROOT/results.tsv"
if [[ ! -s "$RESULTS_TSV" ]]; then
    printf 'run\tvram_mb\tubatch\tcase\tstatus\tttft_ms\ttpot_ms\ttotal_ms\tdecode_tok_s\tdecode_hit_pct\tdecode_ssd_gib\tssd_reads\tnuma_local_pct\tpage_cache_after_prepare_pct\tprompt_token_hash\tcase_dir\n' > "$RESULTS_TSV"
fi

# This body is sent to the container without host-side shell interpolation.
CONTAINER_BODY=$(cat <<'CONTAINER_BODY'
set -u
bench=$1; model=$2; prompt=$3; sidecar=$4; out=$5; case_name=$6
pp=$7; tg=$8; ub=$9; vram=${10}; page=${11}; host=${12}; preload=${13}; threshold=${14}; ctx=${15}; cpus=${16}; cuda=${17}
mkdir -p "$out"
printf 'elapsed_s,rss_kib,hwm_kib,locked_kib,n0_pages,n1_pages\n' > "$out/memory.csv"
sample() {
    p=$1
    rss=$(awk '/^VmRSS:/ {print $2; exit}' "/proc/$p/status" 2>/dev/null || true)
    hwm=$(awk '/^VmHWM:/ {print $2; exit}' "/proc/$p/status" 2>/dev/null || true)
    lck=$(awk '/^VmLck:/ {print $2; exit}' "/proc/$p/status" 2>/dev/null || true)
    nodes=$(awk '{for(i=1;i<=NF;i++) if($i ~ /^N[0-9]+=/) print $i}' "/proc/$p/numa_maps" 2>/dev/null || true)
    n0=$(awk -F= '$1=="N0"{s+=$2} END{print s+0}' <<< "$nodes")
    n1=$(awk -F= '$1=="N1"{s+=$2} END{print s+0}' <<< "$nodes")
    printf '%s,%s,%s,%s,%s,%s\n' "$((SECONDS-started))" "${rss:-0}" "${hwm:-0}" "${lck:-0}" "$n0" "$n1" >> "$out/memory.csv"
}
args=(--model "$model" --prompt-file "$prompt" -ngl 99 -c "$ctx" --pp "$pp" --tg "$tg" --repeat 1 -ub "$ub" --moe-cache-vram-mb "$vram" --moe-predictor lru --moe-host-cache "$host" --moe-host-cache-preload "$preload" --page-cache-policy "$page" --moe-reset-cache-between-repeats --moe-profile-csv "$out/profile.csv" --moe-profile-summary "$out/summary.txt" --token-trace "$out/tokens.csv")
case "$case_name" in
    lru) ;;
    paper-s4) args+=(--moe-sere-path "$sidecar" --moe-sere-top-k 4 --moe-sere-threshold "$threshold" --moe-sere-policy paper) ;;
    miss-s4) args+=(--moe-sere-path "$sidecar" --moe-sere-top-k 4 --moe-sere-threshold "$threshold" --moe-sere-policy miss) ;;
    *) printf 'unknown case: %s\n' "$case_name" >&2; exit 2 ;;
esac
started=$SECONDS
set +e
taskset -c "$cpus" env CUDA_VISIBLE_DEVICES="$cuda" "$bench" "${args[@]}" > "$out/stdout.log" 2> "$out/stderr.log" &
pid=$!
printf '%s\n' "$pid" > "$out/bench.pid"
set -e
while kill -0 "$pid" 2>/dev/null; do sample "$pid"; sleep 1; done
wait "$pid"
rc=$?
printf '%s,%s,%s,%s,%s,%s\n' "$((SECONDS-started))" 0 0 0 0 0 >> "$out/memory.csv"
printf 'container_exit_code=%s\n' "$rc" > "$out/container.status"
exit "$rc"
CONTAINER_BODY
)

case_complete() {
    local d=$1
    [[ "$FORCE" != 1 ]] || return 1
    [[ -s "$d/summary.txt" && -s "$d/profile.csv" && -s "$d/tokens.csv" && -s "$d/stdout.log" && -s "$d/stderr.log" && -s "$d/metadata.txt" && -s "$d/memory.csv" ]] || return 1
    grep -q '^status=complete$' "$d/metadata.txt" || return 1
    grep -Eq '^validation=(pass|warn)$' "$d/metadata.txt" || return 1
    grep -q '^exit_code=0$' "$d/metadata.txt" || return 1
}
archive_case() {
    local d=$1
    [[ -e "$d" ]] || return 0
    local a="${d}.incomplete-$(date -u +%Y%m%dT%H%M%SZ)" n=0
    while [[ -e "$a" ]]; do n=$((n+1)); a="${d}.incomplete-$(date -u +%Y%m%dT%H%M%SZ)-$n"; done
    mv "$d" "$a"
    printf '[resume] archived %s -> %s\n' "$d" "$a"
}
summary_value() { awk -v p="$2" '$0 ~ p {print $2; exit}' "$1"; }
summary_hit() { awk '/^cache hit rate \(decode/ {x=$NF; sub(/%$/, "", x); print x; exit}' "$1"; }
summary_ssd() { awk '/^SSD bytes read \(decode\):/ {print $5; exit}' "$1"; }
summary_reads() { awk '/^SSD reads:/ {print $3; exit}' "$1"; }
summary_after_prepare() { awk '/^Page cache resident pct:/ {for(i=1;i<=NF;i++) if($i ~ /^after_prepare=/){split($i,a,"="); print a[2]; exit}}' "$1"; }
summary_hash() { awk '/^prompt token hash:/ {print $4; exit}' "$1"; }
numa_values() {
    local f=$1 node=$2
    awk -F, -v node="$node" '
        NR==1 {for(i=1;i<=NF;i++){if($i=="rss_kib")r=i;if($i=="n"node"_pages")l=i;if($i~/^n[0-9]+_pages$/)c[++n]=i}next}
        {t=0;for(i=1;i<=n;i++)t+=$(c[i]);if($(r)+0>m){m=$(r)+0;lp=$(l)+0;tp=t}}
        END{if(tp<=0)print "0 0 0 0";else printf "%.0f %.0f %.3f %.0f\n",m,lp,100*lp/tp,tp}' "$f"
}
validate_case() {
    local d=$1 name=$2 ub=$3 err=() f s after rows tokens max_rss local_pages pct total numa_result
    for f in summary.txt profile.csv tokens.csv stdout.log stderr.log metadata.txt memory.csv; do [[ -s "$d/$f" ]] || err+=("missing $f"); done
    ((${#err[@]})) && { printf '%s\n' "${err[@]}" >&2; return 1; }
    s="$d/summary.txt"
    grep -Eq "^n_prompt: $PP  n_gen: $TG  repeats: 1$" "$s" || err+=("prompt/generation/repeat mismatch")
    grep -q "^ubatch: requested=$ub effective=$ub " "$s" || err+=("ubatch mismatch")
    grep -q '^Page cache sample valid: before=1 after_prepare=1 after_model_load=1 after_prefill=1 after_decode=1$' "$s" || err+=("page-cache samples invalid")
    after=$(summary_after_prepare "$s")
    [[ "$after" =~ ^[0-9]+([.][0-9]+)?$ ]] || err+=("page-cache after_prepare missing")
    if [[ "$after" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        case "$PAGE_CACHE_POLICY" in
            cold) awk -v x="$after" 'BEGIN{exit !(x<=1.0)}' || err+=("cold page-cache after_prepare >1%") ;;
            hot) awk -v x="$after" 'BEGIN{exit !(x>=99.0)}' || err+=("hot page-cache after_prepare <99%") ;;
        esac
    fi
    grep -q "^Host cache: mode=$HOST_CACHE preload=$HOST_CACHE_PRELOAD " "$s" || err+=("host-cache mismatch")
    grep -q '^Host cache verification: .*failures=0$' "$s" || err+=("host-cache verification failed")
    rows=$(awk -F, 'NR==1{n=NF;next} NF!=n{bad=1} $1=="layer"{rows++} END{if(n!=62||bad)print -1;else print rows+0}' "$d/profile.csv")
    expected_rows=$((40 * ((PP + ub - 1) / ub) + 40 * TG))
    [[ "$rows" == "$expected_rows" ]] || err+=("profile layer rows=$rows expected=$expected_rows")
    tokens=$(awk -F, 'NR>1{n++} END{print n+0}' "$d/tokens.csv")
    [[ "$tokens" == $((TG + 1)) ]] || err+=("token rows=$tokens expected=$((TG+1))")
    grep -q '^container_exit_code=0$' "$d/container.status" 2>/dev/null || err+=("bench exit code is nonzero")
    case "$name" in
        lru) ! grep -q '^SERE:' "$s" || err+=("LRU unexpectedly has SERE") ;;
        paper-s4) grep -Eq '^SERE: top-k=4 .*policy=paper[[:space:]]+mode=active ' "$s" || err+=("paper S=4 summary mismatch") ;;
        miss-s4) grep -Eq '^SERE: top-k=4 .*policy=miss[[:space:]]+mode=active ' "$s" || err+=("miss S=4 summary mismatch") ;;
    esac
    read -r max_rss local_pages pct total <<< "$(numa_values "$d/memory.csv" "$LOCAL_NUMA_NODE")"
    printf 'numa_peak_rss_kib=%s\nnuma_local_node=%s\nnuma_local_pages=%s\nnuma_total_pages=%s\nnuma_local_pct=%s\n' "$max_rss" "$LOCAL_NUMA_NODE" "$local_pages" "$total" "$pct" >> "$d/metadata.txt"
    numa_result=pass
    if (( total <= 0 )); then
        err+=("NUMA sample has no pages"); numa_result=fail
    elif awk -v x="$pct" -v min="$NUMA_MIN_LOCAL_PCT" 'BEGIN{exit !(x<min)}'; then
        if [[ "$NUMA_GATE" == strict ]]; then
            err+=("NUMA local page ratio ${pct}% < ${NUMA_MIN_LOCAL_PCT}%"); numa_result=fail
        else
            numa_result=warn
            printf '[warn] NUMA local page ratio %s%% < %s%% for %s\n' "$pct" "$NUMA_MIN_LOCAL_PCT" "$d" >&2
        fi
    fi
    printf 'numa_gate_result=%s\n' "$numa_result" >> "$d/metadata.txt"
    ((${#err[@]})) && { printf '%s\n' "${err[@]}" >&2; return 1; }
    printf 'validation=%s\n' "$numa_result" >> "$d/metadata.txt"
    return 0
}
append_result() {
    local run=$1 vram=$2 ub=$3 name=$4 d=$5 status=$6 s="$d/summary.txt" key
    printf -v key '%s\t%s\t%s\t%s\t' "$run" "$vram" "$ub" "$name"
    grep -Fq "$key" "$RESULTS_TSV" && return 0
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$run" "$vram" "$ub" "$name" "$status" "$(summary_value "$s" '^TTFT:' || true)" "$(summary_value "$s" '^TPOT:' || true)" \
        "$(summary_value "$s" '^total:' || true)" "$(awk '/^decode[[:space:]]/{print $NF;exit}' "$s" || true)" "$(summary_hit "$s" || true)" "$(summary_ssd "$s" || true)" "$(summary_reads "$s" || true)" \
        "$(awk -F= '/^numa_local_pct=/{print $2;exit}' "$d/metadata.txt" || true)" "$(summary_after_prepare "$s" || true)" "$(summary_hash "$s" || true)" "$d" >> "$RESULTS_TSV"
}

run_case() {
    local run=$1 vram=$2 ub=$3 name=$4
    local d="$RUN_ROOT/vram-$vram/ub$ub/$name/run-$run" d_container
    if (( IN_CONTAINER )); then
        d_container=$d
    else
        [[ "$d" == "$REPO_ROOT"/* ]] || die "case output is not visible inside the container: $d"
        d_container="$CONTAINER_ROOT/${d#"$REPO_ROOT"/}"
    fi
    local rc pid apps gpu_before gpu_after before after delta violation=0 own_gpu_pid=
    if case_complete "$d"; then
        printf '[skip complete] %s\n' "$d"
        append_result "$run" "$vram" "$ub" "$name" "$d" complete
        return 0
    fi
    archive_case "$d"
    mkdir -p "$d"
    wait_gpu_idle "before run-$run $name vram=$vram ub=$ub"
    gpu_before=$(nvidia_smi -i "$GPU_INDEX" --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,power.draw --format=csv,noheader 2>&1 || true)
    before=$(gpu_mem || echo 0)
    {
        printf 'status=running\nrun=%s\nvram_cache_mb=%s\nubatch=%s\ncase=%s\n' "$run" "$vram" "$ub" "$name"
        printf 'started_utc=%s\nmodel=%s\nprompt=%s\nsidecar=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODEL" "$PROMPT" "$SIDECAR"
        printf 'gpu_index=%s\ncuda_visible_devices=%s\nnuma_local_node=%s\ncpu_list=%s\n' "$GPU_INDEX" "$CUDA_VISIBLE_DEVICES" "$LOCAL_NUMA_NODE" "$CPU_LIST"
        printf 'pp=%s\ntg=%s\nctx=%s\nrepeat=1\npage_cache_policy=%s\nhost_cache=%s\nhost_cache_preload=%s\n' "$PP" "$TG" "$CTX" "$PAGE_CACHE_POLICY" "$HOST_CACHE" "$HOST_CACHE_PRELOAD"
        printf 'bench=%s\npredictor=%s\nsere_threshold=%s\ngpu_before=%s\n' "$BENCH" "$PREDICTOR" "$SERE_THRESHOLD" "$gpu_before"
        printf 'no_logits_bin=1\n'
    } > "$d/metadata.txt"
    : > "$d/gpu-monitor.log"

    set +e
    if (( IN_CONTAINER )); then
        printf '%s\n' "$CONTAINER_BODY" | bash -s -- "$BENCH" "$MODEL" "$PROMPT" "$SIDECAR" "$d" "$name" "$PP" "$TG" "$ub" "$vram" "$PAGE_CACHE_POLICY" "$HOST_CACHE" "$HOST_CACHE_PRELOAD" "$SERE_THRESHOLD" "$CTX" "$CPU_LIST" "$CUDA_VISIBLE_DEVICES" &
    else
        printf '%s\n' "$CONTAINER_BODY" | docker exec -i -e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$CONTAINER_NAME" bash -s -- "$BENCH" "$MODEL" "$PROMPT" "$SIDECAR" "$d_container" "$name" "$PP" "$TG" "$ub" "$vram" "$PAGE_CACHE_POLICY" "$HOST_CACHE" "$HOST_CACHE_PRELOAD" "$SERE_THRESHOLD" "$CTX" "$CPU_LIST" "$CUDA_VISIBLE_DEVICES" &
    fi
    pid=$!
    set -e
    ACTIVE_PID=$pid
    while kill -0 "$pid" 2>/dev/null; do
        # The wrapper writes this marker immediately after its bench wait
        # returns. Stop monitoring before nvidia-smi can expose the transient
        # [No data] row for that completed CUDA context.
        [[ -s "$d/container.status" ]] && break
        {
            printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            nvidia_smi -i "$GPU_INDEX" --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,power.draw --format=csv,noheader 2>&1 || true
            printf 'compute_processes\n'; gpu_apps
        } >> "$d/gpu-monitor.log"
        apps=$(gpu_apps)
        if [[ -s "$d/bench.pid" ]]; then
            own_gpu_pid=$(sed -n '1p' "$d/bench.pid" | tr -d '[:space:]')
        fi
        if foreign_apps "$apps" "$own_gpu_pid"; then
            violation=1
            printf 'foreign_gpu_process=%s\n' "${apps//$'\n'/; }" >> "$d/metadata.txt"
            kill "$pid" 2>/dev/null || true
            break
        fi
        sleep "$GPU_MONITOR_INTERVAL_SECONDS"
    done
    if wait "$pid"; then rc=0; else rc=$?; fi
    ACTIVE_PID=
    after=$(gpu_mem || echo 0)
    gpu_after=$(nvidia_smi -i "$GPU_INDEX" --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,power.draw --format=csv,noheader 2>&1 || true)
    printf 'finished_utc=%s\nexit_code=%s\ngpu_after=%s\ngpu_used_before_mib=%s\ngpu_used_after_mib=%s\ngpu_monitor_violation=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$gpu_after" "${before:-0}" "${after:-0}" "$violation" >> "$d/metadata.txt"
    if (( rc != 0 || violation )); then
        printf 'status=failed\nvalidation=fail\n' >> "$d/metadata.txt"
        [[ -s "$d/summary.txt" ]] && append_result "$run" "$vram" "$ub" "$name" "$d" failed
        wait_gpu_idle "after failed case" || true
        return 1
    fi
    if [[ "$before" =~ ^[0-9]+$ && "$after" =~ ^[0-9]+$ ]]; then
        delta=$((after-before)); ((delta<0)) && delta=$((-delta))
        if (( delta > GPU_MEMORY_DELTA_MIB )); then
            printf 'gpu_memory_gate=fail\nstatus=failed\nvalidation=fail\n' >> "$d/metadata.txt"
            append_result "$run" "$vram" "$ub" "$name" "$d" failed
            return 1
        fi
    fi
    printf 'gpu_memory_gate=pass\n' >> "$d/metadata.txt"
    if ! validate_case "$d" "$name" "$ub"; then
        printf 'validation=fail\nstatus=failed\n' >> "$d/metadata.txt"
        append_result "$run" "$vram" "$ub" "$name" "$d" invalid
        wait_gpu_idle "after invalid case" || true
        return 1
    fi
    printf 'exit_code=0\nstatus=complete\n' >> "$d/metadata.txt"
    append_result "$run" "$vram" "$ub" "$name" "$d" complete
    wait_gpu_idle "after run-$run $name" || return 1
    printf '[done] %s\n' "$d"
}

ACTIVE_PID=
on_exit() {
    rc=$?
    if [[ -n "${ACTIVE_PID:-}" ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
        kill "$ACTIVE_PID" 2>/dev/null || true
        wait "$ACTIVE_PID" 2>/dev/null || true
    fi
    write_gpu_snapshot end || true
    if ((rc==0)); then printf 'COMPLETED\n' > "$RUN_ROOT/status"; else printf 'FAILED\n' > "$RUN_ROOT/status"; fi
    trap - EXIT
    exit "$rc"
}
trap on_exit EXIT

overall_fail=0
IFS=',' read -r -a vrams <<< "$TARGET_VRAMS"
IFS=',' read -r -a ubs <<< "$TARGET_UBATCHES"
for run in $(seq 1 "$RUNS"); do
    [[ -z "$TARGET_RUNS" ]] || contains "$TARGET_RUNS" "$run" || continue
    if ((run%2)); then order=(lru paper-s4 miss-s4); else order=(miss-s4 paper-s4 lru); fi
    for vram in "${vrams[@]}"; do
        for ub in "${ubs[@]}"; do
            for name in "${order[@]}"; do
                contains "$TARGET_CASES" "$name" || continue
                if ! run_case "$run" "$vram" "$ub" "$name"; then overall_fail=1; fi
            done
        done
    done
done
exit "$overall_fail"
