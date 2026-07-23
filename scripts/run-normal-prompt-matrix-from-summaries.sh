#!/usr/bin/env bash
set -euo pipefail

ROOT="${LLAMA_CPP_CLEAN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

OLD_ROOT="${OLD_ROOT:-$ROOT/benchmark-results/qwen3.6-35b-a3b/combined-axis-pp1024-tg1024}"
OUT_ROOT="${OUT_ROOT:-$ROOT/benchmark-results/new-result}"
MODEL="${MODEL:-$ROOT/models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf}"
BIN="${BIN:-$ROOT/build-moe/bin/llama-moe-bench}"
PROMPT_FILE="${PROMPT_FILE:-$OUT_ROOT/prompts/normal-long-prompt.zh-en.txt}"

FORCE="${FORCE:-1}"
CLEAR_OUT="${CLEAR_OUT:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EXIT_GRACE_SECONDS="${EXIT_GRACE_SECONDS:-5}"

export CUDA_VISIBLE_DEVICES
export LD_LIBRARY_PATH="$ROOT/build-moe/bin:${LD_LIBRARY_PATH:-}"

if [[ ! -d "$OLD_ROOT" ]]; then
    echo "missing old result root: $OLD_ROOT" >&2
    exit 1
fi

if [[ ! -x "$BIN" ]]; then
    echo "missing executable: $BIN" >&2
    echo "build it first: cmake --build build-moe --target llama-moe-bench -j\$(nproc)" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "missing model: $MODEL" >&2
    exit 1
fi

if [[ "$CLEAR_OUT" == "1" && -d "$OUT_ROOT" ]]; then
    rm -rf "$OUT_ROOT"
fi

mkdir -p "$(dirname "$PROMPT_FILE")"

if [[ ! -s "$PROMPT_FILE" ]]; then
    python3 - "$PROMPT_FILE" <<'PY'
from pathlib import Path
import sys

out = Path(sys.argv[1])

base = [
    "在一次系统设计评审中，团队需要比较三种缓存策略：固定驻留、最近最少使用替换，以及基于历史路由的预测加载。评审的重点不是单次请求的最好成绩，而是长时间运行时的延迟稳定性、带宽利用率和显存峰值。",
    "The benchmark request describes a production assistant that reads a technical incident report, extracts the timeline, identifies the root cause, and proposes a rollback plan with measurable verification steps.",
    "输入文本包含多个自然段、编号列表、中文说明和英文术语。这样的 prompt 更接近真实 workload，比简单重复 Hello 更容易触发多样化的 MoE 路由分布。",
    "A realistic prompt should include context, constraints, examples, and a concrete task. It should not be random tokens, because random tokens distort tokenization, routing locality, and cache behavior.",
    "第一部分描述背景：某个推理服务部署在单张 GPU 上，专家权重保存在 SSD 中，运行时根据 top-k 路由结果把缺失专家搬运到显存缓存和共享 scratch 区域。",
    "第二部分给出约束：显存预算固定，prefill 输入较长，decode 输出连续生成。测试者希望同时观察 TTFT、TPOT、SSD read、H2D copy、GPU compute 和 cache hit rate。",
    "第三部分要求模型输出结构化分析，包括现象、证据、假设、实验设计、风险和下一步优化方向。回答需要避免空泛判断，并尽量引用可观测指标。",
    "For cache evaluation, the prompt asks the model to compare warm-cache and cold-cache behavior, then explain why a larger resident set improves decode more than first-token prefill.",
    "系统还需要讨论 ubatch 的影响：较大的 ubatch 可以提高 prefill 吞吐，但会放大 compute buffer 和中间张量的显存占用；较小的 ubatch 则更稳但吞吐较低。",
    "A second example asks the assistant to review a CUDA kernel optimization. The assistant must reason about memory coalescing, stream synchronization, graph capture, and quantized matrix multiplication.",
    "报告中包含真实工程语言，例如 configuration drift, memory pressure, page cache effects, pinned host memory, asynchronous copy, and scheduler reserve. 这些词汇可以让 token 分布更接近实际技术文档。",
    "最后，用户要求给出可执行命令、结果目录、命名规范和复现实验的方法。输出应该简洁，但每个结论都要能被日志或 summary 文件验证。",
]

parts = []
for i in range(220):
    p = base[i % len(base)]
    parts.append(
        f"段落 {i + 1}: {p} "
        f"本轮实验编号为 {i + 1}，需要记录输入长度、输出长度、显存预算、ubatch、预测策略和运行时间。"
    )

out.write_text("\n".join(parts) + "\n", encoding="utf-8")
PY
fi

PROMPT_TEXT="$(cat "$PROMPT_FILE")"

mapfile -t SUMMARIES < <(find "$OLD_ROOT" -mindepth 2 -maxdepth 2 -type f -name '*summary.txt' | sort)
if [[ "${#SUMMARIES[@]}" -eq 0 ]]; then
    echo "no summary files found under $OLD_ROOT" >&2
    exit 1
fi

CASE_TSV="$OUT_ROOT/cases.tsv"
mkdir -p "$OUT_ROOT"
{
    echo -e "rel_dir\tprefix\tpp\ttg\trepeat\tcache_mb\tpredictor\tub"
    for summary in "${SUMMARIES[@]}"; do
        rel="${summary#"$OLD_ROOT"/}"
        rel_dir="$(dirname "$rel")"
        prefix="$(basename "$summary" -summary.txt)"

        pp="$(awk '/^n_prompt:/ { print $2; exit }' "$summary")"
        tg="$(awk '/^n_prompt:/ { print $4; exit }' "$summary")"
        repeat="$(awk '/^n_prompt:/ { print $6; exit }' "$summary")"
        predictor="$(awk '/^predictor:/ { print $2; exit }' "$summary")"
        cache_mb="$(awk '/^predictor:/ { print $4; exit }' "$summary")"
        ub="$(awk '/^ubatch:/ { sub(/^requested=/, "", $2); print $2; exit }' "$summary")"

        if [[ -z "$pp" || -z "$tg" || -z "$repeat" || -z "$predictor" || -z "$cache_mb" || -z "$ub" ]]; then
            echo "failed to parse summary: $summary" >&2
            exit 1
        fi

        echo -e "$rel_dir\t$prefix\t$pp\t$tg\t$repeat\t$cache_mb\t$predictor\t$ub"
    done
} > "$CASE_TSV"

echo "root       : $ROOT"
echo "old_root   : $OLD_ROOT"
echo "out_root   : $OUT_ROOT"
echo "model      : $MODEL"
echo "prompt     : $PROMPT_FILE ($(wc -c < "$PROMPT_FILE") bytes)"
echo "cases      : ${#SUMMARIES[@]}"
echo "cuda device: $CUDA_VISIBLE_DEVICES"
echo "force rerun: $FORCE"
echo "clear out  : $CLEAR_OUT"
echo

run_case() {
    local rel_dir="$1"
    local prefix="$2"
    local pp="$3"
    local tg="$4"
    local repeat="$5"
    local cache_mb="$6"
    local predictor="$7"
    local ub="$8"

    local out_dir="$OUT_ROOT/$rel_dir"
    local csv="$out_dir/${prefix}-profile.csv"
    local summary="$out_dir/${prefix}-summary.txt"
    local log="$out_dir/${prefix}-run.log"

    mkdir -p "$out_dir"

    if [[ "$FORCE" != "1" && -s "$summary" ]] && grep -q "profile rows:" "$summary"; then
        echo "===== skipping completed $rel_dir/$prefix ====="
        echo
        return 0
    fi

    if [[ "$FORCE" == "1" ]]; then
        rm -f "$csv" "$summary" "$log"
    fi

    local ctx=$((pp + tg + 16))
    if (( ctx < 4096 )); then
        ctx=4096
    fi

    echo "===== running $rel_dir/$prefix with normal prompt ====="
    echo "pp=$pp tg=$tg ctx=$ctx repeat=$repeat cache_mb=$cache_mb predictor=$predictor ub=$ub"
    echo "log: $log"

    "$BIN" \
        --model "$MODEL" \
        --pp "$pp" \
        --tg "$tg" \
        -c "$ctx" \
        --repeat "$repeat" \
        --moe-cache-vram-mb "$cache_mb" \
        --moe-predictor "$predictor" \
        -ub "$ub" \
        -p "$PROMPT_TEXT" \
        --moe-profile-csv "$csv" \
        --moe-profile-summary "$summary" \
        > "$log" 2>&1 &

    local pid=$!
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

    if [[ "$saw_done" != "1" ]] && grep -q "\\[moe-bench\\] done\\." "$log" 2>/dev/null; then
        saw_done=1
    fi

    if [[ "$saw_done" != "1" || ! -s "$summary" ]]; then
        echo "case failed or summary missing: $rel_dir/$prefix" >&2
        echo "see log: $log" >&2
        return 1
    fi

    echo "===== done $rel_dir/$prefix ====="
    grep -nE "prefill|decode|cache hit rate|VRAM peak|profile rows" "$summary" || true
    echo
}

tail -n +2 "$CASE_TSV" | while IFS=$'\t' read -r rel_dir prefix pp tg repeat cache_mb predictor ub; do
    run_case "$rel_dir" "$prefix" "$pp" "$tg" "$repeat" "$cache_mb" "$predictor" "$ub"
done

echo "all normal-prompt runs completed: $OUT_ROOT"
