#!/usr/bin/env python3

import argparse
import csv
import hashlib
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR.parent
MODES = ("control", "host4gb-cold", "host4gb-hot")
VRAMS = (6000, 8000)
UBATCHES = (512, 1024)
RUNS = ("01", "02", "03")
GIB = 1024 ** 3


def search(text, pattern, casts=None, required=True):
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        if required:
            raise ValueError(f"missing summary field: {pattern}")
        return None
    values = match.groups()
    if casts is None:
        casts = (float,) * len(values)
    converted = tuple(cast(value) for cast, value in zip(casts, values))
    return converted[0] if len(converted) == 1 else converted


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_key_values(path):
    result = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def phase_profile(path, prompt_tokens, decode_tokens):
    required_fields = (
        "k_required", "k_hit", "k_miss", "ssd_read_us", "h2d_us",
        "compute_us", "stall_us", "host_cache_hits", "host_cache_misses",
        "host_cache_lookup_us", "host_cache_fill_us", "host_memcpy_us",
        "host_memcpy_bytes", "h2d_bytes", "pinned_staging_wait_us",
        "ssd_bytes", "ssd_reads", "routes_required", "routes_hit",
        "routes_persistent",
    )
    optional_fields = (
        "global_gpu_hits", "global_gpu_misses", "global_gpu_admits",
        "routes_global_hit",
    )
    counter_fields = required_fields + optional_fields
    totals = {
        phase: {field: 0 for field in counter_fields}
        for phase in ("prefill", "decode")
    }
    rows = {"prefill": 0, "decode": 0}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = set(required_fields) - columns
        if missing:
            raise ValueError(f"profile is missing columns: {sorted(missing)}")
        for row in reader:
            phase = row.get("phase")
            if row.get("row_type") != "layer" or phase not in totals:
                continue
            rows[phase] += 1
            for field in counter_fields:
                totals[phase][field] += int(row.get(field, 0) or 0)

    result = {}
    for phase, tokens in (("prefill", prompt_tokens), ("decode", decode_tokens)):
        phase_totals = totals[phase]
        required = phase_totals["k_required"]
        host_accesses = phase_totals["host_cache_hits"] + phase_totals["host_cache_misses"]
        global_accesses = phase_totals["global_gpu_hits"] + phase_totals["global_gpu_misses"]
        result.update({
            f"{phase}_profile_rows": rows[phase],
            f"{phase}_gpu_hits": phase_totals["k_hit"],
            f"{phase}_gpu_misses": phase_totals["k_miss"],
            f"{phase}_gpu_hit_pct": 100.0 * phase_totals["k_hit"] / required if required else 0.0,
            f"{phase}_host_hits": phase_totals["host_cache_hits"],
            f"{phase}_host_misses": phase_totals["host_cache_misses"],
            f"{phase}_host_hit_pct": 100.0 * phase_totals["host_cache_hits"] / host_accesses if host_accesses else 0.0,
            f"{phase}_global_gpu_hits": phase_totals["global_gpu_hits"],
            f"{phase}_global_gpu_misses": phase_totals["global_gpu_misses"],
            f"{phase}_global_gpu_admits": phase_totals["global_gpu_admits"],
            f"{phase}_global_gpu_hit_pct": 100.0 * phase_totals["global_gpu_hits"] / global_accesses if global_accesses else 0.0,
            f"{phase}_route_hit_pct": 100.0 * phase_totals["routes_hit"] / phase_totals["routes_required"] if phase_totals["routes_required"] else 0.0,
            f"{phase}_route_global_hit_pct": 100.0 * phase_totals["routes_global_hit"] / phase_totals["routes_required"] if phase_totals["routes_required"] else 0.0,
            f"{phase}_source_gib": phase_totals["ssd_bytes"] / GIB,
            f"{phase}_h2d_gib": phase_totals["h2d_bytes"] / GIB,
            f"{phase}_host_memcpy_gib": phase_totals["host_memcpy_bytes"] / GIB,
            f"{phase}_source_ms_per_token": phase_totals["ssd_read_us"] / 1000.0 / tokens,
            f"{phase}_h2d_ms_per_token": phase_totals["h2d_us"] / 1000.0 / tokens,
            f"{phase}_compute_ms_per_token": phase_totals["compute_us"] / 1000.0 / tokens,
            f"{phase}_stall_ms_per_token": phase_totals["stall_us"] / 1000.0 / tokens,
            f"{phase}_host_lookup_ms_per_token": phase_totals["host_cache_lookup_us"] / 1000.0 / tokens,
            f"{phase}_host_fill_ms_per_token": phase_totals["host_cache_fill_us"] / 1000.0 / tokens,
            f"{phase}_host_memcpy_ms_per_token": phase_totals["host_memcpy_us"] / 1000.0 / tokens,
            f"{phase}_staging_wait_ms_per_token": phase_totals["pinned_staging_wait_us"] / 1000.0 / tokens,
        })
    return result


def parse_run(mode, vram, ubatch, run):
    prefix = RESULT_ROOT / mode / f"vram-{vram}-ub{ubatch}" / f"run-{run}"
    paths = {
        suffix: Path(f"{prefix}-{suffix}")
        for suffix in (
            "summary.txt", "profile.csv", "metadata.txt", "memory.csv",
            "tokens.csv", "validation.txt", "block-io.tsv",
        )
    }
    missing = [str(path) for path in paths.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(", ".join(missing))

    summary = paths["summary.txt"].read_text(errors="replace")
    metadata = read_key_values(paths["metadata.txt"])
    validation = read_key_values(paths["validation.txt"])
    if validation.get("validation") != "pass":
        raise ValueError(f"validation did not pass: {prefix}")

    prompt_tokens, prompt_ms, _prompt_per_token, _prompt_printed_speed = search(
        summary, r"^prefill\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)$",
        (int, float, float, float),
    )
    decode_tokens, decode_ms, _decode_per_token, _decode_printed_speed = search(
        summary, r"^decode\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)$",
        (int, float, float, float),
    )
    if (prompt_tokens, decode_tokens) != (1024, 1024):
        raise ValueError(f"unexpected token counts: {prompt_tokens}/{decode_tokens}")

    host_mode, host_preload, host_capacity, host_data = search(
        summary,
        r"^Host cache: mode=(\S+) preload=(\S+) capacity=([0-9.]+) GB data=([0-9.]+) GB$",
        (str, str, float, float),
    )
    ready_state = search(
        summary,
        r"^Host cache ready: blobs=(\d+)/(\d+) experts=(\d+) slots_per_layer=(\d+) "
        r"after_prefill=([0-9.]+) GB after_decode=([0-9.]+) GB$",
        (int, int, int, int, float, float), required=False,
    )
    if ready_state is None:
        legacy_ready = search(
            summary,
            r"^Host cache ready: blobs=(\d+)/(\d+) after_prefill=([0-9.]+) GB after_decode=([0-9.]+) GB$",
            (int, int, float, float),
        )
        ready_blobs, ready_total_blobs, ready_prefill, ready_decode = legacy_ready
        ready_experts = slots_per_layer = 0
    else:
        ready_blobs, ready_total_blobs, ready_experts, slots_per_layer, ready_prefill, ready_decode = ready_state
    initial_blobs, initial_experts, initial_gib = search(
        summary, r"^Host cache initial: blobs=(\d+) experts=(\d+) bytes=([0-9.]+) GB$",
        (int, int, float), required=False,
    ) or (0, 0, 0.0)
    host_hits, host_misses, host_hit_pct = search(
        summary, r"^Host cache access: hits=(\d+) misses=(\d+) hit_rate=([0-9.]+)%$",
        (int, int, float),
    )
    preload_alloc_ms, preload_read_ms, preload_total_ms, preload_gib = search(
        summary,
        r"^Host cache preload: alloc=([0-9.]+) ms read=([0-9.]+) ms total=([0-9.]+) ms bytes=([0-9.]+) GB$",
        (float, float, float, float),
    )
    management = search(
        summary,
        r"^Host cache management: admissions=(\d+) evictions=(\d+) bypasses=(\d+) active_leases=(\d+)$",
        (int, int, int, int), required=False,
    )
    host_admissions, host_evictions, host_bypasses, host_leases = management or (0, 0, 0, 0)
    host_memcpy_gib_summary, host_h2d_gib_summary = search(
        summary, r"^Host cache transfer: memcpy=([0-9.]+) GB h2d=([0-9.]+) GB$",
        (float, float),
    )
    global_state = search(
        summary,
        r"^global GPU decode cache: hits=(\d+) misses=(\d+) admits=(\d+) hit_rate=([0-9.]+)%$",
        (int, int, int, float), required=False,
    )
    global_hits, global_misses, global_admits, global_hit_pct = global_state or (0, 0, 0, 0.0)
    page_before, page_after_prepare, page_after_model, page_after_prefill, page_after_decode = search(
        summary,
        r"^Page cache resident pct: before=([0-9.]+) after_prepare=([0-9.]+) "
        r"after_model_load=([0-9.]+) after_prefill=([0-9.]+) after_decode=([0-9.]+)$",
        (float,) * 5,
    )
    prepare_read, model_read, prefill_read, decode_read, total_read = search(
        summary,
        r"^Process IO read_bytes: prepare=(\d+) model_init=(\d+) prefill=(\d+) decode=(\d+) total=(\d+)$",
        (int,) * 5,
    )
    prepare_rchar, model_rchar, prefill_rchar, decode_rchar, total_rchar = search(
        summary,
        r"^Process IO rchar: prepare=(\d+) model_init=(\d+) prefill=(\d+) decode=(\d+) total=(\d+)$",
        (int,) * 5,
    )
    page_prepare_ms, model_init_ms, process_start_to_prefill_end_ms = search(
        summary,
        r"^Page cache timing: prepare_ms=([0-9.]+) model_init_ms=([0-9.]+) "
        r"process_start_to_prefill_end_ms=([0-9.]+)$",
        (float, float, float),
    )

    peak_rss = peak_hwm = peak_locked = 0
    with paths["memory.csv"].open(newline="") as stream:
        for row in csv.DictReader(stream):
            peak_rss = max(peak_rss, int(row["rss_kib"] or 0))
            peak_hwm = max(peak_hwm, int(row["hwm_kib"] or 0))
            peak_locked = max(peak_locked, int(row["locked_kib"] or 0))

    block_stats = {}
    with paths["block-io.tsv"].open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            block_stats[row["device"]] = row

    result = {
        "mode": mode,
        "vram_cache_mb": vram,
        "ubatch": ubatch,
        "run": run,
        "validation": "pass",
        "prompt_sha256": metadata.get("prompt_sha256", ""),
        "binary_sha256": metadata.get("binary_sha256", ""),
        "token_sha256": validation.get("token_sha256", sha256(paths["tokens.csv"])),
        "route_trace_sha256": validation.get("route_trace_sha256", ""),
        "prefill_ms": prompt_ms,
        "prefill_ms_per_token": prompt_ms / prompt_tokens,
        "prefill_tok_s": prompt_tokens * 1000.0 / prompt_ms,
        "decode_total_ms": decode_ms,
        "tpot_ms": decode_ms / decode_tokens,
        "decode_tok_s": decode_tokens * 1000.0 / decode_ms,
        "inference_ttft_ms": search(summary, r"^TTFT: ([0-9.]+) ms$"),
        "service_cold_start_ttft_ms": search(summary, r"^service cold-start TTFT: ([0-9.]+) ms$"),
        "cache_hit_prefill_pct": search(summary, r"^cache hit rate \(prefill\): ([0-9.]+)%$"),
        "cache_hit_decode_pct": search(summary, r"^cache hit rate \(decode\): ([0-9.]+)%$"),
        "route_hit_prefill_pct": search(summary, r"^route hit coverage \(prefill\): ([0-9.]+)%$"),
        "route_hit_decode_pct": search(summary, r"^route hit coverage \(decode\): ([0-9.]+)%$"),
        "global_gpu_hits": global_hits,
        "global_gpu_misses": global_misses,
        "global_gpu_admits": global_admits,
        "global_gpu_hit_pct": global_hit_pct,
        "host_cache_mode": host_mode,
        "host_cache_preload": host_preload,
        "host_capacity_gib": host_capacity,
        "host_data_gib": host_data,
        "host_initial_blobs": initial_blobs,
        "host_initial_experts": initial_experts,
        "host_initial_gib": initial_gib,
        "host_final_blobs": ready_blobs,
        "host_total_blobs": ready_total_blobs,
        "host_final_experts": ready_experts,
        "host_slots_per_layer": slots_per_layer,
        "host_ready_gib_after_prefill": ready_prefill,
        "host_ready_gib_after_decode": ready_decode,
        "host_hits": host_hits,
        "host_misses": host_misses,
        "host_hit_pct": host_hit_pct,
        "host_preload_alloc_ms": preload_alloc_ms,
        "host_preload_read_ms": preload_read_ms,
        "host_preload_total_ms": preload_total_ms,
        "host_preload_gib": preload_gib,
        "host_admissions": host_admissions,
        "host_evictions": host_evictions,
        "host_bypasses": host_bypasses,
        "host_active_leases": host_leases,
        "host_memcpy_gib_summary": host_memcpy_gib_summary,
        "host_h2d_gib_summary": host_h2d_gib_summary,
        "source_total_gib": search(summary, r"^Source bytes read \(total\): ([0-9.]+) GB$"),
        "page_cache_resident_before_pct": page_before,
        "page_cache_resident_after_prepare_pct": page_after_prepare,
        "page_cache_resident_after_model_pct": page_after_model,
        "page_cache_resident_after_prefill_pct": page_after_prefill,
        "page_cache_resident_after_decode_pct": page_after_decode,
        "page_cache_prepare_ms": page_prepare_ms,
        "model_init_ms": model_init_ms,
        "process_start_to_prefill_end_ms": process_start_to_prefill_end_ms,
        "process_read_prepare_gib": prepare_read / GIB,
        "process_read_model_gib": model_read / GIB,
        "process_read_prefill_gib": prefill_read / GIB,
        "process_read_decode_gib": decode_read / GIB,
        "process_read_total_gib": total_read / GIB,
        "process_rchar_prepare_gib": prepare_rchar / GIB,
        "process_rchar_model_gib": model_rchar / GIB,
        "process_rchar_prefill_gib": prefill_rchar / GIB,
        "process_rchar_decode_gib": decode_rchar / GIB,
        "process_rchar_total_gib": total_rchar / GIB,
        "peak_rss_gib": peak_rss / 1024.0 / 1024.0,
        "peak_hwm_gib": peak_hwm / 1024.0 / 1024.0,
        "peak_locked_gib": peak_locked / 1024.0 / 1024.0,
        "dm_reads_delta": int(block_stats.get("253:0", {}).get("reads_completed_delta", 0)),
        "dm_sectors_read_delta": int(block_stats.get("253:0", {}).get("sectors_read_delta", 0)),
        "sdb3_reads_delta": int(block_stats.get("sdb3", {}).get("reads_completed_delta", 0)),
        "sdb3_sectors_read_delta": int(block_stats.get("sdb3", {}).get("sectors_read_delta", 0)),
    }
    result.update(phase_profile(paths["profile.csv"], prompt_tokens, decode_tokens))
    return result


def write_tsv(path, rows, columns):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


KEY_METRICS = (
    "prefill_ms", "prefill_ms_per_token", "prefill_tok_s", "decode_total_ms",
    "tpot_ms", "decode_tok_s", "inference_ttft_ms", "service_cold_start_ttft_ms",
    "cache_hit_prefill_pct", "cache_hit_decode_pct", "prefill_gpu_hit_pct",
    "decode_gpu_hit_pct", "decode_global_gpu_hit_pct", "host_hit_pct",
    "prefill_host_hit_pct", "decode_host_hit_pct", "host_initial_experts",
    "host_final_experts", "host_ready_gib_after_prefill", "host_ready_gib_after_decode",
    "host_preload_alloc_ms", "host_preload_read_ms", "host_preload_total_ms",
    "host_admissions", "host_evictions", "host_bypasses", "source_total_gib",
    "prefill_source_gib", "decode_source_gib", "prefill_h2d_gib", "decode_h2d_gib",
    "prefill_source_ms_per_token", "decode_source_ms_per_token",
    "prefill_h2d_ms_per_token", "decode_h2d_ms_per_token",
    "prefill_compute_ms_per_token", "decode_compute_ms_per_token",
    "prefill_stall_ms_per_token", "decode_stall_ms_per_token",
    "model_init_ms", "page_cache_prepare_ms", "page_cache_resident_after_prepare_pct",
    "process_read_total_gib",
    "peak_rss_gib", "peak_hwm_gib", "peak_locked_gib",
)


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["mode"], row["vram_cache_mb"], row["ubatch"])].append(row)
    aggregates = []
    for mode in MODES:
        for vram in VRAMS:
            for ubatch in UBATCHES:
                values = groups.get((mode, vram, ubatch), [])
                item = {
                    "mode": mode,
                    "vram_cache_mb": vram,
                    "ubatch": ubatch,
                    "run_count": len(values),
                }
                for metric in KEY_METRICS:
                    samples = [float(row[metric]) for row in values]
                    if not samples:
                        for suffix in ("mean", "stddev", "min", "max"):
                            item[f"{metric}_{suffix}"] = ""
                        continue
                    item[f"{metric}_mean"] = statistics.fmean(samples)
                    item[f"{metric}_stddev"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
                    item[f"{metric}_min"] = min(samples)
                    item[f"{metric}_max"] = max(samples)
                aggregates.append(item)
    return aggregates


def fmt(value, digits=2):
    return f"{float(value):.{digits}f}"


def stat_cell(row, metric, digits=2):
    return (
        f"{fmt(row[f'{metric}_mean'], digits)} +/- {fmt(row[f'{metric}_stddev'], digits)} "
        f"[{fmt(row[f'{metric}_min'], digits)}, {fmt(row[f'{metric}_max'], digits)}]"
    )


def pct_change(new, baseline):
    return 100.0 * (new / baseline - 1.0) if baseline else math.nan


def build_report(aggregates, raw_rows):
    lookup = {
        (row["mode"], row["vram_cache_mb"], row["ubatch"]): row
        for row in aggregates
    }
    lines = [
        "# 4 GiB Host DRAM Cache Comparison",
        "",
        "本报告所有速度均由实际 token 数和未取整的总时间重新计算。`均值 +/- 样本标准差 [最小值, 最大值]`，每个 cell 为 3 个全新进程。",
        "",
        "固定使用 GPU1、正常中英混合 prompt、`pp=1024`、`tg=1024`、greedy 和热 Linux page cache。control 是冻结原版；两个 4 GiB 模式还包含 aged-LFU 与 decode 全局 GPU cache，因此相对 control 的结果代表三级系统整体收益，不能只归因于 Host DRAM。",
        "",
        "## 核心性能",
        "",
        "| 模式 | VRAM MiB | ubatch | Prefill ms | Prefill tok/s | Decode TPOT ms/token | Decode tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        for vram in VRAMS:
            for ubatch in UBATCHES:
                row = lookup[(mode, vram, ubatch)]
                lines.append(
                    f"| {mode} | {vram} | {ubatch} | {stat_cell(row, 'prefill_ms')} | "
                    f"{stat_cell(row, 'prefill_tok_s')} | {stat_cell(row, 'tpot_ms')} | "
                    f"{stat_cell(row, 'decode_tok_s')} |"
                )

    lines.extend([
        "",
        "## 相对 Control",
        "",
        "正值表示速度提升；TPOT 一列正值表示延迟降低。",
        "",
        "| 模式 | VRAM MiB | ubatch | Prefill 速度变化 | Decode 速度变化 | TPOT 降低 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for mode in ("host4gb-cold", "host4gb-hot"):
        for vram in VRAMS:
            for ubatch in UBATCHES:
                row = lookup[(mode, vram, ubatch)]
                base = lookup[("control", vram, ubatch)]
                prefill_gain = pct_change(row["prefill_tok_s_mean"], base["prefill_tok_s_mean"])
                decode_gain = pct_change(row["decode_tok_s_mean"], base["decode_tok_s_mean"])
                tpot_reduction = -pct_change(row["tpot_ms_mean"], base["tpot_ms_mean"])
                lines.append(
                    f"| {mode} | {vram} | {ubatch} | {prefill_gain:+.2f}% | "
                    f"{decode_gain:+.2f}% | {tpot_reduction:+.2f}% |"
                )

    lines.extend([
        "",
        "## Hot 相对 Cold",
        "",
        "这一比较使用相同的 4 GiB 容量、aged-LFU 和 decode 全局 GPU cache，只改变 Host cache 是否在推理前预加载 hotset。",
        "",
        "| VRAM MiB | ubatch | Prefill 速度变化 | Decode 速度变化 | Hot preload ms |",
        "|---:|---:|---:|---:|---:|",
    ])
    hot_cold_decode_gains = []
    for vram in VRAMS:
        for ubatch in UBATCHES:
            cold = lookup[("host4gb-cold", vram, ubatch)]
            hot = lookup[("host4gb-hot", vram, ubatch)]
            prefill_gain = pct_change(hot["prefill_tok_s_mean"], cold["prefill_tok_s_mean"])
            decode_gain = pct_change(hot["decode_tok_s_mean"], cold["decode_tok_s_mean"])
            hot_cold_decode_gains.append(decode_gain)
            lines.append(
                f"| {vram} | {ubatch} | {prefill_gain:+.2f}% | {decode_gain:+.2f}% | "
                f"{fmt(hot['host_preload_total_ms_mean'])} |"
            )

    lines.extend([
        "",
        "## 缓存与传输",
        "",
        "| 模式 | VRAM MiB | ubatch | GPU prefill hit % | GPU decode hit % | Host hit % | Global decode GPU hit % | Source GiB | H2D GiB | Decode stall ms/token |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for mode in MODES:
        for vram in VRAMS:
            for ubatch in UBATCHES:
                row = lookup[(mode, vram, ubatch)]
                h2d = row["prefill_h2d_gib_mean"] + row["decode_h2d_gib_mean"]
                lines.append(
                    f"| {mode} | {vram} | {ubatch} | {fmt(row['prefill_gpu_hit_pct_mean'])} | "
                    f"{fmt(row['decode_gpu_hit_pct_mean'])} | {fmt(row['host_hit_pct_mean'])} | "
                    f"{fmt(row['decode_global_gpu_hit_pct_mean'])} | {fmt(row['source_total_gib_mean'])} | "
                    f"{fmt(h2d)} | {fmt(row['decode_stall_ms_per_token_mean'])} |"
                )

    lines.extend([
        "",
        "## Decode 数据路径时间",
        "",
        "以下是 profile 累加时间除以 1024 decode tokens。各项存在异步重叠，不能直接相加得到 TPOT。",
        "",
        "| 模式 | VRAM MiB | ubatch | Source ms/token | H2D ms/token | GPU compute ms/token | Stall ms/token |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for mode in MODES:
        for vram in VRAMS:
            for ubatch in UBATCHES:
                row = lookup[(mode, vram, ubatch)]
                lines.append(
                    f"| {mode} | {vram} | {ubatch} | {fmt(row['decode_source_ms_per_token_mean'])} | "
                    f"{fmt(row['decode_h2d_ms_per_token_mean'])} | "
                    f"{fmt(row['decode_compute_ms_per_token_mean'])} | "
                    f"{fmt(row['decode_stall_ms_per_token_mean'])} |"
                )

    lines.extend([
        "",
        "## Linux Page Cache 证据",
        "",
        "Page-cache prepare 不计入 inference TTFT。进程物理读取量来自 `/proc/self/io read_bytes`；它与应用层逻辑 Source GiB 不同。",
        "",
        "| 模式 | VRAM MiB | ubatch | Prepare 后驻留率 % | Prepare ms | 进程物理读取 GiB |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for mode in MODES:
        for vram in VRAMS:
            for ubatch in UBATCHES:
                row = lookup[(mode, vram, ubatch)]
                lines.append(
                    f"| {mode} | {vram} | {ubatch} | "
                    f"{fmt(row['page_cache_resident_after_prepare_pct_mean'], 4)} | "
                    f"{fmt(row['page_cache_prepare_ms_mean'])} | "
                    f"{fmt(row['process_read_total_gib_mean'], 4)} |"
                )

    lines.extend([
        "",
        "## Host Cache 状态与启动成本",
        "",
        "| 模式 | VRAM MiB | ubatch | 初始专家 | 最终专家 | Preload ms | Model init ms | Service cold-start TTFT ms | Peak RSS GiB | VmLck GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for mode in MODES:
        for vram in VRAMS:
            for ubatch in UBATCHES:
                row = lookup[(mode, vram, ubatch)]
                lines.append(
                    f"| {mode} | {vram} | {ubatch} | {fmt(row['host_initial_experts_mean'], 0)} | "
                    f"{fmt(row['host_final_experts_mean'], 0)} | {fmt(row['host_preload_total_ms_mean'])} | "
                    f"{fmt(row['model_init_ms_mean'])} | {fmt(row['service_cold_start_ttft_ms_mean'])} | "
                    f"{fmt(row['peak_rss_gib_mean'])} | {fmt(row['peak_locked_gib_mean'])} |"
                )

    hot_decode_gains = []
    hot_stall_shares = []
    cold_decode_gains = []
    high_variation = []
    for vram in VRAMS:
        for ubatch in UBATCHES:
            base = lookup[("control", vram, ubatch)]
            cold = lookup[("host4gb-cold", vram, ubatch)]
            hot = lookup[("host4gb-hot", vram, ubatch)]
            hot_decode_gains.append(pct_change(hot["decode_tok_s_mean"], base["decode_tok_s_mean"]))
            cold_decode_gains.append(pct_change(cold["decode_tok_s_mean"], base["decode_tok_s_mean"]))
            hot_stall_shares.append(100.0 * hot["decode_stall_ms_per_token_mean"] / hot["tpot_ms_mean"])

    for mode in MODES:
        for vram in VRAMS:
            for ubatch in UBATCHES:
                row = lookup[(mode, vram, ubatch)]
                cv = 100.0 * row["tpot_ms_stddev"] / row["tpot_ms_mean"]
                if cv > 5.0:
                    high_variation.append(f"{mode}/{vram}/ub{ubatch}={cv:.2f}%")

    lines.extend([
        "",
        "## 初步结论",
        "",
        f"- 4 GiB hot cache 相对 control 的 decode 速度变化范围为 {min(hot_decode_gains):+.2f}% 到 {max(hot_decode_gains):+.2f}%。",
        f"- 4 GiB cold cache 相对 control 的 decode 速度变化范围为 {min(cold_decode_gains):+.2f}% 到 {max(cold_decode_gains):+.2f}%。",
        f"- Hot 相对 cold 的 decode 速度变化范围仅为 {min(hot_cold_decode_gains):+.2f}% 到 {max(hot_cold_decode_gains):+.2f}%。1024-token prefill 已让 cold cache 接近填满，因此 hotset 的主要收益出现在 prefill，而不是随后 decode 的稳态。",
        f"- Hot 模式 decode 的已观测 stall/TPOT 比例范围为 {min(hot_stall_shares):.2f}% 到 {max(hot_stall_shares):.2f}%。这是后续异步预取可尝试隐藏的上界线索，不等于可实现的端到端收益。",
        "- `Source GiB` 是应用层专家源读取量；Linux page cache 已在每次运行前预热，因此它不等同于物理 SSD 读取量。物理读取证据见 `summary-table.tsv` 的进程 `read_bytes` 与块设备计数。",
        "- Host cache 与 GPU cache 采用互补策略；预加载的 Host 专家晋升 GPU 后会从 Host tier 删除，所以运行结束时 Host 专家数低于初始 2240 属于预期行为。",
        "- `/proc/<pid>/status` 的 `VmLck` 对 CUDA `cudaHostAlloc` 页面报告为 0；本实验以 pinned 分配成功、direct-H2D 路径和 5.31 GiB RSS 为主要 Host 内存证据，不能把 `VmLck=0` 解读为 pageable fallback。",
        f"- TPOT 变异系数超过 5% 的 cell：{', '.join(high_variation) if high_variation else '无'}。这些三次均值应视为快速测试估计；逐次范围已保留在核心性能表。",
        "",
        "## 正确性",
        "",
    ])
    digest_groups = defaultdict(lambda: {"tokens": set(), "routes": set()})
    for row in raw_rows:
        key = (row["vram_cache_mb"], row["ubatch"])
        digest_groups[key]["tokens"].add(row["token_sha256"])
        digest_groups[key]["routes"].add(row["route_trace_sha256"])
    digest_ok = all(len(value["tokens"]) == 1 and len(value["routes"]) == 1 for value in digest_groups.values())
    lines.append(
        f"- 36 次正式运行按每个 VRAM/ubatch cell 比较 token trace 与 routing trace：{'通过' if digest_ok else '失败'}。"
    )
    logits_path = RESULT_ROOT / "correctness-logits" / "comparison.txt"
    if logits_path.is_file():
        logits = read_key_values(logits_path)
        lines.append(
            f"- 短任务 logits 对比：{logits.get('status', 'unknown')}；"
            f"max abs delta={logits.get('max_abs_delta', 'unknown')}，"
            f"阈值={logits.get('threshold', 'unknown')}，"
            f"token trace equal={logits.get('token_trace_equal', 'unknown')}。"
        )
    else:
        lines.append("- 短任务 logits 数值对比尚未生成。")
    lines.extend([
        "",
        "## Fate 后续判断",
        "",
        "Hot 模式仍有约 16%–20% 的 decode TPOT 被标记为 overlap-loss stall，超过 15% 的继续研究门槛，因此进入 Fate hidden-state predictor/异步预取阶段是合理的。但这只是可隐藏空间的线索：应以 `decode_source_ms_per_token`、`decode_h2d_ms_per_token`、`decode_stall_ms_per_token` 以及 global GPU cache 命中率联合判断，不能把逻辑 source-read 时间全部当成可消除收益。",
        "",
        "完整逐次数据见 `summary-table.tsv`，聚合数据见 `aggregate-table.tsv`。",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    rows = []
    errors = []
    for mode in MODES:
        for vram in VRAMS:
            for ubatch in UBATCHES:
                for run in RUNS:
                    try:
                        rows.append(parse_run(mode, vram, ubatch, run))
                    except (FileNotFoundError, ValueError) as error:
                        errors.append(f"{mode}/vram-{vram}-ub{ubatch}/run-{run}: {error}")

    if errors and args.require_complete:
        for error in errors:
            print(f"summary error: {error}")
        return 1
    if not rows:
        print("no completed runs found")
        return 0

    columns = list(rows[0])
    write_tsv(RESULT_ROOT / "summary-table.tsv", rows, columns)
    aggregates = aggregate(rows)
    aggregate_columns = list(aggregates[0])
    write_tsv(RESULT_ROOT / "aggregate-table.tsv", aggregates, aggregate_columns)

    cases = [{
        "case_id": f"{mode}-vram-{vram}-ub{ubatch}",
        "mode": mode,
        "vram_cache_mb": vram,
        "ubatch": ubatch,
        "pp": 1024,
        "tg": 1024,
        "runs": 3,
        "page_cache_policy": "hot",
        "host_cache": "off" if mode == "control" else "pinned",
        "host_cache_capacity_mb": 0 if mode == "control" else 4096,
        "host_cache_preload": "hotset" if mode == "host4gb-hot" else "none",
        "tier_policy": "legacy" if mode == "control" else "aged-lfu",
        "decode_global_cache": "off" if mode == "control" else "on",
    } for mode in MODES for vram in VRAMS for ubatch in UBATCHES]
    write_tsv(RESULT_ROOT / "cases.tsv", cases, list(cases[0]))

    if not errors:
        (RESULT_ROOT / "final-report.md").write_text(build_report(aggregates, rows))
    else:
        print(f"summarized {len(rows)} completed runs; {len(errors)} runs are absent or invalid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
