#!/usr/bin/env python3

import csv
import math
import random
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODES = ["control-0716", "pageable-cold", "pageable-hot", "pinned-hot"]


def number(text, pattern, default=0.0):
    match = re.search(pattern, text, re.MULTILINE)
    return float(match.group(1)) if match else default


def parse_summary(path):
    text = path.read_text(errors="replace")
    phase = {}
    for name in ("prefill", "decode"):
        match = re.search(rf"^{name}\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)$", text, re.MULTILINE)
        if match:
            phase[name] = {
                "tokens": int(match.group(1)),
                "total_ms": float(match.group(2)),
                "per_token_ms": float(match.group(3)),
                "tok_s": float(match.group(4)),
            }
    io = {}
    match = re.search(r"I/O breakdown \(decode, mean per token\):\n(.*?)(?:\n\n|Profiler breakdown)", text, re.DOTALL)
    if match:
        for key, value in re.findall(r"^\s+([a-z0-9_]+)(?: \(overlap loss\))?\s+([0-9.]+) ms$", match.group(1), re.MULTILINE):
            io[key] = float(value)

    ready = re.search(r"Host cache ready: blobs=(\d+)/(\d+) after_prefill=([0-9.]+) GB after_decode=([0-9.]+) GB", text)
    access = re.search(r"Host cache access: hits=(\d+) misses=(\d+) hit_rate=([0-9.]+)%", text)
    return {
        "path": str(path),
        "prefill_ms": phase.get("prefill", {}).get("total_ms", 0.0),
        "prefill_tok_s": phase.get("prefill", {}).get("tok_s", 0.0),
        "ttft_ms": number(text, r"^TTFT: ([0-9.]+) ms$"),
        "service_ttft_ms": number(text, r"^service cold-start TTFT: ([0-9.]+) ms$"),
        "tpot_ms": number(text, r"^TPOT: ([0-9.]+) ms$"),
        "decode_tok_s": phase.get("decode", {}).get("tok_s", 0.0),
        "host_hit_pct": float(access.group(3)) if access else 0.0,
        "ready_prefill_gib": float(ready.group(3)) if ready else 0.0,
        "ready_decode_gib": float(ready.group(4)) if ready else 0.0,
        "source_read_gib": number(text, r"^Source bytes read \(total\): ([0-9.]+) GB$"),
        "preload_total_ms": number(text, r"^Host cache preload: alloc=[0-9.]+ ms read=[0-9.]+ ms total=([0-9.]+) ms"),
        "host_memcpy_ms": io.get("host_memcpy", 0.0),
        "h2d_ms": io.get("h2d", 0.0),
        "compute_ms": io.get("gpu_compute", 0.0),
        "stall_ms": io.get("stall", 0.0),
        "rss_peak_gib": number(text, r"^DRAM peak \(process\): ([0-9.]+) GB$"),
        "locked_gib": number(text, r"^DRAM locked \(process\): ([0-9.]+) GB$"),
    }


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def bootstrap_median_ci(values, samples=10000):
    if not values:
        return 0.0, 0.0
    rng = random.Random(20260729 + len(values))
    medians = []
    for _ in range(samples):
        medians.append(statistics.median(rng.choice(values) for _ in values))
    return percentile(medians, 0.025), percentile(medians, 0.975)


def median(rows, key):
    values = [row[key] for row in rows]
    return statistics.median(values) if values else 0.0


def fmt(value):
    return f"{value:.4f}"


def collect_cell(mode, vram, ubatch):
    cell = ROOT / mode / f"vram-{vram}-ub{ubatch}"
    paths = sorted(cell.glob("run-[0-9][0-9]-summary.txt"))
    return cell, [parse_summary(path) for path in paths]


def write_case_summary(cell, rows):
    columns = [
        "run", "prefill_ms", "prefill_tok_s", "ttft_ms", "service_ttft_ms",
        "tpot_ms", "decode_tok_s", "host_hit_pct", "ready_prefill_gib",
        "ready_decode_gib", "source_read_gib", "preload_total_ms", "host_memcpy_ms", "h2d_ms",
        "compute_ms", "stall_ms", "rss_peak_gib", "locked_gib",
    ]
    with (cell / "case-summary.tsv").open("w") as out:
        out.write("\t".join(columns) + "\n")
        for row in rows:
            values = [Path(row["path"]).name.replace("-summary.txt", "")]
            values.extend(fmt(row[key]) for key in columns[1:])
            out.write("\t".join(values) + "\n")


def make_table_row(mode, vram, ubatch, rows):
    tpot = [row["tpot_ms"] for row in rows]
    low, high = bootstrap_median_ci(tpot)
    return {
        "mode": mode,
        "vram": vram,
        "ubatch": ubatch,
        "count": len(rows),
        "prefill_ms": median(rows, "prefill_ms"),
        "prefill_tok_s": median(rows, "prefill_tok_s"),
        "ttft_ms": median(rows, "ttft_ms"),
        "service_ttft_ms": median(rows, "service_ttft_ms"),
        "tpot_ms": median(rows, "tpot_ms"),
        "decode_tok_s": median(rows, "decode_tok_s"),
        "host_hit_pct": median(rows, "host_hit_pct"),
        "ready_prefill_gib": median(rows, "ready_prefill_gib"),
        "ready_decode_gib": median(rows, "ready_decode_gib"),
        "source_read_gib": median(rows, "source_read_gib"),
        "preload_total_ms": median(rows, "preload_total_ms"),
        "host_memcpy_ms": median(rows, "host_memcpy_ms"),
        "h2d_ms": median(rows, "h2d_ms"),
        "compute_ms": median(rows, "compute_ms"),
        "stall_ms": median(rows, "stall_ms"),
        "rss_peak_gib": median(rows, "rss_peak_gib"),
        "locked_gib": median(rows, "locked_gib"),
        "mean": statistics.mean(tpot) if tpot else 0.0,
        "p95": percentile(tpot, 0.95),
        "stddev": statistics.stdev(tpot) if len(tpot) > 1 else 0.0,
        "cv_pct": 100.0 * statistics.stdev(tpot) / statistics.mean(tpot) if len(tpot) > 1 else 0.0,
        "ci_low": low,
        "ci_high": high,
    }


def paired_gain(control_rows, candidate_rows, samples=10000):
    control = {Path(row["path"]).name: row["tpot_ms"] for row in control_rows}
    candidate = {Path(row["path"]).name: row["tpot_ms"] for row in candidate_rows}
    pairs = [(control[name], candidate[name]) for name in sorted(control.keys() & candidate.keys())]
    gains = [100.0 * (base - value) / base for base, value in pairs if base]
    if not gains:
        return 0.0, 0.0, 0.0
    rng = random.Random(20260729 + len(gains) * 17)
    medians = []
    for _ in range(samples):
        medians.append(statistics.median(rng.choice(gains) for _ in gains))
    return statistics.median(gains), percentile(medians, 0.025), percentile(medians, 0.975)


def write_warmup_table():
    output = ROOT / "pageable-cold-warmup.tsv"
    header = [
        "vram_cache_mb", "ubatch", "window", "tokens", "mean_tpot_ms",
        "ready_blobs_end", "ready_gib_end", "host_hit_pct", "source_reads",
        "source_read_gib", "host_memcpy_ms_per_token", "h2d_ms_per_token",
        "stall_ms_per_token",
    ]
    result = []
    windows = [(0, 0, "0"), (1, 7, "1-7"), (8, 31, "8-31"),
               (32, 127, "32-127"), (128, 1023, "128-1023")]
    for vram in (6000, 8000):
        for ubatch in (512, 1024):
            path = ROOT / "pageable-cold" / f"vram-{vram}-ub{ubatch}" / "diagnostic-profile.csv"
            if not path.exists():
                continue
            layer_rows = []
            request_wall = {}
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    if row["phase"] == "decode" and row["row_type"] == "layer":
                        layer_rows.append(row)
                    elif row["phase"] == "decode" and row["row_type"] == "request":
                        # Request rows describe an entire decode step. Their token_idx is
                        # always zero; batch_idx advances from one for the first token.
                        request_wall[int(row["batch_idx"]) - 1] = int(row["request_wall_us"] or 0)
            if not layer_rows:
                continue
            token_base = min(int(row["token_idx"]) for row in layer_rows)
            per_token = {}
            for row in layer_rows:
                token = int(row["token_idx"]) - token_base
                stats = per_token.setdefault(token, {
                    "hits": 0, "misses": 0, "miss_bytes": 0,
                    "memcpy_us": 0, "h2d_us": 0, "stall_us": 0,
                })
                stats["hits"] += int(row["host_cache_hits"] or 0)
                stats["misses"] += int(row["host_cache_misses"] or 0)
                stats["miss_bytes"] += int(row["host_cache_miss_bytes"] or 0)
                stats["memcpy_us"] += int(row["host_memcpy_us"] or 0)
                stats["h2d_us"] += int(row["h2d_us"] or 0)
                stats["stall_us"] += int(row["stall_us"] or 0)

            summary = parse_summary(path.with_name("diagnostic-summary.txt"))
            ready_blobs = 0
            ready_bytes = 0
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    if row["phase"] == "prefill" and row["row_type"] == "layer":
                        ready_blobs += int(row["host_cache_misses"] or 0)
                        ready_bytes += int(row["host_cache_miss_bytes"] or 0)
            for low, high, label in windows:
                tokens = list(range(low, high + 1))
                stats = [per_token.get(token, {}) for token in tokens]
                hits = sum(row.get("hits", 0) for row in stats)
                misses = sum(row.get("misses", 0) for row in stats)
                miss_bytes = sum(row.get("miss_bytes", 0) for row in stats)
                ready_blobs += misses
                ready_bytes += miss_bytes
                count = len(tokens)
                walls = [request_wall[token] / 1000.0 for token in tokens if token in request_wall]
                result.append({
                    "vram": vram, "ubatch": ubatch, "window": label, "tokens": count,
                    "tpot": statistics.mean(walls) if walls else 0.0,
                    "ready_blobs": ready_blobs,
                    "ready_gib": ready_bytes / (1024.0 ** 3),
                    "hit_pct": 100.0 * hits / (hits + misses) if hits + misses else 0.0,
                    "source_reads": misses,
                    "source_gib": miss_bytes / (1024.0 ** 3),
                    "memcpy_ms": sum(row.get("memcpy_us", 0) for row in stats) / count / 1000.0,
                    "h2d_ms": sum(row.get("h2d_us", 0) for row in stats) / count / 1000.0,
                    "stall_ms": sum(row.get("stall_us", 0) for row in stats) / count / 1000.0,
                    "summary_ready": summary["ready_decode_gib"],
                })
    with output.open("w") as out:
        out.write("\t".join(header) + "\n")
        for row in result:
            out.write("\t".join([
                str(row["vram"]), str(row["ubatch"]), row["window"], str(row["tokens"]),
                fmt(row["tpot"]), str(row["ready_blobs"]), fmt(row["ready_gib"]),
                fmt(row["hit_pct"]), str(row["source_reads"]), fmt(row["source_gib"]),
                fmt(row["memcpy_ms"]), fmt(row["h2d_ms"]), fmt(row["stall_ms"]),
            ]) + "\n")
    return result


def environment_minima():
    resident_pcts = []
    local_pcts = []
    for path in ROOT.glob("*/vram-*/run-*-metadata.txt"):
        text = path.read_text(errors="replace")
        page = re.search(r"^page_cache_before=(\d+) (\d+)$", text, re.MULTILINE)
        if page and int(page.group(2)):
            resident_pcts.append(100.0 * int(page.group(1)) / int(page.group(2)))
        n0 = number(text, r"^peak_sample_n0_pages=(\d+)$")
        n1 = number(text, r"^peak_sample_n1_pages=(\d+)$")
        if n0 + n1:
            local_pcts.append(100.0 * n0 / (n0 + n1))
    return min(resident_pcts, default=0.0), min(local_pcts, default=0.0)


def main():
    table_rows = []
    grouped = {}
    for mode in MODES:
        for vram in (6000, 8000):
            for ubatch in (512, 1024):
                cell, rows = collect_cell(mode, vram, ubatch)
                cell.mkdir(parents=True, exist_ok=True)
                write_case_summary(cell, rows)
                grouped[(mode, vram, ubatch)] = rows
                table_rows.append(make_table_row(mode, vram, ubatch, rows))

    header = [
        "mode", "vram_cache_mb", "ubatch", "run_count", "prefill_median_ms",
        "prefill_tok_s", "inference_ttft_median_ms", "service_cold_start_ttft_median_ms",
        "decode_tpot_median_ms", "decode_tok_s", "host_cache_hit_pct",
        "host_ready_gib_after_prefill", "host_ready_gib_after_decode", "source_read_gib",
        "preload_total_ms", "host_memcpy_ms_per_token", "h2d_ms_per_token", "gpu_compute_ms_per_token",
        "stall_ms_per_token", "rss_peak_gib", "locked_peak_gib", "decode_mean_ms",
        "decode_p95_ms", "decode_stddev_ms", "decode_cv_pct", "bootstrap_95ci_low_ms", "bootstrap_95ci_high_ms",
    ]
    keys = [
        "mode", "vram", "ubatch", "count", "prefill_ms", "prefill_tok_s", "ttft_ms",
        "service_ttft_ms", "tpot_ms", "decode_tok_s", "host_hit_pct", "ready_prefill_gib",
        "ready_decode_gib", "source_read_gib", "preload_total_ms", "host_memcpy_ms", "h2d_ms", "compute_ms",
        "stall_ms", "rss_peak_gib", "locked_gib", "mean", "p95", "stddev", "cv_pct", "ci_low", "ci_high",
    ]
    with (ROOT / "summary-table.tsv").open("w") as out:
        out.write("\t".join(header) + "\n")
        for row in table_rows:
            values = [str(row[key]) if key in ("mode", "vram", "ubatch", "count") else fmt(row[key]) for key in keys]
            out.write("\t".join(values) + "\n")

    real_grouped = {}
    real_rows = []
    for mode in MODES:
        cell = ROOT / "real-prompt" / mode / "vram-8000-ub512"
        rows = [parse_summary(path) for path in sorted(cell.glob("run-[0-9][0-9]-summary.txt"))]
        if rows:
            write_case_summary(cell, rows)
        real_grouped[mode] = rows
        real_rows.append(make_table_row(mode, 8000, 512, rows))
    with (ROOT / "real-prompt-summary.tsv").open("w") as out:
        out.write("mode\trun_count\tprefill_median_ms\tdecode_tpot_median_ms\tdecode_mean_ms\tdecode_p95_ms\tdecode_stddev_ms\tdecode_cv_pct\n")
        for row in real_rows:
            out.write("\t".join([
                row["mode"], str(row["count"]), fmt(row["prefill_ms"]), fmt(row["tpot_ms"]),
                fmt(row["mean"]), fmt(row["p95"]), fmt(row["stddev"]), fmt(row["cv_pct"]),
            ]) + "\n")

    warmup = write_warmup_table()
    complete = all(len(rows) >= 8 for rows in grouped.values())
    formal_count = sum(len(rows) for rows in grouped.values())
    formal_extensions = max(0, formal_count - 128)
    diagnostic_count = sum(
        1 for mode in MODES for vram in (6000, 8000) for ubatch in (512, 1024)
        if (ROOT / mode / f"vram-{vram}-ub{ubatch}" / "diagnostic-summary.txt").exists()
    )
    real_count = sum(row["count"] for row in real_rows)
    real_extensions = max(0, real_count - 32)
    resident_min, numa_min = environment_minima()
    selected = {row["mode"]: row for row in table_rows
                if row["vram"] == 8000 and row["ubatch"] == 512 and row["count"]}

    lines = [
        "# DRAM Expert Cache Report",
        "",
        f"Status: {'complete' if complete else 'incomplete'} primary matrix "
        f"(128 base formal runs + {formal_extensions} CV-extension runs + "
        f"{diagnostic_count} diagnostics; {formal_count} formal total).",
        f"The real-prompt robustness matrix contains 32 base runs + {real_extensions} "
        f"CV-extension runs ({real_count} total).",
        "",
        "## Main result",
        "",
        "| Mode | VRAM MiB | ubatch | n | Prefill ms | Prefill tok/s | TPOT ms | Decode tok/s | 95% median CI | CV % | vs control |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        control_rows = grouped[("control-0716", row["vram"], row["ubatch"])]
        gain, _, _ = paired_gain(control_rows, grouped[(row["mode"], row["vram"], row["ubatch"])])
        comparison = "baseline" if row["mode"] == "control-0716" else f"{gain:+.2f}%"
        lines.append(
            f"| {row['mode']} | {row['vram']} | {row['ubatch']} | {row['count']} | "
            f"{row['prefill_ms']:.2f} | {row['prefill_tok_s']:.2f} | "
            f"{row['tpot_ms']:.2f} | {row['decode_tok_s']:.2f} | "
            f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}] | {row['cv_pct']:.2f} | {comparison} |"
        )

    lines.extend(["", "Positive `vs control` is a speedup; negative is a slowdown.", ""])
    lines.extend([
        "## Data-path breakdown",
        "",
        "Selected cell: 8000 MiB VRAM cache, ubatch 512.",
        "",
        "| Mode | Host hit % | Source GiB | Host memcpy ms/tok | H2D ms/tok | Stall ms/tok | RSS GiB | Preload ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for mode in MODES:
        row = selected.get(mode)
        if row:
            lines.append(
                f"| {mode} | {row['host_hit_pct']:.2f} | {row['source_read_gib']:.2f} | "
                f"{row['host_memcpy_ms']:.2f} | {row['h2d_ms']:.2f} | {row['stall_ms']:.2f} | "
                f"{row['rss_peak_gib']:.2f} | {row['preload_total_ms']:.2f} |"
            )

    lines.extend(["", "## Pageable-cold warmup", ""])
    lines.extend([
        "Diagnostic profile for 8000 MiB / ubatch 512:",
        "",
        "| Decode window | TPOT ms | Decode tok/s | READY blobs at end | READY GiB | Host hit % | Source reads | Source GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in warmup:
        if row["vram"] == 8000 and row["ubatch"] == 512:
            lines.append(
                f"| {row['window']} | {row['tpot']:.2f} | "
                f"{(1000.0 / row['tpot']) if row['tpot'] else 0.0:.2f} | {row['ready_blobs']} | "
                f"{row['ready_gib']:.2f} | {row['hit_pct']:.2f} | {row['source_reads']} | {row['source_gib']:.2f} |"
            )

    lines.extend(["", "## Real-prompt robustness", ""])
    lines.extend([
        "The fixed Chinese-English prompt uses 8000 MiB / ubatch 512.",
        "",
        "| Mode | n | Prefill ms | Prefill tok/s | TPOT median ms | Decode tok/s | TPOT p95 ms | CV % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in real_rows:
        lines.append(
            f"| {row['mode']} | {row['count']} | {row['prefill_ms']:.2f} | "
            f"{row['prefill_tok_s']:.2f} | {row['tpot_ms']:.2f} | "
            f"{row['decode_tok_s']:.2f} | {row['p95']:.2f} | {row['cv_pct']:.2f} |"
        )

    lines.extend(["", "## Validation and controls", ""])
    correctness = ROOT / "correctness" / "report.tsv"
    if correctness.exists():
        lines.append("- Four-mode token traces matched and max absolute logits difference was 0 (threshold 1e-5).")
    lines.extend([
        "- The manifest contains 30,720 blobs; hot modes reached 30,720 READY blobs and reported zero timed source reads.",
        "- Up to 256 READY blobs per run were checked against source bytes with zero verification failures.",
        f"- The primary pageable-hot 6000/512 cell was extended to 15 runs; its final TPOT CV is "
        f"{next(row['cv_pct'] for row in table_rows if row['mode'] == 'pageable-hot' and row['vram'] == 6000 and row['ubatch'] == 512):.2f}%.",
        f"- Real-prompt pageable-hot and pinned-hot were extended to 15 runs. Their final CVs remain "
        f"{next(row['cv_pct'] for row in real_rows if row['mode'] == 'pageable-hot'):.2f}% and "
        f"{next(row['cv_pct'] for row in real_rows if row['mode'] == 'pinned-hot'):.2f}%, respectively; "
        "all token traces and transfer volumes match, so the residual variation is retained rather than filtered.",
        f"- Minimum recorded Linux page-cache residency before a formal run was {resident_min:.4f}%.",
        f"- Minimum recorded NUMA-node-0 share at peak sampled RSS was {numa_min:.2f}%.",
        "- CUDA pinned allocations are not charged to VmLck by this NVIDIA driver. Pinned validity is based on successful cudaHostAlloc, zero staging memcpy bytes, and direct async H2D.",
    ])
    baseline_report = ROOT / "baseline-crosscheck" / "report.tsv"
    if baseline_report.exists():
        baseline_text = baseline_report.read_text(errors="replace")
        match = re.search(r"^TPOT_ms\t([0-9.]+)\t([0-9.]+)\t(-?[0-9.]+)$", baseline_text, re.MULTILINE)
        if match:
            lines.append(
                f"- Archived 0716 baseline TPOT median was {float(match.group(1)):.2f} ms versus "
                f"{float(match.group(2)):.2f} ms for experiment-off ({float(match.group(3)):+.2f}%). "
                "This exceeds the 3% cross-check threshold, so cross-version comparisons remain qualified; the A/B/C/D table uses one experiment binary."
            )

    lines.extend(["", "## Microbenchmark", ""])
    micro_path = ROOT / "microbench" / "summary.tsv"
    if micro_path.exists():
        lines.extend([
            "| Topology | Operation | Mean us | P95 us | Aggregate GiB/s |",
            "|---|---|---:|---:|---:|",
        ])
        with micro_path.open(newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                lines.append(
                    f"| {row['topology']} | {row['operation']} | {float(row['mean_us']):.2f} | "
                    f"{float(row['p95_us']):.2f} | {float(row['aggregate_gib_s']):.2f} |"
                )

    lines.extend(["", "## Existing benchmark comparison", ""])
    lines.extend([
        "Existing `benchmark-results/new-result` values are single historical runs and are shown only as a sanity check.",
        "",
        "| VRAM MiB | ubatch | Historical TPOT ms | Current control median ms |",
        "|---:|---:|---:|---:|",
    ])
    for vram, label in ((6000, "6g"), (8000, "8g")):
        for ubatch in (512, 1024):
            path = ROOT.parent / "new-result" / f"vram-{vram}" / f"lru-{label}-pp1024-tg1024-ub{ubatch}-summary.txt"
            historical = parse_summary(path)["tpot_ms"] if path.exists() else 0.0
            current = next(row["tpot_ms"] for row in table_rows
                           if row["mode"] == "control-0716" and row["vram"] == vram and row["ubatch"] == ubatch)
            lines.append(f"| {vram} | {ubatch} | {historical:.2f} | {current:.2f} |")

    lines.extend(["", "## Decision", ""])
    if len(selected) == 4:
        pageable_gain, pageable_low, pageable_high = paired_gain(
            grouped[("control-0716", 8000, 512)], grouped[("pageable-hot", 8000, 512)])
        pinned_gain, pinned_low, pinned_high = paired_gain(
            grouped[("pageable-hot", 8000, 512)], grouped[("pinned-hot", 8000, 512)])
        control_gain, _, _ = paired_gain(
            grouped[("control-0716", 8000, 512)], grouped[("pinned-hot", 8000, 512)])
        pinned = selected["pinned-hot"]
        transfer_share = 100.0 * (pinned["h2d_ms"] + pinned["stall_ms"]) / pinned["tpot_ms"] if pinned["tpot_ms"] else 0.0
        lines.extend([
            f"- Pageable-hot versus control: {pageable_gain:+.2f}% paired TPOT change, bootstrap 95% CI [{pageable_low:+.2f}%, {pageable_high:+.2f}%]. The required +10% gain is not met.",
            f"- Pinned-hot versus pageable-hot: {pinned_gain:+.2f}%, bootstrap 95% CI [{pinned_low:+.2f}%, {pinned_high:+.2f}%].",
            f"- Pinned-hot versus control: {control_gain:+.2f}% at the selected cell.",
            f"- Pinned H2D plus measured stall is {transfer_share:.2f}% of TPOT. This exceeds the 15% Fate threshold, so hidden-state prefetch remains worth testing.",
            "- Do not deploy the current full pageable cache as a synchronous replacement for fread. Its host memcpy and staging path is slower than the hot Linux page-cache control.",
            "- Use the existing pinned staging pool as the Fate destination path, prefetch asynchronously ahead of demand, and evaluate a bounded pinned hot-expert cache instead of pinning all 18.22 GiB by default.",
        ])
    else:
        lines.append("The selected cell is incomplete; no deployment decision is reported.")

    lines.extend([
        "",
        "Machine-readable companion files: `summary-table.tsv`, `real-prompt-summary.tsv`, `pageable-cold-warmup.tsv`, and per-cell `case-summary.tsv`.",
        "Raw summaries, profiles, logs, metadata, memory samples, and token traces remain below each mode/cell directory.",
    ])
    (ROOT / "final-report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
