#!/usr/bin/env python3
"""Create prompt-level and paired summaries for a prompt-aware matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import statistics
from collections import defaultdict
from pathlib import Path


def meta(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def match(text: str, pattern: str, default: float = 0.0) -> float:
    found = re.search(pattern, text, re.MULTILINE)
    return float(found.group(1)) if found else default


def route_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row.get("row_type") == "layer":
                fields = (row.get(key, "") for key in
                          ("request_idx", "repeat_idx", "batch_idx", "token_idx",
                           "phase", "layer", "routes_required", "route_hash"))
                digest.update(("\t".join(fields) + "\n").encode("ascii", "replace"))
    return digest.hexdigest()


def parse_run(prefix: Path) -> dict[str, object] | None:
    summary_path = Path(f"{prefix}-summary.txt")
    metadata_path = Path(f"{prefix}-metadata.txt")
    profile_path = Path(f"{prefix}-profile.csv")
    validation_path = Path(f"{prefix}-validation.txt")
    if not all(path.is_file() and path.stat().st_size for path in
               (summary_path, metadata_path, profile_path, validation_path)):
        return None
    validation = meta(validation_path)
    if validation.get("validation") != "pass":
        return None
    summary = summary_path.read_text(errors="replace")
    metadata = meta(metadata_path)
    prefill = re.search(r"^prefill\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)$", summary, re.MULTILINE)
    decode = re.search(r"^decode\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)$", summary, re.MULTILINE)
    if not prefill or not decode:
        return None
    totals = defaultdict(int)
    with profile_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("row_type") != "layer":
                continue
            for key in ("k_miss", "h2d_bytes", "host_memcpy_bytes", "ssd_bytes", "ssd_reads", "h2d_us", "stall_us"):
                try:
                    totals[key] += int(row.get(key) or 0)
                except ValueError:
                    pass
    return {
        "mode": metadata.get("mode", ""),
        "vram_cache_mb": int(metadata.get("vram_cache_mb", 0)),
        "ubatch": int(metadata.get("ubatch", 0)),
        "prompt_id": metadata.get("prompt_id", ""),
        "repeat": int(metadata.get("repeat", 0)),
        "prompt_sha256": metadata.get("prompt_sha256", ""),
        "prompt_bytes": int(metadata.get("prompt_bytes", 0)),
        "n_prompt": int(prefill.group(1)),
        "n_gen": int(decode.group(1)),
        "prefill_ms": float(prefill.group(2)),
        "prefill_tok_s": float(prefill.group(1)) * 1000.0 / float(prefill.group(2)),
        "decode_ms": float(decode.group(2)),
        "decode_tok_s": float(decode.group(1)) * 1000.0 / float(decode.group(2)),
        "tpot_ms": match(summary, r"^TPOT:\s*([0-9.]+) ms$"),
        "vram_misses": totals["k_miss"],
        "h2d_bytes": totals["h2d_bytes"],
        "host_memcpy_bytes": totals["host_memcpy_bytes"],
        "ssd_bytes": totals["ssd_bytes"],
        "ssd_reads": totals["ssd_reads"],
        "h2d_ms": totals["h2d_us"] / 1000.0,
        "stall_ms": totals["stall_us"] / 1000.0,
        "token_sha256": validation.get("token_sha256", ""),
        "route_trace_sha256": route_digest(profile_path),
        "path": str(prefix),
    }


def average(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def prompt_means(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], row["vram_cache_mb"], row["ubatch"], row["prompt_id"])].append(row)
    result = []
    for key, values in sorted(grouped.items(), key=lambda item: item[0]):
        mode, vram, ubatch, prompt_id = key
        result.append({
            "mode": mode, "vram_cache_mb": vram, "ubatch": ubatch, "prompt_id": prompt_id,
            "repeat_count": len(values),
            "prefill_tok_s": average([float(v["prefill_tok_s"]) for v in values]),
            "decode_tok_s": average([float(v["decode_tok_s"]) for v in values]),
            "tpot_ms": average([float(v["tpot_ms"]) for v in values]),
            "prefill_ms": average([float(v["prefill_ms"]) for v in values]),
            "decode_ms": average([float(v["decode_ms"]) for v in values]),
        })
    return result


def write_tsv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("\t".join(columns) + "\n")
        for row in rows:
            stream.write("\t".join(str(row.get(column, "")) for column in columns) + "\n")


def aggregates(means: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in means:
        grouped[(row["mode"], row["vram_cache_mb"], row["ubatch"])].append(row)
    result = []
    for key, values in sorted(grouped.items(), key=lambda item: item[0]):
        mode, vram, ubatch = key
        record: dict[str, object] = {
            "mode": mode, "vram_cache_mb": vram, "ubatch": ubatch,
            "prompt_count": len(values),
        }
        for metric in ("prefill_tok_s", "decode_tok_s", "tpot_ms"):
            numbers = [float(value[metric]) for value in values]
            record[f"{metric}_mean"] = statistics.mean(numbers)
            record[f"{metric}_median"] = statistics.median(numbers)
            record[f"{metric}_stdev"] = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
            record[f"{metric}_min"] = min(numbers)
            record[f"{metric}_max"] = max(numbers)
        result.append(record)
    return result


def paired_aggregates(deltas: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in deltas:
        grouped[(row["mode"], row["vram_cache_mb"], row["ubatch"])].append(row)
    result = []
    for key, values in sorted(grouped.items(), key=lambda item: item[0]):
        mode, vram, ubatch = key
        record: dict[str, object] = {
            "mode": mode, "vram_cache_mb": vram, "ubatch": ubatch,
            "prompt_count": len(values),
        }
        for metric in ("prefill_tok_s_pct", "decode_tok_s_pct", "tpot_improvement_pct"):
            numbers = [float(value[metric]) for value in values]
            record[f"{metric}_mean"] = statistics.mean(numbers)
            record[f"{metric}_median"] = statistics.median(numbers)
            record[f"{metric}_min"] = min(numbers)
            record[f"{metric}_max"] = max(numbers)
        result.append(record)
    return result


def matrix_errors(rows: list[dict[str, object]]) -> list[str]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["vram_cache_mb"], row["ubatch"], row["prompt_id"], row["repeat"])].append(row)
    errors = []
    for key, values in sorted(grouped.items()):
        if len(values) != 3:
            errors.append(f"{key}: expected 3 modes, found {len(values)}")
            continue
        for field in ("prompt_sha256", "token_sha256", "route_trace_sha256"):
            if len({str(value[field]) for value in values}) != 1:
                errors.append(f"{key}: {field} differs across modes")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()
    summary_suffix = "-summary.txt"
    summaries = sorted(args.result_root.glob("*/vram-*-ub*/prompt-*/repeat-*-summary.txt"))
    prefixes = [Path(str(path)[:-len(summary_suffix)]) for path in summaries]
    rows = [row for prefix in prefixes if (row := parse_run(prefix)) is not None]
    rows.sort(key=lambda row: (row["vram_cache_mb"], row["ubatch"], row["prompt_id"], row["mode"], row["repeat"]))
    raw_columns = ["mode", "vram_cache_mb", "ubatch", "prompt_id", "repeat", "n_prompt", "n_gen",
                   "prompt_bytes", "prompt_sha256", "prefill_ms", "prefill_tok_s", "decode_ms", "decode_tok_s",
                   "tpot_ms", "vram_misses", "h2d_bytes", "host_memcpy_bytes", "ssd_bytes", "ssd_reads",
                   "h2d_ms", "stall_ms", "token_sha256", "route_trace_sha256", "path"]
    write_tsv(args.result_root / "summary-table.tsv", rows, raw_columns)

    means = prompt_means(rows)
    mean_columns = ["mode", "vram_cache_mb", "ubatch", "prompt_id", "repeat_count", "prefill_ms",
                    "prefill_tok_s", "decode_ms", "decode_tok_s", "tpot_ms"]
    write_tsv(args.result_root / "prompt-means.tsv", means, mean_columns)

    control_names = {"control-original", "control"}
    controls = {(r["vram_cache_mb"], r["ubatch"], r["prompt_id"]): r for r in means if r["mode"] in control_names}
    deltas = []
    for row in means:
        control = controls.get((row["vram_cache_mb"], row["ubatch"], row["prompt_id"]))
        if control is None or row["mode"] in control_names:
            continue
        deltas.append({
            "mode": row["mode"], "vram_cache_mb": row["vram_cache_mb"], "ubatch": row["ubatch"], "prompt_id": row["prompt_id"],
            "prefill_tok_s_delta": float(row["prefill_tok_s"]) - float(control["prefill_tok_s"]),
            "decode_tok_s_delta": float(row["decode_tok_s"]) - float(control["decode_tok_s"]),
            "tpot_ms_delta": float(row["tpot_ms"]) - float(control["tpot_ms"]),
            "prefill_tok_s_pct": 100.0 * (float(row["prefill_tok_s"]) / float(control["prefill_tok_s"]) - 1.0),
            "decode_tok_s_pct": 100.0 * (float(row["decode_tok_s"]) / float(control["decode_tok_s"]) - 1.0),
            "tpot_improvement_pct": 100.0 * (float(control["tpot_ms"]) - float(row["tpot_ms"])) / float(control["tpot_ms"]),
        })
    write_tsv(args.result_root / "paired-deltas.tsv", deltas,
              ["mode", "vram_cache_mb", "ubatch", "prompt_id", "prefill_tok_s_delta", "decode_tok_s_delta", "tpot_ms_delta",
               "prefill_tok_s_pct", "decode_tok_s_pct", "tpot_improvement_pct"])

    aggregate_rows = aggregates(means)
    aggregate_columns = ["mode", "vram_cache_mb", "ubatch", "prompt_count"]
    for metric in ("prefill_tok_s", "decode_tok_s", "tpot_ms"):
        aggregate_columns.extend(f"{metric}_{suffix}" for suffix in ("mean", "median", "stdev", "min", "max"))
    write_tsv(args.result_root / "aggregate-table.tsv", aggregate_rows, aggregate_columns)

    paired_rows = paired_aggregates(deltas)
    paired_columns = ["mode", "vram_cache_mb", "ubatch", "prompt_count"]
    for metric in ("prefill_tok_s_pct", "decode_tok_s_pct", "tpot_improvement_pct"):
        paired_columns.extend(f"{metric}_{suffix}" for suffix in ("mean", "median", "min", "max"))
    write_tsv(args.result_root / "paired-summary.tsv", paired_rows, paired_columns)

    checks = matrix_errors(rows)

    lines = [f"# {args.result_root.name} Report", "", f"Validated runs: {len(rows)}", "",
             "Each prompt_id is a distinct workload sample. A repeat is a fresh-process rerun of the same prompt.",
             "All cache modes must use the same prompt_id before their paired delta is interpreted.", "",
             "## Performance", "",
             "The table reports four-prompt workload means and medians. Speeds are explicit in token/s; ranges are prompt-to-prompt variation, not process-repeat noise.", "",
             "| Mode | VRAM MiB | ubatch | prompts | Prefill tok/s mean | median | range | Decode tok/s mean | median | range | TPOT ms mean |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate_rows:
        lines.append(
            f"| {row['mode']} | {row['vram_cache_mb']} | {row['ubatch']} | {row['prompt_count']} | "
            f"{float(row['prefill_tok_s_mean']):.2f} | {float(row['prefill_tok_s_median']):.2f} | "
            f"{float(row['prefill_tok_s_min']):.2f}-{float(row['prefill_tok_s_max']):.2f} | "
            f"{float(row['decode_tok_s_mean']):.2f} | {float(row['decode_tok_s_median']):.2f} | "
            f"{float(row['decode_tok_s_min']):.2f}-{float(row['decode_tok_s_max']):.2f} | "
            f"{float(row['tpot_ms_mean']):.2f} |"
        )
    lines.extend(["", "## Paired Change Versus Control", "",
                  "Positive speed deltas mean faster. Positive TPOT improvement means lower latency. Every delta uses the same prompt_id.", "",
                  "| Mode | VRAM MiB | ubatch | prompts | Prefill speed mean | median | Decode speed mean | median | TPOT improvement mean | median |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in paired_rows:
        lines.append(
            f"| {row['mode']} | {row['vram_cache_mb']} | {row['ubatch']} | {row['prompt_count']} | "
            f"{float(row['prefill_tok_s_pct_mean']):+.2f}% | {float(row['prefill_tok_s_pct_median']):+.2f}% | "
            f"{float(row['decode_tok_s_pct_mean']):+.2f}% | {float(row['decode_tok_s_pct_median']):+.2f}% | "
            f"{float(row['tpot_improvement_pct_mean']):+.2f}% | {float(row['tpot_improvement_pct_median']):+.2f}% |"
        )
    lines.extend(["", "## Statistics", "",
                  "`summary-table.tsv` has one row per process. `prompt-means.tsv` averages process repeats within each prompt. `paired-deltas.tsv` subtracts control for the same prompt, VRAM budget, and ubatch. Prompt-to-prompt standard deviation is workload variation, not scheduler noise.", ""])
    if rows:
        lines.extend(["## Coverage", "", f"Prompt IDs observed: {', '.join(sorted({str(r['prompt_id']) for r in rows}))}",
                      f"Modes observed: {', '.join(sorted({str(r['mode']) for r in rows}))}", ""])
    else:
        lines.extend(["## Coverage", "", "No validated runs yet; this file is a plan artifact until the matrix is executed.", ""])
    lines.extend(["## Validation", ""])
    if checks:
        lines.append(f"Matrix consistency failures: {len(checks)}")
        lines.extend(f"- {error}" for error in checks)
    else:
        lines.append("All prompt groups contain three modes with matching prompt SHA, token trace SHA, and route trace SHA.")
    (args.result_root / "final-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summarized validated_runs={len(rows)} prompt_means={len(means)} paired_deltas={len(deltas)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
