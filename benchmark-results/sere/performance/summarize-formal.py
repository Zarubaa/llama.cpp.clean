#!/usr/bin/env python3
"""Validate and summarize one corrected SERE performance run."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def value(text: str, pattern: str, *, last: bool = False, default: str = "") -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return default
    item = matches[-1] if last else matches[0]
    return item if isinstance(item, str) else item[0]


def summary_row(run_root: Path, case_dir: Path) -> dict[str, str]:
    summary = (case_dir / "summary.txt").read_text(encoding="utf-8")
    metadata = (case_dir / "metadata.txt").read_text(encoding="utf-8")
    rel = case_dir.relative_to(run_root)
    parts = rel.parts
    if len(parts) != 2 or not parts[0].startswith("vram-"):
        raise ValueError(f"unexpected case directory: {case_dir}")
    vram = parts[0].split("-")[1]
    ubatch = parts[0].split("-ub", 1)[1]
    case = parts[1]

    rows = re.search(r"^profile rows: prefill=(\d+) decode=(\d+)$", summary, re.MULTILINE)
    if rows is None:
        raise ValueError(f"{case_dir}: missing profile row counts")
    prefill_rows, decode_rows = rows.groups()

    rerouted_values = re.findall(r"^  sere_rerouted\s+([0-9.]+) routes/token$", summary, re.MULTILINE)
    original_miss_values = re.findall(r"^  sere_original_miss\s+([0-9.]+) routes/token$", summary, re.MULTILINE)
    rerouted_miss_values = re.findall(r"^  sere_rerouted_miss\s+([0-9.]+) routes/token$", summary, re.MULTILINE)
    similarity_values = re.findall(r"^  sere_avg_similarity\s+([0-9.]+)$", summary, re.MULTILINE)

    return {
        "vram_mb": vram,
        "ubatch": ubatch,
        "case": case,
        "ttft_ms": value(summary, r"^TTFT: ([0-9.]+) ms$"),
        "tpot_ms": value(summary, r"^TPOT: ([0-9.]+) ms$"),
        "total_ms": value(summary, r"^total: ([0-9.]+) ms$"),
        "decode_cache_hit_pct": value(summary, r"^cache hit rate \(decode(?:, effective)?\): ([0-9.]+)%$"),
        "decode_ssd_gb": value(summary, r"^SSD bytes read \(decode\): ([0-9.]+) GB"),
        "h2d_gb": value(summary, r"^Host cache transfer: memcpy=[0-9.]+ GB h2d=([0-9.]+) GB$"),
        "source_total_gb": value(summary, r"^Source bytes read \(total\): ([0-9.]+) GB$"),
        "decode_original_miss_per_token": original_miss_values[-1] if original_miss_values else "",
        "decode_rerouted_per_token": rerouted_values[-1] if rerouted_values else "",
        "decode_rerouted_miss_per_token": rerouted_miss_values[-1] if rerouted_miss_values else "",
        "decode_avg_similarity": similarity_values[-1] if similarity_values else "",
        "vram_peak_gb": value(summary, r"^VRAM peak \(process approx\): ([0-9.]+) GB"),
        "prompt_hash": value(summary, r"^prompt token hash: ([0-9a-f]{16})$"),
        "profile_prefill_rows": prefill_rows,
        "profile_decode_rows": decode_rows,
        "numa_local_pct": value(metadata, r"numa_local_pct=([0-9.]+)"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    case_dirs = sorted(
        p for p in run_root.glob("vram-*-ub*/*")
        if p.is_dir() and (p / "summary.txt").is_file()
    )
    if not case_dirs:
        raise SystemExit(f"no completed case directories under {run_root}")

    rows = [summary_row(run_root, p) for p in case_dirs]
    fields = [
        "vram_mb", "ubatch", "case", "ttft_ms", "tpot_ms", "total_ms",
        "decode_cache_hit_pct", "decode_ssd_gb", "source_total_gb", "h2d_gb",
        "decode_original_miss_per_token", "decode_rerouted_per_token",
        "decode_rerouted_miss_per_token", "decode_avg_similarity", "vram_peak_gb",
        "numa_local_pct", "prompt_hash", "profile_prefill_rows", "profile_decode_rows",
    ]
    with (run_root / "results.tsv").open("w", encoding="utf-8") as out:
        out.write("\t".join(fields) + "\n")
        for row in rows:
            out.write("\t".join(row.get(field, "") for field in fields) + "\n")

    failures: list[str] = []
    hashes = {row["prompt_hash"] for row in rows}
    if len(hashes) != 1:
        failures.append(f"prompt hashes differ: {sorted(hashes)}")
    for row in rows:
        label = f"vram-{row['vram_mb']}-ub{row['ubatch']}/{row['case']}"
        if row["profile_decode_rows"] != "40960":
            failures.append(f"{label}: decode profile rows={row['profile_decode_rows']}")
        if row["case"] == "lru-off" and row["decode_rerouted_per_token"] not in {"", "0.00"}:
            failures.append(f"{label}: baseline rerouted={row['decode_rerouted_per_token']}")
        if row["case"] == "paper-s8-rho0" and row["decode_rerouted_per_token"] != "0.00":
            failures.append(f"{label}: S=8 rerouted={row['decode_rerouted_per_token']}")
        if row["case"] in {"paper-s4-rho0", "miss-s4-rho0"}:
            try:
                if float(row["decode_rerouted_per_token"]) <= 0:
                    failures.append(f"{label}: no active reroutes")
            except ValueError:
                failures.append(f"{label}: missing reroute counter")

    report_lines = [
        f"run_root={run_root}",
        f"cases={len(rows)}",
        f"prompt_hash={next(iter(hashes)) if len(hashes) == 1 else 'MISMATCH'}",
        "classification=mechanism-formal-sized-synthetic-sidecar",
        "passed=" + ("1" if not failures else "0"),
    ]
    report_lines.extend(f"failure={item}" for item in failures)
    (run_root / "validation.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    readme = f"# Formal-sized SERE performance run\n\n"
    readme += f"Run root: `{run_root}`\n\n"
    readme += "The workload follows the corrected single-batch bench protocol: `pp=1024`, `tg=1024`, `repeat=1`, the normal Chinese-English prompt truncated to 1024 used tokens, fresh process per case, LRU, host cache off, and a page-cache warm read outside the measured interval. `TTFT` includes final prefill logits synchronization; `TPOT` is calculated only after all 1024 decode tokens finish. No `logits-bin` output is enabled.\n\n"
    readme += "Cases: exact LRU, paper S=4, cache-aware miss S=4, and paper S=8 no-op. Raw artifacts are kept below each `vram-*/` directory. See `results.tsv` and `validation.txt` for the compact table and gates.\n\n"
    readme += "The sidecar is the deterministic synthetic fixture currently available on this branch. These numbers are formal-sized transfer/timing evidence for the integration mechanism only; they are not a real Q4 similarity, quality, or publishable SERE speedup claim.\n"
    (run_root / "README.md").write_text(readme, encoding="utf-8")

    for row in rows:
        print(
            f"vram={row['vram_mb']} ub={row['ubatch']} case={row['case']} "
            f"TTFT={row['ttft_ms']}ms TPOT={row['tpot_ms']}ms "
            f"decode_ssd={row['decode_ssd_gb']}GB rerouted={row['decode_rerouted_per_token']}"
        )
    print(f"validation={'PASS' if not failures else 'FAIL'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
