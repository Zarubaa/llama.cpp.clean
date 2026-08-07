#!/usr/bin/env python3
"""Validate and summarize a SERE mechanism smoke run."""

import argparse
import csv
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path


CASES = ("lru-off", "paper-s8-rho0", "paper-s4-rho0", "miss-s4-rho0")
EXPECTED_PREFILL_ROWS = 40
EXPECTED_DECODE_ROWS = 640
EXPECTED_LOGITS_RECORDS = 17
LOGITS_MAGIC = b"MOELOG1\0"
LOGITS_HEADER = struct.Struct("<iii")
SERE_COLUMNS = (
    "sere_secondary_routes",
    "sere_rerouted_routes",
    "sere_original_miss_routes",
    "sere_original_unique_required",
    "sere_original_unique_misses",
    "sere_rerouted_miss_routes",
    "sere_threshold_rejects",
    "sere_similarity_sum",
)
EFFECTIVE_COLUMNS = (
    "k_required",
    "k_miss",
    "routes_required",
    "ssd_bytes",
    "ssd_reads",
    "h2d_bytes",
)
NOOP_COLUMNS = (
    "phase",
    "layer",
    "k_required",
    "k_hit",
    "k_miss",
    "routes_required",
    "routes_hit",
    "route_hash",
    "ssd_bytes",
    "ssd_reads",
    "h2d_bytes",
)
PREFILL_COLUMNS = NOOP_COLUMNS + (
    "routes_persistent",
    "k_empty_admit",
    "k_victim_admit",
    "k_scratch",
    "host_cache_hits",
    "host_cache_misses",
    "host_cache_hit_bytes",
    "host_cache_miss_bytes",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_logits(path):
    records = []
    with path.open("rb") as source:
        magic = source.read(len(LOGITS_MAGIC))
        if magic != LOGITS_MAGIC:
            raise RuntimeError(f"invalid logits magic in {path}: {magic!r}")
        while True:
            header = source.read(LOGITS_HEADER.size)
            if not header:
                break
            if len(header) != LOGITS_HEADER.size:
                raise RuntimeError(f"truncated logits record header in {path}")
            repeat, step, vocab_size = LOGITS_HEADER.unpack(header)
            if repeat < 0 or step < 0 or vocab_size <= 0:
                raise RuntimeError(
                    f"invalid logits record in {path}: "
                    f"repeat={repeat} step={step} vocab={vocab_size}"
                )
            payload = source.read(vocab_size * 4)
            if len(payload) != vocab_size * 4:
                raise RuntimeError(
                    f"truncated logits payload in {path}: "
                    f"repeat={repeat} step={step} vocab={vocab_size}"
                )
            records.append(
                {
                    "repeat": repeat,
                    "step": step,
                    "vocab_size": vocab_size,
                    "sha256": hashlib.sha256(header + payload).hexdigest(),
                }
            )
    if not records:
        raise RuntimeError(f"logits file contains no records: {path}")
    return records


def extract_summary(path):
    text = path.read_text(encoding="utf-8")

    def number(pattern):
        match = re.search(pattern, text, flags=re.MULTILINE)
        if not match:
            raise RuntimeError(f"missing summary field {pattern!r} in {path}")
        return float(match.group(1))

    return {
        "ttft_ms": number(r"^TTFT: ([0-9.]+) ms$"),
        "tpot_ms": number(r"^TPOT: ([0-9.]+) ms$"),
        "decode_profiled_source_gib": number(
            r"^SSD bytes read \(decode\): ([0-9.]+) GB"
        ),
        "page_cache_after_prepare_pct": number(
            r"^Page cache resident pct: .* after_prepare=([0-9.]+)"
        ),
        "prompt_used_tokens": int(number(r"^prompt used tokens: ([0-9]+)$")),
        "prompt_token_hash": re.search(
            r"^prompt token hash: ([0-9a-f]+)$", text, flags=re.MULTILINE
        ).group(1),
    }


def git_value(repo, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        parser.error(f"run directory does not exist: {run_root}")

    repo = Path(__file__).resolve().parents[3]
    all_rows = {}
    report = {
        "schema_version": 2,
        "classification": "mechanism-only-synthetic",
        "branch": git_value(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git_value(repo, "rev-parse", "HEAD"),
        "cases": {},
        "checks": {},
        "observations": {},
    }

    for case in CASES:
        case_dir = run_root / case
        required = [
            case_dir / "profile.csv",
            case_dir / "summary.txt",
            case_dir / "tokens.csv",
            case_dir / "logits.bin",
            case_dir / "run.log",
        ]
        for path in required:
            if not path.is_file():
                raise RuntimeError(f"missing result file: {path}")

        with required[0].open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if len(reader.fieldnames or []) != 62:
                raise RuntimeError(
                    f"{case}: expected 62 CSV columns, got {len(reader.fieldnames or [])}"
                )
            layer_rows = [row for row in reader if row["row_type"] == "layer"]
        prefill = [row for row in layer_rows if row["phase"] == "prefill"]
        decode = [row for row in layer_rows if row["phase"] == "decode"]
        sums = {
            column: sum(float(row[column]) for row in decode)
            for column in SERE_COLUMNS
        }
        effective_sums = {
            column: sum(int(row[column]) for row in decode)
            for column in EFFECTIVE_COLUMNS
        }
        logits_records = inspect_logits(required[3])
        prefill_nonzero = sum(
            float(row[column]) != 0.0 for row in prefill for column in SERE_COLUMNS
        )
        all_rows[case] = layer_rows
        report["cases"][case] = {
            "csv_columns": 62,
            "layer_rows": len(layer_rows),
            "prefill_rows": len(prefill),
            "decode_rows": len(decode),
            "prefill_sere_nonzero_fields": prefill_nonzero,
            "decode_sere_sums": sums,
            "decode_effective_sums": effective_sums,
            "tokens_sha256": sha256(required[2]),
            "logits_sha256": sha256(required[3]),
            "logits_record_count": len(logits_records),
            "logits_record_headers": [
                [record["repeat"], record["step"], record["vocab_size"]]
                for record in logits_records
            ],
            "prefill_logits_sha256": logits_records[0]["sha256"],
            "summary": extract_summary(required[1]),
        }

    base = all_rows["lru-off"]
    noop = all_rows["paper-s8-rho0"]
    checks = report["checks"]
    checks["all_prefill_sere_zero"] = all(
        case["prefill_sere_nonzero_fields"] == 0
        for case in report["cases"].values()
    )
    checks["profile_shape_valid"] = all(
        case["layer_rows"] == EXPECTED_PREFILL_ROWS + EXPECTED_DECODE_ROWS
        and case["prefill_rows"] == EXPECTED_PREFILL_ROWS
        and case["decode_rows"] == EXPECTED_DECODE_ROWS
        for case in report["cases"].values()
    )
    checks["logits_shape_valid"] = all(
        case["logits_record_count"] == EXPECTED_LOGITS_RECORDS
        and [header[0] for header in case["logits_record_headers"]]
        == [0] * EXPECTED_LOGITS_RECORDS
        and [header[1] for header in case["logits_record_headers"]]
        == list(range(EXPECTED_LOGITS_RECORDS))
        and len({header[2] for header in case["logits_record_headers"]}) == 1
        for case in report["cases"].values()
    )
    prompt_signatures = {
        (
            case["summary"]["prompt_used_tokens"],
            case["summary"]["prompt_token_hash"],
        )
        for case in report["cases"].values()
    }
    checks["prompt_tokens_equal"] = len(prompt_signatures) == 1
    checks["all_prefill_route_cache_io_equal"] = all(
        len(base) == len(all_rows[case])
        and all(
            [row[column] for row in base if row["phase"] == "prefill"]
            == [
                row[column]
                for row in all_rows[case]
                if row["phase"] == "prefill"
            ]
            for column in PREFILL_COLUMNS
        )
        for case in CASES[1:]
    )
    checks["all_prefill_logits_equal"] = len(
        {
            case["prefill_logits_sha256"]
            for case in report["cases"].values()
        }
    ) == 1
    checks["baseline_s8_logits_equal"] = (
        report["cases"]["lru-off"]["logits_sha256"]
        == report["cases"]["paper-s8-rho0"]["logits_sha256"]
    )
    checks["baseline_s8_token_trace_equal"] = (
        report["cases"]["lru-off"]["tokens_sha256"]
        == report["cases"]["paper-s8-rho0"]["tokens_sha256"]
    )
    report["observations"]["s4_token_traces_equal_to_baseline"] = all(
        report["cases"][case]["tokens_sha256"]
        == report["cases"]["lru-off"]["tokens_sha256"]
        for case in ("paper-s4-rho0", "miss-s4-rho0")
    )
    checks["baseline_s8_route_cache_io_equal"] = len(base) == len(noop) and all(
        [row[column] for row in base] == [row[column] for row in noop]
        for column in NOOP_COLUMNS
    )
    checks["s8_rerouted_zero"] = (
        report["cases"]["paper-s8-rho0"]["decode_sere_sums"][
            "sere_rerouted_routes"
        ]
        == 0
    )
    paper = report["cases"]["paper-s4-rho0"]["decode_sere_sums"]
    miss = report["cases"]["miss-s4-rho0"]["decode_sere_sums"]
    checks["paper_all_secondary_rerouted"] = (
        paper["sere_secondary_routes"] > 0
        and paper["sere_secondary_routes"] == paper["sere_rerouted_routes"]
    )
    checks["miss_only_original_misses_rerouted"] = (
        miss["sere_rerouted_routes"] > 0
        and miss["sere_rerouted_routes"] == miss["sere_rerouted_miss_routes"]
    )
    checks["cold_prepare_below_one_pct"] = all(
        case["summary"]["page_cache_after_prepare_pct"] <= 1.0
        for case in report["cases"].values()
    )
    baseline_effective = report["cases"]["lru-off"]["decode_effective_sums"]
    checks["s4_effective_misses_below_baseline"] = all(
        report["cases"][case]["decode_effective_sums"]["k_miss"]
        < baseline_effective["k_miss"]
        for case in ("paper-s4-rho0", "miss-s4-rho0")
    )
    checks["s4_profiled_source_bytes_below_baseline"] = all(
        report["cases"][case]["decode_effective_sums"]["ssd_bytes"]
        < baseline_effective["ssd_bytes"]
        for case in ("paper-s4-rho0", "miss-s4-rho0")
    )
    checks["original_unique_route_accounting_consistent"] = all(
        report["cases"][case]["decode_sere_sums"][
            "sere_original_unique_required"
        ]
        == report["cases"][case]["decode_effective_sums"]["routes_required"]
        and report["cases"][case]["decode_sere_sums"][
            "sere_original_unique_misses"
        ]
        == report["cases"][case]["decode_sere_sums"][
            "sere_original_miss_routes"
        ]
        for case in ("paper-s8-rho0", "paper-s4-rho0", "miss-s4-rho0")
    )
    checks["rerouted_misses_bounded_by_original_misses"] = all(
        report["cases"][case]["decode_sere_sums"]["sere_rerouted_miss_routes"]
        <= report["cases"][case]["decode_sere_sums"]["sere_original_miss_routes"]
        for case in ("paper-s4-rho0", "miss-s4-rho0")
    )
    checks["s4_original_effective_accounting_consistent"] = all(
        report["cases"][case]["decode_effective_sums"]["k_required"]
        == report["cases"][case]["decode_sere_sums"][
            "sere_original_unique_required"
        ]
        - report["cases"][case]["decode_sere_sums"]["sere_rerouted_routes"]
        and report["cases"][case]["decode_effective_sums"]["k_miss"]
        == report["cases"][case]["decode_sere_sums"][
            "sere_original_unique_misses"
        ]
        - report["cases"][case]["decode_sere_sums"][
            "sere_rerouted_miss_routes"
        ]
        for case in ("paper-s4-rho0", "miss-s4-rho0")
    )
    checks["decode_source_io_accounting_consistent"] = all(
        case["decode_effective_sums"]["ssd_reads"]
        == 3 * case["decode_effective_sums"]["k_miss"]
        and case["decode_effective_sums"]["h2d_bytes"]
        == case["decode_effective_sums"]["ssd_bytes"]
        for case in report["cases"].values()
    )

    failed = [name for name, passed in checks.items() if not passed]
    report["passed"] = not failed
    report["failed_checks"] = failed
    output = run_root / "validation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    if failed:
        print("failed checks: " + ", ".join(failed))
        return 1
    print("all mechanism checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
