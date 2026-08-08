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


CASES = (
    "lru-off",
    "paper-s8-rho0",
    "paper-s4-rho0",
    "paper-s4-rho0-shadow",
    "miss-s4-rho0",
)
EXPECTED_LAYERS = 40
EXPECTED_DECODE_REQUESTS = 16
EXPECTED_REQUESTS = 1 + EXPECTED_DECODE_REQUESTS
EXPECTED_PREFILL_ROWS = EXPECTED_LAYERS
EXPECTED_DECODE_ROWS = EXPECTED_LAYERS * EXPECTED_DECODE_REQUESTS
EXPECTED_LOGITS_RECORDS = EXPECTED_REQUESTS
EXPECTED_TOKEN_RECORDS = EXPECTED_REQUESTS
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
SHADOW_EXECUTION_COLUMNS = (
    "route_hash",
    "k_required",
    "k_hit",
    "k_miss",
    "routes_required",
    "routes_hit",
    "routes_persistent",
    "k_empty_admit",
    "k_victim_admit",
    "k_scratch",
    "host_cache_hits",
    "host_cache_misses",
    "host_cache_hit_bytes",
    "host_cache_miss_bytes",
    "h2d_bytes",
    "ssd_bytes",
    "ssd_reads",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path, repo):
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path.resolve())


def artifact_identity(path, repo, required=True):
    path = path.resolve()
    if not path.is_file():
        if required:
            raise RuntimeError(f"artifact does not exist: {path}")
        return {"path": display_path(path, repo), "available": False}

    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"artifact changed while hashing: {path}")
    return {
        "path": display_path(path, repo),
        "available": True,
        "size_bytes": after.st_size,
        "sha256": digest,
    }


def inspect_tokens(path):
    records = []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["repeat", "step", "token"]:
            raise RuntimeError(
                f"invalid token trace header in {path}: {reader.fieldnames!r}"
            )
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise RuntimeError(f"malformed token trace row {line_number} in {path}")
            try:
                records.append(
                    {
                        "repeat": int(row["repeat"]),
                        "step": int(row["step"]),
                        "token": int(row["token"]),
                    }
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"non-integer token trace row {line_number} in {path}"
                ) from exc
    return records


def inspect_profile(path, case):
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if len(reader.fieldnames or []) != 62:
            raise RuntimeError(
                f"{case}: expected 62 CSV columns, got {len(reader.fieldnames or [])}"
            )
        required_columns = {
            "row_type",
            "request_idx",
            "repeat_idx",
            "batch_idx",
            "phase",
            "layer",
            *SERE_COLUMNS,
            *EFFECTIVE_COLUMNS,
            *PREFILL_COLUMNS,
        }
        missing = sorted(required_columns - set(reader.fieldnames or []))
        if missing:
            raise RuntimeError(f"{case}: profile CSV is missing columns: {missing}")
        rows = list(reader)

    malformed_rows = [
        index
        for index, row in enumerate(rows, start=2)
        if None in row or any(value is None for value in row.values())
    ]
    if malformed_rows:
        raise RuntimeError(f"{case}: malformed profile CSV rows: {malformed_rows}")

    layer_rows = [row for row in rows if row["row_type"] == "layer"]
    request_rows = [row for row in rows if row["row_type"] == "request"]
    unexpected_types = sorted(
        {row["row_type"] for row in rows if row["row_type"] not in {"layer", "request"}}
    )
    errors = []
    request_headers = []
    layer_headers = []
    try:
        request_headers = [
            {
                "request_idx": int(row["request_idx"]),
                "repeat_idx": int(row["repeat_idx"]),
                "batch_idx": int(row["batch_idx"]),
                "phase": row["phase"],
                "layer": int(row["layer"]),
            }
            for row in request_rows
        ]
        layer_headers = [
            {
                "request_idx": int(row["request_idx"]),
                "repeat_idx": int(row["repeat_idx"]),
                "batch_idx": int(row["batch_idx"]),
                "phase": row["phase"],
                "layer": int(row["layer"]),
            }
            for row in layer_rows
        ]
    except ValueError as exc:
        errors.append(f"non-integer profile structure field: {exc}")

    if unexpected_types:
        errors.append(f"unexpected row_type values: {unexpected_types}")
    if len(request_headers) != EXPECTED_REQUESTS:
        errors.append(
            f"expected {EXPECTED_REQUESTS} request rows, got {len(request_headers)}"
        )

    if request_headers:
        ordered_requests = sorted(request_headers, key=lambda row: row["request_idx"])
        expected_request_ids = list(range(1, EXPECTED_REQUESTS + 1))
        if [row["request_idx"] for row in ordered_requests] != expected_request_ids:
            errors.append(
                f"request_idx values are not exactly 1..{EXPECTED_REQUESTS}"
            )
        if [row["repeat_idx"] for row in ordered_requests] != [0] * EXPECTED_REQUESTS:
            errors.append("request rows do not all belong to repeat 0")
        if [row["batch_idx"] for row in ordered_requests] != list(range(EXPECTED_REQUESTS)):
            errors.append(
                f"request batch_idx values are not exactly 0..{EXPECTED_REQUESTS - 1}"
            )
        expected_phases = ["prefill"] + ["decode"] * EXPECTED_DECODE_REQUESTS
        if [row["phase"] for row in ordered_requests] != expected_phases:
            errors.append(
                "request phases are not one prefill followed by "
                f"{EXPECTED_DECODE_REQUESTS} decode requests"
            )
        if any(row["layer"] != -1 for row in ordered_requests):
            errors.append("request rows must use layer=-1")

        layers_by_request = {}
        for row in layer_headers:
            layers_by_request.setdefault(row["request_idx"], []).append(row)
        if set(layers_by_request) != set(expected_request_ids):
            errors.append(
                "layer rows do not reference exactly request_idx "
                f"1..{EXPECTED_REQUESTS}"
            )

        request_by_id = {row["request_idx"]: row for row in ordered_requests}
        for request_idx in expected_request_ids:
            group = layers_by_request.get(request_idx, [])
            request = request_by_id.get(request_idx)
            if len(group) != EXPECTED_LAYERS:
                errors.append(
                    f"request {request_idx} expected {EXPECTED_LAYERS} layer rows, got {len(group)}"
                )
                continue
            if sorted(row["layer"] for row in group) != list(range(EXPECTED_LAYERS)):
                errors.append(
                    f"request {request_idx} does not contain layers "
                    f"0..{EXPECTED_LAYERS - 1} exactly once"
                )
            if request and any(
                row["repeat_idx"] != request["repeat_idx"]
                or row["batch_idx"] != request["batch_idx"]
                or row["phase"] != request["phase"]
                for row in group
            ):
                errors.append(
                    f"request {request_idx} layer rows disagree with its request metadata"
                )

    return {
        "layer_rows": layer_rows,
        "request_rows": request_rows,
        "request_headers": request_headers,
        "structure_errors": errors,
    }


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

    sere_line = re.search(
        r"^SERE: .*\bmode=(active|shadow)\b", text, flags=re.MULTILINE
    )
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
        "sere_mode": sere_line.group(1) if sere_line else "off",
    }


def git_value(repo, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def git_bytes(repo, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def git_worktree_identity(repo):
    tracked_diff = git_bytes(repo, "diff", "--binary", "--no-ext-diff", "HEAD", "--")
    status = git_bytes(
        repo, "status", "--porcelain=v1", "--untracked-files=all"
    )
    return {
        "branch": git_value(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git_value(repo, "rev-parse", "HEAD"),
        "tracked_diff_bytes": len(tracked_diff) if tracked_diff is not None else None,
        "tracked_diff_sha256": (
            hashlib.sha256(tracked_diff).hexdigest()
            if tracked_diff is not None
            else "unavailable"
        ),
        "status_porcelain": (
            status.decode("utf-8", errors="replace").splitlines()
            if status is not None
            else ["unavailable"]
        ),
        "status_sha256": (
            hashlib.sha256(status).hexdigest() if status is not None else "unavailable"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--binary", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        parser.error(f"run directory does not exist: {run_root}")

    repo = Path(__file__).resolve().parents[3]
    model = args.model or repo / "models/Qwen3.6-35B-A3B-UD-Q4_K_M.moe.gguf"
    prompt = args.prompt or repo / "benchmark-results/new-result/prompts/normal-long-prompt.zh-en.txt"
    sidecar = args.sidecar or repo / "benchmark-results/sere/mechanism-only/synthetic/similarity.sere"
    binary = args.binary or repo / "build-sere-cuda/bin/llama-moe-bench"
    git_identity = git_worktree_identity(repo)
    all_rows = {}
    report = {
        "schema_version": 4,
        "classification": "mechanism-only-synthetic",
        "branch": git_identity["branch"],
        "commit": git_identity["commit"],
        "git_diff_sha256": git_identity["tracked_diff_sha256"],
        "git": git_identity,
        "artifacts": {
            "model": artifact_identity(model, repo),
            "prompt": artifact_identity(prompt, repo),
            "sidecar": artifact_identity(sidecar, repo),
            "binary": artifact_identity(binary, repo),
            "runner": artifact_identity(
                repo / "benchmark-results/sere/run-mechanism-smoke.sh", repo
            ),
            "validator": artifact_identity(Path(__file__), repo),
            "gpu_snapshot_start": artifact_identity(
                run_root / "gpu-snapshot-start.txt", repo, required=False
            ),
            "gpu_snapshot_end": artifact_identity(
                run_root / "gpu-snapshot-end.txt", repo, required=False
            ),
        },
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

        profile = inspect_profile(required[0], case)
        layer_rows = profile["layer_rows"]
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
        token_records = inspect_tokens(required[2])
        prefill_nonzero = sum(
            float(row[column]) != 0.0 for row in prefill for column in SERE_COLUMNS
        )
        all_sere_nonzero = sum(
            float(row[column]) != 0.0
            for row in layer_rows
            for column in SERE_COLUMNS
        )
        all_rows[case] = layer_rows
        report["cases"][case] = {
            "csv_columns": 62,
            "layer_rows": len(layer_rows),
            "prefill_rows": len(prefill),
            "decode_rows": len(decode),
            "request_rows": len(profile["request_rows"]),
            "request_headers": profile["request_headers"],
            "profile_structure_errors": profile["structure_errors"],
            "prefill_sere_nonzero_fields": prefill_nonzero,
            "all_sere_nonzero_fields": all_sere_nonzero,
            "decode_sere_sums": sums,
            "decode_effective_sums": effective_sums,
            "tokens_sha256": sha256(required[2]),
            "token_record_count": len(token_records),
            "token_record_headers": [
                [record["repeat"], record["step"]] for record in token_records
            ],
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
    checks["gpu_snapshots_recorded"] = all(
        report["artifacts"][name]["available"]
        and report["artifacts"][name]["size_bytes"] > 0
        for name in ("gpu_snapshot_start", "gpu_snapshot_end")
    )
    checks["all_prefill_sere_zero"] = all(
        case["prefill_sere_nonzero_fields"] == 0
        for case in report["cases"].values()
    )
    checks["baseline_all_sere_zero"] = (
        report["cases"]["lru-off"]["all_sere_nonzero_fields"] == 0
    )
    checks["profile_shape_valid"] = all(
        case["layer_rows"] == EXPECTED_PREFILL_ROWS + EXPECTED_DECODE_ROWS
        and case["prefill_rows"] == EXPECTED_PREFILL_ROWS
        and case["decode_rows"] == EXPECTED_DECODE_ROWS
        for case in report["cases"].values()
    )
    checks["profile_request_structure_valid"] = all(
        case["request_rows"] == EXPECTED_REQUESTS
        and not case["profile_structure_errors"]
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
    checks["token_trace_shape_valid"] = all(
        case["token_record_count"] == EXPECTED_TOKEN_RECORDS
        and [header[0] for header in case["token_record_headers"]]
        == [0] * EXPECTED_TOKEN_RECORDS
        and [header[1] for header in case["token_record_headers"]]
        == list(range(EXPECTED_TOKEN_RECORDS))
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
    checks["prompt_used_tokens_expected"] = all(
        case["summary"]["prompt_used_tokens"] == 128
        for case in report["cases"].values()
    )
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
    shadow_case = "paper-s4-rho0-shadow"
    shadow = all_rows[shadow_case]
    checks["shadow_summary_mode"] = (
        report["cases"][shadow_case]["summary"]["sere_mode"] == "shadow"
    )
    checks["non_shadow_summary_modes"] = (
        report["cases"]["lru-off"]["summary"]["sere_mode"] == "off"
        and all(
            report["cases"][case]["summary"]["sere_mode"] == "active"
            for case in ("paper-s8-rho0", "paper-s4-rho0", "miss-s4-rho0")
        )
    )
    checks["shadow_sere_counters_nonzero"] = (
        report["cases"][shadow_case]["all_sere_nonzero_fields"] > 0
        and report["cases"][shadow_case]["decode_sere_sums"][
            "sere_rerouted_routes"
        ]
        > 0
    )
    checks["shadow_token_trace_equal_baseline"] = (
        report["cases"][shadow_case]["tokens_sha256"]
        == report["cases"]["lru-off"]["tokens_sha256"]
    )
    checks["shadow_logits_equal_baseline"] = (
        report["cases"][shadow_case]["logits_sha256"]
        == report["cases"]["lru-off"]["logits_sha256"]
    )
    checks["shadow_execution_accounting_equal_baseline"] = (
        len(base) == len(shadow)
        and all(
            [row[column] for row in base] == [row[column] for row in shadow]
            for column in SHADOW_EXECUTION_COLUMNS
        )
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
    shadow_paper = report["cases"][shadow_case]["decode_sere_sums"]
    checks["shadow_paper_all_secondary_rerouted"] = (
        shadow_paper["sere_secondary_routes"] > 0
        and shadow_paper["sere_secondary_routes"]
        == shadow_paper["sere_rerouted_routes"]
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
        for case in (
            "paper-s8-rho0",
            "paper-s4-rho0",
            shadow_case,
            "miss-s4-rho0",
        )
    )
    checks["rerouted_misses_bounded_by_original_misses"] = all(
        report["cases"][case]["decode_sere_sums"]["sere_rerouted_miss_routes"]
        <= report["cases"][case]["decode_sere_sums"]["sere_original_miss_routes"]
        for case in ("paper-s4-rho0", shadow_case, "miss-s4-rho0")
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
