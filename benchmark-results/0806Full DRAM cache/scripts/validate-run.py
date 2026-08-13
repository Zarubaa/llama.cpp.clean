#!/usr/bin/env python3

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path


def match(text, pattern, cast=float, default=None):
    found = re.search(pattern, text, re.MULTILINE)
    return cast(found.group(1)) if found else default


def named(text, pattern):
    found = re.search(pattern, text, re.MULTILINE)
    return found.groupdict() if found else None


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(args):
    prefix = Path(args.prefix)
    suffixes = (
        "summary.txt", "profile.csv", "stdout.log", "stderr.log", "metadata.txt",
        "memory.csv", "tokens.csv", "page-cache.txt", "process-io.txt", "block-io.tsv",
    )
    paths = {suffix: Path(f"{prefix}-{suffix}") for suffix in suffixes}
    errors = [f"missing or empty: {path}" for path in paths.values()
              if not path.exists() or path.stat().st_size == 0]
    if errors:
        return errors, {}

    summary = paths["summary.txt"].read_text(errors="replace")
    metadata = paths["metadata.txt"].read_text(errors="replace")
    if match(metadata, r"^exit_code=(\d+)$", int) != 0:
        errors.append("benchmark exit code is not zero")
    gpu_delta = match(metadata, r"^gpu_used_delta_mib=(-?\d+)$", int)
    if gpu_delta is None or abs(gpu_delta) > 64:
        errors.append(f"GPU memory baseline delta is invalid: {gpu_delta} MiB")
    if not re.search(r"^n_prompt: 1024  n_gen: 1024  repeats: 1$", summary, re.MULTILINE):
        errors.append("actual prompt/generation sizes are not 1024/1024")

    policy = match(summary, r"^Page cache policy: (\S+)$", str)
    if policy != args.policy:
        errors.append(f"page-cache policy mismatch: expected={args.policy} actual={policy}")
    if not re.search(
        r"^Page cache sample valid: before=1 after_prepare=1 after_model_load=1 after_prefill=1 after_decode=1$",
        summary, re.MULTILINE,
    ):
        errors.append("one or more page-cache samples are invalid")
    if not re.search(
        r"^Process IO sample valid: start=1 after_prepare=1 after_model_load=1 after_prefill=1 after_decode=1$",
        summary, re.MULTILINE,
    ):
        errors.append("one or more process-I/O samples are invalid")

    residency = named(
        summary,
        r"^Page cache resident pct: before=(?P<before>[0-9.]+) after_prepare=(?P<after>[0-9.]+) "
        r"after_model_load=(?P<model>[0-9.]+) after_prefill=(?P<prefill>[0-9.]+) after_decode=(?P<decode>[0-9.]+)$",
    )
    prepare = named(summary, r"^Page cache prepare: time=(?P<ms>[0-9.]+) ms attempts=(?P<attempts>\d+) bytes_read=(?P<bytes>\d+)$")
    process_io = named(
        summary,
        r"^Process IO read_bytes: prepare=(?P<prepare>\d+) model_init=(?P<model>\d+) "
        r"prefill=(?P<prefill>\d+) decode=(?P<decode>\d+) total=(?P<total>\d+)$",
    )
    if not residency or not prepare or not process_io:
        errors.append("missing page-cache or process-I/O metrics")
    elif args.policy == "hot":
        if float(residency["after"]) < 99.0:
            errors.append(f"hot residency below 99%: {residency['after']}%")
        # Pinned/all populates the application cache during model init. Kernel
        # reclaim may therefore cause some GGUF reads while the 18 GiB pinned
        # allocation is being built, but those reads are preload cost rather
        # than timed inference I/O. Keep reporting them and require the
        # prefill/decode interval itself to remain hot. Page-cache-only runs
        # still validate the complete post-prepare interval.
        io_keys = ("prefill", "decode") if args.mode == "pinned-memory-hot" else ("model", "prefill", "decode")
        checked_reads = sum(int(process_io[key]) for key in io_keys)
        if checked_reads > args.model_size // 100:
            interval = "timed inference" if args.mode == "pinned-memory-hot" else "post-prepare"
            errors.append(f"hot {interval} physical reads exceed 1%: {checked_reads}")
    else:
        if int(prepare["attempts"]) != 0 or int(prepare["bytes"]) != 0:
            errors.append("natural control performed page-cache preparation")

    expected_host = "pinned" if args.mode == "pinned-memory-hot" else "off"
    expected_preload = "all" if args.mode == "pinned-memory-hot" else "none"
    host = named(summary, r"^Host cache: mode=(?P<mode>\S+) preload=(?P<preload>\S+)")
    if not host or host["mode"] != expected_host or host["preload"] != expected_preload:
        errors.append(f"Host cache mismatch: expected={expected_host}/{expected_preload} actual={host}")

    layer_rows = []
    with paths["profile.csv"].open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"route_hash", "ssd_bytes", "ssd_reads", "host_memcpy_bytes", "h2d_bytes"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            errors.append("profile is missing required exact counters")
        else:
            field_count = len(reader.fieldnames)
            for line_number, row in enumerate(reader, 2):
                if None in row or len(row) != field_count:
                    errors.append(f"profile column mismatch at line {line_number}")
                    break
                if row["row_type"] == "layer":
                    layer_rows.append(row)
    if not layer_rows:
        errors.append("profile contains no layer rows")

    route_digest = hashlib.sha256()
    totals = {key: 0 for key in ("vram_misses", "h2d_bytes", "host_memcpy_bytes", "ssd_bytes", "ssd_reads")}
    for row in layer_rows:
        trace = ",".join(row[key] for key in (
            "request_idx", "repeat_idx", "batch_idx", "token_idx", "phase",
            "layer", "routes_required", "route_hash",
        ))
        route_digest.update(trace.encode("ascii") + b"\n")
        totals["vram_misses"] += int(row["k_miss"] or 0)
        for key in ("h2d_bytes", "host_memcpy_bytes", "ssd_bytes", "ssd_reads"):
            totals[key] += int(row[key] or 0)

    if args.mode == "pinned-memory-hot":
        ready = named(summary, r"^Host cache ready: blobs=(?P<ready>\d+)/(?P<total>\d+)")
        verified = named(summary, r"^Host cache verification: verified=(?P<count>\d+) failures=(?P<failures>\d+)$")
        if not ready or (int(ready["ready"]), int(ready["total"])) != (30720, 30720):
            errors.append(f"pinned READY blobs invalid: {ready}")
        if not verified or int(verified["failures"]) != 0:
            errors.append(f"pinned verification invalid: {verified}")
        if totals["ssd_reads"] != 0 or totals["ssd_bytes"] != 0:
            errors.append(f"pinned timed source I/O is nonzero: reads={totals['ssd_reads']} bytes={totals['ssd_bytes']}")
        if totals["host_memcpy_bytes"] != 0:
            errors.append(f"pinned staging memcpy is nonzero: {totals['host_memcpy_bytes']}")

    with paths["tokens.csv"].open(newline="") as stream:
        token_rows = list(csv.DictReader(stream))
    if len(token_rows) != 1025:
        errors.append(f"token trace has {len(token_rows)} rows, expected 1025")

    metrics = {
        "validation": "pass" if not errors else "fail",
        "token_sha256": sha256(paths["tokens.csv"]),
        "route_trace_sha256": route_digest.hexdigest(),
        "profile_layer_rows": len(layer_rows),
        **totals,
    }
    return errors, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--policy", choices=("natural", "hot"), required=True)
    parser.add_argument("--model-size", type=int, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    errors, metrics = validate(args)
    if not args.quiet:
        lines = [f"{key}={value}" for key, value in metrics.items()]
        lines.extend(f"error={error}" for error in errors)
        Path(f"{args.prefix}-validation.txt").write_text("\n".join(lines) + "\n")
    if errors:
        if not args.quiet:
            for error in errors:
                print(f"validation error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
