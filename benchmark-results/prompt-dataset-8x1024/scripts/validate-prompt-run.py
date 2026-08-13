#!/usr/bin/env python3
"""Validate the prompt identity and token boundary of one benchmark run."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--pp", type=int, default=1024)
    parser.add_argument("--tg", type=int, default=1024)
    args = parser.parse_args()

    required = ("summary.txt", "profile.csv", "stdout.log", "stderr.log", "metadata.txt",
                "memory.csv", "tokens.csv", "block-io.tsv")
    errors = [f"missing or empty: {args.prefix}-{suffix}"
              for suffix in required
              if not (args.prefix.parent / f"{args.prefix.name}-{suffix}").is_file()
              or (args.prefix.parent / f"{args.prefix.name}-{suffix}").stat().st_size == 0]
    if errors:
        raise SystemExit("\n".join(errors))

    summary_path = args.prefix.parent / f"{args.prefix.name}-summary.txt"
    metadata_path = args.prefix.parent / f"{args.prefix.name}-metadata.txt"
    summary = summary_path.read_text(errors="replace")
    meta = metadata(metadata_path)
    expected_header = rf"^n_prompt:\s*{args.pp}\s+n_gen:\s*{args.tg}\s+repeats:\s*1$"
    if not re.search(expected_header, summary, re.MULTILINE):
        errors.append(f"summary does not record n_prompt={args.pp}, n_gen={args.tg}, repeats=1")
    if meta.get("prompt_id") != args.prompt_id:
        errors.append(f"prompt_id mismatch: {meta.get('prompt_id')} != {args.prompt_id}")
    actual_sha = sha256(args.prompt_file)
    if meta.get("prompt_sha256") != actual_sha:
        errors.append(f"prompt SHA mismatch: {meta.get('prompt_sha256')} != {actual_sha}")
    if meta.get("prompt_file") != str(args.prompt_file):
        errors.append(f"prompt path mismatch: {meta.get('prompt_file')} != {args.prompt_file}")
    token_rows = args.prefix.parent / f"{args.prefix.name}-tokens.csv"
    token_count = sum(1 for _ in token_rows.open(encoding="utf-8", errors="replace")) - 1
    if token_count != args.pp + 1:
        errors.append(f"token trace has {token_count} data rows; expected {args.pp + 1}")
    validation = "pass" if not errors else "fail"
    validation_path = args.prefix.parent / f"{args.prefix.name}-validation.txt"
    validation_path.write_text(
        f"validation={validation}\n"
        f"prompt_id={args.prompt_id}\n"
        f"prompt_sha256={actual_sha}\n"
        f"token_sha256={sha256(token_rows)}\n"
        f"token_trace_rows={token_count}\n"
        + "".join(f"error={error}\n" for error in errors),
        encoding="utf-8",
    )
    if errors:
        raise SystemExit("\n".join(errors))
    return 0


if __name__ == "__main__":
    main()
