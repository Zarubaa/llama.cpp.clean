#!/usr/bin/env python3
"""Validate the prompt manifest without loading the model."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--min-bytes", type=int, default=3000)
    parser.add_argument(
        "--summary-root",
        type=Path,
        help="optional completed-run tree; checks every summary has n_prompt=1024",
    )
    args = parser.parse_args()
    manifest = args.root / "manifest.tsv"
    rows = []
    with manifest.open(encoding="utf-8") as stream:
        header = stream.readline().rstrip("\n").split("\t")
        required = {"prompt_id", "file", "bytes", "sha256", "qwen35_token_count"}
        missing = required - set(header)
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        for line in stream:
            if line.strip():
                rows.append(dict(zip(header, line.rstrip("\n").split("\t"))))

    ids = [row["prompt_id"] for row in rows]
    files = [row["file"] for row in rows]
    if len(rows) != 8 or len(set(ids)) != 8 or len(set(files)) != 8:
        raise ValueError("manifest must contain eight unique prompt ids and files")

    paragraph_digests = set()
    for row in rows:
        path = args.root / row["file"]
        data = path.read_bytes()
        actual_sha = hashlib.sha256(data).hexdigest()
        if len(data) < args.min_bytes:
            raise ValueError(f"{path.name}: only {len(data)} bytes")
        token_count = int(row["qwen35_token_count"])
        if token_count < 1024:
            raise ValueError(f"{path.name}: recorded token count is {token_count}")
        if str(len(data)) != row["bytes"] or actual_sha != row["sha256"]:
            raise ValueError(f"{path.name}: manifest size or SHA256 mismatch")
        text = data.decode("utf-8")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paragraphs) < 7:
            raise ValueError(f"{path.name}: expected at least seven paragraphs")
        normalized = {re.sub(r"\s+", " ", p).strip() for p in paragraphs}
        if len(normalized) != len(paragraphs):
            raise ValueError(f"{path.name}: duplicate paragraphs detected")
        paragraph_digests.update(hashlib.sha256(p.encode()).hexdigest() for p in normalized)

    if args.summary_root:
        summaries = sorted(args.summary_root.glob("**/run-*-summary.txt"))
        if not summaries:
            raise ValueError("--summary-root contains no run summaries")
        for summary in summaries:
            content = summary.read_text(encoding="utf-8", errors="replace")
            if not re.search(r"^n_prompt:\s*1024\b", content, re.MULTILINE):
                raise ValueError(f"{summary}: n_prompt is not 1024")
        print(f"summary_token_check=pass runs={len(summaries)}")

    print(f"manifest_check=pass prompts={len(rows)} unique_paragraphs={len(paragraph_digests)}")
    print("token_count=pass recorded_qwen35_counts_are_at_least_1024")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"manifest_check=fail error={error}", file=sys.stderr)
        raise SystemExit(1)
