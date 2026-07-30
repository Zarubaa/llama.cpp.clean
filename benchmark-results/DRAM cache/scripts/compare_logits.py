#!/usr/bin/env python3

import argparse
import math
import struct
from array import array
from pathlib import Path


def read_logits(path):
    records = []
    with path.open("rb") as source:
        if source.read(8) != b"MOELOG1\0":
            raise ValueError(f"invalid logits header: {path}")
        while True:
            header = source.read(12)
            if not header:
                break
            if len(header) != 12:
                raise ValueError(f"truncated record header: {path}")
            repeat, step, count = struct.unpack("=iii", header)
            payload = source.read(count * 4)
            if len(payload) != count * 4:
                raise ValueError(f"truncated logits record: {path}")
            values = array("f")
            values.frombytes(payload)
            records.append(((repeat, step, count), values))
    return records


def compare(reference, candidate):
    left = read_logits(reference)
    right = read_logits(candidate)
    if len(left) != len(right):
        raise ValueError("record count mismatch")
    maximum = 0.0
    for (left_key, left_values), (right_key, right_values) in zip(left, right):
        if left_key != right_key:
            raise ValueError(f"record key mismatch: {left_key} != {right_key}")
        for a, b in zip(left_values, right_values):
            if not math.isfinite(a) or not math.isfinite(b):
                if a != b:
                    return math.inf
                continue
            maximum = max(maximum, abs(a - b))
    return maximum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=1e-5)
    args = parser.parse_args()

    modes = ["control-0716", "pageable-cold", "pageable-hot", "pinned-hot"]
    reference_logits = args.root / modes[0] / "logits.bin"
    reference_tokens = (args.root / modes[0] / "tokens.csv").read_bytes()
    rows = []
    passed = True
    for mode in modes:
        logits_path = args.root / mode / "logits.bin"
        tokens_equal = (args.root / mode / "tokens.csv").read_bytes() == reference_tokens
        maximum = compare(reference_logits, logits_path)
        mode_passed = tokens_equal and maximum <= args.threshold
        rows.append((mode, tokens_equal, maximum, mode_passed))
        passed = passed and mode_passed

    with (args.root / "report.tsv").open("w") as report:
        report.write("mode\ttokens_equal\tmax_abs_logit_diff\tthreshold\tpassed\n")
        for mode, tokens_equal, maximum, mode_passed in rows:
            report.write(f"{mode}\t{str(tokens_equal).lower()}\t{maximum:.9g}\t{args.threshold:.9g}\t{str(mode_passed).lower()}\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
