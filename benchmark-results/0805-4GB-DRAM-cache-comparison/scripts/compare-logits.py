#!/usr/bin/env python3

import argparse
import hashlib
import struct
from array import array
from pathlib import Path


MAGIC = b"MOELOG1\0"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def records(path):
    with path.open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"invalid logits magic: {path}")
        while True:
            header = stream.read(12)
            if not header:
                return
            if len(header) != 12:
                raise ValueError(f"truncated logits header: {path}")
            repeat, step, vocab = struct.unpack("=iii", header)
            payload = stream.read(vocab * 4)
            if len(payload) != vocab * 4:
                raise ValueError(f"truncated logits payload: {path}, step={step}")
            values = array("f")
            values.frombytes(payload)
            yield repeat, step, vocab, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--tokens-a", type=Path, required=True)
    parser.add_argument("--tokens-b", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    max_delta = 0.0
    max_location = (-1, -1, -1)
    sum_delta = 0.0
    value_count = 0
    above_threshold = 0
    record_count = 0
    iter_a = iter(records(args.a))
    iter_b = iter(records(args.b))
    while True:
        record_a = next(iter_a, None)
        record_b = next(iter_b, None)
        if record_a is None or record_b is None:
            if record_a is not None or record_b is not None:
                raise ValueError("logits files contain different record counts")
            break
        repeat_a, step_a, vocab_a, values_a = record_a
        repeat_b, step_b, vocab_b, values_b = record_b
        if (repeat_a, step_a, vocab_a) != (repeat_b, step_b, vocab_b):
            raise ValueError(
                f"record header mismatch: {(repeat_a, step_a, vocab_a)} != "
                f"{(repeat_b, step_b, vocab_b)}"
            )
        record_count += 1
        for index, (value_a, value_b) in enumerate(zip(values_a, values_b)):
            delta = abs(value_a - value_b)
            sum_delta += delta
            value_count += 1
            if delta > args.threshold:
                above_threshold += 1
            if delta > max_delta:
                max_delta = delta
                max_location = (repeat_a, step_a, index)

    token_a = digest(args.tokens_a)
    token_b = digest(args.tokens_b)
    status = "pass" if max_delta <= args.threshold and token_a == token_b else "fail"
    lines = [
        f"status={status}",
        f"threshold={args.threshold:.9g}",
        f"records={record_count}",
        f"values={value_count}",
        f"max_abs_delta={max_delta:.9g}",
        f"mean_abs_delta={sum_delta / value_count if value_count else 0.0:.9g}",
        f"values_above_threshold={above_threshold}",
        f"max_delta_repeat={max_location[0]}",
        f"max_delta_step={max_location[1]}",
        f"max_delta_vocab_index={max_location[2]}",
        f"tokens_a_sha256={token_a}",
        f"tokens_b_sha256={token_b}",
        f"token_trace_equal={int(token_a == token_b)}",
    ]
    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
