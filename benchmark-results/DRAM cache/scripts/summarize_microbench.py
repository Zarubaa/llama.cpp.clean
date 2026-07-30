#!/usr/bin/env python3

import csv
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "microbench"


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


rows = []
for family in ("pageable-copy", "pinned-h2d"):
    for topology in ("local", "remote"):
        path = ROOT / family / f"{topology}.csv"
        if not path.exists():
            continue
        grouped = {}
        with path.open() as source:
            for row in csv.DictReader(source):
                grouped.setdefault(row["operation"], []).append(row)
        for operation, values in sorted(grouped.items()):
            latencies = [float(value["latency_us"]) for value in values]
            total_bytes = sum(int(value["bytes"]) for value in values)
            total_seconds = sum(latencies) / 1e6
            rows.append({
                "family": family,
                "topology": topology,
                "operation": operation,
                "samples": len(values),
                "mean_us": statistics.mean(latencies),
                "p50_us": statistics.median(latencies),
                "p95_us": percentile(latencies, 0.95),
                "bandwidth": total_bytes / (1024 ** 3) / total_seconds if total_seconds else 0.0,
            })

with (ROOT / "summary.tsv").open("w") as out:
    out.write("family\ttopology\toperation\tsamples\tmean_us\tp50_us\tp95_us\taggregate_gib_s\n")
    for row in rows:
        out.write(
            f"{row['family']}\t{row['topology']}\t{row['operation']}\t{row['samples']}\t"
            f"{row['mean_us']:.4f}\t{row['p50_us']:.4f}\t{row['p95_us']:.4f}\t{row['bandwidth']:.4f}\n"
        )
