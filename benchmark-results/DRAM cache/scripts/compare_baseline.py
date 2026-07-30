#!/usr/bin/env python3

import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "baseline-crosscheck"


def values(kind, metric):
    result = []
    for path in sorted((ROOT / kind).glob("run-*-summary.txt")):
        text = path.read_text(errors="replace")
        match = re.search(rf"^{metric}: ([0-9.]+) ms$", text, re.MULTILINE)
        if match:
            result.append(float(match.group(1)))
    return result


baseline_tpot = values("baseline", "TPOT")
experiment_tpot = values("experiment-off", "TPOT")
baseline_ttft = values("baseline", "TTFT")
experiment_ttft = values("experiment-off", "TTFT")
if len(baseline_tpot) != 3 or len(experiment_tpot) != 3:
    raise SystemExit("expected three baseline and three experiment runs")

base = statistics.median(baseline_tpot)
experiment = statistics.median(experiment_tpot)
delta = 100.0 * (experiment - base) / base
with (ROOT / "report.tsv").open("w") as out:
    out.write("metric\tbaseline_median\texperiment_off_median\tdelta_pct\n")
    out.write(f"TPOT_ms\t{base:.4f}\t{experiment:.4f}\t{delta:.4f}\n")
    out.write(
        f"TTFT_ms\t{statistics.median(baseline_ttft):.4f}\t"
        f"{statistics.median(experiment_ttft):.4f}\t"
        f"{100.0 * (statistics.median(experiment_ttft) - statistics.median(baseline_ttft)) / statistics.median(baseline_ttft):.4f}\n"
    )
if abs(delta) > 3.0:
    raise SystemExit(f"experiment off TPOT differs from baseline by {delta:.2f}%")
