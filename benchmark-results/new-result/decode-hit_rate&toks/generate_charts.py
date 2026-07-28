#!/usr/bin/env python3
import csv
import html
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TSV = ROOT / "decode_hit_rate_toks.tsv"
PLOTS = ROOT / "plots"


COLORS = {
    6000: "#4c78a8",
    8000: "#f58518",
    10000: "#54a24b",
}

UB_COLORS = {
    8: "#7f7f7f",
    128: "#4c78a8",
    256: "#f58518",
    512: "#54a24b",
    1024: "#e45756",
}


def esc(value):
    return html.escape(str(value), quote=True)


def read_rows():
    with TSV.open(newline="") as f:
        rows = []
        for row in csv.DictReader(f, delimiter="\t"):
            item = {}
            for k, v in row.items():
                if k in {"summary"}:
                    item[k] = v
                elif k in {"vram_mb", "ub", "effective_ub", "decode_rows"}:
                    item[k] = int(float(v))
                else:
                    item[k] = float(v)
            rows.append(item)
    rows.sort(key=lambda r: (r["vram_mb"], r["ub"]))
    return rows


def save_svg(path, width, height, body):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<style>\n'
        'text { font-family: Arial, Helvetica, sans-serif; fill: #1f2933; }\n'
        '.title { font-size: 20px; font-weight: 700; }\n'
        '.subtitle { font-size: 12px; fill: #52606d; }\n'
        '.axis { stroke: #52606d; stroke-width: 1; }\n'
        '.grid { stroke: #e4e7eb; stroke-width: 1; }\n'
        '.tick { font-size: 11px; fill: #52606d; }\n'
        '.label { font-size: 12px; fill: #323f4b; }\n'
        '.legend { font-size: 12px; fill: #323f4b; }\n'
        '.point-label { font-size: 10px; fill: #1f2933; }\n'
        '</style>\n'
        f"{body}\n</svg>\n"
    )
    path.write_text(svg)


def nice_ticks(lo, hi, count=5):
    if lo == hi:
        return [lo]
    step = (hi - lo) / max(count - 1, 1)
    return [lo + i * step for i in range(count)]


def plot_scatter(rows):
    width, height = 920, 560
    ml, mr, mt, mb = 82, 180, 72, 76
    x0, x1 = ml, width - mr
    y0, y1 = height - mb, mt
    xs = [r["decode_hit_rate_pct"] for r in rows]
    ys = [r["decode_tok_s_calc"] for r in rows]
    xmin, xmax = math.floor(min(xs) * 10) / 10 - 0.2, math.ceil(max(xs) * 10) / 10 + 0.2
    ymin, ymax = math.floor(min(ys) / 2) * 2 - 1, math.ceil(max(ys) / 2) * 2 + 1

    def sx(x):
        return x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)

    def sy(y):
        return y0 - (y - ymin) / (ymax - ymin) * (y0 - y1)

    parts = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="34" class="title">Decode hit rate vs throughput</text>',
        '<text x="40" y="54" class="subtitle">Higher hit rate generally reduces SSD/H2D traffic and improves decode tok/s.</text>',
    ]
    for t in nice_ticks(xmin, xmax, 6):
        x = sx(t)
        parts.append(f'<line x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y0}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{y0 + 22}" text-anchor="middle" class="tick">{t:.1f}%</text>')
    for t in nice_ticks(ymin, ymax, 6):
        y = sy(t)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{t:.0f}</text>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="axis"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="axis"/>')
    parts.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{height - 24}" text-anchor="middle" class="label">decode cache hit rate</text>')
    parts.append(f'<text x="22" y="{(y0 + y1) / 2:.1f}" text-anchor="middle" class="label" transform="rotate(-90 22 {(y0 + y1) / 2:.1f})">decode tokens/s</text>')

    for r in rows:
        x, y = sx(r["decode_hit_rate_pct"]), sy(r["decode_tok_s_calc"])
        color = COLORS[r["vram_mb"]]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="#1f2933" stroke-width="0.8"/>')
        parts.append(f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" class="point-label">ub{r["ub"]}</text>')

    lx = width - mr + 34
    parts.append(f'<text x="{lx}" y="{mt + 8}" class="legend" font-weight="700">VRAM cap</text>')
    for i, vram in enumerate(sorted(COLORS)):
        y = mt + 32 + i * 24
        parts.append(f'<rect x="{lx}" y="{y - 10}" width="14" height="14" fill="{COLORS[vram]}"/>')
        parts.append(f'<text x="{lx + 22}" y="{y + 1}" class="legend">{vram} MB</text>')

    save_svg(PLOTS / "hit_rate_vs_decode_toks.svg", width, height, "\n".join(parts))


def plot_vram_scaling(rows):
    width, height = 980, 560
    mt, mb = 76, 70
    panel_w, gap = 390, 90
    left_x, right_x = 78, 78 + panel_w + gap
    y_top, y_bot = mt, height - mb
    vrams = sorted({r["vram_mb"] for r in rows})
    ubs = sorted({r["ub"] for r in rows})
    lookup = {(r["vram_mb"], r["ub"]): r for r in rows}

    def panel(parts, x_start, title, metric, y_label, y_min=None, y_max=None):
        vals = [r[metric] for r in rows]
        lo = min(vals) if y_min is None else y_min
        hi = max(vals) if y_max is None else y_max
        lo -= (hi - lo) * 0.08
        hi += (hi - lo) * 0.08

        def sx(v):
            return x_start + (v - min(vrams)) / (max(vrams) - min(vrams)) * panel_w

        def sy(v):
            return y_bot - (v - lo) / (hi - lo) * (y_bot - y_top)

        parts.append(f'<text x="{x_start}" y="{mt - 24}" class="label" font-weight="700">{esc(title)}</text>')
        for t in nice_ticks(lo, hi, 5):
            y = sy(t)
            parts.append(f'<line x1="{x_start}" y1="{y:.1f}" x2="{x_start + panel_w}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{x_start - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{t:.1f}</text>')
        for v in vrams:
            x = sx(v)
            parts.append(f'<line x1="{x:.1f}" y1="{y_top}" x2="{x:.1f}" y2="{y_bot}" class="grid"/>')
            parts.append(f'<text x="{x:.1f}" y="{y_bot + 22}" text-anchor="middle" class="tick">{v // 1000}G</text>')
        parts.append(f'<line x1="{x_start}" y1="{y_bot}" x2="{x_start + panel_w}" y2="{y_bot}" class="axis"/>')
        parts.append(f'<line x1="{x_start}" y1="{y_bot}" x2="{x_start}" y2="{y_top}" class="axis"/>')
        parts.append(f'<text x="{x_start + panel_w / 2:.1f}" y="{height - 24}" text-anchor="middle" class="label">VRAM cap</text>')
        parts.append(f'<text x="{x_start - 55}" y="{(y_top + y_bot) / 2:.1f}" text-anchor="middle" class="label" transform="rotate(-90 {x_start - 55} {(y_top + y_bot) / 2:.1f})">{esc(y_label)}</text>')
        for ub in ubs:
            pts = [(v, lookup[(v, ub)][metric]) for v in vrams if (v, ub) in lookup]
            if len(pts) < 2:
                continue
            color = UB_COLORS.get(ub, "#000000")
            d = " ".join(f'{sx(v):.1f},{sy(val):.1f}' for v, val in pts)
            parts.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2.4"/>')
            for v, val in pts:
                parts.append(f'<circle cx="{sx(v):.1f}" cy="{sy(val):.1f}" r="4.5" fill="{color}"/>')

    parts = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="34" class="title">VRAM scaling by ubatch</text>',
        '<text x="40" y="54" class="subtitle">Lines compare the same ub while increasing the MoE cache VRAM cap.</text>',
    ]
    panel(parts, left_x, "Decode hit rate", "decode_hit_rate_pct", "hit rate (%)")
    panel(parts, right_x, "Decode throughput", "decode_tok_s_calc", "tokens/s")
    lx = width - 126
    parts.append(f'<text x="{lx}" y="{mt - 24}" class="legend" font-weight="700">ub</text>')
    for i, ub in enumerate([128, 256, 512, 1024, 8]):
        y = mt + i * 24
        color = UB_COLORS[ub]
        parts.append(f'<line x1="{lx}" y1="{y}" x2="{lx + 18}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 26}" y="{y + 4}" class="legend">{ub}</text>')
    save_svg(PLOTS / "vram_scaling_by_ub.svg", width, height, "\n".join(parts))


def plot_hit_vs_ssd(rows):
    width, height = 900, 540
    ml, mr, mt, mb = 82, 160, 70, 76
    x0, x1 = ml, width - mr
    y0, y1 = height - mb, mt
    xs = [r["decode_hit_rate_pct"] for r in rows]
    ys = [r["decode_ssd_read_gb"] for r in rows]
    xmin, xmax = min(xs) - 0.3, max(xs) + 0.3
    ymin, ymax = min(ys) - 3, max(ys) + 3

    def sx(x):
        return x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)

    def sy(y):
        return y0 - (y - ymin) / (ymax - ymin) * (y0 - y1)

    parts = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="34" class="title">Hit rate vs SSD traffic</text>',
        '<text x="40" y="54" class="subtitle">The near-linear inverse relation is expected because each miss reads expert weights from SSD.</text>',
    ]
    for t in nice_ticks(xmin, xmax, 6):
        x = sx(t)
        parts.append(f'<line x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y0}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{y0 + 22}" text-anchor="middle" class="tick">{t:.1f}%</text>')
    for t in nice_ticks(ymin, ymax, 6):
        y = sy(t)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{t:.0f}</text>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="axis"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="axis"/>')
    parts.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{height - 24}" text-anchor="middle" class="label">decode cache hit rate</text>')
    parts.append(f'<text x="22" y="{(y0 + y1) / 2:.1f}" text-anchor="middle" class="label" transform="rotate(-90 22 {(y0 + y1) / 2:.1f})">decode SSD read (GB)</text>')
    for r in rows:
        x, y = sx(r["decode_hit_rate_pct"]), sy(r["decode_ssd_read_gb"])
        color = COLORS[r["vram_mb"]]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="#1f2933" stroke-width="0.8"/>')
        parts.append(f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" class="point-label">{r["vram_mb"] // 1000}G ub{r["ub"]}</text>')
    save_svg(PLOTS / "hit_rate_vs_ssd_read.svg", width, height, "\n".join(parts))


def plot_component_bars(rows):
    width, height = 1120, 620
    ml, mr, mt, mb = 86, 42, 88, 132
    x0, x1 = ml, width - mr
    y0, y1 = height - mb, mt
    selected_ubs = [128, 512, 1024]
    data = [r for r in rows if r["ub"] in selected_ubs]
    data.sort(key=lambda r: (r["vram_mb"], r["ub"]))
    metrics = [
        ("decode_ssd_read_ms_per_token", "SSD read", "#4c78a8"),
        ("decode_h2d_ms_per_token", "H2D", "#f58518"),
        ("decode_gpu_compute_ms_per_token", "GPU compute", "#54a24b"),
        ("decode_callback_wall_ms_per_token", "callback wall", "#e45756"),
    ]
    ymax = max(max(r[m[0]] for m in metrics) for r in data) + 2

    def sy(v):
        return y0 - v / ymax * (y0 - y1)

    parts = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="34" class="title">Decode time components</text>',
        '<text x="40" y="54" class="subtitle">Grouped bars are measured component times per token; components overlap and should not be summed.</text>',
    ]
    for t in nice_ticks(0, ymax, 6):
        y = sy(t)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{t:.0f}</text>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="axis"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="axis"/>')
    group_gap = 18
    group_w = (x1 - x0 - group_gap * (len(data) - 1)) / len(data)
    bar_gap = 3
    bar_w = (group_w - bar_gap * (len(metrics) - 1)) / len(metrics)
    for i, r in enumerate(data):
        gx = x0 + i * (group_w + group_gap)
        for j, (metric, name, color) in enumerate(metrics):
            val = r[metric]
            bx = gx + j * (bar_w + bar_gap)
            by = sy(val)
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{y0 - by:.1f}" fill="{color}"/>')
        tx = gx + group_w / 2
        parts.append(f'<text x="{tx:.1f}" y="{y0 + 18}" text-anchor="middle" class="tick">{r["vram_mb"] // 1000}G</text>')
        parts.append(f'<text x="{tx:.1f}" y="{y0 + 34}" text-anchor="middle" class="tick">ub{r["ub"]}</text>')
        total_y = sy(r["decode_ms_per_token"])
        parts.append(f'<line x1="{gx:.1f}" y1="{total_y:.1f}" x2="{gx + group_w:.1f}" y2="{total_y:.1f}" stroke="#111827" stroke-width="1.8" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="24" y="{(y0 + y1) / 2:.1f}" text-anchor="middle" class="label" transform="rotate(-90 24 {(y0 + y1) / 2:.1f})">ms/token</text>')
    lx = x0
    ly = height - 66
    for i, (_, name, color) in enumerate(metrics):
        x = lx + i * 150
        parts.append(f'<rect x="{x}" y="{ly - 12}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{x + 22}" y="{ly}" class="legend">{esc(name)}</text>')
    parts.append(f'<line x1="{lx + 640}" y1="{ly - 5}" x2="{lx + 670}" y2="{ly - 5}" stroke="#111827" stroke-width="1.8" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{lx + 678}" y="{ly}" class="legend">decode total ms/token</text>')
    save_svg(PLOTS / "decode_time_components.svg", width, height, "\n".join(parts))


def plot_heatmap(rows):
    width, height = 760, 430
    ml, mt = 120, 90
    cell_w, cell_h = 112, 66
    ubs = [8, 128, 256, 512, 1024]
    vrams = [6000, 8000, 10000]
    lookup = {(r["vram_mb"], r["ub"]): r for r in rows}
    vals = [r["decode_tok_s_calc"] for r in rows]
    vmin, vmax = min(vals), max(vals)

    def color(v):
        if v is None:
            return "#f3f4f6"
        t = (v - vmin) / (vmax - vmin)
        # Light blue to dark green.
        r = int(230 - t * 150)
        g = int(242 - t * 70)
        b = int(255 - t * 150)
        return f"#{r:02x}{g:02x}{b:02x}"

    parts = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="34" class="title">Decode throughput heatmap</text>',
        '<text x="40" y="54" class="subtitle">Cell value is calculated decode tokens/s.</text>',
    ]
    for j, ub in enumerate(ubs):
        x = ml + j * cell_w + cell_w / 2
        parts.append(f'<text x="{x:.1f}" y="{mt - 20}" text-anchor="middle" class="label">ub{ub}</text>')
    for i, vram in enumerate(vrams):
        y = mt + i * cell_h + cell_h / 2
        parts.append(f'<text x="{ml - 18}" y="{y + 5:.1f}" text-anchor="end" class="label">{vram // 1000}G</text>')
        for j, ub in enumerate(ubs):
            x = ml + j * cell_w
            y0 = mt + i * cell_h
            r = lookup.get((vram, ub))
            v = None if r is None else r["decode_tok_s_calc"]
            parts.append(f'<rect x="{x}" y="{y0}" width="{cell_w - 3}" height="{cell_h - 3}" rx="3" fill="{color(v)}" stroke="#d9e2ec"/>')
            label = "N/A" if v is None else f"{v:.1f}"
            parts.append(f'<text x="{x + cell_w / 2:.1f}" y="{y0 + 31:.1f}" text-anchor="middle" class="label" font-weight="700">{label}</text>')
            if r is not None:
                parts.append(f'<text x="{x + cell_w / 2:.1f}" y="{y0 + 49:.1f}" text-anchor="middle" class="tick">hit {r["decode_hit_rate_pct"]:.1f}%</text>')
    parts.append(f'<text x="{ml + len(ubs) * cell_w / 2:.1f}" y="{height - 38}" text-anchor="middle" class="label">decode tokens/s by VRAM cap and ubatch</text>')
    save_svg(PLOTS / "decode_toks_heatmap.svg", width, height, "\n".join(parts))


def plot_ub1024_focus(rows):
    width, height = 1060, 560
    ub_rows = [r for r in rows if r["ub"] == 1024]
    ub_rows.sort(key=lambda r: r["vram_mb"])

    left = {
        "x0": 82,
        "x1": 520,
        "y0": 458,
        "y1": 88,
    }
    right = {
        "x0": 670,
        "x1": 1000,
        "y0": 458,
        "y1": 88,
    }
    vrams = [r["vram_mb"] for r in ub_rows]
    hits = [r["decode_hit_rate_pct"] for r in ub_rows]
    toks = [r["decode_tok_s_calc"] for r in ub_rows]

    hit_min, hit_max = min(hits) - 0.4, max(hits) + 0.4
    tok_min, tok_max = min(toks) - 2.0, max(toks) + 2.0

    def lx(v):
        return left["x0"] + (v - min(vrams)) / (max(vrams) - min(vrams)) * (left["x1"] - left["x0"])

    def ly_hit(v):
        return left["y0"] - (v - hit_min) / (hit_max - hit_min) * (left["y0"] - left["y1"])

    def ly_tok(v):
        return left["y0"] - (v - tok_min) / (tok_max - tok_min) * (left["y0"] - left["y1"])

    def rx_hit(v):
        return right["x0"] + (v - hit_min) / (hit_max - hit_min) * (right["x1"] - right["x0"])

    def ry_tok(v):
        return right["y0"] - (v - tok_min) / (tok_max - tok_min) * (right["y0"] - right["y1"])

    parts = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="34" class="title">ub=1024: VRAM cap, hit rate, and decode speed</text>',
        '<text x="40" y="54" class="subtitle">Fixed pp=1024, tg=1024, predictor=lru, single GPU. Speed is calculated from decode ms/token.</text>',
    ]

    # Left panel: VRAM on x-axis, hit and speed on two y scales.
    parts.append(f'<text x="{left["x0"]}" y="{left["y1"] - 28}" class="label" font-weight="700">Scaling with VRAM cap</text>')
    for v in vrams:
        x = lx(v)
        parts.append(f'<line x1="{x:.1f}" y1="{left["y1"]}" x2="{x:.1f}" y2="{left["y0"]}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{left["y0"] + 24}" text-anchor="middle" class="tick">{v // 1000}G</text>')
    for t in nice_ticks(hit_min, hit_max, 5):
        y = ly_hit(t)
        parts.append(f'<line x1="{left["x0"]}" y1="{y:.1f}" x2="{left["x1"]}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left["x0"] - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{t:.1f}%</text>')
    for t in nice_ticks(tok_min, tok_max, 5):
        y = ly_tok(t)
        parts.append(f'<text x="{left["x1"] + 10}" y="{y + 4:.1f}" text-anchor="start" class="tick">{t:.0f}</text>')
    parts.append(f'<line x1="{left["x0"]}" y1="{left["y0"]}" x2="{left["x1"]}" y2="{left["y0"]}" class="axis"/>')
    parts.append(f'<line x1="{left["x0"]}" y1="{left["y0"]}" x2="{left["x0"]}" y2="{left["y1"]}" class="axis"/>')
    parts.append(f'<line x1="{left["x1"]}" y1="{left["y0"]}" x2="{left["x1"]}" y2="{left["y1"]}" class="axis"/>')
    parts.append(f'<text x="{(left["x0"] + left["x1"]) / 2:.1f}" y="{height - 38}" text-anchor="middle" class="label">MoE cache VRAM cap</text>')
    parts.append(f'<text x="22" y="{(left["y0"] + left["y1"]) / 2:.1f}" text-anchor="middle" class="label" transform="rotate(-90 22 {(left["y0"] + left["y1"]) / 2:.1f})">decode hit rate</text>')
    parts.append(f'<text x="{left["x1"] + 56}" y="{(left["y0"] + left["y1"]) / 2:.1f}" text-anchor="middle" class="label" transform="rotate(90 {left["x1"] + 56} {(left["y0"] + left["y1"]) / 2:.1f})">decode tok/s</text>')

    hit_line = " ".join(f'{lx(r["vram_mb"]):.1f},{ly_hit(r["decode_hit_rate_pct"]):.1f}' for r in ub_rows)
    tok_line = " ".join(f'{lx(r["vram_mb"]):.1f},{ly_tok(r["decode_tok_s_calc"]):.1f}' for r in ub_rows)
    parts.append(f'<polyline points="{hit_line}" fill="none" stroke="#4c78a8" stroke-width="3"/>')
    parts.append(f'<polyline points="{tok_line}" fill="none" stroke="#e45756" stroke-width="3"/>')
    for r in ub_rows:
        x = lx(r["vram_mb"])
        yh = ly_hit(r["decode_hit_rate_pct"])
        yt = ly_tok(r["decode_tok_s_calc"])
        parts.append(f'<circle cx="{x:.1f}" cy="{yh:.1f}" r="5.5" fill="#4c78a8"/>')
        parts.append(f'<text x="{x:.1f}" y="{yh - 10:.1f}" text-anchor="middle" class="point-label">{r["decode_hit_rate_pct"]:.1f}%</text>')
        parts.append(f'<circle cx="{x:.1f}" cy="{yt:.1f}" r="5.5" fill="#e45756"/>')
        parts.append(f'<text x="{x:.1f}" y="{yt + 20:.1f}" text-anchor="middle" class="point-label">{r["decode_tok_s_calc"]:.1f}</text>')

    parts.append(f'<line x1="{left["x0"] + 10}" y1="{left["y1"] + 22}" x2="{left["x0"] + 38}" y2="{left["y1"] + 22}" stroke="#4c78a8" stroke-width="3"/>')
    parts.append(f'<text x="{left["x0"] + 46}" y="{left["y1"] + 26}" class="legend">hit rate</text>')
    parts.append(f'<line x1="{left["x0"] + 130}" y1="{left["y1"] + 22}" x2="{left["x0"] + 158}" y2="{left["y1"] + 22}" stroke="#e45756" stroke-width="3"/>')
    parts.append(f'<text x="{left["x0"] + 166}" y="{left["y1"] + 26}" class="legend">decode tok/s</text>')

    # Right panel: hit-rate to throughput relation for ub=1024.
    parts.append(f'<text x="{right["x0"]}" y="{right["y1"] - 28}" class="label" font-weight="700">Hit rate to throughput</text>')
    for t in nice_ticks(hit_min, hit_max, 5):
        x = rx_hit(t)
        parts.append(f'<line x1="{x:.1f}" y1="{right["y1"]}" x2="{x:.1f}" y2="{right["y0"]}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{right["y0"] + 24}" text-anchor="middle" class="tick">{t:.1f}%</text>')
    for t in nice_ticks(tok_min, tok_max, 5):
        y = ry_tok(t)
        parts.append(f'<line x1="{right["x0"]}" y1="{y:.1f}" x2="{right["x1"]}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{right["x0"] - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{t:.0f}</text>')
    parts.append(f'<line x1="{right["x0"]}" y1="{right["y0"]}" x2="{right["x1"]}" y2="{right["y0"]}" class="axis"/>')
    parts.append(f'<line x1="{right["x0"]}" y1="{right["y0"]}" x2="{right["x0"]}" y2="{right["y1"]}" class="axis"/>')
    parts.append(f'<text x="{(right["x0"] + right["x1"]) / 2:.1f}" y="{height - 38}" text-anchor="middle" class="label">decode cache hit rate</text>')
    parts.append(f'<text x="{right["x0"] - 54}" y="{(right["y0"] + right["y1"]) / 2:.1f}" text-anchor="middle" class="label" transform="rotate(-90 {right["x0"] - 54} {(right["y0"] + right["y1"]) / 2:.1f})">decode tok/s</text>')
    scatter_line = " ".join(f'{rx_hit(r["decode_hit_rate_pct"]):.1f},{ry_tok(r["decode_tok_s_calc"]):.1f}' for r in ub_rows)
    parts.append(f'<polyline points="{scatter_line}" fill="none" stroke="#111827" stroke-width="2" stroke-dasharray="5 4"/>')
    for r in ub_rows:
        x = rx_hit(r["decode_hit_rate_pct"])
        y = ry_tok(r["decode_tok_s_calc"])
        color = COLORS[r["vram_mb"]]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" stroke="#111827" stroke-width="0.8"/>')
        parts.append(f'<text x="{x + 10:.1f}" y="{y - 10:.1f}" class="point-label">{r["vram_mb"] // 1000}G</text>')
    parts.append(f'<text x="{right["x0"]}" y="{right["y0"] + 58}" class="subtitle">ub1024: 6G -> 8G gives the main speed jump; 8G -> 10G improves hit rate but speed saturates.</text>')

    save_svg(PLOTS / "ub1024_vram_hit_speed.svg", width, height, "\n".join(parts))


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    plot_ub1024_focus(rows)
    plot_scatter(rows)
    plot_vram_scaling(rows)
    plot_hit_vs_ssd(rows)
    plot_component_bars(rows)
    plot_heatmap(rows)
    print(f"generated {len(list(PLOTS.glob('*.svg')))} SVG charts in {PLOTS}")


if __name__ == "__main__":
    main()
