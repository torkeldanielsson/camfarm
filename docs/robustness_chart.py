#!/usr/bin/env python3
"""Draw docs/images/robustness.svg from docs/robustness.csv (written by docs/robustness.sh)."""
import csv
import math
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "robustness.csv")
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "images", "robustness.svg")
rows = list(csv.DictReader(open(src)))

SIZES = [(1280, "720p", "#2a78d6"), (960, "540p", "#eb6834"), (640, "360p, 270p, 180p", "#1baf7a")]
CODECS = [("h264", "H.264"), ("h265", "H.265")]
X_TICKS = [30, 60, 125, 250, 500, 1000]
PANEL_W, PANEL_H, GAP = 330, 230, 40
ML, MR, MT, MB = 48, 12, 84, 44
W = ML + 2 * PANEL_W + GAP + MR
H = MT + PANEL_H + MB
X0, X1 = math.log(26), math.log(1150)


def x_at(panel, kbps):
    return ML + panel * (PANEL_W + GAP) + (math.log(kbps) - X0) / (X1 - X0) * PANEL_W


def y_at(pct):
    return MT + PANEL_H - pct / 100 * (PANEL_H - 10)


o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
     'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif" font-size="12">',
     f'<rect width="{W}" height="{H}" fill="#fcfcfb"/>',
     f'<text x="{ML}" y="18" font-size="14" font-weight="600" fill="#0b0b0b">'
     'Pictures with a readable code, after re-encoding at a lower bitrate</text>']
lx = ML
for _, name, col in SIZES:
    o.append(f'<rect x="{lx}" y="30" width="12" height="12" rx="2" fill="{col}"/>'
             f'<text x="{lx + 17}" y="40" fill="#52514e">{name}</text>')
    lx += 30 + 7 * len(name)
for panel, (codec, cname) in enumerate(CODECS):
    left = ML + panel * (PANEL_W + GAP)
    o.append(f'<text x="{left + PANEL_W / 2}" y="{MT - 8}" text-anchor="middle" font-weight="600" fill="#0b0b0b">{cname}</text>')
    for pct in (0, 25, 50, 75, 100):
        y = y_at(pct)
        o.append(f'<line x1="{left}" x2="{left + PANEL_W}" y1="{y}" y2="{y}" stroke="#e4e3df"/>')
        if panel == 0:
            o.append(f'<text x="{left - 6}" y="{y + 4}" text-anchor="end" fill="#52514e">{pct} %</text>')
    for t in X_TICKS:
        o.append(f'<text x="{x_at(panel, t):.1f}" y="{MT + PANEL_H + 16}" text-anchor="middle" fill="#52514e">{t}</text>')
    o.append(f'<line x1="{left}" x2="{left + PANEL_W}" y1="{MT + PANEL_H}" y2="{MT + PANEL_H}" stroke="#8a8984"/>')
    o.append(f'<text x="{left + PANEL_W / 2}" y="{H - 8}" text-anchor="middle" fill="#52514e">kbit/s (log scale)</text>')
    for width, name, col in reversed(SIZES):
        pts = sorted((int(r["kbps"]), 100 * int(r["readable"]) / int(r["frames"]))
                     for r in rows if r["codec"] == codec and int(r["width"]) == width)
        d = " ".join(f"{'M' if i == 0 else 'L'}{x_at(panel, k):.1f},{y_at(p):.1f}" for i, (k, p) in enumerate(pts))
        o.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2"/>')
        for k, p in pts:
            o.append(f'<circle cx="{x_at(panel, k):.1f}" cy="{y_at(p):.1f}" r="4" fill="{col}" stroke="#fcfcfb" stroke-width="2">'
                     f'<title>{cname} {name}: {k} kbit/s, {p:.0f} % readable</title></circle>')
o.append('</svg>')
os.makedirs(os.path.dirname(dst), exist_ok=True)
open(dst, "w").write("\n".join(o) + "\n")
