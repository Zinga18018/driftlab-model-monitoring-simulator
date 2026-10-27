#!/usr/bin/env python3
"""Render a dependency-free, accessible SVG from checked-in synthetic trials.

No model is trained and no source data is changed. Run from any directory:
    python scripts/render_monitoring_evidence.py
    python scripts/render_monitoring_evidence.py --check
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from html import escape
import json
from math import isclose, sqrt
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/monitoring_benchmark.json"
OUTPUT = ROOT / "docs/images/drift-vs-degradation.svg"
SOURCE_COMMIT = "b25bbd7d0f0d589360350e6df8bf5f4391013c7f"
INPUT_SHA256 = "bdfffa3a451fefe6cf8ec0b4a39ea8f36db4ddf75c1e7c85621d574dab037c7a"
SCENARIOS = [
    ("none", "No drift", "#475569", "circle"),
    ("mean_shift", "Mean shift", "#1d4ed8", "square"),
    ("variance_shift", "Variance shift", "#0f766e", "diamond"),
    ("class_prior_shift", "Positive-rate scenario*", "#a16207", "triangle"),
    ("mixed", "Mixed drift", "#9333ea", "cross"),
]


def close(a: float, b: float) -> None:
    if not isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"inconsistent artifact values: {a} != {b}")


def wilson(count: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = count / n
    den = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / den
    half = z * sqrt(p*(1-p)/n + z**2/(4*n**2)) / den
    return max(0.0, center-half), min(1.0, center+half)


def validate(data: dict) -> None:
    """Independently check every plotted point and summary, not just labels."""
    trials = data["trials"]
    expected = [s[0] for s in SCENARIOS]
    assert data["protocol"]["scenario_order"] == expected
    assert data["protocol"]["seeds"] == list(range(1000, 1040))
    assert Counter(t["scenario"] for t in trials) == {s: 40 for s in expected}
    assert len({(t["scenario"], t["seed"]) for t in trials}) == 200
    paired = {}
    for t in trials:
        cfg = t["config"]
        assert cfg["psi_alert_threshold"] == .20
        assert cfg["auc_drop_threshold"] == .08
        assert cfg["rows"] == 2000 and cfg["label_noise"] == .08
        close(t["auc_drop"], t["reference_metrics"]["auc"]-t["current_metrics"]["auc"])
        flags = t["alert_flags"]
        assert flags["distribution_drift"] == (t["max_psi"] >= .20)
        assert flags["model_degradation"] == (t["auc_drop"] >= .08)
        assert flags["positive_rate_shift"] == (abs(t["current_metrics"]["positive_rate"]-t["reference_metrics"]["positive_rate"]) > .08)
        assert t["any_alert"] == any(flags.values())
        assert 0 <= t["max_psi"] <= 1.10 and -.12 <= t["auc_drop"] <= .52
        for split in ("training", "reference", "current"):
            m = t["split_manifest"][split]
            assert m["rows"] == 2000
            assert m["used_for_fitting"] == (split == "training")
        pair = tuple(t["split_manifest"][s]["sha256_csv"] for s in ("training", "reference"))
        assert paired.setdefault(t["seed"], pair) == pair
    for summary in data["scenario_summary"]:
        group = [t for t in trials if t["scenario"] == summary["scenario"]]
        assert sorted(t["seed"] for t in group) == list(range(1000, 1040))
        assert summary["trials"] == summary["valid_auc_comparisons"] == 40
        assert summary["unavailable_auc_comparisons"] == 0
        close(mean(t["max_psi"] for t in group), summary["mean_max_psi"])
        close(mean(t["auc_drop"] for t in group), summary["mean_auc_drop"])
        close(stdev(t["auc_drop"] for t in group), summary["std_auc_drop"])
        for kind, report in summary["alert_rates"].items():
            count = sum(t["any_alert"] if kind == "any_alert" else t["alert_flags"][kind] for t in group)
            assert report["count"] == count and report["trials"] == 40
            close(report["rate"], count/40)
            for a, b in zip(wilson(count, 40), report["wilson_95_interval"]):
                close(a, b)


def render(data: dict) -> str:
    parts = []
    def put(s: str) -> None:
        parts.append(s)
    def txt(x, y, value, size=21, weight=400, fill="#263449", anchor="start", extra=""):
        put(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {extra}>{escape(str(value))}</text>')
    def line(x1,y1,x2,y2,stroke="#dbe2ea",width=1,extra=""):
        put(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}" {extra}/>')
    def marker(x,y,color,shape,r=4.2,extra=""):
        common=f'fill="{color}" stroke="{color}" stroke-width="1" {extra}'
        if shape=="circle": put(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{r}" {common}/>')
        elif shape=="square": put(f'<rect x="{x-r:.3f}" y="{y-r:.3f}" width="{2*r}" height="{2*r}" {common}/>')
        elif shape=="diamond": put(f'<path d="M {x:.3f},{y-r*1.3:.3f} l {r*1.3},{r*1.3} l {-r*1.3},{r*1.3} l {-r*1.3},{-r*1.3} Z" {common}/>')
        elif shape=="triangle": put(f'<path d="M {x:.3f},{y-r*1.4:.3f} l {r*1.3},{r*2.3} h {-r*2.6} Z" {common}/>')
        else:
            line(x-r,y,x+r,y,color,2.1,extra); line(x,y-r,x,y+r,color,2.1,extra)
    put('<?xml version="1.0" encoding="UTF-8"?>')
    put('<svg xmlns="http://www.w3.org/2000/svg" width="720" height="930" viewBox="0 0 720 930" role="img" aria-labelledby="chart-title chart-desc">')
    put('<title id="chart-title">Drift is not the same as model degradation</title>')
    put('<desc id="chart-desc">All 200 archived synthetic trials: 40 seeds in each of five scenarios. Horizontal axis is maximum feature PSI; vertical axis is reference AUC minus current AUC. Dashed thresholds are PSI 0.20 and AUC drop 0.08. Mean, variance and positive-rate scenarios triggered distribution alerts in 40 of 40 trials each, but no AUC-degradation alerts. Mixed drift triggered both in 40 of 40. No drift triggered neither. Negative AUC drops indicate improved AUC. This is synthetic development evidence, not production monitoring or a locked benchmark.</desc>')
    put(f'<metadata>Source: outputs/monitoring_benchmark.json; SHA-256: {INPUT_SHA256}; repository commit: {SOURCE_COMMIT}. Generated by scripts/render_monitoring_evidence.py; no jitter or sampling.</metadata>')
    put('<rect width="720" height="930" rx="16" fill="#ffffff"/>')
    put('<g font-family="DejaVu Sans, Arial, sans-serif">')
    txt(34,42,"DRIFTLAB  /  REPRODUCIBLE EVIDENCE",18,700,"#0f766e")
    txt(34,89,"Drift ≠ degradation",38,700,"#111c30")
    txt(34,126,"200 synthetic trials · 40 seeds per scenario",23)
    txt(34,158,"One marker = one trial. No jitter or sampling.",20,400,"#526176")
    for (scenario,label,color,shape), (x,y) in zip(SCENARIOS,[(42,197),(232,197),(442,197),(42,235),(422,235)]):
        marker(x,y-7,color,shape,5.5)
        txt(x+14,y,label,20)
    x0,x1,y0,y1=112,679,281,684
    def X(v): return x0+v/1.10*(x1-x0)
    def Y(v): return y1-(v+.12)/.64*(y1-y0)
    threshold_x,threshold_y=X(.20),Y(.08)
    put(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="#f8fafc"/>')
    put(f'<rect x="{threshold_x}" y="{y0}" width="{x1-threshold_x}" height="{threshold_y-y0}" fill="#faf3fc"/>')
    put(f'<rect x="{threshold_x}" y="{threshold_y}" width="{x1-threshold_x}" height="{y1-threshold_y}" fill="#f0fdfa"/>')
    for x in [0,.2,.4,.6,.8,1.0]:
        line(X(x),y0,X(x),y1)
        txt(round(X(x),2),715,f"{x:.1f}",20,anchor="middle")
    for y in [-.1,0,.1,.2,.3,.4,.5]:
        line(x0,Y(y),x1,Y(y))
        txt(96,round(Y(y)+7,2),f"{y:.1f}",20,anchor="end")
    line(x0,Y(0),x1,Y(0),"#6b7789",1.6)
    line(threshold_x,y0,threshold_x,y1,"#65718b",2,'stroke-dasharray="7 6"')
    line(x0,threshold_y,x1,threshold_y,"#65718b",2,'stroke-dasharray="7 6"')
    txt(228,271,"PSI ≥ 0.20",18,600,"#526176")
    txt(667,round(threshold_y-10,2),"AUC drop ≥ 0.08",18,600,"#526176",anchor="end")
    txt(665,319,"Both alerts",20,600,"#6b3388",anchor="end")
    for scenario,label,color,shape in SCENARIOS:
        put(f'<g data-scenario="{scenario}" aria-label="{escape(label)}: 40 trials">')
        for trial in data["trials"]:
            if trial["scenario"]==scenario:
                put(f'<g data-seed="{trial["seed"]}" data-psi="{trial["max_psi"]:.17g}" data-auc-drop="{trial["auc_drop"]:.17g}">')
                marker(X(trial["max_psi"]),Y(trial["auc_drop"]),color,shape,4.2,'opacity="0.68"')
                put('</g>')
        put('</g>')
    txt(665,round(Y(-.09),2),"Distribution alert only",19,600,"#0f766e",anchor="end")
    line(x0,y1,x1,y1,"#6b7789",1.2)
    txt(396,752,"Maximum feature PSI",23,600,anchor="middle")
    txt(28,482,"AUC drop: reference − current",22,600,anchor="middle",extra='transform="rotate(-90 28 482)"')
    line(34,779,686,779)
    txt(34,813,"Negative AUC drop = improved AUC.",22,600,"#111c30")
    txt(34,848,"2,000 rows per independent window · fixed thresholds",20)
    txt(34,878,"*Not a pure label-shift experiment.",19,400,"#526176")
    txt(34,907,"Synthetic development evidence — not production risk.",19,400,"#526176")
    put('</g></svg>')
    return '\n'.join(parts)+'\n'


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='validate data and require exact checked-in SVG bytes')
    args=parser.parse_args()
    raw=INPUT.read_bytes()
    if sha256(raw).hexdigest()!=INPUT_SHA256:
        raise SystemExit('Benchmark input changed. Review the new evidence and update pinned provenance before rendering.')
    data=json.loads(raw)
    validate(data)
    svg=render(data).encode('utf-8')
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_bytes()!=svg:
            raise SystemExit('SVG is missing or stale; run this script without --check.')
        print('PASS: input hash, 200 trials, summaries, thresholds, Wilson intervals, paired fingerprints, and exact SVG bytes')
    else:
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        OUTPUT.write_bytes(svg)
        print(f'Wrote {OUTPUT.relative_to(ROOT)} ({len(svg):,} bytes) from 200 verified synthetic trials.')


if __name__=='__main__':
    main()
