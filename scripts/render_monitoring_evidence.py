#!/usr/bin/env python3
"""Render an accessible Matplotlib SVG from checked-in synthetic trials.

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


def render(data: dict) -> tuple[bytes, bytes]:
    """Use Matplotlib's Agg/SVG backends; verify plotted offsets before export."""
    from io import BytesIO
    import os
    import tempfile

    # A writable cache is useful in read-only development containers.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "driftlab-matplotlib"))
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    markers = {"circle": "o", "square": "s", "diamond": "D", "triangle": "^", "cross": "+"}
    title = "PSI and AUC change across 200 synthetic trials"
    description = (
        "All 200 archived synthetic trials: 40 seeds in each of five scenarios. "
        "Horizontal axis: maximum feature PSI. Vertical axis: reference AUC minus current AUC. "
        "Dashed thresholds: PSI 0.20 and AUC drop 0.08. Negative AUC drop means improved AUC. "
        "Synthetic development results, not production estimates or biological validation. "
        "The positive-rate scenario is not pure label shift. "
        f"Input SHA-256: {INPUT_SHA256}; source commit: {SOURCE_COMMIT}."
    )
    settings = {
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelsize": 13, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#777777", "axes.linewidth": .8,
        "xtick.color": "#333333", "ytick.color": "#333333",
        "svg.fonttype": "none", "svg.hashsalt": "driftlab-archived-trials-20261027",
        "axes.unicode_minus": True,
    }
    with matplotlib.rc_context(settings):
        fig = Figure(figsize=(8, 7.1), dpi=100, facecolor="white")
        FigureCanvasAgg(fig)
        ax = fig.add_axes([.13, .205, .825, .60])
        ax.set(xlim=(-.025, 1.10), ylim=(-.12, .52),
               xlabel="Maximum feature PSI", ylabel="AUC drop (reference − current)",
               xticks=[0, .2, .4, .6, .8, 1.0], yticks=[-.1, 0, .1, .2, .3, .4, .5])
        ax.grid(color="#e6e6e6", linewidth=.6)
        ax.set_axisbelow(True)
        ax.axhline(0, color="#999999", linewidth=.8, zorder=1)
        ax.axvline(.20, color="#555555", linewidth=1, linestyle=(0, (5, 4)))
        ax.axhline(.08, color="#555555", linewidth=1, linestyle=(0, (5, 4)))
        ax.text(.215, .535, "PSI ≥ 0.20", fontsize=10, color="#444444")
        ax.text(1.085, .095, "AUC drop ≥ 0.08", fontsize=10,
                color="#444444", ha="right")
        for scenario, label, color, shape in SCENARIOS:
            trials = [t for t in data["trials"] if t["scenario"] == scenario]
            xs, ys = [t["max_psi"] for t in trials], [t["auc_drop"] for t in trials]
            plotted = ax.scatter(xs, ys, s=27, c=color, marker=markers[shape],
                                 alpha=.72, linewidths=.6, label=label, zorder=3)
            plotted.set_gid("scenario-" + scenario)
            # These offsets are the actual Matplotlib artist coordinates.
            assert [tuple(map(float, p)) for p in plotted.get_offsets()] == list(zip(xs, ys))
            assert len(plotted.get_offsets()) == 40
        fig.suptitle(title, y=.975, fontsize=15, fontweight="normal")
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
                   bbox_to_anchor=(.53, .94), ncol=3, frameon=False,
                   fontsize=10.5, columnspacing=1.8, handletextpad=.3)
        fig.text(.04, .085,
                 "40 seeds per scenario; synthetic development results, not production estimates.",
                 fontsize=10, color="#555555")
        fig.text(.04, .05, "*Positive-rate scenario is not pure label shift.",
                 fontsize=10, color="#555555")
        svg_buffer, png_buffer = BytesIO(), BytesIO()
        fig.savefig(svg_buffer, format="svg", facecolor="white",
                    metadata={"Date": None, "Creator": f"Matplotlib {matplotlib.__version__}",
                              "Title": title, "Description": description})
        fig.savefig(png_buffer, format="png", dpi=150, facecolor="white",
                    metadata={"Software": f"Matplotlib {matplotlib.__version__}",
                              "Title": title, "Description": description})
        # Only accessibility metadata is added; Matplotlib draws every plot mark.
        svg = svg_buffer.getvalue().decode("utf-8")
        svg = svg.replace('<svg ', '<svg role="img" aria-labelledby="chart-title chart-desc" ', 1)
        start = svg.index('>', svg.index('<svg ')) + 1
        svg = (svg[:start] + '\n<title id="chart-title">' + escape(title) + '</title>'
               + '\n<desc id="chart-desc">' + escape(description) + '</desc>' + svg[start:])
        return svg.encode("utf-8"), png_buffer.getvalue()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='require exact SVG bytes in the pinned plotting environment')
    parser.add_argument('--png', type=Path, help='optional PNG path to write, or compare with --check')
    args=parser.parse_args()
    raw=INPUT.read_bytes()
    if sha256(raw).hexdigest()!=INPUT_SHA256:
        raise SystemExit('Benchmark input changed. Review the new evidence and update pinned provenance before rendering.')
    data=json.loads(raw)
    validate(data)
    svg, png = render(data)
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_bytes()!=svg:
            raise SystemExit('SVG is missing or stale; run this script without --check.')
        if args.png and (not args.png.exists() or args.png.read_bytes() != png):
            raise SystemExit('PNG is missing or stale in this plotting environment.')
        print('PASS: input hash, 200 Matplotlib offsets, summaries, thresholds, Wilson intervals, paired fingerprints, and exact SVG bytes')
    else:
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        OUTPUT.write_bytes(svg)
        if args.png:
            args.png.parent.mkdir(parents=True, exist_ok=True)
            args.png.write_bytes(png)
        print(f'Wrote {OUTPUT.relative_to(ROOT)} ({len(svg):,} bytes) from 200 verified synthetic trials.')


if __name__=='__main__':
    main()
