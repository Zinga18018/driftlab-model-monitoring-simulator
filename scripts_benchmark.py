"""Write reproducible scenario trials and an interpretable no-drift baseline."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from driftlab.benchmark import run_benchmark
from driftlab.simulator import DriftConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    config = DriftConfig(rows=args.rows, drift_strength=1.15, label_noise=0.08)
    result = run_benchmark(config, list(range(args.start_seed, args.start_seed + args.runs)))
    root = Path(__file__).resolve().parent
    result["reproducibility"] = {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "source_sha256": {name: sha256((root / name).read_bytes()).hexdigest()
                          for name in ("driftlab/simulator.py", "driftlab/benchmark.py", "scripts_benchmark.py")},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "monitoring_benchmark.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# Repeated synthetic monitoring benchmark", "",
        f"{args.runs} independent seeds per scenario; {args.rows:,} rows in each training, held-out reference, and current window.",
        "PSI threshold 0.20; AUC-drop threshold 0.08; absolute positive-rate-change threshold > 0.08. Thresholds were fixed before this run, with no tuning.", "",
        "| Scenario | Mean max PSI | Mean AUC drop | Distribution alerts | Degradation alerts | Any alert |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in result["scenario_summary"]:
        rates = summary["alert_rates"]
        drop = "unavailable" if summary["mean_auc_drop"] is None else f"{summary['mean_auc_drop']:.4f}"
        lines.append(f"| {summary['scenario']} | {summary['mean_max_psi']:.4f} | {drop} | "
                     f"{rates['distribution_drift']['count']}/{args.runs} | {rates['model_degradation']['count']}/{args.runs} | {rates['any_alert']['count']}/{args.runs} |")
    null = result["scenario_summary"][0]["alert_rates"]["any_alert"]
    lower, upper = null["wilson_95_interval"]
    lines.extend(["", f"No-drift false-alert rate (any alert): **{null['count']}/{args.runs} = {null['rate']:.1%}**, Wilson 95% interval **[{lower:.1%}, {upper:.1%}]**.",
                  "", "A positive AUC drop means performance fell. Feature drift can occur while AUC stays stable or improves; a distribution alert is not proof of model degradation.",
                  "", "These estimates describe independent synthetic windows at the stated sample size. They do not estimate production false alarms per day, repeated testing over overlapping windows, or delayed-label behavior. KS p-values are approximate diagnostics and do not drive alerts. Single-class AUC is unavailable and excluded from AUC means; counts are recorded in the JSON.",
                  "", "See [full trials, split fingerprints, class counts, seeds, thresholds, and source hashes](monitoring_benchmark.json). These runs are development evidence, not a locked benchmark or a claim of untouched evaluation data.", ""])
    (args.output_dir / "MONITORING_BENCHMARK.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
