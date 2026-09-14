"""Repeated synthetic monitoring trials with a fixed-threshold null baseline."""

from __future__ import annotations

from dataclasses import asdict, replace
from math import sqrt
from typing import Any

import numpy as np

from driftlab.simulator import DRIFT_MODES, DriftConfig, run_experiment, serializable_result


def binomial_interval(count: int, trials: int) -> list[float]:
    """Wilson 95% interval; zero observed alerts does not imply zero risk."""
    if trials < 1 or not 0 <= count <= trials:
        raise ValueError("require 0 <= count <= trials and trials >= 1")
    z = 1.959963984540054
    rate = count / trials
    divisor = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / divisor
    margin = z * sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / divisor
    return [max(0.0, center - margin), min(1.0, center + margin)]


def run_benchmark(config: DriftConfig, seeds: list[int]) -> dict[str, Any]:
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("provide at least two distinct seeds for independent repeated trials")
    trials: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    # Complete the no-drift baseline before evaluating changed distributions.
    # No threshold fitting or selection is performed using either result.
    for scenario in DRIFT_MODES:
        scenario_trials = []
        for seed in seeds:
            trial_config = replace(config, seed=seed, drift_mode=scenario,
                                   drift_strength=0.0 if scenario == "none" else config.drift_strength)
            result = serializable_result(run_experiment(trial_config))
            auc_ref, auc_cur = result["reference_metrics"]["auc"], result["current_metrics"]["auc"]
            record = {
                "seed": seed,
                "scenario": scenario,
                "config": result["config"],
                "split_manifest": result["split_manifest"],
                "reference_metrics": result["reference_metrics"],
                "current_metrics": result["current_metrics"],
                "auc_drop": None if auc_ref is None or auc_cur is None else auc_ref - auc_cur,
                "max_psi": max(row["psi"] for row in result["drift_table"]),
                "alert_flags": result["alert_flags"],
                "any_alert": any(result["alert_flags"].values()),
            }
            scenario_trials.append(record)
        rates = {}
        for alert in ("distribution_drift", "model_degradation", "positive_rate_shift", "any_alert"):
            count = sum(r["any_alert"] if alert == "any_alert" else r["alert_flags"][alert] for r in scenario_trials)
            rates[alert] = {"count": count, "trials": len(seeds), "rate": count / len(seeds),
                            "wilson_95_interval": binomial_interval(count, len(seeds))}
        valid_drops = [r["auc_drop"] for r in scenario_trials if r["auc_drop"] is not None]
        summaries.append({
            "scenario": scenario,
            "trials": len(seeds),
            "valid_auc_comparisons": len(valid_drops),
            "unavailable_auc_comparisons": len(seeds) - len(valid_drops),
            "mean_auc_drop": float(np.mean(valid_drops)) if valid_drops else None,
            "std_auc_drop": float(np.std(valid_drops, ddof=1)) if len(valid_drops) > 1 else None,
            "mean_max_psi": float(np.mean([r["max_psi"] for r in scenario_trials])),
            "alert_rates": rates,
        })
        trials.extend(scenario_trials)
    return {
        "schema_version": 1,
        "protocol": {
            "data": "synthetic binary classification; independent training, reference, and current windows",
            "base_config": asdict(config),
            "seeds": seeds,
            "scenario_order": list(DRIFT_MODES),
            "threshold_selection": "fixed existing defaults; no threshold tuning on these results",
            "null_definition": "none: unchanged features, conditional label rule, and label noise",
            "repeated_unit": "one newly trained model and two independent monitoring windows per seed",
            "paired_scenarios": "same training and reference data reused across scenarios for each seed",
            "false_alert_definition": "alert triggered in the no-drift scenario",
            "interval": "Wilson 95% binomial interval across independent seeds within each scenario",
            "limitation": "synthetic repeated trials; not production traffic, a locked evaluation set, or a guarantee about future alerts",
        },
        "scenario_summary": summaries,
        "trials": trials,
    }
