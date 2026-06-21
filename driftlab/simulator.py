"""Drift simulation and model monitoring helpers.

The core module avoids heavyweight ML dependencies so the project is easy to
run and test. The Streamlit app imports these functions for the interactive UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, log, sqrt
from typing import Any

import numpy as np
import pandas as pd


FEATURES = ["signal", "income_index", "activity_score", "risk_noise"]


@dataclass(frozen=True)
class DriftConfig:
    rows: int = 4000
    drift_strength: float = 1.0
    drift_mode: str = "mean_shift"
    label_noise: float = 0.08
    seed: int = 42
    psi_alert_threshold: float = 0.2
    auc_drop_threshold: float = 0.08


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-clipped))


def generate_frame(
    rows: int,
    rng: np.random.Generator,
    drift_strength: float = 0.0,
    drift_mode: str = "none",
    label_noise: float = 0.0,
) -> pd.DataFrame:
    """Generate a binary-classification dataset with controlled drift."""
    signal = rng.normal(0, 1, rows)
    income_index = rng.normal(0, 1, rows)
    activity_score = rng.normal(0, 1, rows)
    risk_noise = rng.normal(0, 1, rows)

    if drift_mode == "mean_shift":
        signal = signal + 0.85 * drift_strength
        income_index = income_index - 0.45 * drift_strength
    elif drift_mode == "variance_shift":
        signal = signal * (1.0 + 0.55 * drift_strength)
        activity_score = activity_score * (1.0 + 0.45 * drift_strength)
    elif drift_mode == "class_prior_shift":
        income_index = income_index + 0.65 * drift_strength
        activity_score = activity_score + 0.35 * drift_strength
    elif drift_mode == "mixed":
        signal = signal + 0.65 * drift_strength
        income_index = income_index - 0.35 * drift_strength
        activity_score = activity_score * (1.0 + 0.45 * drift_strength)
        risk_noise = risk_noise + rng.normal(0, 0.25 * drift_strength, rows)

    logits = 1.15 * signal + 0.8 * income_index - 0.65 * activity_score + 0.25 * risk_noise
    if drift_mode == "class_prior_shift":
        logits = logits + 0.8 * drift_strength
    elif drift_mode == "mixed":
        logits = (
            0.45 * signal
            - 0.65 * income_index
            + 0.95 * activity_score
            + 0.25 * risk_noise
            - 0.15 * drift_strength
        )
    probabilities = sigmoid(logits)
    labels = rng.binomial(1, probabilities)

    if label_noise > 0:
        flip_mask = rng.random(rows) < label_noise
        labels = np.where(flip_mask, 1 - labels, labels)

    frame = pd.DataFrame(
        {
            "signal": signal,
            "income_index": income_index,
            "activity_score": activity_score,
            "risk_noise": risk_noise,
            "label": labels.astype(int),
        }
    )
    return frame


def train_logistic_regression(
    features: pd.DataFrame,
    labels: pd.Series,
    learning_rate: float = 0.08,
    steps: int = 600,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train a small logistic regression model with gradient descent."""
    x = features.to_numpy(dtype=float)
    y = labels.to_numpy(dtype=float)
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds[stds == 0] = 1.0
    x_scaled = (x - means) / stds
    x_design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    weights = np.zeros(x_design.shape[1])

    for _ in range(steps):
        predictions = sigmoid(x_design @ weights)
        gradient = x_design.T @ (predictions - y) / len(y)
        weights -= learning_rate * gradient

    return weights, means, stds


def predict_proba(features: pd.DataFrame, weights: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    x = features.to_numpy(dtype=float)
    x_scaled = (x - means) / stds
    x_design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    return sigmoid(x_design @ weights)


def binary_metrics(labels: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    y = labels.to_numpy(dtype=int)
    preds = (probabilities >= 0.5).astype(int)

    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(y), 1)

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "auc": round(float(roc_auc(y, probabilities)), 4),
        "positive_rate": round(float(y.mean()), 4),
        "predicted_positive_rate": round(float(preds.mean()), 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC from ranks with tie handling."""
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)

    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    sum_pos_ranks = ranks[positives].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Compute PSI using reference quantile buckets."""
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference.to_numpy(dtype=float), quantiles))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return round(float(psi), 4)


def ks_test_approx(reference: pd.Series, current: pd.Series) -> dict[str, float]:
    """Two-sample KS statistic with asymptotic p-value approximation."""
    ref = np.sort(reference.to_numpy(dtype=float))
    cur = np.sort(current.to_numpy(dtype=float))
    combined = np.sort(np.concatenate([ref, cur]))
    ref_cdf = np.searchsorted(ref, combined, side="right") / len(ref)
    cur_cdf = np.searchsorted(cur, combined, side="right") / len(cur)
    statistic = float(np.max(np.abs(ref_cdf - cur_cdf)))

    effective_n = len(ref) * len(cur) / (len(ref) + len(cur))
    lambda_value = (sqrt(effective_n) + 0.12 + 0.11 / max(sqrt(effective_n), 1e-12)) * statistic
    p_value = min(max(2 * exp(-2 * lambda_value * lambda_value), 0.0), 1.0)
    return {"ks_stat": round(statistic, 4), "ks_p_value": round(p_value, 6)}


def feature_drift_table(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        psi = population_stability_index(reference[feature], current[feature])
        ks = ks_test_approx(reference[feature], current[feature])
        rows.append(
            {
                "feature": feature,
                "reference_mean": round(float(reference[feature].mean()), 4),
                "current_mean": round(float(current[feature].mean()), 4),
                "mean_delta": round(float(current[feature].mean() - reference[feature].mean()), 4),
                "psi": psi,
                **ks,
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def build_alerts(reference_metrics: dict[str, float], current_metrics: dict[str, float], drift: pd.DataFrame, config: DriftConfig) -> list[str]:
    alerts: list[str] = []
    max_psi = float(drift["psi"].max())
    auc_drop = float(reference_metrics["auc"] - current_metrics["auc"])
    if max_psi >= config.psi_alert_threshold:
        top_feature = str(drift.iloc[0]["feature"])
        alerts.append(f"Distribution drift: {top_feature} PSI={max_psi:.3f}")
    if auc_drop >= config.auc_drop_threshold:
        alerts.append(f"Model degradation: AUC drop={auc_drop:.3f}")
    if current_metrics["positive_rate"] - reference_metrics["positive_rate"] > 0.08:
        alerts.append("Class-prior shift: current positive rate moved up")
    if reference_metrics["positive_rate"] - current_metrics["positive_rate"] > 0.08:
        alerts.append("Class-prior shift: current positive rate moved down")
    if not alerts:
        alerts.append("No major alert under the selected thresholds")
    return alerts


def incident_summary(reference_metrics: dict[str, float], current_metrics: dict[str, float], drift: pd.DataFrame, alerts: list[str]) -> str:
    top = drift.iloc[0]
    auc_delta = reference_metrics["auc"] - current_metrics["auc"]
    return (
        f"Top drift feature is {top['feature']} with PSI {top['psi']:.3f} "
        f"and KS p-value {top['ks_p_value']:.3g}. "
        f"Reference AUC is {reference_metrics['auc']:.3f}; current AUC is "
        f"{current_metrics['auc']:.3f}, a change of {-auc_delta:+.3f}. "
        f"Alert status: {alerts[0]}."
    )


def run_experiment(config: DriftConfig) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed)
    reference = generate_frame(
        rows=config.rows,
        rng=rng,
        drift_strength=0.0,
        drift_mode="none",
        label_noise=config.label_noise,
    )
    current = generate_frame(
        rows=config.rows,
        rng=rng,
        drift_strength=config.drift_strength,
        drift_mode=config.drift_mode,
        label_noise=config.label_noise,
    )

    weights, means, stds = train_logistic_regression(reference[FEATURES], reference["label"])
    reference_probabilities = predict_proba(reference[FEATURES], weights, means, stds)
    current_probabilities = predict_proba(current[FEATURES], weights, means, stds)

    reference_metrics = binary_metrics(reference["label"], reference_probabilities)
    current_metrics = binary_metrics(current["label"], current_probabilities)
    drift = feature_drift_table(reference, current)
    alerts = build_alerts(reference_metrics, current_metrics, drift, config)

    return {
        "config": asdict(config),
        "reference": reference,
        "current": current,
        "reference_metrics": reference_metrics,
        "current_metrics": current_metrics,
        "drift_table": drift,
        "alerts": alerts,
        "summary": incident_summary(reference_metrics, current_metrics, drift, alerts),
    }


def serializable_result(result: dict[str, Any]) -> dict[str, Any]:
    """Convert a full experiment result into JSON-friendly evidence."""
    return {
        "config": result["config"],
        "reference_metrics": result["reference_metrics"],
        "current_metrics": result["current_metrics"],
        "drift_table": result["drift_table"].to_dict(orient="records"),
        "alerts": result["alerts"],
        "summary": result["summary"],
    }
