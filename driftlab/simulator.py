"""Drift simulation and model monitoring helpers.

The core module avoids heavyweight ML dependencies so the project is easy to
run and test. The Streamlit app imports these functions for the interactive UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import exp, sqrt
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd


FEATURES = ["signal", "income_index", "activity_score", "risk_noise"]
DRIFT_MODES = ("none", "mean_shift", "variance_shift", "class_prior_shift", "mixed")


@dataclass(frozen=True)
class DriftConfig:
    rows: int = 4000
    drift_strength: float = 1.0
    drift_mode: str = "mean_shift"
    label_noise: float = 0.08
    seed: int = 42
    psi_alert_threshold: float = 0.2
    auc_drop_threshold: float = 0.08
    train_rows: int | None = None

    def __post_init__(self) -> None:
        for name, value in (("rows", self.rows), ("train_rows", self.train_rows or self.rows)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        if self.train_rows is not None and self.train_rows == 0:
            raise ValueError("train_rows must be an integer >= 2")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.drift_mode not in DRIFT_MODES:
            raise ValueError(f"drift_mode must be one of {DRIFT_MODES}")
        for name in ("drift_strength", "psi_alert_threshold", "auc_drop_threshold"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(self.label_noise) or not 0 <= self.label_noise <= 1:
            raise ValueError("label_noise must be in [0, 1]")


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
    DriftConfig(rows=rows, drift_strength=drift_strength, drift_mode=drift_mode, label_noise=label_noise)
    signal = rng.normal(0, 1, rows)
    income_index = rng.normal(0, 1, rows)
    activity_score = rng.normal(0, 1, rows)
    risk_noise = rng.normal(0, 1, rows)
    # Draw in every mode so strength=0 is identical for the same random seed.
    extra_noise = rng.normal(0, 1, rows)

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
        risk_noise = risk_noise + 0.25 * drift_strength * extra_noise

    logits = 1.15 * signal + 0.8 * income_index - 0.65 * activity_score + 0.25 * risk_noise
    if drift_mode == "class_prior_shift":
        logits = logits + 0.8 * drift_strength
    elif drift_mode == "mixed":
        # Interpolate the conditional label rule; zero strength means no change.
        logits += drift_strength * (-0.70 * signal - 1.45 * income_index + 1.60 * activity_score - 0.15)
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
    if x.ndim != 2 or len(x) < 2 or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError("training features must be finite and have >= 2 rows matching labels")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("training labels must be binary")
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


def binary_metrics(labels: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    y = labels.to_numpy()
    probabilities = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or probabilities.ndim != 1 or len(y) == 0 or len(y) != len(probabilities):
        raise ValueError("labels and probabilities must be non-empty matching vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(probabilities).all():
        raise ValueError("labels must be binary and probabilities finite")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")
    preds = (probabilities >= 0.5).astype(int)

    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(y), 1)

    auc = roc_auc(y, probabilities)
    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "auc": auc,
        "auc_defined": auc is not None,
        "positive_rate": float(y.mean()),
        "predicted_positive_rate": round(float(preds.mean()), 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "rows": len(y),
        "positive_count": int((y == 1).sum()),
        "negative_count": int((y == 0).sum()),
    }


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Compute ROC AUC from ranks with tie handling."""
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return None

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
    if not isinstance(bins, Integral) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    for sample in (reference, current):
        if len(sample) == 0 or not np.isfinite(sample.to_numpy(dtype=float)).all():
            raise ValueError("PSI requires non-empty finite samples")
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference.to_numpy(dtype=float), quantiles))
    if len(edges) == 1:
        # Keep mass at a constant reference value separate from values on either side.
        point = edges[0]
        edges = np.array([-np.inf, point, np.nextafter(point, np.inf), np.inf])
    else:
        # Keep both tails visible even for a two-valued reference sample.
        edges = np.concatenate(([-np.inf], edges[1:-1] if len(edges) > 2 else [edges.mean()], [np.inf]))

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def ks_test_approx(reference: pd.Series, current: pd.Series) -> dict[str, float]:
    """Two-sample KS statistic with asymptotic p-value approximation."""
    for sample in (reference, current):
        if len(sample) == 0 or not np.isfinite(sample.to_numpy(dtype=float)).all():
            raise ValueError("KS requires non-empty finite samples")
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


def alert_flags(reference_metrics: dict[str, Any], current_metrics: dict[str, Any], drift: pd.DataFrame, config: DriftConfig) -> dict[str, bool]:
    auc_ref, auc_cur = reference_metrics["auc"], current_metrics["auc"]
    return {
        "distribution_drift": bool(float(drift["psi"].max()) >= config.psi_alert_threshold),
        "model_degradation": bool(auc_ref is not None and auc_cur is not None and auc_ref - auc_cur >= config.auc_drop_threshold),
        "positive_rate_shift": bool(abs(current_metrics["positive_rate"] - reference_metrics["positive_rate"]) > 0.08),
    }


def build_alerts(reference_metrics: dict[str, Any], current_metrics: dict[str, Any], drift: pd.DataFrame, config: DriftConfig) -> list[str]:
    alerts: list[str] = []
    flags = alert_flags(reference_metrics, current_metrics, drift, config)
    max_psi = float(drift["psi"].max())
    if flags["distribution_drift"]:
        top_feature = str(drift.iloc[0]["feature"])
        alerts.append(f"Distribution drift: {top_feature} PSI={max_psi:.3f}")
    if flags["model_degradation"]:
        auc_drop = float(reference_metrics["auc"] - current_metrics["auc"])
        alerts.append(f"Model degradation: AUC drop={auc_drop:.3f}")
    if current_metrics["positive_rate"] - reference_metrics["positive_rate"] > 0.08:
        alerts.append("Positive-rate shift: current positive rate moved up")
    if reference_metrics["positive_rate"] - current_metrics["positive_rate"] > 0.08:
        alerts.append("Positive-rate shift: current positive rate moved down")
    if not alerts:
        alerts.append("No major alert under the selected thresholds")
    return alerts


def incident_summary(reference_metrics: dict[str, Any], current_metrics: dict[str, Any], drift: pd.DataFrame, alerts: list[str]) -> str:
    top = drift.iloc[0]
    auc_ref, auc_cur = reference_metrics["auc"], current_metrics["auc"]
    auc_text = "AUC comparison unavailable: a window contains only one class. "
    if auc_ref is not None and auc_cur is not None:
        auc_text = f"Held-out reference AUC is {auc_ref:.3f}; current AUC is {auc_cur:.3f}, a change of {auc_cur - auc_ref:+.3f}. "
    return (
        f"Top drift feature is {top['feature']} with PSI {top['psi']:.3f} "
        f"and KS p-value {top['ks_p_value']:.3g}. "
        f"{auc_text}"
        f"Alert status: {alerts[0]}."
    )


def run_experiment(config: DriftConfig) -> dict[str, Any]:
    sequences = np.random.SeedSequence(config.seed).spawn(3)
    training = generate_frame(
        rows=config.train_rows if config.train_rows is not None else config.rows,
        rng=np.random.default_rng(sequences[0]),
        label_noise=config.label_noise,
    )
    reference = generate_frame(
        rows=config.rows,
        rng=np.random.default_rng(sequences[1]),
        drift_strength=0.0,
        drift_mode="none",
        label_noise=config.label_noise,
    )
    current = generate_frame(
        rows=config.rows,
        rng=np.random.default_rng(sequences[2]),
        drift_strength=config.drift_strength,
        drift_mode=config.drift_mode,
        label_noise=config.label_noise,
    )

    weights, means, stds = train_logistic_regression(training[FEATURES], training["label"])
    reference_probabilities = predict_proba(reference[FEATURES], weights, means, stds)
    current_probabilities = predict_proba(current[FEATURES], weights, means, stds)

    reference_metrics = binary_metrics(reference["label"], reference_probabilities)
    current_metrics = binary_metrics(current["label"], current_probabilities)
    drift = feature_drift_table(reference, current)
    alerts = build_alerts(reference_metrics, current_metrics, drift, config)
    flags = alert_flags(reference_metrics, current_metrics, drift, config)
    frames = {"training": training, "reference": reference, "current": current}
    manifest = {
        name: {
            "rows": len(frame),
            "class_counts": {str(label): int((frame["label"] == label).sum()) for label in (0, 1)},
            "sha256_csv": sha256(frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode()).hexdigest(),
            "random_spawn_key": list(sequence.spawn_key),
            "used_for_fitting": name == "training",
        }
        for (name, frame), sequence in zip(frames.items(), sequences)
    }

    return {
        "config": asdict(config),
        "training": training,
        "reference": reference,
        "current": current,
        "reference_metrics": reference_metrics,
        "current_metrics": current_metrics,
        "drift_table": drift,
        "alerts": alerts,
        "alert_flags": flags,
        "alert_count": sum(flags.values()),
        "split_manifest": manifest,
        "model": {"weights": weights.tolist(), "feature_means": means.tolist(), "feature_stds": stds.tolist(), "fitted_on": "training", "features": FEATURES},
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
        "alert_flags": result["alert_flags"],
        "alert_count": result["alert_count"],
        "split_manifest": result["split_manifest"],
        "model": result["model"],
        "summary": result["summary"],
    }
