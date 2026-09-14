from __future__ import annotations

import streamlit as st

from driftlab.simulator import DriftConfig, FEATURES, run_experiment


st.set_page_config(
    page_title="DriftLab",
    page_icon="DL",
    layout="wide",
)


def metric_delta(reference: float | None, current: float | None) -> str | None:
    if reference is None or current is None:
        return None
    return f"{current - reference:+.3f}"


def metric_value(value: float | None) -> str:
    return "Unavailable" if value is None else f"{value:.3f}"


st.title("DriftLab")
st.caption("Interactive model monitoring simulator for data drift, label noise, and performance decay.")

with st.sidebar:
    st.header("Incident Controls")
    drift_mode = st.selectbox(
        "Drift scenario",
        ["none", "mean_shift", "variance_shift", "class_prior_shift", "mixed"],
        index=4,
    )
    drift_strength = st.slider("Drift strength", 0.0, 2.5, 1.15, 0.05)
    label_noise = st.slider("Label noise", 0.0, 0.35, 0.08, 0.01)
    rows = st.slider("Rows per training / reference / current window", 1000, 10000, 4000, 500)
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42, step=1)
    psi_threshold = st.slider("PSI alert threshold", 0.05, 0.5, 0.20, 0.01)
    auc_threshold = st.slider("AUC drop threshold", 0.02, 0.25, 0.08, 0.01)

config = DriftConfig(
    rows=int(rows),
    drift_strength=float(drift_strength),
    drift_mode=drift_mode,
    label_noise=float(label_noise),
    seed=int(seed),
    psi_alert_threshold=float(psi_threshold),
    auc_drop_threshold=float(auc_threshold),
)

result = run_experiment(config)
reference_metrics = result["reference_metrics"]
current_metrics = result["current_metrics"]
drift_table = result["drift_table"]
st.caption(
    f"Separate synthetic windows: {len(result['training']):,} training rows, "
    f"{len(result['reference']):,} held-out reference rows, and {len(result['current']):,} current rows. "
    "The model and feature scaling are fitted only on training data."
)
if drift_mode == "class_prior_shift":
    st.info("This scenario shifts feature means and the conditional label rule. It is not a pure label-prior-shift experiment.")
if not reference_metrics["auc_defined"] or not current_metrics["auc_defined"]:
    st.warning("AUC is unavailable for a single-class window; the degradation alert cannot be assessed.")

st.subheader("Model Health")
metric_cols = st.columns(4)
metric_cols[0].metric("Held-out reference AUC", metric_value(reference_metrics["auc"]))
metric_cols[1].metric("Current AUC", metric_value(current_metrics["auc"]), metric_delta(reference_metrics["auc"], current_metrics["auc"]))
metric_cols[2].metric("Current F1", f"{current_metrics['f1']:.3f}", metric_delta(reference_metrics["f1"], current_metrics["f1"]))
metric_cols[3].metric("Current positive rate", f"{current_metrics['positive_rate']:.3f}", metric_delta(reference_metrics["positive_rate"], current_metrics["positive_rate"]))

left, right = st.columns([1.2, 0.8])

with left:
    st.subheader("Drift Radar")
    chart_data = drift_table.set_index("feature")[["psi", "ks_stat"]]
    st.bar_chart(chart_data)
    st.dataframe(
        drift_table,
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.subheader("Alert Board")
    for alert in result["alerts"]:
        if alert.startswith("No major"):
            st.success(alert)
        else:
            st.warning(alert)
    st.info(result["summary"])

st.subheader("Feature Distribution Playground")
feature = st.selectbox("Feature to inspect", FEATURES)

hist_reference = result["reference"][[feature]].rename(columns={feature: "reference"})
hist_current = result["current"][[feature]].rename(columns={feature: "current"})

dist_cols = st.columns(2)
with dist_cols[0]:
    st.write("Reference window")
    st.line_chart(hist_reference["reference"].sort_values(ignore_index=True))
with dist_cols[1]:
    st.write("Current window")
    st.line_chart(hist_current["current"].sort_values(ignore_index=True))

st.subheader("Confusion Matrix")
cm = {
    "": ["Actual 0", "Actual 1"],
    "Predicted 0": [current_metrics["tn"], current_metrics["fn"]],
    "Predicted 1": [current_metrics["fp"], current_metrics["tp"]],
}
st.table(cm)

st.caption(
    "Synthetic educational simulation. Labels are immediately available; KS p-values are approximate diagnostics. "
    "A distribution alert alone does not establish model degradation or justify automatic retraining. "
    "See outputs/MONITORING_BENCHMARK.md in the repository for repeated no-drift false-alert measurements at fixed thresholds."
)
