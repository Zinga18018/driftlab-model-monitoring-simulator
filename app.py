from __future__ import annotations

import streamlit as st

from driftlab.simulator import DriftConfig, FEATURES, run_experiment


st.set_page_config(
    page_title="DriftLab",
    page_icon="DL",
    layout="wide",
)


def metric_delta(reference: float, current: float) -> str:
    return f"{current - reference:+.3f}"


st.title("DriftLab")
st.caption("Interactive model monitoring simulator for data drift, label noise, and performance decay.")

with st.sidebar:
    st.header("Incident Controls")
    drift_mode = st.selectbox(
        "Drift scenario",
        ["mean_shift", "variance_shift", "class_prior_shift", "mixed"],
        index=3,
    )
    drift_strength = st.slider("Drift strength", 0.0, 2.5, 1.15, 0.05)
    label_noise = st.slider("Label noise", 0.0, 0.35, 0.08, 0.01)
    rows = st.slider("Rows per window", 1000, 10000, 4000, 500)
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

st.subheader("Model Health")
metric_cols = st.columns(4)
metric_cols[0].metric("Reference AUC", f"{reference_metrics['auc']:.3f}")
metric_cols[1].metric("Current AUC", f"{current_metrics['auc']:.3f}", metric_delta(reference_metrics["auc"], current_metrics["auc"]))
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
    "Built as a portfolio project: the simulation is synthetic by design, but the metrics are computed live from the selected settings."
)
