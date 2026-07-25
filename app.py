import streamlit as st
import pandas as pd
import numpy as np
from config import STREAMLIT_PAGE_TITLE, STREAMLIT_LAYOUT, setup_logging
from pipeline import run_full_pipeline, extract_joined_data, transform_data, get_aggregations
from anomaly_detector import AnomalyDetector

logger = setup_logging()

st.set_page_config(page_title=STREAMLIT_PAGE_TITLE, layout=STREAMLIT_LAYOUT, initial_sidebar_state="expanded")

st.markdown("""
<style>
.kpi-card { background-color: #f0f2f6; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
.kpi-value { font-size: 32px; font-weight: bold; color: #1f77b4; }
.kpi-label { font-size: 14px; color: #555; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Controls")
    run_btn = st.button("Run ETL Pipeline", type="primary")
    st.divider()
    st.subheader("ML Config")
    contamination = st.slider("Contamination", 0.01, 0.30, 0.08, 0.01)
    n_estimators = st.slider("Estimators", 50, 300, 150, 10)
    st.divider()
    st.subheader("Features")
    selected_features = st.multiselect(
        "Anomaly features", ["battery_voltage", "current_load_kw", "temperature_c"],
        default=["battery_voltage", "current_load_kw"]
    )

st.title(STREAMLIT_PAGE_TITLE)
st.caption("Smart-grid fleet telemetry with ML anomaly detection")

if run_btn:
    with st.spinner("Running pipeline..."):
        logger.info("Pipeline triggered from UI")
        transformed_df, aggregations = run_full_pipeline()
        st.session_state["transformed_df"] = transformed_df
        st.session_state["aggregations"] = aggregations
        st.session_state["pipeline_run"] = True
        st.success("Pipeline complete!")

if "pipeline_run" not in st.session_state or not st.session_state["pipeline_run"]:
    st.info("Click 'Run ETL Pipeline' to load data")
    st.stop()

transformed_df = st.session_state["transformed_df"]
aggregations = st.session_state["aggregations"]

# KPIs
st.header("KPIs")
c1, c2, c3, c4 = st.columns(4)
total_records = int(aggregations["count_stats"]["total_records"].values[0])
fleet_size = int(aggregations["count_stats"]["fleet_size"].values[0])
criticals = transformed_df["voltage_critical"].sum() + transformed_df["load_critical"].sum() + transformed_df["temp_critical"].sum()
avg_v = transformed_df["battery_voltage"].mean()

with c1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{total_records:,}</div><div class="kpi-label">Records</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{fleet_size}</div><div class="kpi-label">Fleet</div></div>', unsafe_allow_html=True)
with c3:
    color = '#d62728' if criticals > 0 else '#2ca02c'
    st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{color};">{int(criticals)}</div><div class="kpi-label">Critical</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{avg_v:.1f}V</div><div class="kpi-label">Avg Voltage</div></div>', unsafe_allow_html=True)

st.divider()

# Tables
st.header("Aggregations")
tc1, tc2 = st.columns(2)
with tc1:
    st.subheader("Avg Voltage by Vehicle")
    st.dataframe(aggregations["avg_voltage_by_vehicle"].style.format({"avg_voltage": "{:.2f} V"}), use_container_width=True, height=300)
with tc2:
    st.subheader("Max Load by Vehicle")
    st.dataframe(aggregations["max_load_by_vehicle"].style.format({"max_load": "{:.2f} kW"}), use_container_width=True, height=300)

st.divider()

# Time series
st.header("Load Trends")
ts_df = transformed_df.copy()
ts_df["log_timestamp"] = pd.to_datetime(ts_df["log_timestamp"])
ts_df = ts_df.sort_values("log_timestamp")
st.line_chart(ts_df, x="log_timestamp", y="current_load_kw", color="#1f77b4", use_container_width=True)

st.divider()

# Anomaly detection
st.header("Anomaly Detection")
if not selected_features:
    st.warning("Select at least one feature")
    st.stop()

with st.spinner("Training Isolation Forest..."):
    detector = AnomalyDetector(contamination=contamination, n_estimators=int(n_estimators), random_state=42)
    anomaly_df = detector.detect_anomalies(transformed_df, selected_features)
    st.session_state["anomaly_df"] = anomaly_df
    st.session_state["detector"] = detector

anomaly_df = st.session_state["anomaly_df"]
detector = st.session_state["detector"]

n_anom = int(anomaly_df["is_anomaly"].sum())
n_total = len(anomaly_df)
rate = (n_anom / n_total) * 100 if n_total > 0 else 0
avg_score = anomaly_df.loc[anomaly_df["is_anomaly"], "anomaly_score"].mean()
if pd.isna(avg_score):
    avg_score = 0.0

ac1, ac2, ac3 = st.columns(3)
ac1.metric("Anomalies", f"{n_anom}", f"{rate:.1f}%")
ac2.metric("Normal", f"{n_total - n_anom}")
ac3.metric("Avg Score", f"{avg_score:.4f}")

st.divider()

# Score histogram
st.subheader("Score Distribution")
bins = pd.cut(anomaly_df["anomaly_score"], bins=20)
counts = bins.value_counts().sort_index()
hist_df = pd.DataFrame({"score_range": [f"{i.left:.3f}" for i in counts.index], "count": counts.values})
st.bar_chart(hist_df, x="score_range", y="count", color="#ff7f0e", use_container_width=True)

st.divider()

# Feature importance
st.subheader("Feature Importance")
imp = detector.get_feature_importance()
imp_df = pd.DataFrame({"feature": list(imp.keys()), "importance": list(imp.values())}).sort_values("importance", ascending=False)
st.bar_chart(imp_df, x="feature", y="importance", color="#2ca02c", use_container_width=True)

st.divider()

# Data table
st.subheader("Telemetry with Flags")
display_cols = [c for c in ["vehicle_id", "battery_voltage", "current_load_kw", "temperature_c", "log_timestamp", "is_anomaly", "anomaly_score"] if c in anomaly_df.columns]
disp = anomaly_df[display_cols].copy()
if "anomaly_score" in disp.columns:
    disp["anomaly_score"] = disp["anomaly_score"].round(4)
st.dataframe(disp, use_container_width=True, height=400)

st.divider()

# Export
st.subheader("Export")
csv = anomaly_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Full CSV", csv, "telemetry_with_anomalies.csv", "text/csv")

anom_only = anomaly_df[anomaly_df["is_anomaly"]]
if len(anom_only) > 0:
    st.download_button(f"Download Anomalies Only ({len(anom_only)})", anom_only.to_csv(index=False).encode("utf-8"), "anomalies_only.csv", "text/csv")

st.caption("Smart-Grid Fleet Telemetry Pipeline v1.0")
