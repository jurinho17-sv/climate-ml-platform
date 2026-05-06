import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Anomaly Detection", page_icon="🚨", layout="wide")
st.title("Emission Anomaly Detection")
st.markdown(
    "Flag years where reported CO2 emissions deviate from the autoencoder's learned "
    "reconstruction baseline. High reconstruction error = unexpected pattern."
)


@st.cache_data(ttl=300)
def fetch_countries() -> list[dict]:
    resp = httpx.get(f"{API_URL}/data/countries", timeout=30)
    resp.raise_for_status()
    return resp.json()["countries"]


@st.cache_data(ttl=300)
def fetch_anomalies(iso: str) -> dict:
    resp = httpx.get(f"{API_URL}/anomalies/{iso}", timeout=30)
    resp.raise_for_status()
    return resp.json()


try:
    countries = fetch_countries()
except (httpx.HTTPError, httpx.HTTPStatusError) as e:
    st.error(f"❌ Cannot reach API at {API_URL}: {e}")
    st.stop()

names = sorted(c["name"] for c in countries)
name_to_iso = {c["name"]: c["iso_code"] for c in countries}

col_pick, col_refresh = st.columns([3, 1])
with col_pick:
    default_idx = names.index("United States") if "United States" in names else 0
    selected_name = st.selectbox("Country", options=names, index=default_idx)
with col_refresh:
    st.write("")
    st.write("")
    if st.button("Refresh", help="Force re-fetch from API"):
        fetch_anomalies.clear()

iso = name_to_iso[selected_name]

try:
    with st.spinner("Detecting anomalies…"):
        result = fetch_anomalies(iso)
except httpx.HTTPStatusError as e:
    st.error(f"❌ Anomaly detection failed ({e.response.status_code}): {e.response.text}")
    st.stop()
except httpx.HTTPError as e:
    st.error(f"❌ API error: {e}")
    st.stop()

records = result["anomalies"]
total = result["total_anomalies"]

if not records:
    st.warning(f"No anomaly records returned for {selected_name}.")
    st.stop()

st.metric("Total Anomalies Detected", total)

df = pd.DataFrame(records)
anomaly_rows = df[df["is_anomaly"]]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df["year"],
        y=df["reconstruction_error"],
        mode="lines",
        name="Reconstruction error",
        line=dict(color="#1f77b4", width=2),
    )
)
if not anomaly_rows.empty:
    fig.add_trace(
        go.Scatter(
            x=anomaly_rows["year"],
            y=anomaly_rows["reconstruction_error"],
            mode="markers",
            name="Anomaly",
            marker=dict(color="red", size=11, symbol="circle"),
        )
    )
    for _, row in anomaly_rows.iterrows():
        if row["event_label"]:
            fig.add_annotation(
                x=row["year"],
                y=row["reconstruction_error"],
                text=row["event_label"],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                ax=0,
                ay=-40,
                font=dict(size=10, color="black"),
                bgcolor="rgba(255,255,200,0.85)",
                bordercolor="red",
                borderwidth=1,
                borderpad=3,
            )

fig.update_layout(
    title=f"{selected_name} — Reconstruction Error Over Time",
    xaxis_title="Year",
    yaxis_title="Reconstruction error",
    template="plotly_white",
    hovermode="x unified",
    height=500,
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Detected Anomalies")
if anomaly_rows.empty:
    st.info("No anomalies flagged for this country.")
else:
    table = anomaly_rows[["year", "event_label", "reconstruction_error"]].copy()
    table["reconstruction_error"] = table["reconstruction_error"].round(4)
    table = table.sort_values("year").reset_index(drop=True)
    st.dataframe(table, use_container_width=True)
