import os

import httpx
import plotly.graph_objects as go
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="CO2 Forecast", page_icon="📈", layout="wide")
st.title("CO2 Emissions Forecast")
st.markdown(
    "Project future CO2 emissions for a country using the trained NHiTS forecaster, "
    "with optional conformal prediction intervals."
)


@st.cache_data(ttl=300)
def fetch_countries() -> list[dict]:
    resp = httpx.get(f"{API_URL}/data/countries", timeout=30)
    resp.raise_for_status()
    return resp.json()["countries"]


@st.cache_data(ttl=300)
def fetch_history(iso: str) -> tuple[list[int], list[float]]:
    resp = httpx.get(
        f"{API_URL}/data/emissions",
        params={"start_year": 1960, "end_year": 2100, "countries": iso},
        timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json()["data"]
    rows.sort(key=lambda r: r["year"])
    return [r["year"] for r in rows], [r["co2"] for r in rows]


try:
    countries = fetch_countries()
except (httpx.HTTPError, httpx.HTTPStatusError) as e:
    st.error(f"❌ Cannot reach API at {API_URL}: {e}")
    st.stop()

names = sorted(c["name"] for c in countries)
name_to_iso = {c["name"]: c["iso_code"] for c in countries}

col_pick, col_horizon = st.columns([2, 1])
with col_pick:
    default_idx = names.index("United States") if "United States" in names else 0
    selected_name = st.selectbox("Country", options=names, index=default_idx)
with col_horizon:
    horizon = st.slider("Forecast horizon (years)", min_value=5, max_value=20, value=10)

if st.button("Generate Forecast", type="primary"):
    iso = name_to_iso[selected_name]

    try:
        with st.spinner("Running forecaster…"):
            resp = httpx.get(
                f"{API_URL}/forecast/{iso}",
                params={"horizon": horizon},
                timeout=60,
            )
            resp.raise_for_status()
            forecast = resp.json()
            hist_years, hist_values = fetch_history(iso)
    except httpx.HTTPStatusError as e:
        st.error(f"❌ Forecast failed ({e.response.status_code}): {e.response.text}")
        st.stop()
    except httpx.HTTPError as e:
        st.error(f"❌ API error: {e}")
        st.stop()

    pred_years = forecast["prediction_years"]
    predictions = forecast["predictions"]
    intervals = forecast.get("intervals")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist_years,
            y=hist_values,
            mode="lines",
            name="Historical",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pred_years,
            y=predictions,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        )
    )
    if intervals is not None:
        fig.add_trace(
            go.Scatter(
                x=pred_years,
                y=intervals["upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        coverage_pct = int(round(intervals["coverage"] * 100))
        fig.add_trace(
            go.Scatter(
                x=pred_years,
                y=intervals["lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(255,127,14,0.2)",
                name=f"{coverage_pct}% interval",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"{selected_name} — CO2 Emissions Forecast ({forecast['model']})",
        xaxis_title="Year",
        yaxis_title="CO2 Emissions (Mt)",
        template="plotly_white",
        hovermode="x unified",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    next_year_value = predictions[0]
    next_year = pred_years[0]
    delta = predictions[-1] - predictions[0]
    if abs(delta) < 1.0:
        trend_label = "→ Flat"
    elif delta > 0:
        trend_label = "↑ Increasing"
    else:
        trend_label = "↓ Decreasing"

    m1, m2 = st.columns(2)
    with m1:
        st.metric(f"Next-year forecast ({next_year})", f"{next_year_value:,.1f} Mt")
    with m2:
        st.metric(
            f"{horizon}-year trend direction",
            trend_label,
            delta=f"{delta:+.1f} Mt over horizon",
        )
else:
    st.info("Pick a country and horizon, then click **Generate Forecast**.")
