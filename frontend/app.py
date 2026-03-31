import os

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
API_URL = os.environ.get("API_URL", "http://localhost:8000")
DEFAULT_YEAR_WINDOW = 10
MIN_EMISSION_THRESHOLD = 10.0  # Mt CO2 (was 10,000 kt)
TOP_N_OPTIONS = [5, 10, 15, 20]

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Global CO2 Emissions", page_icon="🌍", layout="wide")


# -----------------------------------------------------------------------------
# 2. API Client Functions
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_metadata() -> dict:
    """Fetch dataset metadata from the API."""
    resp = httpx.get(f"{API_URL}/data/metadata", timeout=30)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def fetch_countries() -> list[dict]:
    """Fetch list of countries from the API."""
    resp = httpx.get(f"{API_URL}/data/countries", timeout=30)
    resp.raise_for_status()
    return resp.json()["countries"]


@st.cache_data(ttl=300)
def fetch_emissions(start_year: int, end_year: int, countries: str | None = None) -> pd.DataFrame:
    """Fetch emissions data from the API and return as DataFrame."""
    params: dict[str, int | str] = {"start_year": start_year, "end_year": end_year}
    if countries:
        params["countries"] = countries
    resp = httpx.get(f"{API_URL}/data/emissions", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]
    if not data:
        return pd.DataFrame(columns=["country", "iso_code", "year", "co2"])
    return pd.DataFrame(data)


# -----------------------------------------------------------------------------
# 3. Load Data with Error Handling
# -----------------------------------------------------------------------------
try:
    metadata = fetch_metadata()
    countries_list = fetch_countries()
except (httpx.ConnectError, httpx.HTTPStatusError):
    st.error(f"❌ Cannot connect to API at {API_URL}")
    st.info("Please ensure the API server is running.")
    st.stop()

min_year = metadata["min_year"]
max_year = metadata["max_year"]
all_country_names = sorted([c["name"] for c in countries_list])
# Build name → iso_code lookup
name_to_iso = {c["name"]: c["iso_code"] for c in countries_list}

# Get Top 10 emitters for default selection
df_latest_for_default = fetch_emissions(max_year, max_year)
if not df_latest_for_default.empty:
    top_10_default = df_latest_for_default.nlargest(10, "co2")["country"].tolist()
else:
    top_10_default = all_country_names[:10]

# -----------------------------------------------------------------------------
# 4. Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")

# Year Range Slider
year_range = st.sidebar.slider(
    "Select Year Range",
    min_year,
    max_year,
    (max_year - DEFAULT_YEAR_WINDOW, max_year),
)

# Country Multi-select
selected_countries = st.sidebar.multiselect(
    "Select Countries (for Trend Chart)",
    options=all_country_names,
    default=top_10_default[:5],
    help="Choose countries to display in the time series chart",
)

# Top N selector
top_n = st.sidebar.selectbox("Top N Countries (Bar Chart)", options=TOP_N_OPTIONS, index=1)

# Chart type selector
chart_type = st.sidebar.radio("Time Series Chart Type", ["Area", "Line"], index=0)


# -----------------------------------------------------------------------------
# 5. Main Dashboard UI
# -----------------------------------------------------------------------------
st.title("Global CO2 Emissions Dashboard")

st.markdown(
    f"""
Interactive exploration of CO2 emissions across countries ({min_year}–{max_year}).<br>
**Data source:** <a href="https://github.com/owid/co2-data" target="_blank">Our World in Data</a> (Updated to {max_year})<br>
**Exploratory Data Analysis (EDA) Report:** <a href="https://github.com/jurinho17-sv/global-co2-insight/blob/main/notebooks/01_data_eda.ipynb" target="_blank">Jupyter Notebook</a><br>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 6. Fetch data for selected year range
# -----------------------------------------------------------------------------
df = fetch_emissions(year_range[0], year_range[1])
df_latest_year = df[df["year"] == year_range[1]]

# -----------------------------------------------------------------------------
# 7. Key Metrics Row
# -----------------------------------------------------------------------------
if not df_latest_year.empty:
    total_emissions = df_latest_year["co2"].sum()
    top_emitter_row = df_latest_year.loc[df_latest_year["co2"].idxmax()]

    # Calculate YoY change
    if year_range[1] > min_year:
        df_prev_year = df[df["year"] == year_range[1] - 1]
        prev_total = df_prev_year["co2"].sum()
        yoy_change = ((total_emissions - prev_total) / prev_total) * 100 if prev_total > 0 else 0
    else:
        yoy_change = 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Selected Period", f"{year_range[0]}–{year_range[1]}")
    with col2:
        st.metric(f"Total Emissions ({year_range[1]})", f"{total_emissions:,.1f} Mt", delta=f"{yoy_change:+.1f}% YoY")
    with col3:
        st.metric("Top Emitter", top_emitter_row["country"])
    with col4:
        st.metric("Countries Tracked", f"{df_latest_year['country'].nunique()}")

    st.markdown("---")

# -----------------------------------------------------------------------------
# 8. TABS: Volume Analysis | Growth Analysis
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["Volume Analysis", "Growth Analysis"])

# =============================================================================
# TAB 1: Volume Analysis
# =============================================================================
with tab1:
    chart_col1, chart_col2 = st.columns([1.2, 1])

    # -- LEFT: Time Series Chart --
    with chart_col1:
        st.subheader(f"CO2 Emissions Trend ({year_range[0]}–{year_range[1]})")

        if not selected_countries:
            st.warning("⚠️ Please select at least one country from the sidebar.")
        else:
            # Fetch only selected countries
            selected_isos = [name_to_iso[c] for c in selected_countries if c in name_to_iso]
            df_trend = fetch_emissions(year_range[0], year_range[1], ",".join(selected_isos))

            if not df_trend.empty:
                # Sort by total emissions (descending) for consistent ordering
                country_order = df_trend.groupby("country")["co2"].sum().sort_values(ascending=False).index.tolist()
                df_trend["country"] = pd.Categorical(df_trend["country"], categories=country_order, ordered=True)
                df_trend = df_trend.sort_values(["year", "country"])

                if chart_type == "Area":
                    fig_trend = px.area(
                        df_trend,
                        x="year",
                        y="co2",
                        color="country",
                        labels={"co2": "CO2 Emissions (Mt)", "year": "Year", "country": "Country"},
                        template="plotly_white",
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        category_orders={"country": country_order},
                    )
                else:
                    fig_trend = px.line(
                        df_trend,
                        x="year",
                        y="co2",
                        color="country",
                        labels={"co2": "CO2 Emissions (Mt)", "year": "Year", "country": "Country"},
                        template="plotly_white",
                        markers=True,
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        category_orders={"country": country_order},
                    )

                fig_trend.update_layout(hovermode="x unified", legend_title="Country", height=450)
                st.plotly_chart(fig_trend, use_container_width=True)

    # -- RIGHT: Top N Bar Chart --
    with chart_col2:
        st.subheader(f"Top {top_n} Emitters ({year_range[1]})")

        top_n_data = df_latest_year.nlargest(top_n, "co2")

        fig_bar = px.bar(
            top_n_data,
            x="co2",
            y="country",
            orientation="h",
            text_auto=".2s",
            labels={"co2": "CO2 Emissions (Mt)", "country": ""},
            template="plotly_white",
            color="co2",
            color_continuous_scale="Reds",
        )
        fig_bar.update_layout(
            yaxis={"categoryorder": "total ascending"}, showlegend=False, height=450, coloraxis_showscale=False
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

# =============================================================================
# TAB 2: Growth Analysis
# =============================================================================
with tab2:
    if year_range[0] < min_year:
        st.warning(f"⚠️ Start year {year_range[0]} is before data begins ({min_year}). Adjusting to {min_year}.")
        effective_start = min_year
    else:
        effective_start = year_range[0]

    st.subheader(f"Fastest Growing Emitters ({effective_start}→{year_range[1]})")

    # Calculate growth rates
    df_start = df[df["year"] == effective_start].set_index("country")["co2"]
    df_end = df[df["year"] == year_range[1]].set_index("country")["co2"]

    # Filter significant countries (> threshold Mt in end year)
    significant = df_end[df_end > MIN_EMISSION_THRESHOLD].index

    # Only calculate for countries that exist in BOTH years
    valid_countries = df_start.index.intersection(significant)

    if valid_countries.empty:
        st.warning("⚠️ No significant countries found for comparison. Try adjusting the year range.")
    else:
        growth_rate = (
            (df_end.loc[valid_countries] - df_start.loc[valid_countries]) / df_start.loc[valid_countries] * 100
        ).dropna()

        # Top 5 growth countries
        top_5_growth = growth_rate.nlargest(5)
        top_5_growth_list = top_5_growth.index.tolist()

        # -- Growth Chart --
        growth_col1, growth_col2 = st.columns([1.5, 1])

        with growth_col1:
            df_growth_viz = df[
                (df["country"].isin(top_5_growth_list)) & (df["year"].between(effective_start, year_range[1]))
            ].copy()

            country_order_growth = top_5_growth_list

            df_growth_viz["country"] = pd.Categorical(
                df_growth_viz["country"], categories=country_order_growth, ordered=True
            )
            df_growth_viz = df_growth_viz.sort_values(["year", "country"])

            if chart_type == "Area":
                fig_growth = px.area(
                    df_growth_viz,
                    x="year",
                    y="co2",
                    color="country",
                    title=f"CO2 Emissions: Fastest Growing Countries ({effective_start}-{year_range[1]})",
                    labels={"co2": "CO2 Emissions (Mt)", "year": "Year", "country": "Country"},
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                    category_orders={"country": country_order_growth},
                )
            else:
                fig_growth = px.line(
                    df_growth_viz,
                    x="year",
                    y="co2",
                    color="country",
                    title=f"CO2 Emissions: Fastest Growing Countries ({effective_start}-{year_range[1]})",
                    labels={"co2": "CO2 Emissions (Mt)", "year": "Year", "country": "Country"},
                    template="plotly_white",
                    markers=True,
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                    category_orders={"country": country_order_growth},
                )

            # Add growth rate annotations
            if chart_type == "Area":
                end_year_values = {}
                for country in country_order_growth:
                    country_data = df_growth_viz[df_growth_viz["country"] == country]
                    last_row = country_data[country_data["year"] == year_range[1]]
                    if not last_row.empty:
                        end_year_values[country] = last_row["co2"].values[0]

                cumulative = 0
                cumulative_positions = {}
                for country in country_order_growth:
                    if country in end_year_values:
                        cumulative += end_year_values[country]
                        cumulative_positions[country] = cumulative

                for country in country_order_growth:
                    if country in cumulative_positions:
                        growth_pct = top_5_growth[country]
                        fig_growth.add_annotation(
                            x=year_range[1],
                            y=cumulative_positions[country],
                            text=f"+{growth_pct:.1f}%",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            ax=50,
                            ay=0,
                            font=dict(size=10, color="black", family="Arial Black"),
                            bgcolor="rgba(255,255,255,0.9)",
                            borderwidth=2,
                            borderpad=3,
                        )
            else:
                for i, country in enumerate(country_order_growth):
                    country_data = df_growth_viz[df_growth_viz["country"] == country]
                    growth_pct = top_5_growth[country]

                    last_row = country_data[country_data["year"] == year_range[1]]
                    if not last_row.empty:
                        fig_growth.add_annotation(
                            x=year_range[1],
                            y=last_row["co2"].values[0],
                            text=f"+{growth_pct:.1f}%",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            ax=40,
                            ay=-20,
                            font=dict(size=11, color="black", family="Arial Black"),
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor=fig_growth.data[i].line.color
                            if i < len(fig_growth.data) and hasattr(fig_growth.data[i], "line")
                            else "gray",
                            borderwidth=2,
                            borderpad=4,
                        )

            fig_growth.update_traces(line=dict(width=3) if chart_type == "Line" else {})
            if chart_type == "Line":
                fig_growth.update_traces(mode="lines+markers", marker=dict(size=6))

            fig_growth.update_layout(hovermode="x unified", height=500, legend_title="Country", title_font_size=16)
            st.plotly_chart(fig_growth, use_container_width=True)

        with growth_col2:
            st.markdown("#### Growth Rate Rankings")

            # Fastest Growing
            st.markdown(f"**🔺 Fastest Growing ({effective_start}→{year_range[1]})**")
            for i, (country, rate) in enumerate(top_5_growth.items(), 1):
                start_val = df_start.get(country, 0)
                end_val = df_end.get(country, 0)
                st.markdown(f"{i}. **{country}**: +{rate:.1f}%")
                st.caption(f"   {start_val:,.1f} → {end_val:,.1f} Mt")

            st.markdown("---")

            # Biggest Decreases
            st.markdown(f"**🔻 Biggest Decreases ({effective_start}→{year_range[1]})**")
            bottom_5_growth = growth_rate.nsmallest(5)
            for i, (country, rate) in enumerate(bottom_5_growth.items(), 1):
                start_val = df_start.get(country, 0)
                end_val = df_end.get(country, 0)
                st.markdown(f"{i}. **{country}**: {rate:.1f}%")
                st.caption(f"   {start_val:,.1f} → {end_val:,.1f} Mt")

# -----------------------------------------------------------------------------
# 9. Data Preview
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("View Raw Data"):
    df_preview = (
        df[df["year"] == year_range[1]][["country", "iso_code", "year", "co2"]]
        .sort_values("co2", ascending=False)
        .head(50)
        .copy()
    )
    df_preview["co2"] = df_preview["co2"].round(2)
    st.dataframe(df_preview, use_container_width=True)

# -----------------------------------------------------------------------------
# 10. Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    Built by <strong>Ju Ho Kim</strong> |
    <a href="https://github.com/jurinho17-sv/global-co2-insight" target="_blank">GitHub</a> |
    Data: <a href="https://github.com/owid/co2-data" target="_blank">Our World in Data</a>
    ({min_year}-{max_year})
</div>
""",
    unsafe_allow_html=True,
)
