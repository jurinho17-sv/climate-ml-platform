import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from co2_ml.config import DATA_PATH, MIN_EMISSION_THRESHOLD, DEFAULT_YEAR_WINDOW, TOP_N_OPTIONS

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Global CO2 Emissions",
    page_icon="🌍",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. Data Loading & Cleaning Function
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data() -> pd.DataFrame:
    """
    Loads the CO2 dataset and applies robust cleaning to remove aggregates.
    Updated: Now supports 2023 data from World Bank.
    """
    df = pd.read_csv(DATA_PATH)

    # --- Robust Cleaning Logic (Verified in EDA Notebook) ---
    ignore_list = [
        'World', 'Arab World', 'Central Europe and the Baltics',
        'East Asia & Pacific', 'East Asia & Pacific (excluding high income)', 
        'East Asia & Pacific (IDA & IBRD countries)',
        'Europe & Central Asia', 'Europe & Central Asia (excluding high income)', 
        'Europe & Central Asia (IDA & IBRD countries)',
        'European Union', 'Euro area',
        'Latin America & Caribbean', 'Latin America & Caribbean (excluding high income)', 
        'Latin America & the Caribbean (IDA & IBRD countries)',
        'Middle East & North Africa', 'Middle East & North Africa (excluding high income)', 
        'Middle East & North Africa (IDA & IBRD countries)',
        'North America', 'South Asia', 'South Asia (IDA & IBRD)',
        'Sub-Saharan Africa', 'Sub-Saharan Africa (excluding high income)', 
        'Sub-Saharan Africa (IDA & IBRD countries)',
        'Africa Eastern and Southern', 'Africa Western and Central',
        'High income', 'Low & middle income', 'Low income', 'Lower middle income', 
        'Middle income', 'Upper middle income', 'OECD members',
        'IDA & IBRD total', 'IBRD only', 'IDA total', 'IDA blend', 'IDA only',
        'Heavily indebted poor countries (HIPC)', 
        'Least developed countries: UN classification',
        'Fragile and conflict affected situations', 
        'Early-demographic dividend', 'Late-demographic dividend', 
        'Pre-demographic dividend', 'Post-demographic dividend',
        'Small states', 'Pacific island small states', 'Caribbean small states', 
        'Other small states', 'Not classified',
        # Additional aggregates in new dataset
        'Africa', 'Asia', 'Asia (excl. China and India)', 'Europe',
        'Europe (excl. EU-27)', 'Europe (excl. EU-28)', 
        'European Union (27)', 'European Union (28)',
        'North America (excl. USA)', 'Oceania', 'South America',
        'Non-OECD (GCP)', 'OECD (GCP)',
        'High-income countries', 'Low-income countries',
        'Lower-middle-income countries', 'Upper-middle-income countries'
    ]

    aggregate_codes = [
        'ARB', 'CSS', 'EAP', 'EAS', 'ECA', 'ECS', 'EMU', 'EUU',
        'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDX', 'INX',
        'LAC', 'LCN', 'LDC', 'LIC', 'LMC', 'LMY', 'LTE', 'MEA',
        'MIC', 'MNA', 'NAC', 'OED', 'OSS', 'PRE', 'PSS', 'PST',
        'SAS', 'SSA', 'SSF', 'SST', 'TEA', 'TEC', 'TLA', 'TMN',
        'TSA', 'TSS', 'UMC', 'WLD',
        # Additional codes
        'OWID_AFR', 'OWID_ASI', 'OWID_EUR', 'OWID_EUN', 'OWID_NAM',
        'OWID_OCE', 'OWID_SAM', 'OWID_WRL', 'OWID_HIC', 'OWID_LIC',
        'OWID_LMC', 'OWID_UMC', 'OWID_NON_OECD', 'OWID_OECD'
    ]

    def looks_like_aggregate(name):
        if pd.isna(name):
            return True
        patterns = ['income', 'dividend', 'IBRD', 'IDA', 'excluding', 'excl.', 
                   '(GCP)', 'European Union', 'OECD']
        return any(p in str(name) for p in patterns)

    df_clean = df[
        (~df['country_name'].isin(ignore_list)) &
        (~df['country_code'].isin(aggregate_codes)) &
        (~df['country_name'].apply(looks_like_aggregate))
    ].copy()
    
    return df_clean

# -----------------------------------------------------------------------------
# 3. Load Data with Error Handling
# -----------------------------------------------------------------------------
try:
    df = load_and_clean_data()
except FileNotFoundError:
    st.error("❌ Error: Data file not found.")
    st.info(f"Please ensure '{DATA_PATH}' exists.")
    st.stop()

# Get data ranges
min_year = int(df['year'].min())
max_year = int(df['year'].max())
all_countries = sorted(df['country_name'].unique())

# Get Top 10 emitters (for default selection)
df_latest = df[df['year'] == max_year]
top_10_default = df_latest.nlargest(10, 'value')['country_name'].tolist()

# -----------------------------------------------------------------------------
# 4. Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")

# Year Range Slider
year_range = st.sidebar.slider(
    "Select Year Range",
    min_year,
    max_year,
    (max_year - DEFAULT_YEAR_WINDOW, max_year)  # Default: last N years
)

# Country Multi-select
selected_countries = st.sidebar.multiselect(
    "Select Countries (for Trend Chart)",
    options=all_countries,
    default=top_10_default[:5],
    help="Choose countries to display in the time series chart"
)

# Top N selector
top_n = st.sidebar.selectbox(
    "Top N Countries (Bar Chart)",
    options=TOP_N_OPTIONS,
    index=1
)

# Chart type selector
chart_type = st.sidebar.radio(
    "Time Series Chart Type",
    ["Area", "Line"],
    index=0
)


# -----------------------------------------------------------------------------
# 5. Main Dashboard UI
# -----------------------------------------------------------------------------
st.title("Global CO2 Emissions Dashboard") 

st.markdown(f"""
Interactive exploration of CO2 emissions across countries ({min_year}–{max_year}).<br>
**Data source:** <a href="https://data360.worldbank.org/en/dataset/OWID_CB" target="_blank">World Bank / Our World in Data</a> (Updated to {max_year})<br>
**Exploratory Data Analysis (EDA) Report:** <a href="https://github.com/jurinho17-sv/global-co2-insight/blob/main/notebooks/01_data_eda.ipynb" target="_blank">Jupyter Notebook</a><br>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. Key Metrics Row
# -----------------------------------------------------------------------------
df_latest_year = df[df['year'] == year_range[1]]
total_emissions = df_latest_year['value'].sum()
top_emitter = df_latest_year.loc[df_latest_year['value'].idxmax()]

# Calculate YoY change
if year_range[1] > min_year:
    df_prev_year = df[df['year'] == year_range[1] - 1]
    prev_total = df_prev_year['value'].sum()
    yoy_change = ((total_emissions - prev_total) / prev_total) * 100
else:
    yoy_change = 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Selected Period", f"{year_range[0]}–{year_range[1]}")
with col2:
    st.metric(
        f"Total Emissions ({year_range[1]})", 
        f"{total_emissions/1e6:.2f}M kt",
        delta=f"{yoy_change:+.1f}% YoY"
    )
with col3:
    st.metric("Top Emitter", top_emitter['country_name'])
with col4:
    st.metric("Countries Tracked", f"{df_latest_year['country_name'].nunique()}")

st.markdown("---")

# -----------------------------------------------------------------------------
# 7. TABS: Volume Analysis | Growth Analysis
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
            df_trend = df[
                (df['country_name'].isin(selected_countries)) &
                (df['year'].between(year_range[0], year_range[1]))
            ].copy()
            
            # Sort by total emissions (descending) for consistent ordering
            country_order = df_trend.groupby('country_name')['value'].sum().sort_values(ascending=False).index.tolist()
            df_trend['country_name'] = pd.Categorical(df_trend['country_name'], categories=country_order, ordered=True)
            df_trend = df_trend.sort_values(['year', 'country_name'])
            
            if chart_type == "Area":
                fig_trend = px.area(
                    df_trend, x='year', y='value', color='country_name',
                    labels={'value': 'CO2 Emissions (kt)', 'year': 'Year', 'country_name': 'Country'},
                    template='plotly_white',
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    category_orders={'country_name': country_order}
                )
            else:
                fig_trend = px.line(
                    df_trend, x='year', y='value', color='country_name',
                    labels={'value': 'CO2 Emissions (kt)', 'year': 'Year', 'country_name': 'Country'},
                    template='plotly_white', markers=True,
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    category_orders={'country_name': country_order}
                )
            
            fig_trend.update_layout(hovermode='x unified', legend_title='Country', height=450)
            st.plotly_chart(fig_trend, use_container_width=True)

    # -- RIGHT: Top N Bar Chart --
    with chart_col2:
        st.subheader(f"Top {top_n} Emitters ({year_range[1]})")
        
        top_n_data = df_latest_year.nlargest(top_n, 'value')
        
        fig_bar = px.bar(
            top_n_data, x='value', y='country_name', orientation='h',
            text_auto='.2s',
            labels={'value': 'CO2 Emissions (kt)', 'country_name': ''},
            template='plotly_white', color='value',
            color_continuous_scale='Reds'
        )
        fig_bar.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False, height=450, coloraxis_showscale=False
        )
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

# =============================================================================
# TAB 2: Growth Analysis (NEW - from EDA Chart 2)
# =============================================================================
with tab2:
    # [Safety Check] Verify both start and end year data exist
    if year_range[0] < min_year:
        st.warning(f"⚠️ Start year {year_range[0]} is before data begins ({min_year}). Adjusting to {min_year}.")
        effective_start = min_year
    else:
        effective_start = year_range[0]
    
    st.subheader(f"Fastest Growing Emitters ({effective_start}→{year_range[1]})")
    
    # Calculate growth rates
    df_start = df[df['year'] == effective_start].set_index('country_name')['value']
    df_end = df[df['year'] == year_range[1]].set_index('country_name')['value']
    
    # Filter significant countries (> 10,000 kt in end year)
    significant = df_end[df_end > MIN_EMISSION_THRESHOLD].index
    
    # [Safety] Only calculate for countries that exist in BOTH years
    valid_countries = df_start.index.intersection(significant)
    
    if valid_countries.empty:
        st.warning("⚠️ No significant countries found for comparison. Try adjusting the year range.")
    else:
        growth_rate = ((df_end.loc[valid_countries] - df_start.loc[valid_countries]) / df_start.loc[valid_countries] * 100).dropna()
        
        # Top 5 growth countries
        top_5_growth = growth_rate.nlargest(5)
        top_5_growth_list = top_5_growth.index.tolist()
        
        # -- Growth Chart --
        growth_col1, growth_col2 = st.columns([1.5, 1])
        
        with growth_col1:
            # Filter data for growth countries
            df_growth_viz = df[
                (df['country_name'].isin(top_5_growth_list)) &
                (df['year'].between(effective_start, year_range[1]))
            ].copy()
            
            # Always sort by growth rate (top_5_growth_list is already sorted by growth %)
            # This keeps consistency with the Rankings on the right side
            country_order_growth = top_5_growth_list
            
            df_growth_viz['country_name'] = pd.Categorical(
                df_growth_viz['country_name'], 
                categories=country_order_growth, 
                ordered=True
            )
            df_growth_viz = df_growth_viz.sort_values(['year', 'country_name'])
            
            # Create chart based on chart_type selection
            if chart_type == "Area":
                fig_growth = px.area(
                    df_growth_viz,
                    x='year', y='value', color='country_name',
                    title=f"CO2 Emissions: Fastest Growing Countries ({effective_start}-{year_range[1]})",
                    labels={'value': 'CO2 Emissions (kt)', 'year': 'Year', 'country_name': 'Country'},
                    template='plotly_white',
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                    category_orders={'country_name': country_order_growth}
                )
            else:
                fig_growth = px.line(
                    df_growth_viz,
                    x='year', y='value', color='country_name',
                    title=f"CO2 Emissions: Fastest Growing Countries ({effective_start}-{year_range[1]})",
                    labels={'value': 'CO2 Emissions (kt)', 'year': 'Year', 'country_name': 'Country'},
                    template='plotly_white',
                    markers=True,
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                    category_orders={'country_name': country_order_growth}
                )
            
            # Add growth rate annotations
            # For Area chart, calculate cumulative y-positions
            if chart_type == "Area":
                # Get end year values for each country in order
                end_year_values = {}
                for country in country_order_growth:
                    country_data = df_growth_viz[df_growth_viz['country_name'] == country]
                    last_row = country_data[country_data['year'] == year_range[1]]
                    if not last_row.empty:
                        end_year_values[country] = last_row['value'].values[0]
                
                # Calculate cumulative positions (stacked from bottom)
                cumulative = 0
                cumulative_positions = {}
                for country in country_order_growth:
                    if country in end_year_values:
                        cumulative += end_year_values[country]
                        cumulative_positions[country] = cumulative
                
                # Add annotations at cumulative positions
                for i, country in enumerate(country_order_growth):
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
                            font=dict(size=10, color='black', family='Arial Black'),
                            bgcolor='rgba(255,255,255,0.9)',
                            borderwidth=2,
                            borderpad=3
                        )
            else:
                # For Line chart, use individual y-values
                for i, country in enumerate(country_order_growth):
                    country_data = df_growth_viz[df_growth_viz['country_name'] == country]
                    growth_pct = top_5_growth[country]
                    
                    last_row = country_data[country_data['year'] == year_range[1]]
                    if not last_row.empty:
                        fig_growth.add_annotation(
                            x=year_range[1],
                            y=last_row['value'].values[0],
                            text=f"+{growth_pct:.1f}%",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            ax=40,
                            ay=-20,
                            font=dict(size=11, color='black', family='Arial Black'),
                            bgcolor='rgba(255,255,255,0.8)',
                            bordercolor=fig_growth.data[i].line.color if i < len(fig_growth.data) and hasattr(fig_growth.data[i], 'line') else 'gray',
                            borderwidth=2,
                            borderpad=4
                        )
            
            fig_growth.update_traces(line=dict(width=3) if chart_type == "Line" else {})
            if chart_type == "Line":
                fig_growth.update_traces(mode='lines+markers', marker=dict(size=6))
            
            fig_growth.update_layout(
                hovermode='x unified',
                height=500,
                legend_title='Country',
                title_font_size=16
            )
            st.plotly_chart(fig_growth, use_container_width=True)
        
        with growth_col2:
            st.markdown("#### Growth Rate Rankings")
            
            # Fastest Growing
            st.markdown(f"**🔺 Fastest Growing ({effective_start}→{year_range[1]})**")
            for i, (country, rate) in enumerate(top_5_growth.items(), 1):
                start_val = df_start.get(country, 0)
                end_val = df_end.get(country, 0)
                st.markdown(f"{i}. **{country}**: +{rate:.1f}%")
                st.caption(f"   {start_val:,.0f} → {end_val:,.0f} kt")
            
            st.markdown("---")
            
            # Biggest Decreases
            st.markdown(f"**🔻 Biggest Decreases ({effective_start}→{year_range[1]})**")
            bottom_5_growth = growth_rate.nsmallest(5)
            for i, (country, rate) in enumerate(bottom_5_growth.items(), 1):
                start_val = df_start.get(country, 0)
                end_val = df_end.get(country, 0)
                st.markdown(f"{i}. **{country}**: {rate:.1f}%")
                st.caption(f"   {start_val:,.0f} → {end_val:,.0f} kt")

# -----------------------------------------------------------------------------
# 8. Data Preview
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("View Raw Data"):
    df_preview = (
        df[df['year'] == year_range[1]][['country_name', 'country_code', 'year', 'value']]
        .sort_values('value', ascending=False)
        .head(50)
        .copy()
    )
    df_preview['value'] = df_preview['value'].round(0).astype(int)
    st.dataframe(df_preview, use_container_width=True)

# -----------------------------------------------------------------------------
# 9. Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    Built by <strong>Ju Ho Kim</strong> | 
    <a href="https://github.com/jurinho17-sv/global-co2-insight" target="_blank">GitHub</a> | 
    Data: <a href="https://data360.worldbank.org/en/dataset/OWID_CB" target="_blank">World Bank / Our World in Data</a> ({min_year}-{max_year})
</div>
""", unsafe_allow_html=True)
