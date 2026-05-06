import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Policy Impact", page_icon="🏛️", layout="wide")
st.title("Paris Agreement Policy Impact")
st.markdown(
    "Causal analysis of the **Paris Agreement (2016)** on CO2 emissions. "
    "Estimates the average treatment effect on the treated (ATT) for ratifying countries "
    "relative to a non-ratifier counterfactual."
)
st.caption("Based on staggered DiD with a DoWhy causal graph.")

if st.button("Run Causal Analysis", type="primary"):
    try:
        with st.spinner("Running causal analysis (this may take ~30s)…"):
            resp = httpx.post(
                f"{API_URL}/policy_effect",
                json={},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
    except httpx.ReadTimeout:
        st.error("⌛ Analysis took too long — please try again.")
        st.stop()
    except httpx.HTTPStatusError as e:
        st.error(f"❌ Analysis failed ({e.response.status_code}): {e.response.text}")
        st.stop()
    except httpx.HTTPError as e:
        st.error(f"❌ API error: {e}")
        st.stop()

    method = result.get("method", "did")

    if method == "did":
        att = result["att"]
        ci_lower = result["ci_lower"]
        ci_upper = result["ci_upper"]
        n_countries = result.get("n_countries")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric(
                "Average Treatment Effect (ATT)",
                f"{att:+.3f} Mt",
                delta=f"95% CI: [{ci_lower:+.3f}, {ci_upper:+.3f}]",
                delta_color="off",
            )
        with col2:
            if n_countries is not None:
                st.metric("Countries in Analysis", n_countries)

        if ci_upper < 0:
            interpretation = (
                "**Negative effect** — the Paris Agreement is associated with a statistically "
                "significant reduction in CO2 emissions among ratifying countries."
            )
        elif ci_lower > 0:
            interpretation = (
                "**Positive effect** — ratifying countries show a statistically significant "
                "increase in CO2 emissions post-agreement."
            )
        else:
            interpretation = (
                "**Inconclusive** — the 95% confidence interval crosses zero, so we cannot "
                "rule out a null effect at this confidence level."
            )
        st.markdown("### Interpretation")
        st.markdown(interpretation)

        with st.expander("Raw estimate"):
            st.json(result)
    else:
        st.warning(f"Unexpected method in response: {method}")
        st.json(result)
else:
    st.info("Click **Run Causal Analysis** to estimate the treatment effect.")
