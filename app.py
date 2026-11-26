import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SaaS Revenue Risk & Unit Economics Engine", layout="wide")

st.title("SaaS Dark Churn & Unit Economics Risk Engine")
st.markdown(
    "Early-warning system detecting silent contract downgrades 6 months prior to renewal "
    "and isolating negative-margin accounts."
)

# Load trained model artifacts if available; fallback to analytical scoring if run standalone
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("churn_risk_model.pkl")
        scaler = joblib.load("feature_scaler.pkl")
        return model, scaler
    except Exception:
        return None, None

model, scaler = load_artifacts()

tab1, tab2 = st.tabs(["1. Account Risk & Margin Simulator", "2. Portfolio Audit Overview"])

# --- TAB 1: INTERACTIVE SIMULATOR ---
with tab1:
    st.subheader("Simulate Client Usage & Cloud Cost")
    col1, col2 = st.columns([1, 1.3])

    with col1:
        tier = st.selectbox("Subscription Tier", ["Basic ($20/seat)", "Pro ($50/seat)", "Enterprise ($120/seat)"])
        price_map = {"Basic ($20/seat)": 20, "Pro ($50/seat)": 50, "Enterprise ($120/seat)": 120}
        unit_price = price_map[tier]

        paid_seats = st.slider("Paid Seats", min_value=5, max_value=500, value=100)
        active_seats = st.slider("Current Active Seats (Month 6)", min_value=0, max_value=paid_seats, value=42)
        util_drop_3m = st.slider("3-Month Utilization Change (%)", min_value=-60.0, max_value=20.0, value=-22.0)
        api_calls_per_user = st.slider("Avg Monthly API Calls per Active User", min_value=50, max_value=60000, value=120)

    # Calculate Real-Time Unit Economics
    monthly_revenue = paid_seats * unit_price
    total_api_calls = active_seats * api_calls_per_user
    server_cost = total_api_calls * 0.002  # $0.002 per API call
    net_profit = monthly_revenue - server_cost
    gross_margin_pct = (net_profit / monthly_revenue) * 100 if monthly_revenue > 0 else 0
    seat_util_pct = (active_seats / paid_seats) * 100

    # Predict Downgrade Probability using Trained Logistic Regression
    if model is not None and scaler is not None:
        input_features = np.array([[seat_util_pct, util_drop_3m, total_api_calls, gross_margin_pct, paid_seats]])
        scaled_features = scaler.transform(input_features)
        downgrade_prob = model.predict_proba(scaled_features)[0][1] * 100
    else:
        # Fallback analytical formula if .pkl files haven't been generated locally yet
        raw_score = (60 - seat_util_pct) * 0.06 + (-util_drop_3m) * 0.05
        downgrade_prob = (1 / (1 + np.exp(-raw_score))) * 100

    with col2:
        st.markdown("### Executive Account Diagnosis")
        m1, m2, m3 = st.columns(3)
        m1.metric("Monthly Revenue (MRR)", f"${monthly_revenue:,.0f}")
        m2.metric("Cloud Server Cost", f"${server_cost:,.0f}", delta=f"{gross_margin_pct:.1f}% Margin", delta_color="normal" if gross_margin_pct >= 20 else "inverse")
        m3.metric("Downgrade Risk (Month 12)", f"{downgrade_prob:.1f}%")

        st.divider()

        # Actionable Business Recommendation Engine
        if gross_margin_pct < 20:
            st.error("**ALERT: Toxic Whale (Margin Erosion)** — Cloud API consumption is eating over 80% of this client's subscription fee. Recommended Action: Enforce API rate limits or transition client to usage-based overage billing.")
        elif downgrade_prob >= 60 or seat_util_pct < 45:
            st.warning("**ALERT: High Dark Churn Risk** — Active seat utilization has dropped sharply. Recommended Action: Trigger Customer Success onboarding intervention immediately (6 months prior to renewal).")
        else:
            st.success("**HEALTHY ACCOUNT** — Strong seat adoption and healthy gross margin. Candidate for upsell/expansion.")

        # Visual Breakdown of Revenue vs Cost
        fig, ax = plt.subplots(figsize=(6, 2.5))
        bars = ax.barh(["Cloud Cost", "Net Profit", "Total MRR"], [server_cost, max(0, net_profit), monthly_revenue], color=["#e74c3c", "#2ecc71", "#3498db"])
        ax.set_xlabel("USD ($)")
        ax.set_title("Monthly Account Unit Economics Breakdown")
        st.pyplot(fig)

# --- TAB 2: PORTFOLIO OVERVIEW ---
with tab2:
    st.subheader("Portfolio-Wide Audit Findings")
    try:
        df = pd.read_csv("enriched_saas_metrics.csv")
        m12 = df[df["month"] == 12]

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Accounts Audited", len(m12))
        at_risk_mrr = m12[m12["health_segment"] == "Dark Churn Risk (Silent Drop)"]["monthly_revenue"].sum()
        c2.metric("Monthly Revenue at Risk", f"${at_risk_mrr:,.0f}")
        toxic_loss = m12[m12["health_segment"] == "Toxic Whale (Low/Negative Margin)"]["net_profit"].sum()
        c3.metric("Toxic Whale Net Profit Impact", f"${toxic_loss:,.0f}")

        st.dataframe(m12[["account_id", "tier", "monthly_revenue", "server_cost", "gross_margin_pct", "seat_utilization_pct", "health_segment"]].head(20), use_container_width=True)
    except FileNotFoundError:
        st.info("Run `01_generate_data.py` and `02_unit_economics.py` locally to populate portfolio-wide CSV tables here.")
