# SaaS "Dark Churn" & Unit Economics Risk Engine

An end-to-end B2B SaaS revenue intelligence and predictive risk pipeline designed to detect **silent contract downgrades ("Dark Churn")** 6 months prior to renewal and isolate **negative-margin accounts ("Toxic Whales")**.

---

## 1. The Business Problem
Standard churn models wait until a customer cancels their subscription—by which point the revenue is already lost. In enterprise B2B SaaS, two hidden issues quietly erode profitability long before renewal:
1. **Silent Contraction (Dark Churn):** Clients maintain active subscriptions while seat utilization steadily decays over 6–9 months, resulting in severe contract downgrades at renewal.
2. **Gross Margin Cannibalization (Toxic Whales):** High-tier clients generate heavy automated API traffic whose underlying cloud infrastructure costs exceed their monthly subscription fees.

---

## 2. End-to-End Pipeline Architecture

```text
[01_generate_data.py] ──▶ Relational Synthetic Logs (500 Accounts × 12 Months)
          │
          ▼
[02_unit_economics.py] ──▶ KPI Engineering (Rolling Seat Utilization, Net Profit, Gross Margin)
          │
          ▼
[03_sql_analytics.py]  ──▶ SQLite Star Schema (dim_accounts & fact_monthly_usage + LAG/RANK Queries)
          │
          ▼
[04_risk_model.py]     ──▶ Month-6 Logistic Regression Classifier (Leakage-Free Early Warning)
          │
          ▼
[app.py]               ──▶ Interactive Streamlit Executive Simulator & Portfolio Audit Dashboard
