import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Load and Merge the Contract & Usage Tables
accounts = pd.read_csv("accounts.csv")
usage = pd.read_csv("monthly_usage.csv")

# Left join usage logs with account contract details
df = pd.merge(usage, accounts, on="account_id", how="left")

# 2. Engineer Core SaaS Business KPIs
# Seat Utilization Rate: What % of paid licenses are actually being used?
df["seat_utilization_pct"] = (df["active_seats"] / df["paid_seats"]) * 100

# Net Profit & Gross Margin: Revenue minus cloud infrastructure cost
df["net_profit"] = df["monthly_revenue"] - df["server_cost"]
df["gross_margin_pct"] = (df["net_profit"] / df["monthly_revenue"]) * 100

# Sort chronologically per account to track month-over-month changes
df = df.sort_values(by=["account_id", "month"]).reset_index(drop=True)

# 3-Month Rolling Utilization: Smooths out random monthly holidays/vacations
df["rolling_3m_utilization"] = df.groupby("account_id")[
    "seat_utilization_pct"
].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

# Usage Velocity (Drop Rate): Difference between current utilization and 3 months ago
df["utilization_drop_3m"] = df.groupby("account_id")[
    "seat_utilization_pct"
].diff(periods=3)


# 3. Rule-Based Business Segmentation (Tagging Risk Cohorts)
def classify_account_health(row):
  if row["gross_margin_pct"] < 20:
    return "Toxic Whale (Low/Negative Margin)"
  elif row["seat_utilization_pct"] < 45 or (
      pd.notna(row["utilization_drop_3m"]) and row["utilization_drop_3m"] < -15
  ):
    return "Dark Churn Risk (Silent Drop)"
  else:
    return "Healthy Account"


df["health_segment"] = df.apply(classify_account_health, axis=1)

# Save the enriched analytical dataset for SQL & Machine Learning steps
df.to_csv("enriched_saas_metrics.csv", index=False)

# 4. Print Executive Summary for Month 12
m12 = df[df["month"] == 12]
print("--- MONTH 12 EXECUTIVE HEALTH AUDIT ---")
print(m12["health_segment"].value_counts())
print(
    "\nTotal Monthly Revenue at Risk (Dark Churn):"
    f" ${m12[m12['health_segment'] == 'Dark Churn Risk (Silent Drop)']['monthly_revenue'].sum():,.2f}"
)
print(
    "Total Monthly Loss from Toxic Whales:"
    f" ${m12[m12['health_segment'] == 'Toxic Whale (Low/Negative Margin)']['net_profit'].sum():,.2f}"
)

# 5. Generate Publication-Ready Executive Visuals
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Chart A: Seat Utilization Decay Over 12 Months (Downgraded vs Retained)
trend_data = (
    df.groupby(["month", "account_id"])
    .agg(
        seat_utilization_pct=("seat_utilization_pct", "mean"),
        ever_downgraded=("downgraded_at_renewal", "max"),
    )
    .reset_index()
)

# Propagate the Month 12 downgrade flag across all 12 months per account
downgrade_map = (
    df.groupby("account_id")["downgraded_at_renewal"].max().to_dict()
)
df["will_downgrade"] = df["account_id"].map(downgrade_map)

sns.lineplot(
    data=df,
    x="month",
    y="seat_utilization_pct",
    hue="will_downgrade",
    palette={0: "#2ecc71", 1: "#e74c3c"},
    linewidth=2.5,
    ax=axes[0],
)
axes[0].set_title(
    "Early Warning Signal: Seat Utilization Decay (12 Months)",
    fontsize=12,
    fontweight="bold",
)
axes[0].set_xlabel("Contract Month")
axes[0].set_ylabel("Active Seat Utilization (%)")
axes[0].legend(
    title="Renewal Outcome", labels=["Renewed (Healthy)", "Downgraded (Churn)"]
)
axes[0].axvline(x=6, color="gray", linestyle="--", alpha=0.7)
axes[0].text(6.2, 80, "6-Month Early\nWarning Window", color="gray", fontsize=9)

# Chart B: Revenue vs. Server Cost (Exposing Toxic Whales in Month 12)
sns.scatterplot(
    data=m12,
    x="monthly_revenue",
    y="server_cost",
    hue="health_segment",
    palette={
        "Healthy Account": "#2ecc71",
        "Dark Churn Risk (Silent Drop)": "#f39c12",
        "Toxic Whale (Low/Negative Margin)": "#8e44ad",
    },
    alpha=0.8,
    s=70,
    ax=axes[1],
)
# Plot Break-Even Line (Where Server Cost = Monthly Revenue)
max_val = max(m12["monthly_revenue"].max(), m12["server_cost"].max())
axes[1].plot(
    [0, max_val],
    [0, max_val],
    color="red",
    linestyle="--",
    label="Break-Even Line (0% Margin)",
)
axes[1].set_title(
    "Unit Economics Audit: Monthly Revenue vs. Cloud Cost",
    fontsize=12,
    fontweight="bold",
)
axes[1].set_xlabel("Monthly Subscription Revenue ($)")
axes[1].set_ylabel("Monthly Cloud Server Cost ($)")
axes[1].legend(loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig("unit_economics_analysis.png", dpi=300)
print("\nSaved executive chart: unit_economics_analysis.png")
