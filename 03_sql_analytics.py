import sqlite3
import pandas as pd

# 1. Load Enriched Data & Build a Relational SQLite Warehouse
df = pd.read_csv("enriched_saas_metrics.csv")

# Connect to a local SQLite database (creates 'saas_warehouse.db' automatically)
conn = sqlite3.connect("saas_warehouse.db")

# Split into a Star Schema: Dimension Table (Contracts) & Fact Table (Monthly Logs)
dim_accounts = df[
    ["account_id", "tier", "paid_seats", "monthly_revenue"]
].drop_duplicates()
fact_usage = df[[
    "account_id",
    "month",
    "active_seats",
    "api_calls",
    "server_cost",
    "seat_utilization_pct",
    "net_profit",
    "gross_margin_pct",
    "downgraded_at_renewal",
]]

dim_accounts.to_sql("dim_accounts", conn, if_exists="replace", index=False)
fact_usage.to_sql("fact_monthly_usage", conn, if_exists="replace", index=False)

print(
    "SQLite Warehouse built successfully: 'dim_accounts' & 'fact_monthly_usage'"
    " tables loaded.\n"
)

# 2. SQL Query 1: Executive Tier Profitability Audit (JOIN + GROUP BY + HAVING)
# Business Question: Which subscription tiers generate real profit, and which have margin leakage?
query_1 = """
SELECT 
    a.tier,
    COUNT(DISTINCT a.account_id) AS total_accounts,
    ROUND(SUM(a.monthly_revenue), 2) AS total_mrr,
    ROUND(SUM(u.server_cost), 2) AS total_cloud_cost,
    ROUND(SUM(u.net_profit), 2) AS total_net_profit,
    ROUND(AVG(u.gross_margin_pct), 1) AS avg_gross_margin_pct
FROM dim_accounts a
JOIN fact_monthly_usage u ON a.account_id = u.account_id
WHERE u.month = 12
GROUP BY a.tier
HAVING total_accounts > 0
ORDER BY total_net_profit DESC;
"""
print("--- SQL QUERY 1: MONTH 12 PROFITABILITY BY SUBSCRIPTION TIER ---")
print(pd.read_sql_query(query_1, conn).to_string(index=False))

# 3. SQL Query 2: Catching Silent Downgrades Early (Window Function: LAG)
# Business Question: Track exact month-over-month seat drops for each company to catch Dark Churn at Month 6.
query_2 = """
WITH UsageTrajectory AS (
    SELECT 
        u.account_id,
        a.tier,
        a.monthly_revenue,
        u.month,
        ROUND(u.seat_utilization_pct, 1) AS current_util_pct,
        ROUND(LAG(u.seat_utilization_pct, 3) OVER (
            PARTITION BY u.account_id ORDER BY u.month
        ), 1) AS util_3_months_ago
    FROM fact_monthly_usage u
    JOIN dim_accounts a ON u.account_id = a.account_id
)
SELECT 
    account_id,
    tier,
    monthly_revenue,
    util_3_months_ago AS month_3_util,
    current_util_pct AS month_6_util,
    ROUND(current_util_pct - util_3_months_ago, 1) AS utilization_change_pct
FROM UsageTrajectory
WHERE month = 6 
  AND (current_util_pct - util_3_months_ago) < -15
ORDER BY monthly_revenue DESC
LIMIT 10;
"""
print(
    "\n--- SQL QUERY 2: TOP 10 HIGH-VALUE ACCOUNTS SILENTLY DROPPING AT MONTH 6"
    " ---"
)
print(pd.read_sql_query(query_2, conn).to_string(index=False))

# 4. SQL Query 3: Exposing the Worst "Toxic Whales" (Window Function: RANK)
# Business Question: Rank the top 5 money-losing accounts per tier so Finance can renegotiate contracts.
query_3 = """
WITH MarginRanking AS (
    SELECT 
        a.account_id,
        a.tier,
        a.monthly_revenue,
        u.server_cost,
        u.net_profit,
        RANK() OVER (PARTITION BY a.tier ORDER BY u.net_profit ASC) AS loss_rank
    FROM dim_accounts a
    JOIN fact_monthly_usage u ON a.account_id = u.account_id
    WHERE u.month = 12 AND u.net_profit < 0
)
SELECT 
    tier,
    loss_rank,
    account_id,
    monthly_revenue,
    server_cost,
    net_profit AS monthly_loss
FROM MarginRanking
WHERE loss_rank <= 3
ORDER BY tier, loss_rank;
"""
print("\n--- SQL QUERY 3: TOP MONEY-LOSING 'TOXIC WHALES' BY TIER ---")
print(pd.read_sql_query(query_3, conn).to_string(index=False))

# Close database connection cleanly
conn.close()
print("\nSQL Executive Audit Complete.")
