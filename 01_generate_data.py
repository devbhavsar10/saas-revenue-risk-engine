import numpy as np
import pandas as pd

# Set a random seed so your numbers stay identical every time you run the script
np.random.seed(42)

# 1. Create 500 Corporate Clients (Accounts Table)
num_accounts = 500
account_ids = [f"ACC_{1000 + i}" for i in range(num_accounts)]
tiers = np.random.choice(
    ["Basic", "Pro", "Enterprise"], size=num_accounts, p=[0.4, 0.4, 0.2]
)

# Monthly subscription fee per seat based on tier
price_per_seat = {"Basic": 20, "Pro": 50, "Enterprise": 120}

accounts_df = pd.DataFrame({
    "account_id": account_ids,
    "tier": tiers,
    "paid_seats": np.where(
        tiers == "Basic",
        np.random.randint(5, 20, num_accounts),
        np.where(
            tiers == "Pro",
            np.random.randint(20, 100, num_accounts),
            np.random.randint(100, 500, num_accounts),
        ),
    ),
})

accounts_df["monthly_revenue"] = accounts_df.apply(
    lambda row: row["paid_seats"] * price_per_seat[row["tier"]], axis=1
)

# Assign hidden behavioral archetypes to simulate real-world business problems
# 'Healthy': Normal usage | 'Dark_Churn': Usage drops silently | 'Toxic_Whale': Heavy server usage
archetypes = np.random.choice(
    ["Healthy", "Dark_Churn", "Toxic_Whale"],
    size=num_accounts,
    p=[0.65, 0.25, 0.10],
)
accounts_df["true_behavior"] = archetypes

# 2. Generate 12 Months of Usage & Server Cost Logs
records = []
for _, row in accounts_df.iterrows():
  acc_id = row["account_id"]
  paid = row["paid_seats"]
  behavior = row["true_behavior"]

  for month in range(1, 13):
    if behavior == "Healthy":
      # Uses 75% to 95% of paid seats consistently
      active_seats = int(paid * np.random.uniform(0.75, 0.95))
      api_calls = active_seats * np.random.randint(80, 150)

    elif behavior == "Dark_Churn":
      # Starts strong (85%), then silently drops each month down to ~25%
      decay_factor = max(0.20, 0.85 - (month * 0.055))
      active_seats = int(paid * decay_factor * np.random.uniform(0.9, 1.05))
      api_calls = active_seats * np.random.randint(50, 100)

    elif behavior == "Toxic_Whale":
      # High seat usage AND massive automated API calls that burn server money
      active_seats = int(paid * np.random.uniform(0.85, 1.0))
      api_calls = active_seats * np.random.randint(1500, 3000)

    # Server cost: Each API call costs the company $0.002 in cloud infrastructure
    server_cost = round(api_calls * 0.002, 2)

    # Target Flag: Did this client downgrade at the end of Month 12?
    downgraded = 1 if (behavior == "Dark_Churn" and month == 12) else 0

    records.append({
        "account_id": acc_id,
        "month": month,
        "active_seats": active_seats,
        "api_calls": api_calls,
        "server_cost": server_cost,
        "downgraded_at_renewal": downgraded,
    })

usage_df = pd.DataFrame(records)

# Drop the hidden 'true_behavior' label before saving (in real life, you don't get this hint!)
accounts_df.drop(columns=["true_behavior"]).to_csv("accounts.csv", index=False)
usage_df.to_csv("monthly_usage.csv", index=False)

print(
    "Data generated successfully: accounts.csv (500 rows) & monthly_usage.csv"
    " (6,000 rows)"
)
