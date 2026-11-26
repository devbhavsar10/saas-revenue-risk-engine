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
