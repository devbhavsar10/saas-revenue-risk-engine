import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load Enriched Dataset & Prevent Data Leakage
df = pd.read_csv("enriched_saas_metrics.csv")

# Map the final Month 12 outcome back to each account so we can predict it early
target_map = df.groupby("account_id")["downgraded_at_renewal"].max().to_dict()
df["target_downgrade"] = df["account_id"].map(target_map)

# Filter strictly to Month 6: Can we predict a Month 12 downgrade 6 months early?
m6_data = df[df["month"] == 6].copy().dropna()

# Select clear, business-interpretable features
features = [
    "seat_utilization_pct",
    "utilization_drop_3m",
    "api_calls",
    "gross_margin_pct",
    "paid_seats",
]

X = m6_data[features]
y = m6_data["target_downgrade"]

# 2. Stratified Train-Test Split (80% Train, 20% Test)
# 'stratify=y' ensures both train and test sets have the exact same % of churners
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale features so large numbers (like api_calls = 50,000) don't overpower percentages
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train Explainable Logistic Regression Model
# 'class_weight="balanced"' penalizes missing an actual downgrade heavily
model = LogisticRegression(class_weight="balanced", random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Evaluate Model Performance on Unseen Test Clients
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
print(f"--- EARLY-WARNING MODEL EVALUATION (MONTH 6 -> MONTH 12) ---")
print(f"ROC-AUC Score: {roc_auc:.3f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Retained", "Downgraded"]))

# 5. Extract Plain-English Business Drivers (Odds Ratios)
odds_ratios = np.exp(model.coef_[0])
driver_df = pd.DataFrame({
    "Business_Metric": features,
    "Model_Weight": np.round(model.coef_[0], 3),
    "Odds_Ratio": np.round(odds_ratios, 3),
}).sort_values(by="Model_Weight", ascending=True)

print("\n--- KEY BUSINESS DRIVERS OF SILENT DOWNGRADES ---")
print(driver_df.to_string(index=False))

# 6. Save Model Artifacts for the Live Streamlit App (Step 5)
joblib.dump(model, "churn_risk_model.pkl")
joblib.dump(scaler, "feature_scaler.pkl")

# 7. Generate Model Validation & Feature Importance Chart
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Chart A: Precision-Recall Curve (Industry Standard for Imbalanced Churn)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
axes[0].plot(recall, precision, color="#2980b9", linewidth=2.5)
axes[0].set_title(
    f"Precision-Recall Curve (ROC-AUC = {roc_auc:.2f})",
    fontsize=12,
    fontweight="bold",
)
axes[0].set_xlabel("Recall (% of Actual Downgrades Caught)")
axes[0].set_ylabel("Precision (% of Alerts That Were True Risks)")

# Chart B: Feature Impact Chart (What drives downgrades?)
colors = ["#e74c3c" if w > 0 else "#2ecc71" for w in driver_df["Model_Weight"]]
axes[1].barh(driver_df["Business_Metric"], driver_df["Model_Weight"], color=colors)
axes[1].axvline(0, color="black", linewidth=1)
axes[1].set_title(
    "Model Drivers: Red Increases Risk | Green Protects Account",
    fontsize=12,
    fontweight="bold",
)
axes[1].set_xlabel("Standardized Log-Odds Impact")

plt.tight_layout()
plt.savefig("model_performance.png", dpi=300)
print("\nSaved model artifacts (.pkl) & validation chart: model_performance.png")
