import os
import pandas as pd
import numpy as np

from churn_model import pipeline, x

# ----------------------------
# Predict churn probability
# ----------------------------
churn_prob = pipeline.predict_proba(x)[:, 1]

# ----------------------------
# Create dashboard dataset
# ----------------------------
dashboard_df = x.copy()
dashboard_df["churn_probability"] = churn_prob

RISK_THRESHOLD = 0.7
dashboard_df["churn_risk"] = np.where(
    churn_prob > RISK_THRESHOLD, "High", "Low"
)

# ----------------------------
# Ensure output directory exists
# ----------------------------
os.makedirs("data", exist_ok=True)

# ----------------------------
# Export for BI tools
# ----------------------------
dashboard_df.to_csv("data/churn_dashboard_data.csv", index=False)

print("Dashboard data exported successfully.")
