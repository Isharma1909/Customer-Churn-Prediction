import shap
import pandas as pd
import matplotlib.pyplot as plt

from churn_model import pipeline, X_train

# ----------------------------
# Extract trained model
# ----------------------------
model = pipeline.named_steps["model"]
preprocessor = pipeline.named_steps["preprocess"]

# Transform data (important for SHAP)
X_train_transformed = preprocessor.transform(X_train)

# Get feature names
ohe = preprocessor.named_transformers_["cat"]
cat_features = ohe.get_feature_names_out()
num_features = X_train.select_dtypes(exclude=["object"]).columns
feature_names = list(cat_features) + list(num_features)

# ----------------------------
# SHAP Explainer
# ----------------------------
explainer = shap.LinearExplainer(
    model,
    X_train_transformed,
    feature_names=feature_names
)
shap_values = explainer.shap_values(X_train_transformed)

# ----------------------------
# Global Explanation
# ----------------------------
shap.summary_plot(
    shap_values,
    X_train_transformed,
    feature_names=feature_names,
    show=False
)
plt.show()

# ----------------------------
# Individual Customer Explanation
# ----------------------------
customer_index = 5

shap.force_plot(
    explainer.expected_value,
    shap_values[customer_index],
    X_train_transformed[customer_index],
    feature_names=feature_names,
    matplotlib=True
)
