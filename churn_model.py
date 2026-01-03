import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report,roc_auc_score

# load the csv file
df = pd.read_csv(
    r"E:\Git\Customer-Churn-Prediction\Telco-Customer-churn.csv"
)

# data preprocessing
if "customerID" in df.columns:
    df.drop(columns=['customerID'])

y = df['Churn'].map({"Yes": 1, "No": 0}).astype(int)
x = df.drop(columns=["Churn"])

## feature type identification
cat_field = x.select_dtypes(include=['object']).columns
num_field = x.select_dtypes(exclude=['object']).columns

## encoding and transformation
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_field),
        ("num", "passthrough", num_field)
    ]
)

## model building
model = LogisticRegression(
    max_iter=1000,
    solver="liblinear",
    class_weight="balanced",
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])

## train and test the dataset

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

## train
pipeline.fit(X_train,y_train)

## evaluate 
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))