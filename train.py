
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False

from sklearn.ensemble import RandomForestClassifier

# =========================
# 1. Load Dataset
# =========================
DATA_PATH = "data/application_train.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset Shape: {df.shape}")

# =========================
# 2. Age & Feature Engineering
# =========================
if "DAYS_BIRTH" in df.columns:
    df["AGE"] = (-df["DAYS_BIRTH"] / 365.25).astype(int)
    df["AGE_GROUP"] = pd.cut(
        df["AGE"],
        bins=[17, 25, 40, 60, 120],
        labels=["18-25", "26-40", "41-60", "60+"],
    ).astype(str)

candidate_features = [
    "CODE_GENDER",
    "AGE_GROUP",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "REGION_POPULATION_RELATIVE",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
]

features = [col for col in candidate_features if col in df.columns]
missing_cols = [col for col in candidate_features if col not in df.columns]
if missing_cols:
    print(f"Skipping missing columns: {missing_cols}")

target = "TARGET"
if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in dataset.")

df = df[features + [target]].copy()

# =========================
# 3. Train / Test Split
# =========================
X = df.drop(columns=[target])
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print(f"\nTraining Samples: {len(X_train)}")
print(f"Testing Samples : {len(X_test)}")

neg_count = int((y_train == 0).sum())
pos_count = int((y_train == 1).sum())
scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
print(f"Class 0: {neg_count} | Class 1: {pos_count} | scale_pos_weight: {scale_pos_weight:.2f}")

# =========================
# 4. Preprocessing & Model
# =========================
cat_cols = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
num_cols = X_train.select_dtypes(exclude=["object", "category", "string"]).columns.tolist()

print("\nCategorical Features:", cat_cols)
print("Numerical Features  :", num_cols)

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

preprocessor = ColumnTransformer([
    ("categorical", categorical_pipeline, cat_cols),
    ("numerical", numerical_pipeline, num_cols),
])

if HAS_XGBOOST:
    print("\nUsing classifier: XGBClassifier")
    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
else:
    print("\nXGBoost unavailable; using fallback RandomForestClassifier")
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", classifier),
])

# =========================
# 5. Train & Evaluate
# =========================
print("\nTraining model...")
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

auc = roc_auc_score(y_test, probs)
print("\n==============================")
print(f"ROC-AUC Score: {auc:.4f}")
print("==============================")

print("\nPrediction Counts on Test Set:")
print(pd.Series(preds).value_counts().to_dict())

print("\nClassification Report:")
print(classification_report(y_test, preds, zero_division=0))

# =========================
# 6. Save Artifacts & Quick Summary
# =========================
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

model_path = "models/credit_fairness_model.pkl"
preds_path = "models/test_predictions.csv"
summary_path = "reports/train_fairness_summary.txt"

joblib.dump(model, model_path)

test_df = X_test.copy()
test_df["TARGET"] = y_test.values
test_df["PREDICTION"] = preds
test_df["PROBABILITY"] = probs
test_df.to_csv(preds_path, index=False)

summary_lines = [f"ROC-AUC Score: {auc:.4f}"]
for group_col in ["CODE_GENDER", "AGE_GROUP"]:
    if group_col in test_df.columns:
        grp = test_df.groupby(group_col, observed=False).agg(
            Count=("PREDICTION", "count"),
            High_Risk_Rate=("PREDICTION", "mean"),
            Avg_Prob=("PROBABILITY", "mean"),
        )
        gap = float(grp["High_Risk_Rate"].max() - grp["High_Risk_Rate"].min()) if len(grp) > 1 else 0.0
        summary_lines.append(f"\n--- {group_col} (Max-Min Rate Gap: {gap:.4f}) ---")
        summary_lines.append(grp.to_string())

with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines))

print(f"\nModel saved to       : {model_path}")
print(f"Predictions saved to : {preds_path}")
print(f"Summary saved to     : {summary_path}")
print("Done running train.py.")
