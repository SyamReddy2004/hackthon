import os
import joblib
import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except Exception:
    shap = None
    HAS_SHAP = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False

MODEL_PATH = "models/credit_fairness_model.pkl"
PREDS_PATH = "models/test_predictions.csv"
REPORTS_DIR = "reports"

if not os.path.exists(MODEL_PATH) or not os.path.exists(PREDS_PATH):
    raise FileNotFoundError("Missing model or test predictions. Run python3 train.py first.")

print("Loading saved model and test predictions...")
pipeline = joblib.load(MODEL_PATH)
df = pd.read_csv(PREDS_PATH)

drop_cols = [c for c in ["TARGET", "PREDICTION", "PROBABILITY"] if c in df.columns]
X_sample = df.drop(columns=drop_cols).sample(n=min(1000, len(df)), random_state=42)

preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]

X_trans = preprocessor.transform(X_sample)
if hasattr(X_trans, "toarray"):
    X_trans = X_trans.toarray()

try:
    feature_names = preprocessor.get_feature_names_out().tolist()
except Exception:
    feature_names = [f"feat_{i}" for i in range(X_trans.shape[1])]

os.makedirs(REPORTS_DIR, exist_ok=True)

if HAS_SHAP:
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_trans)

    if isinstance(shap_values, list):
        shap_vals_pos = shap_values[1]
    elif getattr(shap_values, "ndim", 0) == 3:
        shap_vals_pos = shap_values[:, :, 1]
    else:
        shap_vals_pos = shap_values

    mean_abs = np.abs(shap_vals_pos).mean(axis=0)
    imp_df = pd.DataFrame({
        "feature": feature_names[:len(mean_abs)],
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    if HAS_MATPLOTLIB:
        plt.figure(figsize=(9, 5.5))
        shap.summary_plot(
            shap_vals_pos,
            X_trans,
            feature_names=feature_names,
            show=False,
            max_display=12,
        )
        plt.tight_layout()
        plot_path = os.path.join(REPORTS_DIR, "shap_summary.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved SHAP plot: {plot_path}")
else:
    print("SHAP not available; using classifier feature_importances_ fallback.")
    importances = getattr(classifier, "feature_importances_", np.zeros(len(feature_names)))
    imp_df = pd.DataFrame({
        "feature": feature_names[:len(importances)],
        "mean_abs_shap": importances,
    }).sort_values("mean_abs_shap", ascending=False)

csv_path = os.path.join(REPORTS_DIR, "shap_top_features.csv")
imp_df.to_csv(csv_path, index=False)

print("\nTop 10 Features Driving High-Risk Predictions:")
print(imp_df.head(10).to_string(index=False))
print(f"\nSaved feature importance table: {csv_path}")
