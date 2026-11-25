import os
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False

try:
    from fairlearn.metrics import (
        MetricFrame,
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
        equalized_odds_ratio,
        false_positive_rate,
        selection_rate,
        true_positive_rate,
    )
    HAS_FAIRLEARN = True
except Exception:
    HAS_FAIRLEARN = False

# =========================
# 1. Load Predictions
# =========================
PREDS_PATH = "models/test_predictions.csv"
REPORTS_DIR = "reports"

if not os.path.exists(PREDS_PATH):
    raise FileNotFoundError(
        f"Predictions file not found at '{PREDS_PATH}'. Run python3 train.py first."
    )

print("Loading test predictions...")
df = pd.read_csv(PREDS_PATH)
print(f"Loaded shape: {df.shape}")

required_cols = ["TARGET", "PREDICTION"]
missing_req = [c for c in required_cols if c not in df.columns]
if missing_req:
    raise ValueError(f"Missing required columns in {PREDS_PATH}: {missing_req}")

y_true = df["TARGET"].astype(int)
y_pred = df["PREDICTION"].astype(int)

os.makedirs(REPORTS_DIR, exist_ok=True)
report_lines = [
    "PS173 Credit Decision Fairness Audit Report",
    "=" * 44,
    f"Total Test Records : {len(df)}",
    f"Overall Default Rate (TARGET=1)   : {y_true.mean():.4f}",
    f"Overall High-Risk Rate (PRED=1)   : {y_pred.mean():.4f}",
]

if "PROBABILITY" in df.columns:
    report_lines.append(f"Mean Predicted Default Probability: {df['PROBABILITY'].mean():.4f}")

# =========================
# 2. Group Fairness Metrics
# =========================
sensitive_cols = ["CODE_GENDER", "AGE_GROUP"]
available_groups = [c for c in sensitive_cols if c in df.columns]
skipped_groups = [c for c in sensitive_cols if c not in df.columns]

if skipped_groups:
    msg = f"Skipping missing sensitive columns: {skipped_groups}"
    print(msg)
    report_lines.append(f"\n[Note] {msg}")

for col in available_groups:
    group_series = df[col].fillna("Unknown").astype(str)
    unique_groups = group_series.unique()

    report_lines.append("\n" + "-" * 44)
    report_lines.append(f"SENSITIVE ATTRIBUTE: {col}")
    report_lines.append("-" * 44)

    # Fallback / basic group table
    agg_dict = {
        "Count": ("PREDICTION", "count"),
        "High_Risk_Rate": ("PREDICTION", "mean"),
        "Actual_Default_Rate": ("TARGET", "mean"),
    }
    if "PROBABILITY" in df.columns:
        agg_dict["Avg_Probability"] = ("PROBABILITY", "mean")

    group_summary = df.assign(_GROUP=group_series).groupby("_GROUP").agg(**agg_dict)
    report_lines.append(group_summary.to_string())

    if len(unique_groups) < 2:
        report_lines.append(f"Only {len(unique_groups)} group present; disparity metrics require at least 2 groups.")
        continue

    if HAS_FAIRLEARN:
        dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=group_series)
        dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=group_series)
        eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=group_series)
        eo_ratio = equalized_odds_ratio(y_true, y_pred, sensitive_features=group_series)

        mf = MetricFrame(
            metrics={
                "selection_rate": selection_rate,
                "tpr": true_positive_rate,
                "fpr": false_positive_rate,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=group_series,
        )

        report_lines.append("\nFairlearn Summary Metrics:")
        report_lines.append(f"  Demographic Parity Difference : {dp_diff:.4f}")
        report_lines.append(f"  Demographic Parity Ratio      : {dp_ratio:.4f}")
        report_lines.append(f"  Equalized Odds Difference     : {eo_diff:.4f}")
        report_lines.append(f"  Equalized Odds Ratio          : {eo_ratio:.4f}")
        report_lines.append("\nPer-Group Rates (Selection Rate, TPR, FPR):")
        report_lines.append(mf.by_group.to_string())
    else:
        max_min_gap = float(group_summary["High_Risk_Rate"].max() - group_summary["High_Risk_Rate"].min())
        report_lines.append(f"\nFairlearn not installed; High-Risk Rate Max-Min Gap: {max_min_gap:.4f}")

    # =========================
    # 3. Save Plot Per Group
    # =========================
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        rates = group_summary["High_Risk_Rate"] * 100.0
        bars = ax.bar(rates.index.astype(str), rates.values, color="#2563EB", edgecolor="none")
        ax.set_title(f"High-Risk Flag Rate by {col}")
        ax.set_ylabel("Predicted High-Risk Rate (%)")
        ax.set_xlabel(col)
        ax.set_ylim(0, max(rates.max() * 1.25, 10.0))
        for bar, val in zip(bars, rates.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + 0.8,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        plt.tight_layout()
        plot_path = os.path.join(REPORTS_DIR, f"fairness_{col.lower()}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        report_lines.append(f"\nSaved chart: {plot_path}")

report_lines.append(
    "\nGovernance Note: Model risk flags assist human reviewers; final credit decisions remain subject to human review."
)

report_path = os.path.join(REPORTS_DIR, "fairness_report.txt")
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))

print("\n" + "\n".join(report_lines))
print(f"\nSaved report to: {report_path}")
