import os
import pandas as pd

REPORTS_DIR = "reports"
TRAIN_SUMMARY_PATH = os.path.join(REPORTS_DIR, "train_fairness_summary.txt")
FAIRNESS_REPORT_PATH = os.path.join(REPORTS_DIR, "fairness_report.txt")
SHAP_CSV_PATH = os.path.join(REPORTS_DIR, "shap_top_features.csv")
OUTPUT_PATH = os.path.join(REPORTS_DIR, "ps173_final_audit_report.txt")

os.makedirs(REPORTS_DIR, exist_ok=True)

sections = [
    "=" * 60,
    "PS173 CREDIT DECISION FAIRNESS & EXPLAINABILITY FINAL REPORT",
    "=" * 60,
]

# 1. Model Training & Baseline Summary
sections.append("\n[1. MODEL TRAINING SUMMARY]")
if os.path.exists(TRAIN_SUMMARY_PATH):
    with open(TRAIN_SUMMARY_PATH, "r") as f:
        sections.append(f.read().strip())
else:
    sections.append(f"Missing {TRAIN_SUMMARY_PATH} (run train.py to generate).")

# 2. Fairlearn Group Fairness Audit
sections.append("\n" + "-" * 60)
sections.append("[2. FAIRNESS AUDIT REPORT]")
if os.path.exists(FAIRNESS_REPORT_PATH):
    with open(FAIRNESS_REPORT_PATH, "r") as f:
        sections.append(f.read().strip())
else:
    sections.append(f"Missing {FAIRNESS_REPORT_PATH} (run fairness_report.py to generate).")

# 3. Top SHAP Feature Drivers
sections.append("\n" + "-" * 60)
sections.append("[3. TOP SHAP RISK DRIVERS]")
if os.path.exists(SHAP_CSV_PATH):
    shap_df = pd.read_csv(SHAP_CSV_PATH)
    if not shap_df.empty:
        top_n = shap_df.head(10).copy()
        if "mean_abs_shap" in top_n.columns:
            top_n["mean_abs_shap"] = top_n["mean_abs_shap"].round(6)
        sections.append(top_n.to_string(index=False))
    else:
        sections.append(f"{SHAP_CSV_PATH} is empty.")
else:
    sections.append(f"Missing {SHAP_CSV_PATH} (run shap_explainer.py to generate).")

# 4. Human-in-the-Loop Governance & Mitigation Notes
sections.append("\n" + "-" * 60)
sections.append("[4. GOVERNANCE & REVIEW RECOMMENDATIONS]")
sections.append(
    "- Advisory Screening Only: High-risk model flags trigger human review rather than automated rejection.\n"
    "- Subgroup Sample Check: Exclude or flag tiny categories (such as XNA with n=2) before interpreting parity ratios.\n"
    "- Disparity & Error Follow-Up: Review high false-positive rates in younger age brackets (18-25) and gender gaps."
)

final_text = "\n".join(sections) + "\n"

with open(OUTPUT_PATH, "w") as f:
    f.write(final_text)

print(final_text)
print(f"Combined report saved to: {OUTPUT_PATH}")
