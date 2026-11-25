import os
from datetime import datetime
import pandas as pd
import streamlit as st

try:
    from google import genai
    HAS_GENAI = True
except Exception:
    genai = None
    HAS_GENAI = False

st.set_page_config(page_title="PS173 Credit Fairness Reviewer", layout="centered")
DEFAULT_API_KEY = ""

st.title("PS173 Credit Decision Explanation Fairness Reviewer")
st.warning(
    "**Mandatory Policy Guardrail:** Automated lending decisions are **PROHIBITED**. "
    "All model outputs are advisory risk flags requiring human credit officer review."
)

PREDS_PATH = "models/test_predictions.csv"
SHAP_CSV = "reports/shap_top_features.csv"
REVIEW_LOG_CSV = "reports/human_review_log.csv"

if not os.path.exists(PREDS_PATH):
    st.error("Missing `models/test_predictions.csv`. Run `python3 train.py` first.")
    st.stop()

full_df = pd.read_csv(PREDS_PATH)

with st.sidebar:
    st.header("Data Slice & LLM Settings")
    sample_mode = st.radio("Dataset Slice", ["Full Test Set (61,503)", "Custom Sample"], index=0)
    if sample_mode == "Custom Sample":
        sample_n = st.slider("Sample Rows", min_value=100, max_value=min(10000, len(full_df)), value=1000, step=100)
        seed_val = st.number_input("Random Seed (change for new rows)", min_value=1, max_value=99999, value=42)
        df = full_df.sample(n=int(sample_n), random_state=int(seed_val)).reset_index(drop=True)
    else:
        df = full_df.reset_index(drop=True)

    env_key = os.environ.get("GEMINI_API_KEY", "")
    api_key = st.text_input("Gemini API Key", value=env_key, type="password")
    model_name = st.selectbox(
        "Gemini Model",
        [
            "gemini-4-hydrogen",
            "gemini-3.8-flash",
            "gemini-3.5-flash-lite",
            "gemini-4-carbon",
            "gemini-4-argon",
        ],
        index=0,
    )
    st.caption("Uses Gemini when authenticated; falls back cleanly to a structured audit report.")

c1, c2, c3 = st.columns(3)
c1.metric("Applications Audited", f"{len(df):,}")
c2.metric("Actual Default Rate", f"{df['TARGET'].mean() * 100:.1f}%")
c3.metric("Flagged for Review", f"{df['PREDICTION'].mean() * 100:.1f}%")

st.divider()
st.subheader("1. Group Disparity Check")
group_choice = st.radio("Check fairness by:", ["CODE_GENDER", "AGE_GROUP"], horizontal=True)

def compute_group_table(data, col):
    rows = []
    for grp_val, sub in data.assign(_G=data[col].fillna("Unknown").astype(str)).groupby("_G"):
        n = len(sub)
        if n < 20:
            continue
        flagged_pct = sub["PREDICTION"].mean() * 100
        default_pct = sub["TARGET"].mean() * 100
        non_defaults = sub[sub["TARGET"] == 0]
        fpr_pct = (non_defaults["PREDICTION"].mean() * 100) if len(non_defaults) > 0 else 0.0
        rows.append({
            "Group": grp_val,
            "Applicants": n,
            "Flagged (%)": round(flagged_pct, 1),
            "Actual Default (%)": round(default_pct, 1),
            "False-Positive Rate (%)": round(fpr_pct, 1),
        })
    return pd.DataFrame(rows)

grp_df = compute_group_table(df, group_choice)
gap = round(grp_df["Flagged (%)"].max() - grp_df["Flagged (%)"].min(), 1) if len(grp_df) > 1 else 0.0
fpr_gap = round(grp_df["False-Positive Rate (%)"].max() - grp_df["False-Positive Rate (%)"].min(), 1) if len(grp_df) > 1 else 0.0

st.dataframe(grp_df, width="stretch", hide_index=True)
st.info(
    f"**Fairness Takeaway ({group_choice}):** High-risk flag gap is **{gap}%** "
    f"and false-positive rate gap is **{fpr_gap}%** across groups (excluding tiny groups < 20 rows)."
)

st.divider()
st.subheader("2. Why a Loan Leans Toward Acceptance vs. Rejection")
st.caption("Advisory explanation for human reviewers — not an automated approval or rejection.")

row_id = st.number_input("Select Applicant Row Index", min_value=0, max_value=max(len(df) - 1, 0), value=0)
row = df.iloc[int(row_id)]

prob = float(row.get("PROBABILITY", 0.0))
pred = int(row.get("PREDICTION", 0))
actual_target = int(row.get("TARGET", 0))

m1, m2, m3 = st.columns(3)
m1.metric("Default Risk Score", f"{prob:.2f}")
m2.metric("Advisory Direction", "Leans Toward Rejection / Review" if pred == 1 else "Leans Toward Acceptance")
m3.metric("Historical Outcome", "Defaulted (1)" if actual_target == 1 else "Repaid (0)")

def explain_loan_leaning(applicant_row):
    reasons_accept = []
    reasons_reject = []

    ext2 = applicant_row.get("EXT_SOURCE_2", None)
    ext3 = applicant_row.get("EXT_SOURCE_3", None)
    income = applicant_row.get("AMT_INCOME_TOTAL", None)
    credit = applicant_row.get("AMT_CREDIT", None)
    annuity = applicant_row.get("AMT_ANNUITY", None)
    days_emp = applicant_row.get("DAYS_EMPLOYED", None)

    if pd.notna(ext2):
        if ext2 >= 0.5:
            reasons_accept.append(f"Strong external score `EXT_SOURCE_2` ({ext2:.2f}) lowers estimated default risk.")
        else:
            reasons_reject.append(f"Low external score `EXT_SOURCE_2` ({ext2:.2f}) increases estimated default risk.")

    if pd.notna(ext3):
        if ext3 >= 0.5:
            reasons_accept.append(f"Strong external score `EXT_SOURCE_3` ({ext3:.2f}) supports repayment reliability.")
        else:
            reasons_reject.append(f"Low external score `EXT_SOURCE_3` ({ext3:.2f}) raises default concern.")

    if pd.notna(income) and pd.notna(credit) and income > 0:
        cti = credit / income
        if cti <= 2.5:
            reasons_accept.append(f"Moderate credit-to-income ratio ({cti:.1f}x) fits income well.")
        else:
            reasons_reject.append(f"High credit-to-income ratio ({cti:.1f}x) increases loan burden relative to income.")

    if pd.notna(income) and pd.notna(annuity) and income > 0:
        burden = (annuity / income) * 100
        if burden <= 20:
            reasons_accept.append(f"Annuity is {burden:.1f}% of income, leaving repayment cushion.")
        else:
            reasons_reject.append(f"Annuity takes {burden:.1f}% of income, tightening monthly cash flow.")

    if pd.notna(days_emp):
        if days_emp == 365243:
            reasons_reject.append("Employment field contains Home Credit code `365243`; verify manually.")
        elif days_emp < 0:
            years_emp = abs(days_emp) / 365.25
            if years_emp >= 3:
                reasons_accept.append(f"Employed for ~{years_emp:.1f} years, showing work stability.")
            else:
                reasons_reject.append(f"Shorter tenure (~{years_emp:.1f} years) offers less multi-year stability.")

    if not reasons_accept:
        reasons_accept.append("No strong positive credit-score or income-buffer signals stood out.")
    if not reasons_reject:
        reasons_reject.append("No major risk red flags stood out among inspected fields.")

    return reasons_accept, reasons_reject

accept_list, reject_list = explain_loan_leaning(row)

col_acc, col_rej = st.columns(2)
with col_acc:
    st.markdown("**Why It May Lean Toward Acceptance**")
    for item in accept_list:
        st.markdown(f"- {item}")

with col_rej:
    st.markdown("**Why It May Lean Toward Rejection / Review**")
    for item in reject_list:
        st.markdown(f"- {item}")

context_cols = [
    c for c in [
        "CODE_GENDER", "AGE_GROUP", "AMT_INCOME_TOTAL", "AMT_CREDIT",
        "AMT_ANNUITY", "DAYS_EMPLOYED", "EXT_SOURCE_2", "EXT_SOURCE_3"
    ] if c in df.columns
]
with st.expander("View Selected Applicant Details"):
    st.dataframe(pd.DataFrame([row[context_cols]]), width="stretch", hide_index=True)

st.divider()
st.subheader("3. Generate & Download LLM Fairness Report")

top_features_str = "- SHAP drivers not generated yet."
if os.path.exists(SHAP_CSV):
    shap_df = pd.read_csv(SHAP_CSV).head(6).copy()
    shap_df["Feature"] = (
        shap_df["feature"]
        .str.replace("numerical__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
    )
    top_features_str = "\n".join(
        f"- {r['Feature']}: mean |SHAP| = {r['mean_abs_shap']:.4f}" for _, r in shap_df.iterrows()
    )

gender_tbl_text = compute_group_table(df, "CODE_GENDER").to_string(index=False)
age_tbl_text = compute_group_table(df, "AGE_GROUP").to_string(index=False)

def build_fallback_doc():
    return (
        "# PS173 Credit Decision Explanation & Fairness Audit Report\n\n"
        "> **POLICY GUARDRAIL:** Automated lending decisions are strictly **PROHIBITED**. Final decisions require human credit officer review.\n\n"
        "## 1. SDG Alignment (SDG 1, 8, 10)\n"
        "- **SDG 1 & 10:** Flags potential over-rejection or false-positive risk flags across age and gender groups.\n"
        "- **SDG 8:** Promotes transparent, explainable lending workflows with human accountability.\n\n"
        "## 2. Group Fairness Summary\n"
        f"### By Gender (`CODE_GENDER`)\n```text\n{gender_tbl_text}\n```\n\n"
        f"### By Age Group (`AGE_GROUP`)\n```text\n{age_tbl_text}\n```\n\n"
        f"## 3. Top Model Drivers (SHAP)\n{top_features_str}\n\n"
        f"## 4. Sample Applicant Decision Rationale (Row #{int(row_id)})\n"
        f"- **Risk Score:** {prob:.2f} | **Advisory Direction:** {'Leans Toward Rejection / Review' if pred == 1 else 'Leans Toward Acceptance'}\n"
        f"- **Acceptance Factors:** {'; '.join(accept_list)}\n"
        f"- **Rejection / Review Factors:** {'; '.join(reject_list)}\n"
    )

def generate_llm_doc(key, chosen_model):
    if not key or not HAS_GENAI:
        return build_fallback_doc(), "Structured Audit Report Ready"
    prompt = (
        "You are a Credit Fairness & Explainability Auditor for PS 173 (SDG 1, 8, and 10). "
        "State clearly that automated lending decisions are PROHIBITED and human review is mandatory. "
        f"Summarize group disparities across CODE_GENDER and AGE_GROUP, top SHAP drivers, "
        f"and why Row #{int(row_id)} leans toward acceptance vs. rejection.\n\n"
        f"Applications in Current Slice: {len(df):,}\n"
        f"Gender Table:\n{gender_tbl_text}\n\nAge Table:\n{age_tbl_text}\n\n"
        f"Top SHAP Drivers:\n{top_features_str}\n\n"
        f"Row #{int(row_id)} Prob={prob:.2f}, Flag={pred}, Accept={accept_list}, Reject={reject_list}"
    )
    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(model=chosen_model, contents=prompt)
        return response.text, f"Generated with {chosen_model}"
    except Exception as e:
        return build_fallback_doc() + f"\n\n*(Fallback report used: {e})*", "Structured Fallback Report Ready"

if "report_text" not in st.session_state:
    txt, status = build_fallback_doc(), "Ready — click Generate Report"
    st.session_state["report_text"] = txt
    st.session_state["doc_status"] = status

if st.button("Generate Report"):
    with st.spinner("Preparing fairness & loan rationale report..."):
        txt, status = generate_llm_doc(api_key.strip(), model_name)
        st.session_state["report_text"] = txt
        st.session_state["doc_status"] = status

st.caption(f"Report Status: {st.session_state['doc_status']}")

d1, d2 = st.columns(2)
d1.download_button("Download Report (.txt)", data=st.session_state["report_text"], file_name="PS173_Credit_Fairness_Report.txt", mime="text/plain")
d2.download_button("Download Report (.md)", data=st.session_state["report_text"], file_name="PS173_Credit_Fairness_Report.md", mime="text/markdown")

with st.expander("Preview Report", expanded=True):
    st.markdown(st.session_state["report_text"])

st.divider()
st.subheader("4. Human Reviewer Sign-Off")
human_choice = st.radio("Final Human Decision:", ["Approve Loan", "Hold for Verification", "Reject Loan"], horizontal=True)
reviewer_note = st.text_input("Reviewer Note")

if st.button("Save Human Decision"):
    os.makedirs("reports", exist_ok=True)
    entry = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "row_index": int(row_id),
        "probability": round(prob, 4),
        "advisory_flag": pred,
        "human_decision": human_choice,
        "note": reviewer_note,
    }])
    write_header = not os.path.exists(REVIEW_LOG_CSV)
    entry.to_csv(REVIEW_LOG_CSV, mode="a", header=write_header, index=False)
    st.success(f"Saved **{human_choice}** for Row #{int(row_id)}.")

if os.path.exists(REVIEW_LOG_CSV):
    st.download_button("Download Reviewer Log (.csv)", data=pd.read_csv(REVIEW_LOG_CSV).to_csv(index=False), file_name="PS173_Human_Reviewer_Log.csv", mime="text/csv")
