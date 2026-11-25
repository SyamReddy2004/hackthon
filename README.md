# Credit Decision Explanation Fairness Reviewer

**Live Deployment:** [https://credit-fairness-reviewer.onrender.com/](https://credit-fairness-reviewer.onrender.com/)  
**Repository:** [https://github.com/SyamReddy2004/hackthon](https://github.com/SyamReddy2004/hackthon)  
**SDG Alignment:** SDG 1 (No Poverty) · SDG 8 (Decent Work & Economic Growth) · SDG 10 (Reduced Inequalities)

> **Mandatory Governance Guardrail:** Automated lending decisions are strictly **PROHIBITED**. All model outputs, SHAP attributions, and Gemini-generated rationales act exclusively as advisory risk flags requiring mandatory human credit officer review and audit sign-off.

---

## Overview

**Credit Decision Explanation Fairness Reviewer** is an end-to-end explainable AI (XAI) and algorithmic fairness auditing workbench built for retail credit underwriting. Trained on the **Home Credit Default Risk** dataset (`application_train.csv`, `scale_pos_weight = 11.39`), the system evaluates credit default risk using **XGBoost** (ROC-AUC **0.7426** across 61,503 holdout test records), audits demographic parity and equalized odds across protected cohorts (`CODE_GENDER`, `AGE_GROUP`) via **Fairlearn**, extracts feature-level risk drivers via **SHAP**, and generates plain-language applicant explanations and regulatory audit reports using **Google Gemini** (`google-genai`).

---

## System Architecture & Workflow

```
+-----------------------------------------------------------------------------------+
|                        1. DATA INGESTION & PREPROCESSING                          |
|  - Upload custom CSV (with binary TARGET) or load built-in Home Credit sample     |
|  - Automated missing-value imputation, categorical encoding, and feature scaling  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      2. RISK MODELING & HYPERPARAMETER TUNING                     |
|  - XGBoost Classifier (scale_pos_weight = 11.39 for class imbalance handling)     |
|  - Evaluates ROC-AUC (0.7426 on 61,503 test records) & calibrated default risk    |
+-----------------------------------------+-----------------------------------------+
                                          |
                     +--------------------+--------------------+
                     |                                         |
                     v                                         v
+---------------------------------------+   +---------------------------------------+
|   3A. FAIRNESS AUDIT (FAIRLEARN)      |   |   3B. EXPLAINABILITY (SHAP)           |
|  - Demographic Parity Difference      |   |  - TreeExplainer global & local values|
|  - Equalized Odds Difference          |   |  - Top positive/negative risk drivers |
|  - Cohort breakdowns (Gender & Age)   |   |  - Visual summary & cohort charts     |
+--------------------+------------------+   +------------------+--------------------+
                     |                                         |
                     +--------------------+--------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                 4. LLM EXPLANATION & AUDIT REPORT (GOOGLE GEMINI)                 |
|  - Translates SHAP drivers & fairness metrics into plain-language explanations    |
|  - Enforces non-discriminatory phrasing & structured compliance audit summaries   |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                   5. HUMAN-IN-THE-LOOP REVIEW & PDF EXPORT                        |
|  - Mandatory human credit officer override/approval logging                       |
|  - Downloadable PDF & text audit reports via fpdf2                                |
+-----------------------------------------------------------------------------------+
```

### End-to-End Execution Flow
1. **Step 1 — Dataset Ingestion:** Users upload a credit application CSV containing a binary `TARGET` column or load the built-in Home Credit dataset sample via the FastAPI `/api` layer (`backend/services.py`).
2. **Step 2 — Parameter Tuning & Scoring:** The XGBoost pipeline scores applicant default probability while accounting for severe class imbalance (`scale_pos_weight = 11.39`).
3. **Step 3 — Fairness & SHAP Diagnostics:** Fairlearn computes Demographic Parity and Equalized Odds differences across gender and age cohorts, while SHAP (`TreeExplainer`) isolates the top drivers behind each risk flag.
4. **Step 4 — Gemini Narrative Synthesis:** `google-genai` converts quantitative SHAP contributions and disparity metrics into transparent, applicant-friendly adverse-action/review explanations and formal audit summaries.
5. **Step 5 — Mandatory Human Sign-Off:** Every flagged case requires explicit human reviewer disposition (`Approve`, `Reject`, or `Escalate`) with logged justification and exportable PDF/TXT audit artifacts (`backend/pdf_generator.py`).

---

## Folder Structure

```text
CreditFairnessReviewer/
├── app.py                              # Interactive Streamlit workbench & Gemini audit UI
├── train.py                            # XGBoost model training & validation pipeline (ROC-AUC: 0.7426)
├── fairness_report.py                  # Fairlearn demographic parity & equalized odds evaluation
├── shap_explainer.py                   # Global & local SHAP feature attribution generator
├── final_report.py                     # Consolidated PS 173 compliance & SDG audit report builder
├── requirements.txt                    # Python dependencies for local & Render deployment
├── .gitignore                          # Excludes .env, large CSVs (>100 MB), and local review logs
├── backend/
│   ├── __init__.py                     # Backend package initializer
│   ├── main.py                         # FastAPI application entry point & REST API routes
│   ├── services.py                     # Dataset loading, model inference, Fairlearn & SHAP services
│   └── pdf_generator.py                # Structured PDF audit report exporter (fpdf2)
├── frontend/
│   └── index.html                      # 3-step HTML/JS Credit Fairness & Explainability Workbench UI
├── models/
│   └── credit_fairness_model.pkl       # Serialized XGBoost model & preprocessing artifacts
├── reports/
│   ├── fairness_age_group.png          # Fairlearn selection & error rate breakdown by age cohort
│   ├── fairness_code_gender.png        # Fairlearn selection & error rate breakdown by gender
│   ├── fairness_report.txt             # Detailed Fairlearn statistical fairness audit output
│   ├── ps173_final_audit_report.txt    # Complete governance, SDG, and model audit report
│   ├── shap_summary.png                # Global SHAP feature importance beeswarm/summary plot
│   ├── shap_top_features.csv           # Ranked SHAP feature attribution table
│   └── train_fairness_summary.txt      # Training metrics & baseline cohort parity summary
└── data/                               # Local dataset directory (sample CSVs / application_train.csv)
```

---

## Tech Stack

* **Backend & Serving:** Python 3, FastAPI, Uvicorn, Pydantic
* **Frontend & Prototyping:** HTML5/CSS3/Vanilla JS (`frontend/index.html`), Streamlit (`app.py`)
* **Machine Learning & Fairness:** XGBoost, Scikit-Learn, Fairlearn, SHAP, Pandas, NumPy, Joblib
* **Generative AI & Reporting:** Google Gemini API (`google-genai`), Matplotlib, FPDF2

---

## Local Setup & Running

### 1. Clone & Install Dependencies
```bash
git clone [https://github.com/SyamReddy2004/hackthon.git](https://github.com/SyamReddy2004/hackthon.git)
cd hackthon
pip install -r requirements.txt
```
*(macOS users running XGBoost/SHAP locally should ensure OpenMP is installed via `brew install libomp`.)*

### 2. Configure Environment Variable
Create a `.env` file (never commit API keys to Git):
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

### 3. Launch the FastAPI Workbench
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in your browser.

*(Optional)* To launch the standalone Streamlit dashboard instead:
```bash
streamlit run app.py
```

---

## Cloud Deployment (Render)

* **Build Command:** `pip install -r requirements.txt`
* **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
* **Environment Variables:** Set `GEMINI_API_KEY` in the Render Web Service **Environment** settings.
