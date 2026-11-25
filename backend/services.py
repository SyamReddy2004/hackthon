import io
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    shap = None
    HAS_SHAP = False

try:
    from fairlearn.metrics import (
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
        equalized_odds_ratio,
    )
    HAS_FAIRLEARN = True
except Exception:
    HAS_FAIRLEARN = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "application_train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_fairness_model.pkl")
PREDS_PATH = os.path.join(BASE_DIR, "models", "test_predictions.csv")
SHAP_CSV_PATH = os.path.join(BASE_DIR, "reports", "shap_top_features.csv")
REVIEW_LOG_CSV = os.path.join(BASE_DIR, "reports", "human_review_log.csv")

POLICY_NOTICE = (
    "Mandatory Policy Guardrail: Automated lending decisions are PROHIBITED. "
    "All model outputs are advisory risk flags requiring human credit officer review."
)

CANDIDATE_FEATURES = [
    "CODE_GENDER",
    "AGE_GROUP",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE",
    "FLAG_OWN_REALTY",
    "FLAG_OWN_CAR",
    "DAYS_EMPLOYED",
    "DAYS_EMPLOYED_CLEAN",
    "DAYS_ID_PUBLISH",
    "DAYS_REGISTRATION",
    "DAYS_LAST_PHONE_CHANGE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "REGION_POPULATION_RELATIVE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "EXT_SOURCE_MEAN",
    "CREDIT_TO_INCOME",
    "ANNUITY_TO_INCOME",
    "CREDIT_TERM",
    "CREDIT_TO_GOODS",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
]

FEATURE_LABELS = {
    "CODE_GENDER": "Gender",
    "AGE_GROUP": "Age Group",
    "AMT_INCOME_TOTAL": "Annual Income",
    "AMT_CREDIT": "Loan Credit Amount",
    "AMT_ANNUITY": "Annualized Loan Payment (Annuity)",
    "AMT_GOODS_PRICE": "Goods Price",
    "NAME_EDUCATION_TYPE": "Education Level",
    "NAME_FAMILY_STATUS": "Family Status",
    "NAME_HOUSING_TYPE": "Housing Type",
    "NAME_INCOME_TYPE": "Income Type",
    "OCCUPATION_TYPE": "Occupation",
    "FLAG_OWN_REALTY": "Owns Real Estate",
    "FLAG_OWN_CAR": "Owns Car",
    "DAYS_EMPLOYED": "Days Employed",
    "DAYS_EMPLOYED_CLEAN": "Employment Tenure (Days)",
    "DAYS_ID_PUBLISH": "ID Document Age (Days)",
    "DAYS_REGISTRATION": "Registration Age (Days)",
    "DAYS_LAST_PHONE_CHANGE": "Last Phone Change (Days)",
    "CNT_CHILDREN": "Number of Children",
    "CNT_FAM_MEMBERS": "Family Size",
    "REGION_POPULATION_RELATIVE": "Region Population Density",
    "EXT_SOURCE_1": "External Credit Score 1",
    "EXT_SOURCE_2": "External Credit Score 2",
    "EXT_SOURCE_3": "External Credit Score 3",
    "EXT_SOURCE_MEAN": "Average External Credit Score",
    "CREDIT_TO_INCOME": "Credit-to-Income Ratio",
    "ANNUITY_TO_INCOME": "Annuity-to-Income Ratio",
    "CREDIT_TERM": "Repayment Rate (Annuity / Credit)",
    "CREDIT_TO_GOODS": "Credit-to-Goods Ratio",
    "REGION_RATING_CLIENT": "Region Risk Rating",
    "REGION_RATING_CLIENT_W_CITY": "Region & City Risk Rating",
}


class CreditFairnessService:
    """Manages dataset upload, parameter-tuned training, fairness metrics, and SHAP explanations."""

    def __init__(self) -> None:
        self.pipeline = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        self.full_df = pd.read_csv(PREDS_PATH) if os.path.exists(PREDS_PATH) else pd.DataFrame()
        self.raw_df: Optional[pd.DataFrame] = None
        self.dataset_name: str = "Preloaded Test Set (models/test_predictions.csv)"
        self.last_training_metrics: Optional[Dict[str, Any]] = None
        self.global_shap_drivers: List[Dict[str, Any]] = self._load_saved_global_shap()

        self._sync_model_refs()

    def _sync_model_refs(self) -> None:
        if self.pipeline is not None:
            self.preprocessor = self.pipeline.named_steps["preprocessor"]
            self.classifier = self.pipeline.named_steps["classifier"]
            try:
                self.transformed_feature_names = list(self.preprocessor.get_feature_names_out())
            except Exception:
                self.transformed_feature_names = []
            self.explainer = shap.TreeExplainer(self.classifier) if HAS_SHAP else None
        else:
            self.preprocessor = None
            self.classifier = None
            self.transformed_feature_names = []
            self.explainer = None

        drop_cols = {"TARGET", "PREDICTION", "PROBABILITY"}
        if not self.full_df.empty:
            self.feature_cols = [c for c in self.full_df.columns if c not in drop_cols]
            self.feature_defaults = {}
            for col in self.feature_cols:
                series = self.full_df[col].dropna()
                if pd.api.types.is_numeric_dtype(self.full_df[col]):
                    self.feature_defaults[col] = float(series.median()) if len(series) else 0.0
                else:
                    self.feature_defaults[col] = str(series.mode().iloc[0]) if len(series) else "Unknown"
        else:
            self.feature_cols = []
            self.feature_defaults = {}

    def _load_saved_global_shap(self, limit: int = 10) -> List[Dict[str, Any]]:
        if not os.path.exists(SHAP_CSV_PATH):
            return []
        shap_df = pd.read_csv(SHAP_CSV_PATH).head(limit)
        drivers = []
        for _, row in shap_df.iterrows():
            clean_name = str(row["feature"]).replace("numerical__", "").replace("categorical__", "")
            drivers.append({
                "feature": clean_name,
                "mean_abs_shap": round(float(row["mean_abs_shap"]), 4),
            })
        return drivers

    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "DAYS_BIRTH" in df.columns and "AGE_GROUP" not in df.columns:
            age = (-pd.to_numeric(df["DAYS_BIRTH"], errors="coerce") / 365.25)
            df["AGE_GROUP"] = pd.cut(
                age,
                bins=[17, 25, 40, 60, 120],
                labels=["18-25", "26-40", "41-60", "60+"],
            ).astype(str)

        if "DAYS_EMPLOYED" in df.columns:
            emp = pd.to_numeric(df["DAYS_EMPLOYED"], errors="coerce")
            df["DAYS_EMPLOYED_CLEAN"] = emp.replace(365243, np.nan)

        ext_cols = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in df.columns]
        if ext_cols:
            df["EXT_SOURCE_MEAN"] = df[ext_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

        if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            inc = pd.to_numeric(df["AMT_INCOME_TOTAL"], errors="coerce").replace(0, np.nan)
            crd = pd.to_numeric(df["AMT_CREDIT"], errors="coerce")
            df["CREDIT_TO_INCOME"] = crd / inc

        if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            inc = pd.to_numeric(df["AMT_INCOME_TOTAL"], errors="coerce").replace(0, np.nan)
            ann = pd.to_numeric(df["AMT_ANNUITY"], errors="coerce")
            df["ANNUITY_TO_INCOME"] = ann / inc

        if "AMT_ANNUITY" in df.columns and "AMT_CREDIT" in df.columns:
            crd = pd.to_numeric(df["AMT_CREDIT"], errors="coerce").replace(0, np.nan)
            ann = pd.to_numeric(df["AMT_ANNUITY"], errors="coerce")
            df["CREDIT_TERM"] = ann / crd

        if "AMT_CREDIT" in df.columns and "AMT_GOODS_PRICE" in df.columns:
            goods = pd.to_numeric(df["AMT_GOODS_PRICE"], errors="coerce").replace(0, np.nan)
            crd = pd.to_numeric(df["AMT_CREDIT"], errors="coerce")
            df["CREDIT_TO_GOODS"] = crd / goods

        return df

    def _prepare_raw_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "TARGET" not in df.columns:
            raise ValueError("Dataset must contain a binary 'TARGET' column (0 = Repaid, 1 = Default).")
        return self._add_engineered_features(df)

    def _summarize_raw_dataset(self) -> Dict[str, Any]:
        if self.raw_df is None:
            return {
                "loaded": not self.full_df.empty,
                "filename": self.dataset_name,
                "rows": int(len(self.full_df)),
                "columns": int(len(self.full_df.columns)),
                "target_default_rate_pct": round(float(self.full_df["TARGET"].mean() * 100), 2) if not self.full_df.empty else 0.0,
                "sensitive_columns": [c for c in ["CODE_GENDER", "AGE_GROUP"] if c in self.full_df.columns],
                "feature_columns": self.feature_cols,
                "ready_for_training": os.path.exists(DEFAULT_DATA_PATH),
            }

        df = self.raw_df
        usable_features = [c for c in CANDIDATE_FEATURES if c in df.columns]
        if not usable_features:
            usable_features = [c for c in df.columns if c not in ("TARGET", "SK_ID_CURR", "DAYS_BIRTH")][:20]

        return {
            "loaded": True,
            "filename": self.dataset_name,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "target_default_rate_pct": round(float(pd.to_numeric(df["TARGET"], errors="coerce").mean() * 100), 2),
            "sensitive_columns": [c for c in ["CODE_GENDER", "AGE_GROUP"] if c in df.columns],
            "feature_columns": usable_features,
            "ready_for_training": True,
        }

    def load_dataset_from_bytes(self, filename: str, raw_bytes: bytes) -> Dict[str, Any]:
        df = pd.read_csv(io.BytesIO(raw_bytes))
        self.raw_df = self._prepare_raw_dataframe(df)
        self.dataset_name = filename or "uploaded_dataset.csv"
        return self._summarize_raw_dataset()

    def load_default_dataset(self) -> Dict[str, Any]:
        if not os.path.exists(DEFAULT_DATA_PATH):
            raise FileNotFoundError(f"Default dataset not found at {DEFAULT_DATA_PATH}")
        df = pd.read_csv(DEFAULT_DATA_PATH)
        self.raw_df = self._prepare_raw_dataframe(df)
        self.dataset_name = "application_train.csv (Built-in Default)"
        return self._summarize_raw_dataset()

    def get_dataset_status(self) -> Dict[str, Any]:
        return self._summarize_raw_dataset()

    def train_and_evaluate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.raw_df is None:
            self.load_default_dataset()

        df = self.raw_df
        assert df is not None

        test_size = float(params.get("test_size", 0.2))
        test_size = min(max(test_size, 0.1), 0.4)
        n_estimators = int(params.get("n_estimators", 150))
        n_estimators = min(max(n_estimators, 20), 500)
        max_depth = int(params.get("max_depth", 5))
        max_depth = min(max(max_depth, 2), 12)
        learning_rate = float(params.get("learning_rate", 0.05))
        learning_rate = min(max(learning_rate, 0.01), 0.5)
        raw_thresh = params.get("decision_threshold", "auto")
        auto_threshold = raw_thresh in ("auto", None, "")
        threshold = 0.5 if auto_threshold else min(max(float(raw_thresh), 0.05), 0.95)
        max_rows_raw = params.get("max_rows", 20000)
        max_rows = int(max_rows_raw) if max_rows_raw not in (None, "", 0, "0") else None

        features = [c for c in CANDIDATE_FEATURES if c in df.columns]
        if not features:
            features = [c for c in df.columns if c not in ("TARGET", "SK_ID_CURR", "DAYS_BIRTH")][:20]

        work_df = df[features + ["TARGET"]].dropna(subset=["TARGET"]).copy()
        work_df["TARGET"] = work_df["TARGET"].astype(int)

        if max_rows is not None and 200 <= max_rows < len(work_df):
            work_df, _ = train_test_split(
                work_df,
                train_size=max_rows,
                stratify=work_df["TARGET"],
                random_state=42,
            )

        X = work_df[features]
        y = work_df["TARGET"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        raw_ratio = (neg_count / pos_count) if pos_count > 0 else 1.0
        class_weight_mode = str(params.get("class_weight_mode", "calibrated")).lower()
        if class_weight_mode == "balanced":
            scale_pos_weight = raw_ratio
        elif class_weight_mode == "mild":
            scale_pos_weight = float(np.sqrt(raw_ratio))
        else:
            scale_pos_weight = 1.0

        cat_cols = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        num_cols = X_train.select_dtypes(exclude=["object", "category", "string"]).columns.tolist()

        preprocessor = ColumnTransformer([
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
            ("numerical", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), num_cols),
        ])

        if HAS_XGBOOST:
            classifier = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.85,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.5,
                reg_lambda=1.5,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
            model_type = "XGBClassifier"
        else:
            classifier = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight=None if class_weight_mode == "calibrated" else "balanced",
                random_state=42,
                n_jobs=-1,
            )
            model_type = "RandomForestClassifier"

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

        if auto_threshold and len(np.unique(y_train)) > 1 and len(X_train) >= 200:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
            )
            pipeline.fit(X_tr, y_tr)
            val_probs = pipeline.predict_proba(X_val)[:, 1]
            prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_val, val_probs)
            if len(thresh_arr) > 0:
                f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / np.maximum(prec_arr[:-1] + rec_arr[:-1], 1e-9)
                threshold = float(np.clip(thresh_arr[int(np.argmax(f1_arr))], 0.08, 0.85))

        pipeline.fit(X_train, y_train)

        probs = pipeline.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)

        test_df = X_test.copy().reset_index(drop=True)
        test_df["TARGET"] = y_test.values
        test_df["PREDICTION"] = preds
        test_df["PROBABILITY"] = probs

        self.pipeline = pipeline
        self.full_df = test_df
        self._sync_model_refs()

        # Compute global SHAP drivers on up to 500 test rows
        self.global_shap_drivers = self._compute_global_shap(X_test.head(500))

        auc = float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 0.0
        metrics = {
            "model_type": model_type,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "roc_auc": round(auc, 4),
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
            "parameters": {
                "test_size": test_size,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "decision_threshold": round(threshold, 4),
                "auto_threshold": auto_threshold,
                "class_weight_mode": class_weight_mode,
                "max_rows": max_rows or len(df),
            },
        }
        print(
            f"[Backend Evaluation] {model_type} | ROC-AUC={metrics['roc_auc']} | "
            f"Acc={metrics['accuracy']} | Prec={metrics['precision']} | "
            f"Rec={metrics['recall']} | F1={metrics['f1']} | Threshold={round(threshold, 4)}"
        )
        self.last_training_metrics = metrics
        return metrics

    def _compute_global_shap(self, X_sample: pd.DataFrame, limit: int = 10) -> List[Dict[str, Any]]:
        if self.explainer is None or self.preprocessor is None or X_sample.empty:
            return []
        X_trans = self.preprocessor.transform(X_sample)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()

        shap_values = self.explainer.shap_values(X_trans)
        if isinstance(shap_values, list):
            vals = shap_values[1]
        elif getattr(shap_values, "ndim", 0) == 3:
            vals = shap_values[:, :, 1]
        else:
            vals = shap_values

        mean_abs = np.abs(vals).mean(axis=0)
        names = self.transformed_feature_names or [f"feat_{i}" for i in range(len(mean_abs))]

        # Aggregate one-hot features back to original feature names for clear visualization
        agg: Dict[str, float] = {}
        for t_name, val in zip(names, mean_abs):
            orig = self._map_transformed_to_original(t_name)
            agg[orig] = agg.get(orig, 0.0) + float(val)

        sorted_items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return [
            {
                "feature": k,
                "label": FEATURE_LABELS.get(k, k),
                "mean_abs_shap": round(v, 4),
            }
            for k, v in sorted_items
        ]

    def get_slice(self, sample_size: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
        if sample_size is not None and 0 < sample_size < len(self.full_df):
            return self.full_df.sample(n=int(sample_size), random_state=int(seed)).reset_index(drop=True)
        return self.full_df

    def get_health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": self.pipeline is not None,
            "total_test_records": int(len(self.full_df)),
            "has_shap": HAS_SHAP and self.explainer is not None,
            "has_fairlearn": HAS_FAIRLEARN,
            "policy_notice": POLICY_NOTICE,
        }

    def get_overview(self, sample_size: Optional[int] = None, seed: int = 42) -> Dict[str, Any]:
        df = self.get_slice(sample_size, seed)
        return {
            "policy_notice": POLICY_NOTICE,
            "applications_audited": int(len(df)),
            "total_available": int(len(self.full_df)),
            "actual_default_rate_pct": round(float(df["TARGET"].mean() * 100), 2) if not df.empty else 0.0,
            "flagged_for_review_pct": round(float(df["PREDICTION"].mean() * 100), 2) if not df.empty else 0.0,
            "mean_risk_score_pct": round(float(df["PROBABILITY"].mean() * 100), 2) if not df.empty else 0.0,
            "top_global_shap_drivers": self.global_shap_drivers,
            "training_metrics": self.last_training_metrics,
        }

    def compute_fairness(
        self,
        group_col: str = "CODE_GENDER",
        sample_size: Optional[int] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        df = self.get_slice(sample_size, seed)
        if group_col not in df.columns:
            return {
                "group_by": group_col,
                "groups": [],
                "excluded_tiny_groups": [],
                "flag_rate_gap_pct": 0.0,
                "false_positive_rate_gap_pct": 0.0,
                "fairlearn_metrics": {},
                "takeaway": f"Column '{group_col}' is not present in the active dataset.",
            }

        group_series = df[group_col].fillna("Unknown").astype(str)
        rows: List[Dict[str, Any]] = []
        excluded_tiny_groups: List[Dict[str, Any]] = []

        for grp_val, sub in df.assign(_G=group_series).groupby("_G"):
            n = int(len(sub))
            flagged_pct = round(float(sub["PREDICTION"].mean() * 100), 2)
            default_pct = round(float(sub["TARGET"].mean() * 100), 2)
            non_defaults = sub[sub["TARGET"] == 0]
            fpr_pct = round(float(non_defaults["PREDICTION"].mean() * 100), 2) if len(non_defaults) > 0 else 0.0
            defaults = sub[sub["TARGET"] == 1]
            tpr_pct = round(float(defaults["PREDICTION"].mean() * 100), 2) if len(defaults) > 0 else 0.0

            entry = {
                "group": grp_val,
                "applicants": n,
                "flagged_pct": flagged_pct,
                "actual_default_pct": default_pct,
                "false_positive_rate_pct": fpr_pct,
                "true_positive_rate_pct": tpr_pct,
                "avg_risk_score_pct": round(float(sub["PROBABILITY"].mean() * 100), 2),
            }
            if n < 20:
                excluded_tiny_groups.append(entry)
            else:
                rows.append(entry)

        flag_gap = round(max(r["flagged_pct"] for r in rows) - min(r["flagged_pct"] for r in rows), 2) if len(rows) > 1 else 0.0
        fpr_gap = round(max(r["false_positive_rate_pct"] for r in rows) - min(r["false_positive_rate_pct"] for r in rows), 2) if len(rows) > 1 else 0.0

        fairlearn_metrics: Dict[str, Any] = {}
        valid_mask = group_series.isin([r["group"] for r in rows])
        if HAS_FAIRLEARN and valid_mask.sum() > 0 and len(rows) >= 2:
            y_t = df.loc[valid_mask, "TARGET"].astype(int)
            y_p = df.loc[valid_mask, "PREDICTION"].astype(int)
            g_s = group_series.loc[valid_mask]
            fairlearn_metrics = {
                "demographic_parity_difference": round(float(demographic_parity_difference(y_t, y_p, sensitive_features=g_s)), 4),
                "demographic_parity_ratio": round(float(demographic_parity_ratio(y_t, y_p, sensitive_features=g_s)), 4),
                "equalized_odds_difference": round(float(equalized_odds_difference(y_t, y_p, sensitive_features=g_s)), 4),
                "equalized_odds_ratio": round(float(equalized_odds_ratio(y_t, y_p, sensitive_features=g_s)), 4),
            }

        highest_flag = max(rows, key=lambda r: r["flagged_pct"]) if rows else None
        lowest_flag = min(rows, key=lambda r: r["flagged_pct"]) if rows else None
        takeaway = (
            f"Across {group_col} (excluding groups with <20 applicants), "
            f"high-risk flag rates differ by {flag_gap}% "
            f"(highest: {highest_flag['group']} at {highest_flag['flagged_pct']}%, "
            f"lowest: {lowest_flag['group']} at {lowest_flag['flagged_pct']}%). "
            f"False-positive rate gap is {fpr_gap}%."
            if highest_flag and lowest_flag
            else "Not enough group data to compute disparities."
        )

        return {
            "group_by": group_col,
            "groups": rows,
            "excluded_tiny_groups": excluded_tiny_groups,
            "flag_rate_gap_pct": flag_gap,
            "false_positive_rate_gap_pct": fpr_gap,
            "fairlearn_metrics": fairlearn_metrics,
            "takeaway": takeaway,
        }

    def _map_transformed_to_original(self, trans_name: str) -> str:
        name = trans_name.replace("numerical__", "").replace("categorical__", "")
        for col in sorted(self.feature_cols, key=len, reverse=True):
            if name == col or name.startswith(col + "_"):
                return col
        return name

    def _compute_row_shap(self, X_row_df: pd.DataFrame) -> Dict[str, float]:
        if self.explainer is None or self.preprocessor is None:
            return {}
        X_trans = self.preprocessor.transform(X_row_df)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()

        shap_vals = self.explainer.shap_values(X_trans)
        if isinstance(shap_vals, list):
            vals = shap_vals[1][0]
        elif getattr(shap_vals, "ndim", 0) == 3:
            vals = shap_vals[0, :, 1]
        else:
            vals = shap_vals[0]

        aggregated: Dict[str, float] = {}
        names = self.transformed_feature_names or [f"feat_{i}" for i in range(len(vals))]
        for t_name, val in zip(names, vals):
            orig = self._map_transformed_to_original(t_name)
            aggregated[orig] = aggregated.get(orig, 0.0) + float(val)
        return aggregated

    def _format_feature_value(self, col: str, val: Any) -> str:
        if pd.isna(val):
            return "Missing"
        if col in ("AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"):
            return f"{float(val):,.0f}"
        if col in ("EXT_SOURCE_2", "EXT_SOURCE_3", "REGION_POPULATION_RELATIVE"):
            return f"{float(val):.3f}"
        if col == "DAYS_EMPLOYED":
            v = float(val)
            if v == 365243:
                return "365243 (Unemployed/Pensioner code)"
            if v < 0:
                return f"{abs(v) / 365.25:.1f} yrs ({int(v)} days)"
        if isinstance(val, (float, np.floating)) and val.is_integer():
            return str(int(val))
        return str(val)

    def _explain_feature_reason(
        self,
        feat: str,
        val: Any,
        s_val: float,
        row_dict: Dict[str, Any],
    ) -> str:
        val_str = self._format_feature_value(feat, val)
        reduces_risk = s_val < 0

        income = row_dict.get("AMT_INCOME_TOTAL")
        credit = row_dict.get("AMT_CREDIT")
        annuity = row_dict.get("AMT_ANNUITY")

        if feat in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_MEAN"):
            name = FEATURE_LABELS.get(feat, feat)
            if pd.isna(val):
                return f"{name} is missing, which adds uncertainty to repayment assessment."
            score = float(val)
            if reduces_risk:
                return f"Strong {name} ({score:.3f}) indicates reliable credit history and lowers default risk."
            return f"Lower {name} ({score:.3f}) signals weaker external credit history and raises default risk."

        if feat == "CREDIT_TO_INCOME" and pd.notna(val):
            v = float(val)
            if reduces_risk:
                return f"Credit-to-income ratio ({v:.1f}x) is manageable relative to annual income."
            return f"High credit-to-income ratio ({v:.1f}x) increases loan repayment burden."

        if feat == "ANNUITY_TO_INCOME" and pd.notna(val):
            v = float(val) * 100
            if reduces_risk:
                return f"Annuity-to-income ratio ({v:.1f}%) leaves comfortable cash-flow cushion."
            return f"High annuity-to-income ratio ({v:.1f}%) tightens monthly cash flow."

        if feat == "CREDIT_TERM" and pd.notna(val):
            v = float(val) * 100
            if reduces_risk:
                return f"Annual repayment pace ({v:.1f}% of loan) indicates a well-structured loan term."
            return f"Repayment pace ({v:.1f}% of loan) increases near-term payment pressure."

        if feat == "CREDIT_TO_GOODS" and pd.notna(val):
            v = float(val)
            if reduces_risk:
                return f"Loan-to-goods ratio ({v:.2f}x) shows reasonable collateral coverage."
            return f"High loan-to-goods ratio ({v:.2f}x) means little or no down payment."

        if feat == "DAYS_EMPLOYED_CLEAN" and pd.notna(val):
            yrs = abs(float(val)) / 365.25
            if reduces_risk:
                return f"Stable employment tenure (~{yrs:.1f} years) supports steady repayment."
            return f"Shorter employment tenure (~{yrs:.1f} years) provides less multi-year income stability."

        if feat == "AMT_CREDIT":
            if pd.notna(income) and float(income) > 0 and pd.notna(val):
                cti = float(val) / float(income)
                if reduces_risk:
                    return f"Loan amount ({val_str}, {cti:.1f}x annual income) is manageable relative to income."
                return f"High loan amount ({val_str}, {cti:.1f}x annual income) increases repayment burden."
            return f"Loan amount ({val_str}) {'supports' if reduces_risk else 'increases'} estimated default risk."

        if feat == "AMT_ANNUITY":
            if pd.notna(income) and float(income) > 0 and pd.notna(val):
                burden = (float(val) / float(income)) * 100
                if reduces_risk:
                    return f"Annualized payment ({val_str}, {burden:.1f}% of income) leaves comfortable cash-flow cushion."
                return f"Annualized payment ({val_str}, {burden:.1f}% of income) tightens monthly cash flow."
            return f"Annualized payment ({val_str}) {'lowers' if reduces_risk else 'raises'} repayment strain."

        if feat == "AMT_INCOME_TOTAL":
            if reduces_risk:
                return f"Reported annual income ({val_str}) provides solid repayment capacity."
            return f"Reported annual income ({val_str}) offers limited buffer for the requested loan size."

        if feat == "DAYS_EMPLOYED":
            if pd.notna(val) and float(val) == 365243:
                return "Employment field uses code 365243 (pensioner/unemployed); verify income source manually."
            if pd.notna(val) and float(val) < 0:
                yrs = abs(float(val)) / 365.25
                if reduces_risk:
                    return f"Stable employment tenure (~{yrs:.1f} years) supports steady repayment."
                return f"Shorter employment tenure (~{yrs:.1f} years) provides less multi-year income stability."

        if feat == "NAME_EDUCATION_TYPE":
            if reduces_risk:
                return f"Education level ({val_str}) is associated with lower historical default rates."
            return f"Education level ({val_str}) correlates with slightly higher historical default risk."

        if feat == "NAME_HOUSING_TYPE":
            if reduces_risk:
                return f"Housing situation ({val_str}) indicates residential stability."
            return f"Housing situation ({val_str}) is associated with higher historical risk."

        if feat == "NAME_FAMILY_STATUS":
            if reduces_risk:
                return f"Family status ({val_str}) is associated with lower historical default risk."
            return f"Family status ({val_str}) correlates with higher historical default risk."

        label = FEATURE_LABELS.get(feat, feat)
        if reduces_risk:
            return f"{label} ({val_str}) lowers estimated default risk."
        return f"{label} ({val_str}) increases estimated default risk."

    def _build_explanation_payload(
        self,
        X_row_df: pd.DataFrame,
        prob: float,
        pred: int,
        row_index: Optional[int] = None,
        actual_target: Optional[int] = None,
    ) -> Dict[str, Any]:
        shap_map = self._compute_row_shap(X_row_df)
        row_dict = X_row_df.iloc[0].to_dict()

        positive_drivers: List[Dict[str, Any]] = []
        risk_drivers: List[Dict[str, Any]] = []
        all_drivers: List[Dict[str, Any]] = []
        fairness_alerts: List[Dict[str, Any]] = []

        for feat, s_val in sorted(shap_map.items(), key=lambda kv: abs(kv[1]), reverse=True):
            label = FEATURE_LABELS.get(feat, feat)
            val_str = self._format_feature_value(feat, row_dict.get(feat))

            # Keep protected attributes out of rejection/acceptance reasons; surface as fairness alert
            if feat in ("CODE_GENDER", "AGE_GROUP"):
                if abs(s_val) >= 0.03:
                    direction = "higher" if s_val > 0 else "lower"
                    impact_pct = abs(s_val)
                    fairness_alerts.append({
                        "title": f"{label} Disparity Detected",
                        "attribute": feat,
                        "value": val_str,
                        "finding": (
                            f"The model assigns {impact_pct:.1%} {direction} default risk weight "
                            f"to applicants with {label} = {val_str} compared to baseline. "
                            f"This indicates a potential {'disadvantage' if s_val > 0 else 'advantage'} "
                            f"for this demographic group in model scoring."
                        ),
                        "directive": (
                            f"Human reviewer must base the final lending decision exclusively on "
                            f"verified financial repayment capacity (income, credit history, "
                            f"employment stability). {label} must not influence the approval outcome."
                        ),
                        "impact_direction": direction,
                        "shap_impact": round(s_val, 4),
                    })
                continue

            summary = self._explain_feature_reason(feat, row_dict.get(feat), s_val, row_dict)
            item = {
                "feature": feat,
                "label": label,
                "value": val_str,
                "shap_impact": round(s_val, 4),
                "summary": summary,
            }
            if s_val < -0.005:
                positive_drivers.append(item)
                all_drivers.append(item)
            elif s_val > 0.005:
                risk_drivers.append(item)
                all_drivers.append(item)

        if prob >= 0.65:
            risk_tier = "High Risk"
        elif prob >= 0.40:
            risk_tier = "Moderate Risk"
        else:
            risk_tier = "Low Risk"

        advisory_direction = (
            "Leans Toward Rejection / Human Review" if pred == 1 else "Leans Toward Acceptance"
        )

        clean_features = {
            col: (None if pd.isna(row_dict[col]) else (
                float(row_dict[col]) if isinstance(row_dict[col], (int, float, np.number)) else str(row_dict[col])
            ))
            for col in self.feature_cols
        }

        return {
            "row_index": row_index,
            "risk_probability": round(prob, 4),
            "risk_percentage": round(prob * 100, 1),
            "advisory_flag": pred,
            "advisory_direction": advisory_direction,
            "risk_tier": risk_tier,
            "historical_outcome": (
                None if actual_target is None else ("Defaulted (1)" if actual_target == 1 else "Repaid (0)")
            ),
            "policy_notice": POLICY_NOTICE,
            "acceptance_drivers": positive_drivers[:5],
            "rejection_drivers": risk_drivers[:5],
            "fairness_alerts": fairness_alerts,
            "top_shap_bars": all_drivers[:10],
            "applicant_features": clean_features,
        }

    def explain_test_applicant(
        self,
        row_index: int,
        sample_size: Optional[int] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        df = self.get_slice(sample_size, seed)
        if df.empty:
            raise ValueError("No test predictions available. Upload a dataset and train the model first.")
        if row_index < 0 or row_index >= len(df):
            raise IndexError(f"row_index {row_index} out of range (0 to {len(df) - 1})")

        row = df.iloc[row_index]
        X_row_df = df.iloc[[row_index]][self.feature_cols]
        prob = float(row["PROBABILITY"])
        pred = int(row["PREDICTION"])
        actual_target = int(row["TARGET"]) if "TARGET" in row else None

        return self._build_explanation_payload(
            X_row_df=X_row_df,
            prob=prob,
            pred=pred,
            row_index=row_index,
            actual_target=actual_target,
        )

    def predict_custom_applicant(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.pipeline is None:
            raise ValueError("Model not trained yet.")
        row_data: Dict[str, Any] = {}
        for col in self.feature_cols:
            val = payload.get(col, None)
            if val is None or val == "":
                row_data[col] = self.feature_defaults[col]
            elif pd.api.types.is_numeric_dtype(self.full_df[col]):
                row_data[col] = float(val)
            else:
                row_data[col] = str(val)

        raw_df = pd.DataFrame([row_data])
        eng_df = self._add_engineered_features(raw_df)
        for col in self.feature_cols:
            if col not in eng_df.columns or pd.isna(eng_df.loc[0, col]):
                eng_df[col] = self.feature_defaults[col]
        X_row_df = eng_df[self.feature_cols]
        prob = float(self.pipeline.predict_proba(X_row_df)[0, 1])
        threshold = (
            self.last_training_metrics["parameters"]["decision_threshold"]
            if self.last_training_metrics else 0.5
        )
        pred = int(prob >= threshold)

        return self._build_explanation_payload(
            X_row_df=X_row_df,
            prob=prob,
            pred=pred,
            row_index=None,
            actual_target=None,
        )

    def find_sample_row_index(
        self,
        flagged: Optional[int] = None,
        sample_size: Optional[int] = None,
        seed: int = 42,
    ) -> int:
        df = self.get_slice(sample_size, seed)
        if df.empty:
            return 0
        sub = df if flagged is None else df[df["PREDICTION"] == int(flagged)]
        if sub.empty:
            return 0
        return int(sub.sample(n=1).index[0])

    REVIEW_LOG_COLUMNS = [
        "timestamp",
        "row_index",
        "applicant_source",
        "probability",
        "risk_percentage",
        "risk_tier",
        "advisory_flag",
        "advisory_direction",
        "historical_outcome",
        "human_decision",
        "note",
        "CODE_GENDER",
        "AGE_GROUP",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "acceptance_reasons",
        "rejection_reasons",
    ]

    def save_review(
        self,
        row_index: Optional[int],
        probability: float,
        advisory_flag: int,
        human_decision: str,
        note: str = "",
        applicant_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        os.makedirs(os.path.dirname(REVIEW_LOG_CSV), exist_ok=True)
        details = applicant_details or {}
        features = details.get("applicant_features") or {}
        prob_val = round(float(probability), 4)
        risk_pct = round(prob_val * 100, 1)
        risk_tier = details.get("risk_tier") or (
            "High Risk" if prob_val >= 0.65 else ("Moderate Risk" if prob_val >= 0.40 else "Low Risk")
        )
        advisory_dir = details.get("advisory_direction") or (
            "Leans Toward Rejection / Human Review" if int(advisory_flag) == 1 else "Leans Toward Acceptance"
        )
        acc_reasons = " | ".join(d.get("summary", "") for d in details.get("acceptance_drivers", []))
        rej_reasons = " | ".join(d.get("summary", "") for d in details.get("rejection_drivers", []))

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "row_index": -1 if row_index is None else int(row_index),
            "applicant_source": f"Test Row #{row_index}" if row_index is not None else "Custom Applicant",
            "probability": prob_val,
            "risk_percentage": risk_pct,
            "risk_tier": risk_tier,
            "advisory_flag": int(advisory_flag),
            "advisory_direction": advisory_dir,
            "historical_outcome": details.get("historical_outcome") or "N/A",
            "human_decision": str(human_decision),
            "note": str(note or "").strip(),
            "CODE_GENDER": features.get("CODE_GENDER", ""),
            "AGE_GROUP": features.get("AGE_GROUP", ""),
            "AMT_INCOME_TOTAL": features.get("AMT_INCOME_TOTAL", ""),
            "AMT_CREDIT": features.get("AMT_CREDIT", ""),
            "AMT_ANNUITY": features.get("AMT_ANNUITY", ""),
            "EXT_SOURCE_2": features.get("EXT_SOURCE_2", ""),
            "EXT_SOURCE_3": features.get("EXT_SOURCE_3", ""),
            "acceptance_reasons": acc_reasons,
            "rejection_reasons": rej_reasons,
        }

        entry_df = pd.DataFrame([record], columns=self.REVIEW_LOG_COLUMNS)
        if os.path.exists(REVIEW_LOG_CSV):
            existing_df = pd.read_csv(REVIEW_LOG_CSV)
            for col in self.REVIEW_LOG_COLUMNS:
                if col not in existing_df.columns:
                    existing_df[col] = ""
            combined = pd.concat([existing_df[self.REVIEW_LOG_COLUMNS], entry_df], ignore_index=True)
            combined.to_csv(REVIEW_LOG_CSV, index=False)
        else:
            entry_df.to_csv(REVIEW_LOG_CSV, index=False)
        return record

    def get_reviews(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not os.path.exists(REVIEW_LOG_CSV):
            return []
        df = pd.read_csv(REVIEW_LOG_CSV)
        if df.empty:
            return []
        df = df.fillna("")
        return df.tail(limit).iloc[::-1].to_dict(orient="records")

    def build_audit_report(
        self,
        row_index: int = 0,
        sample_size: Optional[int] = None,
        seed: int = 42,
        fmt: str = "md",
    ) -> str:
        overview = self.get_overview(sample_size, seed)
        gender = self.compute_fairness("CODE_GENDER", sample_size, seed)
        age = self.compute_fairness("AGE_GROUP", sample_size, seed)
        applicant = self.explain_test_applicant(row_index, sample_size, seed)

        def fmt_groups(groups: List[Dict[str, Any]]) -> str:
            if not groups:
                return "No groups available."
            return pd.DataFrame(groups).to_string(index=False)

        acc_text = "\n".join(f"- {d['summary']}" for d in applicant["acceptance_drivers"]) or "- None"
        rej_text = "\n".join(f"- {d['summary']}" for d in applicant["rejection_drivers"]) or "- None"
        fairness_alerts_list = applicant.get("fairness_alerts", [])
        if fairness_alerts_list:
            fairness_notes_text = "\n".join(
                f"- **{a['title']}**: {a['finding']} *Directive: {a['directive']}*"
                for a in fairness_alerts_list
            )
        else:
            fairness_notes_text = "- None"
        shap_text = "\n".join(
            f"- {d['feature']}: mean |SHAP| = {d['mean_abs_shap']}"
            for d in overview["top_global_shap_drivers"]
        ) or "- SHAP summary not available."

        if fmt == "txt":
            return (
                "============================================================\n"
                "PS173 CREDIT DECISION EXPLANATION & FAIRNESS AUDIT REPORT\n"
                "============================================================\n\n"
                f"POLICY GUARDRAIL:\n{POLICY_NOTICE}\n\n"
                "------------------------------------------------------------\n"
                "1. PORTFOLIO OVERVIEW\n"
                "------------------------------------------------------------\n"
                f"Applications Audited      : {overview['applications_audited']:,}\n"
                f"Actual Default Rate       : {overview['actual_default_rate_pct']}%\n"
                f"Flagged for Human Review  : {overview['flagged_for_review_pct']}%\n"
                f"Mean Predicted Risk Score : {overview['mean_risk_score_pct']}%\n\n"
                "------------------------------------------------------------\n"
                "2. GROUP FAIRNESS SUMMARY\n"
                "------------------------------------------------------------\n"
                f"[By Gender - CODE_GENDER]\n{gender['takeaway']}\n\n"
                f"{fmt_groups(gender['groups'])}\n\n"
                f"[By Age Group - AGE_GROUP]\n{age['takeaway']}\n\n"
                f"{fmt_groups(age['groups'])}\n\n"
                "------------------------------------------------------------\n"
                "3. TOP GLOBAL MODEL DRIVERS (SHAP)\n"
                "------------------------------------------------------------\n"
                f"{shap_text}\n\n"
                "------------------------------------------------------------\n"
                f"4. APPLICANT RATIONALE (TEST ROW #{row_index})\n"
                "------------------------------------------------------------\n"
                f"Default Risk Score  : {applicant['risk_percentage']}% ({applicant['risk_tier']})\n"
                f"Advisory Direction  : {applicant['advisory_direction']}\n"
                f"Historical Outcome  : {applicant['historical_outcome']}\n\n"
                f"Why It Leans Toward Acceptance:\n{acc_text}\n\n"
                f"Why It Leans Toward Rejection / Review:\n{rej_text}\n\n"
                f"Protected-Attribute Fairness Check:\n{fairness_notes_text}\n"
            )

        return (
            "# PS173 Credit Decision Explanation & Fairness Audit Report\n\n"
            f"> **{POLICY_NOTICE}**\n\n"
            "## 1. Portfolio Overview\n"
            f"- **Applications Audited:** {overview['applications_audited']:,}\n"
            f"- **Actual Default Rate:** {overview['actual_default_rate_pct']}%\n"
            f"- **Flagged for Human Review:** {overview['flagged_for_review_pct']}%\n"
            f"- **Mean Predicted Default Risk:** {overview['mean_risk_score_pct']}%\n\n"
            "## 2. Group Fairness Summary\n"
            f"### By Gender (`CODE_GENDER`)\n{gender['takeaway']}\n\n```text\n{fmt_groups(gender['groups'])}\n```\n\n"
            f"### By Age Group (`AGE_GROUP`)\n{age['takeaway']}\n\n```text\n{fmt_groups(age['groups'])}\n```\n\n"
            f"## 3. Top Global Model Drivers (SHAP)\n{shap_text}\n\n"
            f"## 4. Applicant Rationale (Row #{row_index})\n"
            f"- **Default Risk Score:** {applicant['risk_percentage']}% ({applicant['risk_tier']})\n"
            f"- **Advisory Direction:** {applicant['advisory_direction']}\n"
            f"- **Historical Outcome:** {applicant['historical_outcome']}\n\n"
            f"### Why It Leans Toward Acceptance\n{acc_text}\n\n"
            f"### Why It Leans Toward Rejection / Review\n{rej_text}\n\n"
            f"### Protected-Attribute Fairness Check\n{fairness_notes_text}\n"
        )
