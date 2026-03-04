"""
SmartDCA ML (offline training) — COMPLETE train.py

Trains 3 models:
1) recovered_within_60d (classification)  -> probability
2) recovered_amount_usd (regression)      -> amount if recovered (trained on recovered cases)
3) recovery_days_from_allocation (regression) -> days if recovered (trained on recovered cases)

✅ Works with:
- a direct path to cases.csv (recommended for your unzipped dataset), OR
- a zip file containing cases.csv at top level

Prints (for professors):
- Confusion Matrix with TN/FP/FN/TP
- Accuracy, Precision, Recall with formula + substituted values
- Actual vs Predicted table (first 20)
- Saves full test set predictions CSV: artifacts/test_actual_vs_predicted.csv

Run (your case):
python train.py --dataset_path data/fedex_dca_synthetic_dataset/cases.csv --artifacts_dir artifacts
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =============================
# Targets
# =============================
TARGET_CLASS = "recovered_within_60d"
TARGET_AMT = "recovered_amount_usd"
TARGET_DAYS = "recovery_days_from_allocation"

# =============================
# Features (inputs)
# =============================
DEFAULT_FEATURES = [
    "region",
    "industry",
    "invoice_amount_usd",
    "num_open_invoices",
    "overdue_days_at_allocation",
    "previous_default_count",
    "previous_recovery_rate",
    "payment_history_score",
    "credit_score_band",
    "contact_attempts_last_30d",
    "last_contact_channel",
    "dispute_flag",
    "promised_to_pay_flag",
    "assigned_dca_id",
    "sla_days",
]

CATEGORICAL = [
    "region",
    "industry",
    "credit_score_band",
    "last_contact_channel",
    "assigned_dca_id",
]

NUMERIC = [c for c in DEFAULT_FEATURES if c not in CATEGORICAL]


@dataclass
class Metrics:
    clf_roc_auc: float
    clf_pr_auc: float
    clf_accuracy: float
    clf_precision: float
    clf_recall: float
    tn: int
    fp: int
    fn: int
    tp: int
    threshold: float
    reg_amount_mae: float
    reg_days_mae: float
    n_rows: int
    n_train: int
    n_test: int


# =============================
# Data loading
# =============================
def load_cases(dataset_path: str) -> pd.DataFrame:
    """
    Accepts either:
    - a .zip that contains cases.csv at top level, OR
    - a direct path to cases.csv
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    if dataset_path.endswith(".zip"):
        with zipfile.ZipFile(dataset_path, "r") as z:
            with z.open("cases.csv") as f:
                return pd.read_csv(f)

    # direct CSV path
    return pd.read_csv(dataset_path)


# =============================
# Preprocessing
# =============================
def build_preprocess() -> ColumnTransformer:
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CATEGORICAL),
            ("num", num_pipe, NUMERIC),
        ]
    )


# =============================
# Training
# =============================
def train_models(
    cases: pd.DataFrame,
    random_state: int = 42,
    threshold: float = 0.5,
) -> Tuple[Pipeline, Pipeline, Pipeline, Metrics, pd.DataFrame]:
    """
    Returns:
      - classifier pipeline
      - amount regressor pipeline
      - days regressor pipeline
      - metrics object
      - preview df (actual vs predicted on test split)
    """

    # ---- Basic cleaning for better demo outputs ----
    # Make categorical missing values explicit for clean printing + stable inference
    for col in ["region", "industry", "credit_score_band", "last_contact_channel", "assigned_dca_id"]:
        if col in cases.columns:
            cases[col] = cases[col].fillna("UNKNOWN")

    for col in ["dispute_flag", "promised_to_pay_flag"]:
        if col in cases.columns:
            cases[col] = cases[col].fillna(0).astype(int)

    # ---- Validate required columns ----
    required_targets = [TARGET_CLASS, TARGET_AMT, TARGET_DAYS]
    missing_targets = [t for t in required_targets if t not in cases.columns]
    if missing_targets:
        raise ValueError(f"Missing required target columns in cases.csv: {missing_targets}")

    missing_features = [f for f in DEFAULT_FEATURES if f not in cases.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns in cases.csv: {missing_features}")

    # ---- Classification: recovered_within_60d ----
    X = cases[DEFAULT_FEATURES].copy()
    y = cases[TARGET_CLASS].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    clf_pipe = Pipeline(
        steps=[
            ("preprocess", build_preprocess()),
            (
                "model",
                HistGradientBoostingClassifier(
                    random_state=random_state, max_depth=6, learning_rate=0.08, max_iter=250
                ),
            ),
        ]
    )

    clf_pipe.fit(X_train, y_train)

    proba = clf_pipe.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    # Confusion matrix + metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    roc = float(roc_auc_score(y_test, proba))
    pr_auc = float(average_precision_score(y_test, proba))

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))

    # Actual vs Predicted table (for explanation)
    preview = pd.DataFrame(
        {
            "actual": y_test.values,
            "pred_prob": proba,
            "predicted": y_pred,
        }
    ).reset_index(drop=True)

    # ---- Regression: amount and days (train primarily on recovered cases) ----
    if "recovered_flag" not in cases.columns:
        # fallback: if recovered_flag missing, use any rows with non-null recovery_days
        recovered_mask = cases[TARGET_DAYS].notna()
    else:
        recovered_mask = cases["recovered_flag"].astype(int) == 1

    reg_data = cases.loc[recovered_mask].copy()
    if len(reg_data) < 200:
        # fallback to all rows (shouldn't happen with your dataset)
        reg_data = cases.copy()

    Xr = reg_data[DEFAULT_FEATURES].copy()
    y_amt = reg_data[TARGET_AMT].astype(float)
    y_days = reg_data[TARGET_DAYS].astype(float)

    Xr_train, Xr_test, ya_train, ya_test = train_test_split(
        Xr, y_amt, test_size=0.2, random_state=random_state
    )
    Xd_train, Xd_test, yd_train, yd_test = train_test_split(
        Xr, y_days, test_size=0.2, random_state=random_state
    )

    amt_pipe = Pipeline(
        steps=[
            ("preprocess", build_preprocess()),
            (
                "model",
                HistGradientBoostingRegressor(
                    random_state=random_state, max_depth=6, learning_rate=0.08, max_iter=300
                ),
            ),
        ]
    )

    days_pipe = Pipeline(
        steps=[
            ("preprocess", build_preprocess()),
            (
                "model",
                HistGradientBoostingRegressor(
                    random_state=random_state, max_depth=6, learning_rate=0.08, max_iter=300
                ),
            ),
        ]
    )

    amt_pipe.fit(Xr_train, ya_train)
    days_pipe.fit(Xd_train, yd_train)

    pred_amt = amt_pipe.predict(Xr_test)
    pred_days = days_pipe.predict(Xd_test)

    mae_amt = float(mean_absolute_error(ya_test, pred_amt))
    mae_days = float(mean_absolute_error(yd_test, pred_days))

    metrics = Metrics(
        clf_roc_auc=roc,
        clf_pr_auc=pr_auc,
        clf_accuracy=acc,
        clf_precision=prec,
        clf_recall=rec,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        threshold=float(threshold),
        reg_amount_mae=mae_amt,
        reg_days_mae=mae_days,
        n_rows=int(len(cases)),
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
    )

    return clf_pipe, amt_pipe, days_pipe, metrics, preview


# =============================
# Main
# =============================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_path",
        default="data/fedex_dca_synthetic_dataset/cases.csv",
        help="Path to cases.csv OR a zip containing cases.csv at top level",
    )
    ap.add_argument("--artifacts_dir", default="artifacts", help="Output directory for models + outputs")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for class prediction")
    args = ap.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)

    cases = load_cases(args.dataset_path)

    clf_pipe, amt_pipe, days_pipe, metrics, preview = train_models(
        cases, random_state=args.random_state, threshold=args.threshold
    )

    # Save models
    dump(clf_pipe, os.path.join(args.artifacts_dir, "model_recovery_prob.joblib"))
    dump(amt_pipe, os.path.join(args.artifacts_dir, "model_recovery_amount.joblib"))
    dump(days_pipe, os.path.join(args.artifacts_dir, "model_recovery_days.joblib"))

    # Save metadata
    meta = {
        "features": DEFAULT_FEATURES,
        "categorical": CATEGORICAL,
        "numeric": NUMERIC,
        "targets": {
            "classification": TARGET_CLASS,
            "amount": TARGET_AMT,
            "days": TARGET_DAYS,
        },
        "metrics": asdict(metrics),
    }
    with open(os.path.join(args.artifacts_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ===== Print “professor-friendly” outputs =====
    print("\n================ SMARTDCA — CLASSIFICATION RESULTS ================")
    print(f"Threshold = {metrics.threshold}")

    print("\nConfusion Matrix (rows=Actual, cols=Predicted):")
    print("            Pred=0    Pred=1")
    print(f"Actual=0     {metrics.tn:5d}     {metrics.fp:5d}")
    print(f"Actual=1     {metrics.fn:5d}     {metrics.tp:5d}")

    # Print formulas with substituted values
    tp, tn, fp, fn = metrics.tp, metrics.tn, metrics.fp, metrics.fn
    denom = tp + tn + fp + fn

    print("\nMetrics (with formula):")
    print(
        f"Accuracy  = (TP+TN)/(TP+TN+FP+FN) = ({tp}+{tn})/({tp}+{tn}+{fp}+{fn})"
        f" = {metrics.clf_accuracy:.4f}"
    )
    print(
        f"Precision = TP/(TP+FP)            = {tp}/({tp}+{fp})"
        f" = {metrics.clf_precision:.4f}"
    )
    print(
        f"Recall    = TP/(TP+FN)            = {tp}/({tp}+{fn})"
        f" = {metrics.clf_recall:.4f}"
    )

    print("\nExtra (probability quality metrics):")
    print(f"ROC-AUC = {metrics.clf_roc_auc:.4f}")
    print(f"PR-AUC  = {metrics.clf_pr_auc:.4f}")

    print("\nActual vs Predicted (first 20 rows of TEST set):")
    print(preview.head(20).to_string(index=False))

    # Save full actual vs predicted for the test set
    out_csv = os.path.join(args.artifacts_dir, "test_actual_vs_predicted.csv")
    preview.to_csv(out_csv, index=False)
    print(f"\nSaved file for professor: {out_csv}")

    print("\n================ REGRESSION RESULTS ================")
    print(f"Recovered Amount MAE (USD) = {metrics.reg_amount_mae:.2f}")
    print(f"Recovery Days MAE          = {metrics.reg_days_mae:.2f}")

    print("\n✅ Training complete. Models + metrics saved in:", args.artifacts_dir)


if __name__ == "__main__":
    main()
