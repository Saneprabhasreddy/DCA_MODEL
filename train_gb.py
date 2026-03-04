"""
Classifier: GradientBoostingClassifier (sklearn)
Task: Predict recovered_within_60d (0/1)

Outputs (in artifacts/):
- gb_model.joblib
- metrics_gb.json
- test_predictions_gb.csv

Run:
python3 train_gb.py --dataset_path data/fedex_dca_synthetic_dataset/cases.csv --artifacts_dir artifacts --threshold 0.5
"""

import argparse
import json
import os
import zipfile

import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier


TARGET = "recovered_within_60d"

FEATURES = [
    "region","industry","invoice_amount_usd","num_open_invoices",
    "overdue_days_at_allocation","previous_default_count","previous_recovery_rate",
    "payment_history_score","credit_score_band","contact_attempts_last_30d",
    "last_contact_channel","dispute_flag","promised_to_pay_flag","assigned_dca_id",
    "sla_days"
]

CAT_COLS = ["region","industry","credit_score_band","last_contact_channel","assigned_dca_id"]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]


def load_cases(dataset_path: str) -> pd.DataFrame:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    if dataset_path.endswith(".zip"):
        with zipfile.ZipFile(dataset_path, "r") as z:
            with z.open("cases.csv") as f:
                return pd.read_csv(f)

    return pd.read_csv(dataset_path)


def build_preprocess() -> ColumnTransformer:
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CAT_COLS),
            ("num", num_pipe, NUM_COLS),
        ]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", default="data/fedex_dca_synthetic_dataset/cases.csv")
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)

    df = load_cases(args.dataset_path)

    # Clean
    for c in CAT_COLS:
        df[c] = df[c].fillna("UNKNOWN")
    for c in ["dispute_flag","promised_to_pay_flag"]:
        df[c] = df[c].fillna(0).astype(int)

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state, stratify=y
    )

    gb = GradientBoostingClassifier(
        random_state=args.random_state,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3
    )

    pipe = Pipeline(steps=[("preprocess", build_preprocess()), ("model", gb)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (proba >= args.threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "model": "GradientBoostingClassifier",
        "threshold": args.threshold,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_total": int(len(df)),
    }

    dump(pipe, os.path.join(args.artifacts_dir, "gb_model.joblib"))
    with open(os.path.join(args.artifacts_dir, "metrics_gb.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    out = pd.DataFrame({"actual": y_test.values, "pred_prob": proba, "predicted": y_pred})
    out.to_csv(os.path.join(args.artifacts_dir, "test_predictions_gb.csv"), index=False)

    print("\n✅ GradientBoosting training complete\n")
    print("Confusion Matrix (rows=Actual, cols=Predicted):")
    print("            Pred=0    Pred=1")
    print(f"Actual=0     {tn:5d}     {fp:5d}")
    print(f"Actual=1     {fn:5d}     {tp:5d}")

    print("\nMetrics:")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-score  : {metrics['f1']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC    : {metrics['pr_auc']:.4f}")

    print("\nSaved:")
    print(f"- {args.artifacts_dir}/gb_model.joblib")
    print(f"- {args.artifacts_dir}/metrics_gb.json")
    print(f"- {args.artifacts_dir}/test_predictions_gb.csv")


if __name__ == "__main__":
    main()