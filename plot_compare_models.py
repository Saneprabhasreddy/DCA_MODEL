import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)

def load_preds(path: str):
    df = pd.read_csv(path)

    # support both formats:
    # - test_predictions_*.csv => actual, pred_prob
    # - test_actual_vs_predicted.csv => actual, pred_prob
    if "actual" not in df.columns or "pred_prob" not in df.columns:
        raise ValueError(f"{path} must contain columns: actual, pred_prob")

    y_true = df["actual"].astype(int).values
    y_prob = df["pred_prob"].astype(float).values
    return y_true, y_prob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logreg", default="artifacts/test_predictions_logreg.csv")
    ap.add_argument("--rf", default="artifacts/test_predictions_rf.csv")
    ap.add_argument("--gb", default="artifacts/test_predictions_gb.csv")
    ap.add_argument("--hgb", default="artifacts/test_actual_vs_predicted.csv")
    ap.add_argument("--out_dir", default="artifacts/plots/combined")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    models = [
        ("Logistic Regression", args.logreg),
        ("Random Forest", args.rf),
        ("Gradient Boosting", args.gb),
        ("HistGradientBoosting", args.hgb),
    ]

    # ---------- Combined ROC ----------
    fig = plt.figure()
    for name, path in models:
        y_true, y_prob = load_preds(path)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(args.out_dir, "roc_compare.png")
    fig.savefig(roc_path, dpi=200)
    plt.close(fig)

    # ---------- Combined PR ----------
    fig = plt.figure()
    for name, path in models:
        y_true, y_prob = load_preds(path)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_score = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, label=f"{name} (AP={ap_score:.4f})")

    plt.title("Precision–Recall Curve Comparison")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    pr_path = os.path.join(args.out_dir, "pr_compare.png")
    fig.savefig(pr_path, dpi=200)
    plt.close(fig)

    print("✅ Saved combined plots:")
    print(" -", roc_path)
    print(" -", pr_path)

if __name__ == "__main__":
    main()