import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

TITLE_PREFIX = "Random Forest"

def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm)
    plt.title(f"Confusion Matrix ({TITLE_PREFIX})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_roc(y_true, y_prob, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"ROC-AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curve ({TITLE_PREFIX})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_pr(y_true, y_prob, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig = plt.figure()
    plt.plot(recall, precision, label=f"PR-AUC={ap:.4f}")
    plt.title(f"Precision-Recall Curve ({TITLE_PREFIX})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_threshold_curve(y_true, y_prob, out_path):
    thresholds = np.linspace(0.0, 1.0, 101)
    precs, recs, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    fig = plt.figure()
    plt.plot(thresholds, precs, label="Precision")
    plt.plot(thresholds, recs, label="Recall")
    plt.plot(thresholds, f1s, label="F1")
    plt.title(f"Threshold vs Precision/Recall/F1 ({TITLE_PREFIX})")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", default="artifacts/test_predictions_rf.csv")
    ap.add_argument("--out_dir", default="artifacts/plots/rf")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.pred_csv)
    y_true = df["actual"].astype(int).values
    y_prob = df["pred_prob"].astype(float).values
    y_pred = (y_prob >= args.threshold).astype(int)

    plot_confusion_matrix(y_true, y_pred, os.path.join(args.out_dir, "confusion_matrix.png"))
    plot_roc(y_true, y_prob, os.path.join(args.out_dir, "roc_curve.png"))
    plot_pr(y_true, y_prob, os.path.join(args.out_dir, "pr_curve.png"))
    plot_threshold_curve(y_true, y_prob, os.path.join(args.out_dir, "threshold_curve.png"))

    print("✅ Saved plots in:", args.out_dir)

if __name__ == "__main__":
    main()