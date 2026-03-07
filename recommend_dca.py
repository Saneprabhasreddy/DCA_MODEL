import argparse
import json
import os
import numpy as np
import pandas as pd
from joblib import load

FEATURES = [
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

CAT_COLS = ["region", "industry", "credit_score_band", "last_contact_channel", "assigned_dca_id"]

DEFAULT_DCAS = [f"DCA-{i:02d}" for i in range(1, 11)]  # DCA-01..DCA-10


def invoice_bucket(amount: float) -> str:
    # simple bucketing for segment performance
    if amount < 5000:
        return "S"
    if amount < 20000:
        return "M"
    if amount < 80000:
        return "L"
    return "XL"


def build_dca_segment_table(cases_csv_path: str) -> pd.DataFrame:
    """
    Builds historical performance multiplier per:
      (dca_id, region, industry, invoice_bucket)
    Uses recovered_flag recovery rate as performance signal.
    """
    cases = pd.read_csv(cases_csv_path)

    # clean
    cases["region"] = cases["region"].fillna("UNKNOWN")
    cases["industry"] = cases["industry"].fillna("UNKNOWN")
    cases["assigned_dca_id"] = cases["assigned_dca_id"].fillna("UNKNOWN")

    cases["invoice_bucket"] = cases["invoice_amount_usd"].apply(invoice_bucket)
    cases["recovered_flag"] = cases["recovered_flag"].fillna(0).astype(int)

    # Only use rows that actually had a DCA assignment (skip UNKNOWN)
    hist = cases[cases["assigned_dca_id"].astype(str).str.startswith("DCA-")].copy()

    grp = (
        hist.groupby(["assigned_dca_id", "region", "industry", "invoice_bucket"])["recovered_flag"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"assigned_dca_id": "dca_id", "mean": "recovery_rate", "count": "n"})
    )

    # Convert recovery_rate into a multiplier around 1.0
    # Example: global avg 0.40 -> if segment 0.50 => multiplier > 1
    global_rate = hist["recovered_flag"].mean() if len(hist) else 0.5
    grp["multiplier"] = (grp["recovery_rate"] / max(global_rate, 1e-6)).clip(0.6, 1.6)

    return grp


def score_case_for_dca(case: dict, dca_id: str, clf, reg_amt, reg_days) -> dict:
    # Make a copy and set DCA
    row = dict(case)
    row["assigned_dca_id"] = dca_id

    # Fill missing categorical safely
    for c in CAT_COLS:
        if c not in row or row[c] is None:
            row[c] = "UNKNOWN"

    # Build dataframe in correct feature order
    X = pd.DataFrame([row])[FEATURES]

    prob = float(clf.predict_proba(X)[:, 1][0])
    exp_amt = float(max(0.0, reg_amt.predict(X)[0]))
    exp_days = float(max(0.0, reg_days.predict(X)[0]))

    return {"prob": prob, "exp_amt": exp_amt, "exp_days": exp_days}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--cases_csv", default="data/fedex_dca_synthetic_dataset/cases.csv")
    ap.add_argument("--input_json", required=True, help="New case features as JSON string")
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--capacity_json", default=None, help="Optional: {'DCA-01':120, ...} active capacity left")
    args = ap.parse_args()

    # Load models (must run train.py first)
    clf = load(os.path.join(args.artifacts_dir, "model_recovery_prob.joblib"))
    reg_amt = load(os.path.join(args.artifacts_dir, "model_recovery_amount.joblib"))
    reg_days = load(os.path.join(args.artifacts_dir, "model_recovery_days.joblib"))

    with open(args.input_json) as f:
        case = json.load(f)

    # Fill required keys with defaults if missing
    for c in CAT_COLS:
        case.setdefault(c, "UNKNOWN")
    case.setdefault("sla_days", 14)

    # Build historical DCA segment multipliers
    seg = build_dca_segment_table(args.cases_csv)

    # Optional capacity constraints
    capacity_left = None
    if args.capacity_json:
        capacity_left = json.loads(args.capacity_json)

    # Evaluate each DCA
    results = []
    seg_key = (case.get("region", "UNKNOWN"), case.get("industry", "UNKNOWN"), invoice_bucket(float(case["invoice_amount_usd"])))

    # normalization helpers for suitability
    probs, amts, days_list = [], [], []
    per_dca_scores = {}

    for dca in DEFAULT_DCAS:
        # capacity check (optional)
        if capacity_left is not None and capacity_left.get(dca, 0) <= 0:
            continue

        s = score_case_for_dca(case, dca, clf, reg_amt, reg_days)
        per_dca_scores[dca] = s
        probs.append(s["prob"])
        amts.append(s["exp_amt"])
        days_list.append(s["exp_days"])

    if not per_dca_scores:
        raise RuntimeError("No DCAs available (capacity constraints removed all).")

    # Normalize amount/days for suitability score
    prob_min, prob_max = min(probs), max(probs)
    amt_min, amt_max = min(amts), max(amts)
    day_min, day_max = min(days_list), max(days_list)

    def norm(x, a, b):
        return 0.0 if b - a < 1e-9 else (x - a) / (b - a)

    for dca, s in per_dca_scores.items():
        # base suitability from ML
        p_n = norm(s["prob"], prob_min, prob_max)
        a_n = norm(s["exp_amt"], amt_min, amt_max)
        d_n = norm(s["exp_days"], day_min, day_max)

        # Higher is better: probability & amount high, days low
        base_suitability = 0.6 * p_n + 0.2 * a_n - 0.2 * d_n

        # segment multiplier lookup
        region, industry, bucket = seg_key
        match = seg[
            (seg["dca_id"] == dca)
            & (seg["region"] == region)
            & (seg["industry"] == industry)
            & (seg["invoice_bucket"] == bucket)
        ]

        if len(match) == 0:
            multiplier = 1.0
            reason_mult = "No history for this segment → multiplier=1.0"
        else:
            multiplier = float(match["multiplier"].iloc[0])
            rr = float(match["recovery_rate"].iloc[0])
            n = int(match["n"].iloc[0])
            reason_mult = f"Segment history: recovery_rate={rr:.2f} over n={n} → multiplier={multiplier:.2f}"

        final_score = base_suitability * multiplier

        results.append({
            "dca_id": dca,
            "final_score": float(final_score),
            "ml_prob_60d": s["prob"],
            "ml_exp_amount": s["exp_amt"],
            "ml_exp_days": s["exp_days"],
            "multiplier_reason": reason_mult
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    top = results[: args.top_k]

    print("\n=== Top DCA Recommendations ===")
    for i, r in enumerate(top, 1):
        print(f"\n#{i} {r['dca_id']}  final_score={r['final_score']:.4f}")
        print(f"   ML: prob_60d={r['ml_prob_60d']:.4f}, exp_amt=${r['ml_exp_amount']:.2f}, exp_days={r['ml_exp_days']:.1f}")
        print(f"   {r['multiplier_reason']}")

    print("\n✅ Recommended assignment:", top[0]["dca_id"])


if __name__ == "__main__":
    main()
