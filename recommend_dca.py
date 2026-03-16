import argparse
import json
import os
from typing import Dict, List, Optional

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

CAT_COLS = [
    "region",
    "industry",
    "credit_score_band",
    "last_contact_channel",
    "assigned_dca_id",
]


def invoice_bucket(amount: float) -> str:
    """Bucket invoice amount into coarse segments."""
    if amount < 5000:
        return "S"
    if amount < 20000:
        return "M"
    if amount < 80000:
        return "L"
    return "XL"


def build_dca_segment_table(cases_csv_path: str) -> pd.DataFrame:
    """
    Build historical performance multiplier per:
      (dca_id, region, industry, invoice_bucket)

    Uses recovered_flag recovery rate as performance signal.
    """
    cases = pd.read_csv(cases_csv_path)

    cases["region"] = cases["region"].fillna("UNKNOWN").astype(str).str.strip()
    cases["industry"] = cases["industry"].fillna("UNKNOWN").astype(str).str.strip()
    cases["assigned_dca_id"] = (
        cases["assigned_dca_id"].fillna("UNKNOWN").astype(str).str.strip()
    )

    cases["invoice_bucket"] = cases["invoice_amount_usd"].apply(invoice_bucket)
    cases["recovered_flag"] = cases["recovered_flag"].fillna(0).astype(int)

    # Use only rows with a real DCA id
    hist = cases[
        (cases["assigned_dca_id"] != "")
        & (cases["assigned_dca_id"].str.upper() != "UNKNOWN")
    ].copy()

    grp = (
        hist.groupby(
            ["assigned_dca_id", "region", "industry", "invoice_bucket"],
            dropna=False
        )["recovered_flag"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(
            columns={
                "assigned_dca_id": "dca_id",
                "mean": "recovery_rate",
                "count": "n",
            }
        )
    )

    global_rate = hist["recovered_flag"].mean() if len(hist) else 0.5

    # Convert recovery rate into multiplier around 1.0
    grp["multiplier"] = (grp["recovery_rate"] / max(global_rate, 1e-6)).clip(0.6, 1.6)

    return grp


def get_available_dcas(
    cases_csv_path: str,
    capacity_left: Optional[Dict[str, int]] = None
) -> List[str]:
    """
    Get all available DCAs dynamically from cases.csv.
    If capacity_left contains extra DCAs, include them too.
    """
    cases = pd.read_csv(cases_csv_path)

    dcas = (
        cases["assigned_dca_id"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .tolist()
    )

    valid_dcas = {
        dca for dca in dcas
        if dca and dca.upper() != "UNKNOWN"
    }

    if capacity_left is not None:
        valid_dcas.update(
            dca.strip()
            for dca in capacity_left.keys()
            if str(dca).strip()
        )

    return sorted(valid_dcas)


def load_case(args) -> dict:
    """Load new case from --input_json or --input_file."""
    if args.input_json:
        return json.loads(args.input_json)

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError("Provide either --input_json or --input_file")


def score_case_for_dca(case: dict, dca_id: str, clf, reg_amt, reg_days) -> dict:
    """
    Score one case for one DCA using the three trained models.
    """
    row = dict(case)
    row["assigned_dca_id"] = dca_id

    # Fill missing categorical safely
    for c in CAT_COLS:
        if c not in row or row[c] is None:
            row[c] = "UNKNOWN"

    # Fill missing numeric defaults safely
    numeric_defaults = {
        "invoice_amount_usd": 0.0,
        "num_open_invoices": 0,
        "overdue_days_at_allocation": 0,
        "previous_default_count": 0,
        "previous_recovery_rate": 0.0,
        "payment_history_score": 0.0,
        "contact_attempts_last_30d": 0,
        "dispute_flag": 0,
        "promised_to_pay_flag": 0,
        "sla_days": 14,
    }

    for k, v in numeric_defaults.items():
        if k not in row or row[k] is None:
            row[k] = v

    X = pd.DataFrame([row])[FEATURES]

    prob = float(clf.predict_proba(X)[:, 1][0])
    exp_amt = float(max(0.0, reg_amt.predict(X)[0]))
    exp_days = float(max(0.0, reg_days.predict(X)[0]))

    return {
        "prob": prob,
        "exp_amt": exp_amt,
        "exp_days": exp_days,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--cases_csv", default="data/fedex_dca_synthetic_dataset/cases.csv")

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_json", help="New case features as JSON string")
    group.add_argument("--input_file", help="Path to JSON file containing new case")

    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument(
        "--capacity_json",
        default=None,
        help="Optional: {'DCA-01':120, 'DCA-02':50} active capacity left"
    )
    args = ap.parse_args()

    # Load models
    clf = load(os.path.join(args.artifacts_dir, "model_recovery_prob.joblib"))
    reg_amt = load(os.path.join(args.artifacts_dir, "model_recovery_amount.joblib"))
    reg_days = load(os.path.join(args.artifacts_dir, "model_recovery_days.joblib"))

    # Load case
    case = load_case(args)

    # Fill required categorical defaults
    for c in CAT_COLS:
        case.setdefault(c, "UNKNOWN")

    # Fill required numeric defaults
    case.setdefault("invoice_amount_usd", 0.0)
    case.setdefault("num_open_invoices", 0)
    case.setdefault("overdue_days_at_allocation", 0)
    case.setdefault("previous_default_count", 0)
    case.setdefault("previous_recovery_rate", 0.0)
    case.setdefault("payment_history_score", 0.0)
    case.setdefault("contact_attempts_last_30d", 0)
    case.setdefault("dispute_flag", 0)
    case.setdefault("promised_to_pay_flag", 0)
    case.setdefault("sla_days", 14)

    # Optional capacity constraints
    capacity_left = None
    if args.capacity_json:
        capacity_left = json.loads(args.capacity_json)

    # Get ALL available DCAs dynamically
    available_dcas = get_available_dcas(args.cases_csv, capacity_left)
    if not available_dcas:
        raise RuntimeError("No DCA IDs found in cases.csv")

    # Build historical DCA segment multipliers
    seg = build_dca_segment_table(args.cases_csv)

    # Segment key for multiplier lookup
    seg_key = (
        str(case.get("region", "UNKNOWN")).strip(),
        str(case.get("industry", "UNKNOWN")).strip(),
        invoice_bucket(float(case["invoice_amount_usd"]))
    )

    # Evaluate each DCA
    results = []
    probs, amts, days_list = [], [], []
    per_dca_scores = {}

    for dca in available_dcas:
        # Apply capacity constraint only if provided
        if capacity_left is not None and capacity_left.get(dca, 0) <= 0:
            continue

        s = score_case_for_dca(case, dca, clf, reg_amt, reg_days)
        per_dca_scores[dca] = s
        probs.append(s["prob"])
        amts.append(s["exp_amt"])
        days_list.append(s["exp_days"])

    if not per_dca_scores:
        raise RuntimeError("No DCAs available after capacity filtering.")

    # Normalize helpers
    prob_min, prob_max = min(probs), max(probs)
    amt_min, amt_max = min(amts), max(amts)
    day_min, day_max = min(days_list), max(days_list)

    def norm(x, a, b):
        return 0.0 if abs(b - a) < 1e-9 else (x - a) / (b - a)

    for dca, s in per_dca_scores.items():
        p_n = norm(s["prob"], prob_min, prob_max)
        a_n = norm(s["exp_amt"], amt_min, amt_max)
        d_n = norm(s["exp_days"], day_min, day_max)

        # Higher better: probability high, amount high, days low
        base_suitability = 0.6 * p_n + 0.2 * a_n - 0.2 * d_n

        region, industry, bucket = seg_key
        match = seg[
            (seg["dca_id"] == dca)
            & (seg["region"] == region)
            & (seg["industry"] == industry)
            & (seg["invoice_bucket"] == bucket)
        ]

        if len(match) == 0:
            multiplier = 1.0
            reason_mult = "No history for this segment -> multiplier=1.0"
        else:
            multiplier = float(match["multiplier"].iloc[0])
            rr = float(match["recovery_rate"].iloc[0])
            n = int(match["n"].iloc[0])
            reason_mult = (
                f"Segment history: recovery_rate={rr:.2f} over n={n} "
                f"-> multiplier={multiplier:.2f}"
            )

        final_score = base_suitability * multiplier

        results.append({
            "dca_id": dca,
            "final_score": float(final_score),
            "ml_prob_60d": s["prob"],
            "ml_exp_amount": s["exp_amt"],
            "ml_exp_days": s["exp_days"],
            "multiplier_reason": reason_mult,
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    top = results[:args.top_k]

    print("\n=== Top DCA Recommendations ===")
    print(f"Checked {len(results)} available DCAs\n")

    for i, r in enumerate(top, 1):
        print(f"#{i} {r['dca_id']}  final_score={r['final_score']:.4f}")
        print(
            f"   ML: prob_60d={r['ml_prob_60d']:.4f}, "
            f"exp_amt=${r['ml_exp_amount']:.2f}, "
            f"exp_days={r['ml_exp_days']:.1f}"
        )
        print(f"   {r['multiplier_reason']}\n")

    print("✅ Recommended assignment:", top[0]["dca_id"])


if __name__ == "__main__":
    main()