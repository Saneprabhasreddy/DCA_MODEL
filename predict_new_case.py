import argparse
import json
import numpy as np
import pandas as pd
from joblib import load

DEFAULT_ARTIFACTS_DIR = "artifacts"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", default=DEFAULT_ARTIFACTS_DIR)
    ap.add_argument("--input_json", required=True)
    args = ap.parse_args()

    # Load models
    prob_model = load(f"{args.artifacts_dir}/model_recovery_prob.joblib")
    amount_model = load(f"{args.artifacts_dir}/model_recovery_amount.joblib")
    days_model = load(f"{args.artifacts_dir}/model_recovery_days.joblib")
    

    with open(args.input_json) as f:
        case = json.load(f)

    features = [
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

    for c in ["region","industry","credit_score_band","last_contact_channel","assigned_dca_id"]:
        if c not in case or case[c] is None:
            case[c] = "UNKNOWN"

    for c in features:
        if c not in case:
            case[c] = np.nan

    X = pd.DataFrame([case])[features]

    # Predictions
    prob = prob_model.predict_proba(X)[:,1][0]
    amount = amount_model.predict(X)[0]
    days = days_model.predict(X)[0]
   

    print("\n=== New Case Prediction ===")
    print("Recovery Probability :", round(float(prob),2))
    print("Predicted Recovery Amount :", round(float(amount),2),"USD")
    print("Estimated Recovery Days :", int(days))

if __name__ == "__main__":
    main()