# SmartDCA — ML Model Only (with Dataset)

This package contains:
- **Dataset ZIP**: `data/fedex_dca_synthetic_dataset.zip` (includes `cases.csv`, `interactions.csv`)
- Training script: `train.py`
- Scoring script: `score.py`

## 1) Setup (Python 3.10+ recommended)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Train the models

```bash
python train.py --dataset_zip data/fedex_dca_synthetic_dataset.zip --artifacts_dir artifacts
```

This will create:
- `artifacts/model_recovery_prob.joblib`
- `artifacts/model_recovery_amount.joblib`
- `artifacts/model_recovery_days.joblib`
- `artifacts/metadata.json` (features + metrics)

## 3) Score cases (run predictions)

### Score 10 random cases
```bash
python score.py --dataset_zip data/fedex_dca_synthetic_dataset.zip --artifacts_dir artifacts --n 10
```

### Score a specific case_id
```bash
python score.py --dataset_zip data/fedex_dca_synthetic_dataset.zip --artifacts_dir artifacts --case_id C000001
```

### Save predictions to CSV
```bash
python score.py --dataset_zip data/fedex_dca_synthetic_dataset.zip --artifacts_dir artifacts --n 200 --out_csv predictions.csv
```

## What the models predict
1) **Recovery Probability (60d)**: `recovered_within_60d` (classification)
2) **Expected Recovered Amount**: `recovered_amount_usd` (regression)
3) **Expected Recovery Days**: `recovery_days_from_allocation` (regression)

> Note: amount/days regressors are trained primarily on **recovered** cases (where labels make sense).
