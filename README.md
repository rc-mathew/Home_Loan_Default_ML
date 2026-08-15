# Home Loan Default Risk & Customer Eligibility — End-to-End Deployment

This repository converts the 8 GB RAM XGBoost notebook into a deployable ML project.

## Architecture

```text
Raw Home Credit CSVs
        |
        v
Memory-efficient feature engineering
        |
        v
Applicant-level feature table
        |
        +--> Logistic Regression
        |
        +--> XGBoost (hist, CPU, constrained)
        |
        v
Model comparison
        |
        v
Best model artifact (.joblib)
        |
        +--> Batch scoring
        |
        +--> FastAPI REST API
        |
        +--> Docker
```

## Models

- Logistic Regression — interpretable baseline.
- XGBoost — nonlinear production candidate.
- XGBoost is the actual `xgboost.XGBClassifier`, configured conservatively for an 8 GB laptop:
  - `tree_method="hist"`
  - `max_depth=3`
  - `n_estimators=120`
  - `max_bin=64`
  - `subsample=0.80`
  - `colsample_bytree=0.70`
  - `n_jobs=2`

## Feature set

The deployment uses the same compact feature set as the supplied 8 GB notebook: 33 selected numeric features covering application affordability, external risk scores, bureau history, previous applications, POS/Cash, credit cards and installment behavior.

`bureau_balance.csv` is intentionally not expanded in this 8 GB implementation, matching the notebook's resource-constrained design.

## 1. Project structure

```text
home_loan_default_deployment/
├── api/
│   └── main.py
├── src/
│   └── features.py
├── scripts/
│   ├── train.py
│   └── predict_batch.py
├── tests/
│   └── test_api.py
├── data/
├── models/
├── artifacts/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── notebooks_Home_Loan_Default_8GB_XGBoost.ipynb
```

## 2. Dataset

Place the supplied CSV files in `data/`:

- application_train.csv
- bureau.csv
- bureau_balance.csv (not used by the model in this optimized version)
- POS_CASH_balance.csv
- credit_card_balance.csv
- previous_application.csv
- installments_payments.csv

The training script only reads the selected columns needed for the compact feature set.

## 3. Install

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Train

From the project root:

```bash
python scripts/train.py --data-dir data --model-dir models
```

The script:

1. Loads application data.
2. Sequentially processes historical tables.
3. Builds applicant-level features.
4. Splits the data.
5. Trains Logistic Regression and XGBoost.
6. Calculates ROC-AUC, PR-AUC, Accuracy, Precision, Recall and F1.
7. Selects the highest ROC-AUC model.
8. Refits that model on the full labeled feature matrix.
9. Saves:

```text
models/home_loan_default_model.joblib
models/metadata.json
models/model_metrics.csv
```

**Important:** The final production artifact is trained only after the validation comparison. For a regulated production system, use a separate untouched validation/test set and a formal model-governance process before deployment.

## 5. Run the API locally

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Useful endpoints:

- `GET /health`
- `GET /metadata`
- `POST /predict`

## 6. Example API request

The API accepts the compact applicant-level feature vector. Example:

```json
{
  "SK_ID_CURR": 100001,
  "AMT_INCOME_TOTAL": 180000,
  "AMT_CREDIT": 500000,
  "AMT_ANNUITY": 25000,
  "AMT_GOODS_PRICE": 450000,
  "CNT_CHILDREN": 1,
  "CNT_FAM_MEMBERS": 3,
  "DAYS_BIRTH": -14000,
  "DAYS_EMPLOYED": -3000,
  "EXT_SOURCE_1": 0.45,
  "EXT_SOURCE_2": 0.60,
  "EXT_SOURCE_3": 0.55,
  "REGION_RATING_CLIENT": 2,
  "REGION_RATING_CLIENT_W_CITY": 2,
  "bureau_credit_count": 4,
  "bureau_AMT_CREDIT_SUM_mean": 300000,
  "bureau_AMT_CREDIT_SUM_DEBT_mean": 100000,
  "bureau_AMT_CREDIT_SUM_OVERDUE_max": 0,
  "previous_application_count": 3,
  "prev_AMT_CREDIT_mean": 250000,
  "prev_AMT_APPLICATION_mean": 270000,
  "pos_SK_DPD_mean": 2,
  "pos_SK_DPD_DEF_mean": 1,
  "pos_record_count": 20,
  "cc_AMT_BALANCE_mean": 50000,
  "cc_AMT_CREDIT_LIMIT_ACTUAL_mean": 150000,
  "cc_SK_DPD_mean": 0,
  "inst_payment_delay_mean": 2,
  "inst_payment_shortfall_mean": 1000,
  "installment_record_count": 30
}
```

The response contains:

```json
{
  "SK_ID_CURR": 100001,
  "default_probability": 0.123456,
  "risk_segment": "Eligible / Lower Risk"
}
```

The numeric probability above is only an example. It is not a claim about this customer.

## 7. Batch scoring

After training:

```bash
python scripts/predict_batch.py --data-dir data --model-path models/home_loan_default_model.joblib --output artifacts/predictions.csv
```

This creates:

```text
artifacts/predictions.csv
```

with:

- SK_ID_CURR
- Predicted_Default_Probability
- Risk_Segment

## 8. Docker deployment

Build:

```bash
docker build -t home-loan-risk-api .
```

Run:

```bash
docker run --rm -p 8000:8000 home-loan-risk-api
```

Or:

```bash
docker compose up --build
```

Then visit:

```text
http://localhost:8000/docs
```

## 9. Production architecture

For a real FinTech deployment, extend this prototype to:

```text
Loan application
      |
      v
API Gateway / Authentication
      |
      v
Feature validation
      |
      v
Feature store / historical feature service
      |
      v
Versioned model artifact
      |
      v
Default probability
      |
      v
Policy engine
      |
      +--> Eligible / Lower Risk
      +--> Manual Review
      +--> High Risk
      |
      v
Decision logging + monitoring
```

The model should be a risk signal, not an uncontrolled replacement for underwriting policy.

## 10. Monitoring

Recommended production metrics:

- ROC-AUC
- PR-AUC
- Population Stability Index / feature drift
- Prediction distribution drift
- Realized default rate
- Approval rate
- Bad rate by risk segment
- Calibration
- Missing-feature rate
- API latency/error rate
- Model and feature version lineage

## 11. Important model-governance considerations

Before using this for real lending:

- Validate temporal stability.
- Ensure no post-decision leakage.
- Calibrate probabilities.
- Define an explicit cost matrix for threshold selection.
- Validate fairness according to applicable law and company policy.
- Maintain a champion/challenger process.
- Version datasets, features, model artifacts and code.
- Keep human review where required.
- Perform security, privacy and access-control reviews.

## 12. Relationship to the notebook

The included notebook remains the exploratory/model-development artifact. The `src/`, `scripts/` and `api/` components turn the same modeling concept into a reproducible deployment workflow.

The notebook's 8 GB constraints are preserved: sequential historical aggregation, selected columns, compact feature engineering, numeric-only modeling and constrained XGBoost.
