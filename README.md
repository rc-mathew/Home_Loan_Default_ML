# Home Loan Default Risk & Customer Eligibility — Production-Ready ML Platform

A memory-conscious, end-to-end credit-risk ML project based on the Home Credit dataset. The project is designed for an **8 GB RAM laptop** while adding the engineering and model-governance pieces expected in an ML Engineer / FinTech portfolio.

> **Important:** This is a portfolio/research system, not a real lending decision engine. Real credit use requires validated data lineage, privacy/security controls, calibration, fairness assessment, policy integration, monitoring, human oversight and regulatory/model-risk governance.

## What was fixed from the earlier version

| Gap | Resolution |
|---|---|
| Only a couple of commits | Recommended commit plan below creates meaningful milestones instead of one giant commit. |
| No visible test coverage | Added unit tests + `pytest-cov`; GitHub Actions publishes a coverage artifact on every push/PR. Do not claim a percentage until CI reports it. |
| No CI/CD | Added `.github/workflows/ci.yml`. It installs dependencies and runs the test suite with coverage. |
| No evaluation plots | Training now creates ROC, Precision-Recall and model-comparison plots under `artifacts/`. |
| Notebook outputs not shown | Notebook remains the development artifact; README documents the exact commands that generate reproducible evaluation artifacts. Do not fabricate metrics before running on the real data. |
| `bureau_balance.csv` excluded | Added a chunked `bureau_balance` aggregation using only `SK_ID_BUREAU` and `STATUS`, producing compact applicant-level delinquency features while remaining memory-conscious. |
| Class imbalance not explicit | Added prevalence reporting, `class_weight='balanced'` for Logistic Regression and `scale_pos_weight` for XGBoost. PR-AUC is reported and used as the primary model-selection metric. |
| Fixed 0.50 threshold | Threshold is selected on validation using F2, then frozen before the untouched test evaluation. |
| No temporal validation | Added `src/validation.py` and `scripts/temporal_validation.py`. The current Kaggle-style `application_train.csv` has no true application/decision timestamp, so the project **does not falsely claim temporal validation**. When an approved timestamp is available, the chronological split can be executed directly. |
| Test set used too early | Data is now split into 60% train / 20% validation / 20% untouched holdout test. Model and threshold selection happen on validation; the holdout is evaluated afterward. |

## Architecture

```text
Home Credit CSVs
      |
      v
Sequential, memory-conscious feature engineering
      |
      v
Applicant-level feature table
      |
      +----------------------+----------------------+
      |                                             |
      v                                             v
Logistic Regression                              XGBoost
balanced baseline                         class-weighted via
interpretable model                       scale_pos_weight
      |                                             |
      +----------------------+----------------------+
                             v
                 Validation model selection
                  PR-AUC -> ROC-AUC tie-break
                             |
                      Threshold tuning
                        (F2 on val)
                             |
                             v
                   Untouched holdout test
                             |
               +-------------+-------------+
               |                           |
               v                           v
          Model artifact              Evaluation plots
               |                           |
               v                           v
            FastAPI                 ROC / PR / comparison
               |
               v
       Probability + risk segment
               |
               v
     Monitoring / policy / review
```

## Repository structure

```text
Home_Loan_Default_ML/
├── api/main.py
├── src/
│   ├── features.py
│   ├── metrics.py
│   └── validation.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict_batch.py
│   └── temporal_validation.py
├── notebooks/
│   └── Home_Loan_Default_8GB_XGBoost.ipynb
├── tests/
│   ├── test_api.py
│   ├── test_features.py
│   └── test_metrics.py
├── .github/workflows/ci.yml
├── configs/
├── artifacts/
├── models/
├── data/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 1. Data and memory strategy

The historical tables are much larger than the application table. To keep the pipeline viable on an 8 GB laptop:

- read only required columns;
- aggregate one historical source at a time;
- process `bureau_balance.csv` in chunks;
- merge only compact applicant-level features;
- delete intermediate frames and call garbage collection;
- use `float32` model matrices;
- use CPU `tree_method='hist'`, shallow trees and `n_jobs=2`.

### bureau_balance decision

The previous implementation skipped `bureau_balance.csv`. That was a resource-saving shortcut, not a modeling principle. This version uses it in a controlled way:

- `SK_ID_BUREAU` is mapped to `SK_ID_CURR` using a two-column bureau mapping;
- `STATUS` is converted into a compact delinquency indicator;
- the file is processed in chunks;
- applicant-level record count, delinquency rate and delinquency maximum are retained.

If memory becomes tight, the chunk size in `src/features.py` can be reduced from `200_000`.

## 2. Class imbalance

Home Credit default prediction is highly imbalanced. Accuracy alone can therefore be misleading.

The pipeline explicitly records:

- positive/default rate;
- negative/non-default rate;
- training class counts;
- XGBoost `scale_pos_weight = negatives / positives`;
- Logistic Regression `class_weight='balanced'`;
- ROC-AUC and PR-AUC;
- precision, recall and F1;
- a validation-selected threshold.

PR-AUC is especially useful here because it focuses attention on the quality of the positive/default class under imbalance.

After training, inspect:

```text
artifacts/class_distribution.json
models/validation_metrics.csv
models/holdout_test_metrics.csv
```

## 3. Train / validation / test design

The pipeline uses:

```text
60% train
20% validation
20% untouched holdout test
```

The validation set is used to:

1. compare Logistic Regression and XGBoost;
2. select the model using PR-AUC, with ROC-AUC as a tie-break;
3. tune the classification threshold using F2.

The holdout test set is evaluated only after these choices are frozen.

The selected model is then refit on the development data (train + validation) for the production artifact. The holdout test remains untouched for the reported evaluation.

## 4. Temporal validation — important limitation

A true temporal split requires a real **application/decision timestamp**. The supplied Home Credit `application_train.csv` does not provide a direct application timestamp suitable for claiming temporal validation.

Therefore this repository intentionally **does not pretend that `SK_ID_CURR`, `DAYS_BIRTH`, or another relative feature is a calendar timestamp**.

When an approved application date is available, run:

```bash
python scripts/temporal_validation.py --date-column APPLICATION_DATE
```

The implementation creates chronological train/validation/test periods and reports the boundaries. In an interview, the correct explanation is:

> "The public dataset lacks a true application timestamp, so I did not manufacture temporal validation. I implemented the temporal split interface and would run it on the production application decision date before deployment."

## 5. Train

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/train.py --data-dir data --model-dir models
```

Linux/macOS:

```bash
source .venv/bin/activate
python scripts/train.py --data-dir data --model-dir models
```

The training command produces:

```text
models/home_loan_default_model.joblib
models/metadata.json
models/model_metrics.csv
models/validation_metrics.csv
models/holdout_test_metrics.csv
artifacts/class_distribution.json
artifacts/roc_curves.png
artifacts/precision_recall_curves.png
artifacts/model_comparison.png
```

### Evaluation artifacts

After running training locally, commit the generated **small, non-sensitive** evaluation images if you want them visible on GitHub:

```bash
git add artifacts/*.png models/*metrics.csv artifacts/class_distribution.json
```

Do not commit raw customer data or secrets.

## 6. Tests and coverage

Run locally:

```bash
pytest --cov=src --cov=api --cov-report=term-missing --cov-report=html
```

Open:

```text
htmlcov/index.html
```

GitHub Actions runs the same coverage command on every push and pull request and stores `coverage.xml` as a workflow artifact.

**Do not put a made-up coverage percentage in the README.** Once the workflow runs, use the reported number/badge if desired.

## 7. CI/CD

The current workflow is continuous integration:

```text
Push / Pull Request
        |
        v
GitHub Actions
        |
        v
Install dependencies
        |
        v
Run tests + coverage
```

For a full production CI/CD pipeline, add after the tests:

```text
Build Docker image
        |
Security scan
        |
Push image to registry
        |
Deploy to staging
        |
Smoke test
        |
Manual approval
        |
Production deployment
```

This separation is deliberate: a portfolio repository should not automatically deploy an unvalidated credit model to production.

## 8. API

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Endpoints:

- `GET /health`
- `GET /metadata`
- `POST /predict`

## 9. Docker

```bash
docker build -t home-loan-risk-api .
docker run --rm -p 8000:8000 home-loan-risk-api
```

or:

```bash
docker compose up --build
```

## 10. GitHub commit strategy

Do not create meaningless commits just to increase the count. Use commits that represent real engineering milestones.

Recommended history:

```text
1. Initial project structure and README
2. Add memory-efficient feature engineering
3. Add Logistic Regression and XGBoost training pipeline
4. Add class-imbalance handling and threshold tuning
5. Add holdout evaluation and plots
6. Add bureau_balance chunked features
7. Add FastAPI inference service
8. Add tests and GitHub Actions CI
9. Add temporal validation framework
10. Add Docker deployment configuration
```

Example:

```bash
git add .
git commit -m "Add memory-efficient feature engineering"
```

Then later:

```bash
git add .
git commit -m "Add class imbalance handling and holdout evaluation"
```

This gives interviewers a meaningful iteration trail.

## 11. Model governance checklist

Before a real lending deployment:

- [ ] true temporal validation with application/decision date;
- [ ] leakage audit;
- [ ] probability calibration;
- [ ] threshold/cost analysis;
- [ ] fairness analysis;
- [ ] stability/drift monitoring;
- [ ] champion/challenger model process;
- [ ] feature/data/model version lineage;
- [ ] security and privacy controls;
- [ ] audit logging;
- [ ] human-review policy;
- [ ] model-risk approval.

## 12. Interview-ready explanation

**Why XGBoost?** It captures nonlinear interactions and usually provides a stronger tabular baseline than a linear model. The histogram tree method, shallow trees and limited parallelism keep memory/CPU usage reasonable on an 8 GB machine.

**Why Logistic Regression?** It is an interpretable, strong baseline and provides a useful comparison against a nonlinear model.

**How did you handle imbalance?** I measured prevalence, used `class_weight='balanced'` for Logistic Regression and `scale_pos_weight` for XGBoost, reported PR-AUC in addition to ROC-AUC, and selected the operating threshold on validation data rather than blindly using 0.50.

**Why is bureau_balance now included?** The earlier version excluded it to protect 8 GB memory. The improved pipeline processes it in chunks and keeps only compact applicant-level delinquency statistics.

**How did you handle temporal validation?** I did not falsely use an ID or relative date as an application timestamp. The public dataset lacks a suitable decision timestamp, so the repository includes a temporal split interface that can be run as soon as a true application date is available.

**What is the most important production caveat?** The model probability is a risk signal. Lending eligibility must be determined by a governed policy layer using validated thresholds, affordability rules, fairness controls and human review where appropriate.
