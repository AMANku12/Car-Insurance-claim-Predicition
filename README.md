# Car Insurance Claim Prediction

A full, production‑minded **machine learning project** that predicts:

1. **Whether a customer will file a car insurance claim (classification)**
2. **The monetary value of the claim if it occurs (regression)**

Built from raw Kaggle data using **scikit‑learn, XGBoost, and structured ML pipelines** with rigorous preprocessing, feature engineering, cross‑validation, and hyperparameter tuning.

---

## Data Source

**Dataset:** Car Insurance Claim Data (Kaggle)
**Records:** ~8,000 policyholders
**Targets:**

* `is_claim` → Binary classification (claim/no claim)
* `new_claim_value` → Regression target (claim cost)

Original features include:

* Demographics (age, income, marital status, gender)
* Vehicle attributes (type, value, age, usage)
* Driving history (license points, prior claim frequency, license revocations)
* Policy details (tenure, commute distance, home ownership)

---

## Project Objectives

* Build a **high‑performance claim prediction model** for underwriting and fraud risk.
* Estimate **expected loss value** conditional on claim occurrence.
* Create a **fully reproducible ML pipeline** suitable for real deployment.

Business relevance:

* Risk‑based pricing
* Loss‑prevention targeting
* Claims investigation prioritization

---

## Technical Stack

* Python, NumPy, Pandas
* Scikit‑Learn
* XGBoost, CatBoost
* Seaborn, Matplotlib
* Statsmodels (VIF)

---

## Machine Learning Pipeline

### 1. Data Cleaning

* Removed duplicate records
* Stripped currency symbols and converted numeric fields
* Standardized categorical values
* Dropped redundant identifiers and leakage‑prone features

### 2. Stratified Train/Test Split

* Stratification based on binned claim value
* Preserved class distribution across splits

### 3. Missing Value Handling

* **Numerical:** KNN Imputation
* **Categorical:** Most‑frequent imputation

### 4. Feature Encoding

* Binary → Ordinal Encoding
* Ordinal → Education ranking
* Nominal → One‑Hot Encoding (baseline category dropped)

### 5. Multicollinearity Control

* Variance Inflation Factor (VIF)
* Removed redundant dummy variables

### 6. Feature Engineering

* Square‑root transforms for skewed financial features
* Standard scaling for numerical stability

### 7. Model Pipelines

All preprocessing implemented via **ColumnTransformer + Pipelines** for:

* Numerical features
* Binary categorical features
* Ordinal categorical features
* One‑hot encoded features

This guarantees identical transformations in training and inference.

---

## Classification Modeling (Claim Occurrence)

### Candidate Models

* Logistic Regression
* KNN
* Decision Tree
* Random Forest
* Linear SVM
* AdaBoost
* Gradient Boosting
* Bagging
* CatBoost
* **XGBoost (final model)**

### Validation Strategy

* 10‑Fold Cross Validation
* Metric: **Weighted F1‑Score**

### Hyperparameter Optimization

* **RandomizedSearchCV (2,000 iterations)**
* **GridSearchCV for fine‑tuning**

### Final Evaluation

* Test‑set F1 Score
* Confusion Matrix
* Generalization gap monitored

---

## Regression Modeling (Claim Cost Prediction)

Trained only on records where `new_claim_value > 0`.

### Candidate Models

* Linear Regression
* SGD Regressor *(final model)*
* Random Forest Regressor
* KNN Regressor
* Support Vector Regressor
* XGBoost Regressor

### Metric

* **Root Mean Squared Error (RMSE)**
* Mean Absolute Error (MAE)

### Optimization

* RandomizedSearchCV
* GridSearchCV

Final evaluation performed on completely unseen test data.

---

## Results (Replace With Your Final Metrics)

**Classification:**

* Model: XGBoost
* Test F1 Score: 0.77328

**Regression:**

* RMSE: 8378.846
* MAE: 8378.846

---

## How to Run

```bash
git clone <repo-url>
cd car-insurance-claim-prediction
pip install -r requirements.txt
python main.py
```

---

## Project Structure

```
├── main.ipynb
├── data/
│   └── car_insurance_claim.csv
├── requirements.txt
└── README.md
```

---

## Reproducibility Practices

* Fixed random seeds
* Data leakage controlled through pipelines
* Cross‑validated all major models
* Hyperparameter search isolated from test data

---

## Resume‑Ready Highlights

* Built a **dual‑task ML system** predicting both claim probability and claim cost.
* Designed **fully automated preprocessing pipelines** with KNN imputation and categorical encoders.
* Identified and mitigated **multicollinearity using VIF analysis**.
* Optimized **XGBoost and SGD models using large‑scale hyperparameter search**.
* Evaluated models with **stratified cross‑validation and strict test isolation**.

---

## Future Improvements

* Class‑imbalance cost‑sensitive optimization
* SHAP‑based global feature interpretability
* Time‑aware modeling for policy risk evolution
* Deployment using FastAPI + Docker

---

This project demonstrates strong foundations in **tabular ML modeling, production pipelines, statistical validation, and business‑driven evaluation**, making it suitable for **Data Scientist / ML Engineer** roles.
