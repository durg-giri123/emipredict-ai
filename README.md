# EMIPredict AI — Financial Risk Assessment Platform

A machine learning project that predicts whether someone is eligible for an EMI loan — and if yes, how much they can safely afford to pay per month.

Built as part of a FinTech capstone project using a dataset of **400,000 realistic loan applicant profiles** across 5 EMI categories (e-commerce, home appliances, vehicle, personal loan, education).

---

## What this project does

Most loan decisions today still rely heavily on manual review. This platform automates two things:

1. **Eligibility Classification** — Predicts one of three outcomes: `Eligible`, `High_Risk`, or `Not_Eligible`
2. **EMI Amount Regression** — Estimates the maximum monthly EMI a person can safely handle (between ₹500 and ₹50,000)

Both predictions run together in real-time using a Streamlit web app.

---

## Tech Stack

- **Python 3.11**
- **Scikit-learn, XGBoost** — model training
- **MLflow** — experiment tracking and model registry
- **Streamlit** — interactive web application
- **SQLite + SQLAlchemy** — application data persistence
- **Pandas, NumPy, Plotly** — data processing and visualization

---

## Project Structure

```
emipredict-ai/
│
├── src/
│   ├── data/              # Dataset generation and preprocessing
│   ├── features/          # Feature engineering (FOIR, DTI, affordability ratios)
│   ├── models/            # Training scripts for classifiers and regressors
│   ├── mlops/             # MLflow tracking, pipeline runner, model registry
│   └── database/          # SQLite CRUD operations
│
├── app/
│   ├── app.py             # Main Streamlit entry point
│   ├── components/        # Reusable UI cards and layout helpers
│   └── pages/
│       ├── 1 - Executive Dashboard
│       ├── 2 - Real-Time Predictor
│       ├── 3 - Exploratory Data Analysis
│       ├── 4 - MLflow Model Hub
│       ├── 5 - Application CRUD
│       └── 6 - Batch Prediction & Stress Testing
│
├── artifacts/             # Saved models, preprocessor, and benchmark results
├── data/                  # Raw and processed datasets
├── tests/                 # Unit tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Dataset

- **400,000 records** generated with realistic financial distributions
- **22 input features** — salary, credit score, existing loans, family size, expenses, etc.
- **5 loan scenarios** with different amount ranges and tenures:

| Scenario | Records | Amount Range | Tenure |
|----------|---------|-------------|--------|
| E-commerce Shopping | 80,000 | ₹10K–₹2L | 3–24 months |
| Home Appliances | 80,000 | ₹20K–₹3L | 6–36 months |
| Vehicle | 80,000 | ₹80K–₹15L | 12–84 months |
| Personal Loan | 80,000 | ₹50K–₹10L | 12–60 months |
| Education | 80,000 | ₹50K–₹5L | 6–48 months |

---

## Models Trained

### Classification (EMI Eligibility)

| Model | Val Accuracy | Val F1 | Test Accuracy |
|-------|-------------|--------|--------------|
| Logistic Regression | 94.5% | 0.930 | 94.5% |
| Random Forest | 98.4% | 0.983 | 98.4% |
| **XGBoost** ✅ | **98.7%** | **0.987** | **98.6%** |

### Regression (Max Safe EMI)

| Model | Val RMSE | Val R² | Test RMSE |
|-------|----------|--------|----------|
| Ridge Regression | ₹2,438 | 0.960 | ₹2,374 |
| **Random Forest** ✅ | **₹284** | **0.9995** | **₹279** |
| XGBoost | ₹291 | 0.999 | ₹286 |

Both models comfortably meet the project targets (accuracy > 90%, RMSE < ₹2,000).

---

## Running Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the models (generates dataset + runs full pipeline)

```bash
python -X utf8 src/mlops/pipeline.py --records 400000
```

This takes about 8–10 minutes. It generates the dataset, trains all 6 models, logs everything to MLflow, and saves the best models to `artifacts/production_models/`.

To retrain from scratch (overwrite existing data):

```bash
python -X utf8 src/mlops/pipeline.py --records 400000 --force-regenerate
```

### 3. Launch the web app

```bash
python -m streamlit run app/app.py
```

Open http://localhost:8501 in your browser.

### 4. (Optional) View MLflow experiment dashboard

```bash
mlflow ui --port 5000
```

Open http://localhost:5000 to compare all model runs.

---

## Feature Engineering

Beyond the 22 raw features, the system derives several financial ratios that significantly improve model accuracy:

- **DTI** (Debt-to-Income) — existing EMI burden relative to monthly salary
- **ETI** (Expense-to-Income) — total monthly expenses as % of income
- **FOIR** (Fixed Obligation to Income Ratio) — standard banking risk metric
- **Affordability Index** — composite score of how comfortably someone can take on new EMI
- **Emergency Runway** — months of expenses covered by emergency fund
- **Interaction features** — e.g., credit score × salary band combinations

---

## App Pages

| Page | What it does |
|------|-------------|
| Executive Dashboard | Portfolio-level KPIs, eligibility breakdowns by scenario |
| Real-Time Predictor | Enter any applicant's details, get instant dual prediction |
| EDA | Explore the dataset — distributions, correlations, demographic patterns |
| MLflow Hub | Compare all 6 model runs, check metrics, inspect registered models |
| CRUD Manager | Add, search, update, delete applicant records in the database |
| Batch & Stress Test | Upload CSV for bulk scoring, run macroeconomic shock simulations |

---

## Docker

```bash
docker-compose up --build -d
```

- Streamlit app: http://localhost:8501
- MLflow UI: http://localhost:5000

---

## Notes

- The pipeline uses a 70/15/15 train/validation/test split
- All MLflow runs are stored in a local SQLite file (`mlflow.db`)
- Production models are saved as `.joblib` files in `artifacts/production_models/`
- The stress testing simulator lets you simulate scenarios like salary cuts, rate hikes, and inflation to see portfolio-level impact
