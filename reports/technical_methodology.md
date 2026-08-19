# Technical Methodology & MLOps Architecture

## 1. Executive Overview
The **EMIPredict AI Platform** is an enterprise-grade financial risk assessment and loan underwriting system designed to tackle dual machine learning problems in the FinTech and Banking sectors:
1. **Multi-Class Classification**: Predict loan repayment eligibility (`Eligible`, `High_Risk`, `Not_Eligible`).
2. **Continuous Regression**: Predict the maximum safe monthly EMI capacity (`max_monthly_emi` in INR, bounded between 500 and 50,000 INR).

The platform operates across **400,000 realistic financial profiles** distributed equally (80,000 each) across five lending categories:
- **E-commerce Shopping EMI** (₹10K - ₹200K, 3 - 24 months)
- **Home Appliances EMI** (₹20K - ₹300K, 6 - 36 months)
- **Vehicle EMI** (₹80K - ₹1.5M, 12 - 84 months)
- **Personal Loan EMI** (₹50K - ₹1.0M, 12 - 60 months)
- **Education EMI** (₹50K - ₹500K, 6 - 48 months)

---

## 2. Mathematical Formulations & Feature Engineering

### 2.1 Standard Loan Amortization Equation
The requested loan monthly repayment is computed using the standard annuity formula:
$$EMI = P \times r \times \frac{(1+r)^n}{(1+r)^n - 1}$$
Where:
- $P$ = Principal loan amount requested
- $r$ = Monthly interest rate ($\text{Annual Rate} / 12$)
- $n$ = Loan tenure in months

### 2.2 Derived Banking Ratios
1. **Fixed Obligation to Income Ratio (FOIR)**:
   $$FOIR = \frac{\text{Current EMI} + \text{Requested EMI} + \text{Monthly Rent}}{\text{Monthly Gross Salary}}$$
   *Standard banking threshold: $FOIR \le 50\%$ (Prime), $50\% < FOIR \le 65\%$ (Marginal/High Risk), $FOIR > 65\%$ (Critical Risk).*

2. **Debt-to-Income Ratio (DTI)**:
   $$DTI = \frac{\text{Current Active EMIs} + \text{Requested EMI}}{\text{Monthly Gross Salary}}$$

3. **Expense-to-Income Ratio (ETI)**:
   $$ETI = \frac{\sum \text{Living Expenses (Rent + Groceries + Utilities + Travel + Education)}}{\text{Monthly Gross Salary}}$$

4. **Disposable Monthly Surplus**:
   $$\text{Disposable Surplus} = \text{Salary} - \left(\sum \text{Living Expenses} + \text{Current EMIs}\right)$$

5. **Affordability Index**:
   $$\text{Affordability Index} = \frac{\text{Disposable Surplus}}{\text{Requested EMI}}$$

6. **Emergency Runway (Months)**:
   $$\text{Runway} = \frac{\text{Emergency Savings Fund}}{\text{Total Monthly Living Expenses} + \text{Current EMIs}}$$

---

## 3. Data Preprocessing & Validation Pipeline

1. **Quality Audit**:
   - Automated range validation on 22 input variables (age $[21, 75]$, salary $> 0$, credit score $[300, 850]$).
   - Duplicate deduplication and median/mode imputation for continuous/categorical attributes.
2. **Train / Validation / Test Splitting**:
   - Stratified splitting based on `emi_eligibility` with 70% Train ($N=280,000$), 15% Validation ($N=60,000$), and 15% Test ($N=60,000$).
3. **Encoding & Scaling**:
   - Categorical features (`gender`, `marital_status`, `education`, `employment_type`, `company_type`, `house_type`, `existing_loans`, `emi_scenario`) transformed via `OneHotEncoder(handle_unknown='ignore')`.
   - Numerical and engineered ratios normalized via `StandardScaler`.

---

## 4. Machine Learning Model Suites

### 4.1 Classification Candidates (EMI Eligibility)
- **Logistic Regression**: Multinomial baseline with $L_2$ regularization. Serves as linear benchmark.
- **Random Forest Classifier**: Non-linear ensemble with 100 decision trees, Gini impurity splitting, and max depth of 16.
- **XGBoost Classifier**: Extreme Gradient Boosting with multiclass log-loss objective (`multi:softprob`), learning rate $\eta=0.08$, subsample ratio $0.85$, and colsample $0.85$.

### 4.2 Regression Candidates (Maximum Safe Monthly EMI)
- **Linear Ridge Regression**: $L_2$-penalized linear baseline with regularization $\alpha=10.0$.
- **Random Forest Regressor**: Ensemble of 100 regression trees with MSE criterion.
- **XGBoost Regressor**: Gradient boosted trees optimizing mean squared error.

---

## 5. MLOps Lifecycle & MLflow Tracking Architecture

1. **Experiment Tracking**:
   - Distinct experiment namespaces: `EMIPredict_Classification_Suite` and `EMIPredict_Regression_Suite`.
   - Parameter logging: All tree hyperparameters, regularization weights, and split depths.
   - Metric tracking: Train, Validation, and Test Accuracy, Precision, Recall, F1-Score (macro & weighted), ROC-AUC; RMSE, MAE, $R^2$, and MAPE.
2. **Artifact Versioning**:
   - Automatic generation and persistence of Confusion Matrices, Feature Importance Bar Charts, Multiclass ROC Curves, and Residual Distribution Plots.
3. **Model Registry & Promotion**:
   - Programmatic selection of champion models: Highest Validation F1 for Classifier, Lowest Validation RMSE for Regressor.
   - Tagged and promoted to `Production` stage for inference in the multi-page Streamlit web app.
