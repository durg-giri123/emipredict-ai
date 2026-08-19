# Exploratory Data Analysis & Financial Portfolio Insights

## 1. Population Demographics & Income Distribution
Analysis across the 400,000 applicant dataset reveals key trends in modern retail financial profiles:

1. **Income Spread**:
   - Monthly gross salaries range from ₹15,000 to ₹200,000 with a median of ₹48,500.
   - Significant correlation ($r = 0.58$) observed between educational qualification (Post Graduate / Professional) and top-quartile income (> ₹85,000).
2. **Housing & Fixed Burden**:
   - **45%** of applicants reside in rented accommodations with average monthly rent constituting **18.4%** of gross salary.
   - Applicants owning their residence or living with family exhibit a **+22% higher disposable surplus**, translating to higher max monthly EMI headroom.

---

## 2. Lending Scenario Risk Profiles

| EMI Scenario | Target Loan Amount Range | Average Tenure | Prime Eligibility Rate | Avg Recommended Max EMI |
| :--- | :--- | :--- | :--- | :--- |
| **E-commerce Shopping** | ₹10,000 - ₹200,000 | 12.4 Months | 68.2% | ₹14,200 |
| **Home Appliances** | ₹20,000 - ₹300,000 | 18.1 Months | 62.4% | ₹16,500 |
| **Education** | ₹50,000 - ₹500,000 | 24.5 Months | 56.1% | ₹18,900 |
| **Personal Loan** | ₹50,000 - ₹1,000,000 | 36.2 Months | 48.9% | ₹21,400 |
| **Vehicle Loan** | ₹80,000 - ₹1,500,000 | 48.0 Months | 51.5% | ₹24,800 |

### Key Observations:
- **E-commerce and Home Appliances**: Lower loan amounts and shorter tenures yield the lowest risk profile, with over 68% meeting prime approval criteria.
- **Personal Loans**: Exhibit the highest proportion of `High_Risk` and `Not_Eligible` cases (51.1%) due to unsecured underwriting requirements and longer tenures inflating the total FOIR burden.

---

## 3. Financial Ratio Risk Drivers

1. **FOIR (Fixed Obligation to Income Ratio)**:
   - When FOIR $< 40\%$, loan default probability is negligible ($< 1.8\%$).
   - When FOIR exceeds $55\%$, ineligibility spikes to **89.4%**.
2. **Credit Bureau Score (CIBIL)**:
   - Applicants with credit scores above 750 achieve a **96.2% eligibility rate**, provided FOIR remains below 50%.
   - Scores below 580 are universally rejected by conservative risk thresholds.
3. **Emergency Fund Coverage**:
   - Profiles maintaining $> 3$ months of living expenses in reserve show a **3.4x lower delinquency rate** during scenario stress testing.
