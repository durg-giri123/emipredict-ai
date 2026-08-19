"""Data Preprocessor and Pipeline Transformer for EMIPredict AI.

Handles data quality audits, missing value imputation, categorical encoding,
feature scaling, train-val-test splitting, and pipeline serialization.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

from src.features.feature_engineering import FinancialFeatureEngineer


def load_raw_dataset(file_path: str) -> pd.DataFrame:
    """Load raw dataset from Parquet or CSV with memory optimization."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
        
    if file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)


class DataQualityAuditor:
    """Performs validation checks and quality assessments on financial data."""
    
    @staticmethod
    def audit(df: pd.DataFrame) -> Dict[str, Any]:
        report = {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_records": int(df.duplicated().sum()),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "quality_score": 100.0,
            "anomalies": []
        }
        
        # Check invalid ranges
        if "age" in df.columns:
            invalid_age = int(((df["age"] < 18) | (df["age"] > 80)).sum())
            if invalid_age > 0:
                report["anomalies"].append(f"Invalid age records: {invalid_age}")
                
        if "credit_score" in df.columns:
            invalid_cs = int(((df["credit_score"] < 300) | (df["credit_score"] > 850)).sum())
            if invalid_cs > 0:
                report["anomalies"].append(f"Invalid credit score records: {invalid_cs}")
                
        if "monthly_salary" in df.columns:
            neg_salary = int((df["monthly_salary"] <= 0).sum())
            if neg_salary > 0:
                report["anomalies"].append(f"Non-positive salary records: {neg_salary}")
                
        # Calculate quality penalty
        missing_count = sum(report["missing_values"].values())
        penalty = (missing_count / (len(df) * max(1, len(df.columns)))) * 50 + (len(report["anomalies"]) * 10)
        report["quality_score"] = max(0.0, round(100.0 - penalty, 2))
        
        return report


class DataPreprocessor:
    """Production data preprocessing and transformation pipeline."""
    
    def __init__(self):
        self.feature_engineer = FinancialFeatureEngineer()
        
        self.categorical_cols = [
            "gender", "marital_status", "education",
            "employment_type", "company_type", "house_type",
            "existing_loans", "emi_scenario"
        ]
        
        self.numerical_cols = [
            "age", "monthly_salary", "years_of_employment",
            "monthly_rent", "family_size", "dependents",
            "school_fees", "college_fees", "travel_expenses",
            "groceries_utilities", "other_monthly_expenses",
            "current_emi_amount", "credit_score", "bank_balance",
            "emergency_fund", "requested_amount", "requested_tenure",
            "total_monthly_expenses", "disposable_income",
            "estimated_requested_emi", "debt_to_income_ratio",
            "expense_to_income_ratio", "foir", "affordability_ratio",
            "savings_to_income_ratio", "emergency_runway_months",
            "employment_stability_score", "loan_to_income_ratio",
            "interaction_salary_credit", "interaction_dti_credit",
            "credit_score_normalized"
        ]
        
        self.label_encoder = LabelEncoder()
        self.label_mapping = {"Eligible": 0, "High_Risk": 1, "Not_Eligible": 2}
        self.target_names = ["Eligible", "High_Risk", "Not_Eligible"]
        
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.is_fitted = False
        self.feature_names_out = []

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset by filling nulls and clipping logical bounds."""
        clean_df = df.copy()
        
        # Deduplicate
        clean_df = clean_df.drop_duplicates().reset_index(drop=True)
        
        # Impute missing values
        for col in clean_df.columns:
            if clean_df[col].dtype in [np.float64, np.int64, float, int]:
                clean_df[col] = clean_df[col].fillna(clean_df[col].median())
            else:
                clean_df[col] = clean_df[col].fillna(clean_df[col].mode()[0] if not clean_df[col].empty else "Unknown")
                
        # Validate ranges
        if "credit_score" in clean_df.columns:
            clean_df["credit_score"] = np.clip(clean_df["credit_score"], 300, 850)
        if "age" in clean_df.columns:
            clean_df["age"] = np.clip(clean_df["age"], 21, 75)
        if "monthly_salary" in clean_df.columns:
            clean_df["monthly_salary"] = np.clip(clean_df["monthly_salary"], 10000, 1000000)
            
        return clean_df

    def fit(self, df: pd.DataFrame):
        """Fit feature transformations, encoders, and scalers."""
        cleaned = self.clean_data(df)
        engineered = self.feature_engineer.transform(cleaned)
        
        # Fit Categorical Encoder
        self.encoder.fit(engineered[self.categorical_cols])
        encoded_cat_names = list(self.encoder.get_feature_names_out(self.categorical_cols))
        
        # Fit Numerical Scaler
        self.scaler.fit(engineered[self.numerical_cols])
        
        self.feature_names_out = self.numerical_cols + encoded_cat_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform raw dataframe into preprocessed numpy array ready for ML models."""
        if not self.is_fitted:
            raise RuntimeError("DataPreprocessor must be fitted before calling transform()")
            
        cleaned = self.clean_data(df)
        engineered = self.feature_engineer.transform(cleaned)
        
        num_scaled = self.scaler.transform(engineered[self.numerical_cols])
        cat_encoded = self.encoder.transform(engineered[self.categorical_cols])
        
        X = np.hstack([num_scaled, cat_encoded])
        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def prepare_train_test_val_splits(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.15, 
        val_size: float = 0.15,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Split dataset into 70% Train, 15% Validation, and 15% Test sets.
        Returns preprocessed X arrays, y_class arrays, and y_reg arrays.
        """
        cleaned = self.clean_data(df)
        
        # Extract targets
        has_clf_target = "emi_eligibility" in cleaned.columns
        has_reg_target = "max_monthly_emi" in cleaned.columns
        
        if not has_clf_target or not has_reg_target:
            raise ValueError("Dataset must contain 'emi_eligibility' and 'max_monthly_emi' columns")
            
        # Fit preprocessor on full train split
        # Initial train+val vs test split
        strat_col = cleaned["emi_eligibility"]
        train_val_df, test_df = train_test_split(
            cleaned, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=strat_col
        )
        
        # Train vs Val split
        adjusted_val_size = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=adjusted_val_size, 
            random_state=random_state, 
            stratify=train_val_df["emi_eligibility"]
        )
        
        print(f"Splits Created: Train={len(train_df):,} (70%), Val={len(val_df):,} (15%), Test={len(test_df):,} (15%)")
        
        # Fit transformer on training set only
        self.fit(train_df)
        
        X_train = self.transform(train_df)
        X_val = self.transform(val_df)
        X_test = self.transform(test_df)
        
        # Target Encoding
        y_train_clf = np.array([self.label_mapping[y] for y in train_df["emi_eligibility"]])
        y_val_clf = np.array([self.label_mapping[y] for y in val_df["emi_eligibility"]])
        y_test_clf = np.array([self.label_mapping[y] for y in test_df["emi_eligibility"]])
        
        y_train_reg = train_df["max_monthly_emi"].values.astype(float)
        y_val_reg = val_df["max_monthly_emi"].values.astype(float)
        y_test_reg = test_df["max_monthly_emi"].values.astype(float)
        
        return {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train_clf": y_train_clf, "y_val_clf": y_val_clf, "y_test_clf": y_test_clf,
            "y_train_reg": y_train_reg, "y_val_reg": y_val_reg, "y_test_reg": y_test_reg,
            "train_df": train_df, "val_df": val_df, "test_df": test_df,
            "feature_names": self.feature_names_out
        }

    def save(self, filepath: str):
        """Serialize preprocessor to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> "DataPreprocessor":
        """Load serialized preprocessor."""
        return joblib.load(filepath)
