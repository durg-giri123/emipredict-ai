from setuptools import setup, find_packages

setup(
    name="emipredict-ai",
    version="1.0.0",
    description="Intelligent Financial Risk Assessment & EMI Prediction Platform",
    author="EMIPredict AI Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "mlflow>=2.10.0",
        "streamlit>=1.30.0",
        "plotly>=5.18.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "joblib>=1.3.0",
        "pydantic>=2.0.0",
        "pytest>=7.4.0",
        "sqlalchemy>=2.0.0",
    ],
)
