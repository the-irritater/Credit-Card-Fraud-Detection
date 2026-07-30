"""
Credit Card Fraud Detection Pipeline
====================================
Authors: Sanman Kadam, Varsha Gupta

Executes end-to-end machine learning workflow:
  1. Data Loading & Deduplication
  2. Feature Engineering
  3. Stratified Train/Test Split
  4. Separate RobustScaling for Amount and Time (No Leakage)
  5. 5-Fold Stratified Cross-Validation & Optuna Tuning
  6. Threshold Optimization & Multi-Metric Evaluation
  7. Model Serialization & Reporting
"""

import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_dataset
from src.feature_engineering import engineer_features
from src.preprocessing import prepare_data_splits, save_scalers_and_features
from src.train import train_and_benchmark_models
from src.utils import generate_evaluation_plots, run_shap_analysis

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, 'creditcard.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')


def print_banner(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    print("\nAI-BASED CREDIT CARD FRAUD DETECTION PIPELINE")
    print("Authors: Sanman Kadam, Varsha Gupta")
    print("=" * 60)

    # 1. Load & Deduplicate
    print_banner("Step 1: Data Loading")
    df = load_dataset(DATA_RAW)

    # 2. Feature Engineering
    print_banner("Step 2: Feature Engineering")
    df_feat = engineer_features(df)

    # 3. Train/Test Split & Scaler Normalization (No Leakage)
    print_banner("Step 3: Preprocessing & Leakage Prevention Split")
    X_train, X_test, y_train, y_test, amount_scaler, time_scaler = prepare_data_splits(
        df_feat, test_size=0.2, random_state=42
    )

    # Save fitted scalers & feature names
    save_scalers_and_features(
        amount_scaler, time_scaler, X_train.columns.tolist(), MODELS_DIR
    )

    # 4. Training, 5-Fold Cross-Validation & Optimization
    print_banner("Step 4: Model Training & Benchmark")
    results_df, trained_models = train_and_benchmark_models(
        X_train, y_train, X_test, y_test, MODELS_DIR, REPORTS_DIR
    )

    # 5. Visualizations & SHAP Analysis
    print_banner("Step 5: Visualizations & Interpretability")
    generate_evaluation_plots(trained_models, X_test, y_test, IMAGES_DIR)

    best_model = trained_models.get('Random Forest', list(trained_models.values())[0])
    run_shap_analysis(best_model, X_test, IMAGES_DIR)

    print_banner("PIPELINE EXECUTION COMPLETE")
    print(f"Saved artifacts to {MODELS_DIR}/ and {REPORTS_DIR}/")


if __name__ == '__main__':
    main()
