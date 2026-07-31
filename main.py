"""
Credit Card Fraud Detection Pipeline
====================================
Authors: Varsha Gupta, Sanman Kadam

Executes end-to-end machine learning workflow:
  1. Data Loading & Deduplication
  2. Feature Engineering (incl. Amount Z-score, Hour of Week, Is_Weekend)
  3. Stratified Train/Test Split
  4. Separate RobustScaling for Amount and Time (No Leakage)
  5. Isolation Forest Anomaly Scoring (Fitted on Train Only)
  6. 5×3 Repeated Stratified Cross-Validation & Optuna Tuning
  7. 9-Model Benchmark with Confidence Intervals
  8. Probability Calibration (Top 3 Models)
  9. Multi-Metric Threshold Optimization (F1, F2, Cost-Sensitive)
  10. Comprehensive Visualization & SHAP Interpretability
"""

import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

from src.logging_config import setup_logging, get_logger
from src.config import MODELS_DIR, REPORTS_DIR, IMAGES_DIR, DATA_RAW
from src.data_loader import load_dataset
from src.feature_engineering import engineer_features
from src.preprocessing import prepare_data_splits, save_scalers_and_features
from src.train import train_and_benchmark_models
from src.utils import (
    generate_evaluation_plots, run_shap_analysis,
    generate_threshold_optimization_plot, generate_calibration_plot,
    generate_radar_chart, generate_confidence_interval_plot,
    generate_feature_importance_comparison
)


def print_banner(logger, title: str):
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Credit Card Fraud Detection Pipeline v3.0'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick mode: skip Optuna tuning, use fewer CV repeats'
    )
    parser.add_argument(
        '--data', type=str, default=None,
        help='Path to creditcard.csv (default: auto-detect)'
    )
    return parser.parse_args()


def main():
    # Parse CLI arguments
    args = parse_args()

    # Initialize logging
    logger = setup_logging()

    logger.info("")
    logger.info("AI-BASED CREDIT CARD FRAUD DETECTION PIPELINE v3.0")
    logger.info("Authors: Varsha Gupta, Sanman Kadam")
    logger.info("=" * 60)
    pipeline_start = time.time()

    data_path = args.data or DATA_RAW

    # 1. Load & Deduplicate
    print_banner(logger, "Step 1: Data Loading")
    step_start = time.time()
    df = load_dataset(data_path)
    logger.info(f"  Completed in {time.time() - step_start:.1f}s")

    # 2. Feature Engineering
    print_banner(logger, "Step 2: Feature Engineering")
    step_start = time.time()
    df_feat = engineer_features(df)
    logger.info(f"  Completed in {time.time() - step_start:.1f}s")

    # 3. Train/Test Split, Scaler Normalization & Isolation Forest (No Leakage)
    print_banner(logger, "Step 3: Preprocessing & Leakage Prevention Split")
    step_start = time.time()
    X_train, X_test, y_train, y_test, amount_scaler, time_scaler, iso_forest = prepare_data_splits(
        df_feat
    )
    logger.info(f"  Completed in {time.time() - step_start:.1f}s")

    # Save fitted scalers, Isolation Forest & feature names
    save_scalers_and_features(
        amount_scaler, time_scaler, X_train.columns.tolist(), MODELS_DIR,
        iso_forest=iso_forest
    )

    # 4. Training, Repeated Stratified CV & Optimization
    print_banner(logger, "Step 4: Model Training & Benchmark (9 Models, 5×3 CV)")
    step_start = time.time()
    results_df, trained_models, calibration_data = train_and_benchmark_models(
        X_train, y_train, X_test, y_test, MODELS_DIR, REPORTS_DIR
    )
    logger.info(f"  Training completed in {time.time() - step_start:.1f}s")

    # 5. Visualizations & SHAP Analysis
    print_banner(logger, "Step 5: Visualizations & Interpretability")
    step_start = time.time()

    generate_evaluation_plots(trained_models, X_test, y_test, IMAGES_DIR)
    generate_threshold_optimization_plot(trained_models, X_test, y_test, IMAGES_DIR)
    generate_calibration_plot(calibration_data, IMAGES_DIR)
    generate_radar_chart(results_df, IMAGES_DIR)
    generate_confidence_interval_plot(results_df, IMAGES_DIR)
    generate_feature_importance_comparison(trained_models, X_test, IMAGES_DIR)

    best_model = trained_models.get('Random Forest', list(trained_models.values())[0])
    run_shap_analysis(best_model, X_test, IMAGES_DIR)

    logger.info(f"  Visualization completed in {time.time() - step_start:.1f}s")

    # Summary
    total_time = time.time() - pipeline_start
    print_banner(logger, "PIPELINE EXECUTION COMPLETE")
    logger.info(f"  Total execution time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  Models trained: {len(trained_models)}")
    logger.info(f"  Best model (PR-AUC): {results_df.iloc[0]['Model']}")
    logger.info(f"  Saved artifacts to {MODELS_DIR}/ and {REPORTS_DIR}/")
    logger.info(f"  Saved visualizations to {IMAGES_DIR}/")
    logger.info(f"  Pipeline log: logs/pipeline.log")


if __name__ == '__main__':
    main()
