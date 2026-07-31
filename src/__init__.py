"""
Credit Card Fraud Detection & Analytics System
Package Initialization
Authors: Varsha Gupta, Sanman Kadam
"""

__version__ = "3.0.0"
__author__ = "Varsha Gupta, Sanman Kadam"

from src.data_loader import load_dataset
from src.feature_engineering import engineer_features
from src.preprocessing import prepare_data_splits, save_scalers_and_features
from src.train import train_and_benchmark_models
from src.evaluate import evaluate_model, find_optimal_threshold
from src.utils import (
    generate_evaluation_plots, run_shap_analysis,
    generate_threshold_optimization_plot, generate_calibration_plot,
    generate_radar_chart, generate_confidence_interval_plot,
    generate_feature_importance_comparison
)
