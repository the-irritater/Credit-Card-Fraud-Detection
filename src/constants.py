"""
Constants Module
================
Static constants and enumerations used across the pipeline.
Separates configuration (tunable) from constants (fixed domain knowledge).

Authors: Sanman Kadam, Varsha Gupta
"""

import numpy as np

# ── Amount Categorization ────────────────────────────────────────────────────
AMOUNT_BINS = [-np.inf, 10, 50, 100, 500, 1000, np.inf]
AMOUNT_LABELS = [0, 1, 2, 3, 4, 5]
AMOUNT_LABEL_NAMES = ['$0-$10', '$10-$50', '$50-$100', '$100-$500', '$500-$1K', '$1K+']

# ── Risk Tier Thresholds (Streamlit App) ─────────────────────────────────────
RISK_THRESHOLDS = {
    'HIGH': 80.0,
    'MEDIUM': 50.0,
    'LOW': 20.0,
    'SAFE': 0.0
}

RISK_RECOMMENDATIONS = {
    'HIGH': 'BLOCK transaction immediately. Alert fraud investigation team.',
    'MEDIUM': 'Flag for manual review. Hold transaction pending verification.',
    'LOW': 'Monitor closely. Send verification SMS to cardholder.',
    'SAFE': 'Transaction appears legitimate. Approve normally.'
}

# ── Metric Names (for reporting & display) ───────────────────────────────────
PRIMARY_METRICS = ['PR-AUC', 'Recall', 'MCC', 'F1-Score', 'F2-Score']
SECONDARY_METRICS = ['ROC-AUC', 'Precision', 'Balanced Accuracy', 'Cohen Kappa']
ALL_METRICS = PRIMARY_METRICS + SECONDARY_METRICS + ['Accuracy']

# ── Model Display Names ─────────────────────────────────────────────────────
MODEL_DISPLAY_NAMES = {
    'logistic_regression': 'Logistic Regression',
    'decision_tree': 'Decision Tree',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM',
    'catboost': 'CatBoost',
    'hist_gradient_boosting': 'HistGradientBoosting',
    'balanced_random_forest': 'BalancedRandomForest',
    'easy_ensemble': 'EasyEnsemble'
}

# ── Hours for Night Classification ───────────────────────────────────────────
NIGHT_START_HOUR = 22  # 10 PM
NIGHT_END_HOUR = 5     # 5 AM

# ── Seconds in a Day/Week ───────────────────────────────────────────────────
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400
HOURS_PER_WEEK = 168
