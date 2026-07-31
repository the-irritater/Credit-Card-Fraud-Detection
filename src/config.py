"""
Centralized Configuration Module
=================================
Single source of truth for all pipeline hyperparameters, paths, and settings.
Eliminates hard-coded values throughout the codebase.

Authors: Varsha Gupta, Sanman Kadam
"""

import os

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Data Splitting ───────────────────────────────────────────────────────────
TEST_SIZE = 0.2

# ── Cross-Validation ─────────────────────────────────────────────────────────
CV_FOLDS = 5
CV_REPEATS = 3  # RepeatedStratifiedKFold: 5 folds × 3 repeats = 15 evaluations

# ── SMOTE ────────────────────────────────────────────────────────────────────
SMOTE_RANDOM_STATE = 42

# ── Optuna ───────────────────────────────────────────────────────────────────
OPTUNA_TRIALS = 10
OPTUNA_CV_FOLDS = 3

# ── Threshold Optimization ───────────────────────────────────────────────────
THRESHOLD_MIN = 0.01
THRESHOLD_MAX = 0.99
THRESHOLD_STEPS = 99

# ── Cost-Sensitive Optimization ──────────────────────────────────────────────
# In fraud detection, missing a fraud (FN) is typically 10× more costly
# than a false alarm (FP). Adjust for domain-specific cost structure.
COST_FN = 10.0   # Cost of missing a fraud (False Negative)
COST_FP = 1.0    # Cost of a false alarm (False Positive)

# ── Probability Calibration ──────────────────────────────────────────────────
CALIBRATION_METHOD = 'isotonic'   # 'sigmoid' (Platt) or 'isotonic'
CALIBRATION_CV_FOLDS = 5

# ── Isolation Forest ─────────────────────────────────────────────────────────
ISOLATION_FOREST_CONTAMINATION = 'auto'
ISOLATION_FOREST_N_ESTIMATORS = 100

# ── Model Training ───────────────────────────────────────────────────────────
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = None
DT_MAX_DEPTH = 10
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, 'creditcard.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# ── SHAP ─────────────────────────────────────────────────────────────────────
SHAP_SAMPLE_SIZE = 500

# ── Visualization ────────────────────────────────────────────────────────────
PLOT_DPI = 150
PLOT_STYLE = 'whitegrid'
TOP_N_FEATURES = 15
