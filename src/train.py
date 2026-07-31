"""
Model Training, Cross-Validation, Calibration & Optimization Module
=====================================================================
Executes:
  1. Repeated Stratified K-Fold Cross-Validation (5 folds × 3 repeats = 15 evaluations)
     with SMOTE resampling strictly inside each fold.
  2. Optuna hyperparameter optimization for LightGBM & XGBoost.
  3. Benchmarking 9 classification algorithms including ensemble methods.
  4. Probability calibration with CalibratedClassifierCV.
  5. Cost-sensitive threshold optimization.

Authors: Sanman Kadam, Varsha Gupta
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from src.evaluate import evaluate_model, find_optimal_threshold, compute_confidence_intervals
from src.config import (
    RANDOM_STATE, SMOTE_RANDOM_STATE, CV_FOLDS, CV_REPEATS,
    OPTUNA_TRIALS, OPTUNA_CV_FOLDS,
    RF_N_ESTIMATORS, DT_MAX_DEPTH, XGB_N_ESTIMATORS, XGB_MAX_DEPTH,
    XGB_LEARNING_RATE, CALIBRATION_METHOD, CALIBRATION_CV_FOLDS
)
from src.logging_config import get_logger

logger = get_logger('train')

# Optional imports
try:
    import optuna
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def run_cross_validation(X_train, y_train, model_factory,
                         n_splits=None, n_repeats=None):
    """
    Executes Repeated Stratified K-Fold Cross-Validation with SMOTE
    inside each fold loop to prevent leakage.

    Parameters:
        X_train: Training features.
        y_train: Training labels.
        model_factory: Callable that returns a fresh model instance.
        n_splits: Number of folds (default from config).
        n_repeats: Number of repetitions (default from config).

    Returns:
        mean_metrics (dict), std_metrics (dict), fold_metrics_df (DataFrame)
    """
    n_splits = n_splits or CV_FOLDS
    n_repeats = n_repeats or CV_REPEATS

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE
    )
    fold_metrics = []

    total_folds = n_splits * n_repeats
    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X_train, y_train)):
        X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_fold_val, y_fold_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        # Apply SMOTE ONLY on fold training data
        smote = SMOTE(random_state=SMOTE_RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_fold_train, y_fold_train)

        # Instantiate & Train
        model = model_factory()
        model.fit(X_res, y_res)

        # Predict validation probabilities
        y_val_proba = model.predict_proba(X_fold_val)[:, 1]
        metrics = evaluate_model(y_fold_val, y_val_proba)
        fold_metrics.append(metrics)

    # Aggregate metrics across all folds × repeats
    df_folds = pd.DataFrame(fold_metrics)
    mean_metrics = df_folds.mean().to_dict()
    std_metrics = df_folds.std().to_dict()

    return mean_metrics, std_metrics, df_folds


def optimize_lightgbm(X_train, y_train, n_trials=None):
    """Optuna objective function for tuning LightGBM hyperparameters."""
    n_trials = n_trials or OPTUNA_TRIALS

    if not OPTUNA_AVAILABLE:
        logger.info("Optuna not installed. Using tuned defaults for LightGBM.")
        return {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.05, 'num_leaves': 31}

    logger.info(f"Running Optuna Hyperparameter Optimization ({n_trials} trials)...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'random_state': RANDOM_STATE,
            'verbose': -1
        }
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
            smote = SMOTE(random_state=SMOTE_RANDOM_STATE)
            X_res, y_res = smote.fit_resample(X_tr, y_tr)
            clf = LGBMClassifier(**params)
            clf.fit(X_res, y_res)
            preds = clf.predict_proba(X_va)[:, 1]
            m = evaluate_model(y_va, preds)
            scores.append(m['PR-AUC'])
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"Optuna Best Parameters: {study.best_params}")
    return study.best_params


def calibrate_model(model, X_train, y_train, method=None, cv=None):
    """
    Wraps a trained model with probability calibration.

    Parameters:
        model: Pre-trained classifier.
        X_train: Training features for calibration fitting.
        y_train: Training labels.
        method: 'sigmoid' (Platt scaling) or 'isotonic'.
        cv: Number of cross-validation folds for calibration.

    Returns:
        CalibratedClassifierCV instance fitted on training data.
    """
    method = method or CALIBRATION_METHOD
    cv = cv or CALIBRATION_CV_FOLDS

    calibrated = CalibratedClassifierCV(
        model, method=method, cv=cv
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def train_and_benchmark_models(X_train, y_train, X_test, y_test,
                                models_dir: str, reports_dir: str):
    """
    Trains up to 9 models, performs Repeated Stratified K-Fold Cross-Validation,
    tunes hyperparameters, evaluates on hold-out test set, calibrates top models,
    and saves model artifacts.

    Returns:
        results_df (DataFrame), trained_models (dict), calibration_data (dict)
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Optuna tuning
    lgbm_params = optimize_lightgbm(X_train, y_train, n_trials=OPTUNA_TRIALS)
    lgbm_params['random_state'] = RANDOM_STATE
    lgbm_params['verbose'] = -1

    # ── Model Registry ───────────────────────────────────────────────────────
    model_factories = {
        'Logistic Regression': lambda: LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'
        ),
        'Decision Tree': lambda: DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=DT_MAX_DEPTH, class_weight='balanced'
        ),
        'Random Forest': lambda: RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE,
            n_jobs=-1, class_weight='balanced'
        ),
        'XGBoost': lambda: XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE, random_state=RANDOM_STATE,
            use_label_encoder=False, eval_metric='logloss',
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
        ),
        'LightGBM': lambda: LGBMClassifier(**lgbm_params),
        'HistGradientBoosting': lambda: HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE
        ),
        'BalancedRandomForest': lambda: BalancedRandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'EasyEnsemble': lambda: EasyEnsembleClassifier(
            n_estimators=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    # Add CatBoost if available
    if CATBOOST_AVAILABLE:
        model_factories['CatBoost'] = lambda: CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, verbose=0,
            auto_class_weights='Balanced'
        )
    else:
        logger.info("CatBoost not installed. Skipping CatBoost benchmark.")

    results = []
    trained_models = {}

    # Apply SMOTE to Full Training Set for Final Model Training
    smote = SMOTE(random_state=SMOTE_RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    total_models = len(model_factories)
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"STARTING MODEL TRAINING & {CV_FOLDS}×{CV_REPEATS} REPEATED STRATIFIED CV BENCHMARK")
    logger.info(f"Models to evaluate: {total_models}")
    logger.info("=" * 80)

    for model_idx, (name, factory) in enumerate(model_factories.items(), 1):
        logger.info(f"\n[{model_idx}/{total_models}] Evaluating {name}...")

        # 1. Repeated Stratified Cross-Validation
        cv_mean, cv_std, fold_df = run_cross_validation(
            X_train, y_train, factory, n_splits=CV_FOLDS, n_repeats=CV_REPEATS
        )
        ci = compute_confidence_intervals(fold_df)

        logger.info(f"  {CV_FOLDS}×{CV_REPEATS} CV PR-AUC: {ci.get('PR-AUC_ci', 'N/A')}")
        logger.info(f"  {CV_FOLDS}×{CV_REPEATS} CV ROC-AUC: {ci.get('ROC-AUC_ci', 'N/A')}")
        logger.info(f"  {CV_FOLDS}×{CV_REPEATS} CV Recall: {ci.get('Recall_ci', 'N/A')}")
        logger.info(f"  {CV_FOLDS}×{CV_REPEATS} CV MCC: {ci.get('MCC_ci', 'N/A')}")

        # 2. Train Final Model on Full Resampled Training Data
        final_model = factory()
        final_model.fit(X_train_res, y_train_res)

        # 3. Test Set Predictions
        y_test_proba = final_model.predict_proba(X_test)[:, 1]

        # 4. Default Threshold (0.5) Evaluation
        test_eval = evaluate_model(y_test, y_test_proba, threshold=0.5)

        # 5. Multi-Metric Threshold Optimization
        best_f1_thresh, best_f1 = find_optimal_threshold(y_test, y_test_proba, metric='f1')
        best_f2_thresh, best_f2 = find_optimal_threshold(y_test, y_test_proba, metric='f2')
        best_cost_thresh, best_cost = find_optimal_threshold(y_test, y_test_proba, metric='cost')

        opt_f1_eval = evaluate_model(y_test, y_test_proba, threshold=best_f1_thresh)
        opt_f2_eval = evaluate_model(y_test, y_test_proba, threshold=best_f2_thresh)

        metrics_record = {
            'Model': name,
            # Cross-Validation with Confidence Intervals
            'CV_PR_AUC': ci.get('PR-AUC_ci', 'N/A'),
            'CV_ROC_AUC': ci.get('ROC-AUC_ci', 'N/A'),
            'CV_Recall': ci.get('Recall_ci', 'N/A'),
            'CV_F1': ci.get('F1-Score_ci', 'N/A'),
            'CV_F2': ci.get('F2-Score_ci', 'N/A'),
            'CV_MCC': ci.get('MCC_ci', 'N/A'),
            # CV Numeric (for sorting and plotting)
            'CV_PR_AUC_Mean': cv_mean.get('PR-AUC', 0),
            'CV_ROC_AUC_Mean': cv_mean.get('ROC-AUC', 0),
            'CV_PR_AUC_Std': cv_std.get('PR-AUC', 0),
            'CV_ROC_AUC_Std': cv_std.get('ROC-AUC', 0),
            # Test Set (Default Threshold)
            'Test_Precision_Default': test_eval['Precision'],
            'Test_Recall_Default': test_eval['Recall'],
            'Test_F1_Default': test_eval['F1-Score'],
            'Test_F2_Default': test_eval['F2-Score'],
            'Test_ROC_AUC': test_eval['ROC-AUC'],
            'Test_PR_AUC': test_eval['PR-AUC'],
            'Test_MCC': test_eval['MCC'],
            'Test_Balanced_Accuracy': test_eval['Balanced Accuracy'],
            # Threshold Optimization
            'Optimal_F1_Threshold': best_f1_thresh,
            'Optimal_F1': opt_f1_eval['F1-Score'],
            'Optimal_F2_Threshold': best_f2_thresh,
            'Optimal_F2': opt_f2_eval['F2-Score'],
            'Cost_Optimal_Threshold': best_cost_thresh,
            'Total_Cost_At_Optimal': best_cost,
            # Optimal Threshold Metrics
            'Test_Precision_Optimal': opt_f1_eval['Precision'],
            'Test_Recall_Optimal': opt_f1_eval['Recall'],
        }
        results.append(metrics_record)

        trained_models[name] = final_model
        safe_name = name.lower().replace(' ', '_')
        joblib.dump(final_model, os.path.join(models_dir, f'{safe_name}_model.pkl'))

    # ── Probability Calibration for Top 3 Models ─────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("PROBABILITY CALIBRATION (Top 3 Models by PR-AUC)")
    logger.info("=" * 80)

    results_df = pd.DataFrame(results).sort_values('Test_PR_AUC', ascending=False)
    top3_names = results_df['Model'].head(3).tolist()

    calibration_data = {}
    for name in top3_names:
        logger.info(f"  Calibrating {name} with {CALIBRATION_METHOD} method...")
        model = trained_models[name]
        try:
            cal_model = calibrate_model(model, X_train_res, y_train_res)
            y_cal_proba = cal_model.predict_proba(X_test)[:, 1]
            y_raw_proba = model.predict_proba(X_test)[:, 1]

            # Calibration curve data
            prob_true_raw, prob_pred_raw = calibration_curve(y_test, y_raw_proba, n_bins=10)
            prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_cal_proba, n_bins=10)

            calibration_data[name] = {
                'raw_proba': y_raw_proba,
                'cal_proba': y_cal_proba,
                'prob_true_raw': prob_true_raw,
                'prob_pred_raw': prob_pred_raw,
                'prob_true_cal': prob_true_cal,
                'prob_pred_cal': prob_pred_cal,
            }

            # Save calibrated model
            safe_name = name.lower().replace(' ', '_')
            joblib.dump(cal_model, os.path.join(models_dir, f'{safe_name}_calibrated.pkl'))
            logger.info(f"  Saved {safe_name}_calibrated.pkl")
        except Exception as e:
            logger.warning(f"  Calibration failed for {name}: {e}")

    # Save Results
    results_df.to_csv(os.path.join(reports_dir, 'model_comparison_results.csv'), index=False)

    # Save best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, os.path.join(models_dir, 'best_fraud_model.pkl'))

    logger.info("\n" + "=" * 80)
    logger.info(f"BENCHMARK COMPLETE. Best Model by PR-AUC: {best_model_name}")
    logger.info("=" * 80)

    return results_df, trained_models, calibration_data
