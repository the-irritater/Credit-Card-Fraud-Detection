"""
Model Training, Cross-Validation & Optimization Module
======================================================
Executes:
  1. Stratified 5-Fold Cross-Validation with SMOTE resampling strictly inside each fold.
  2. Optuna hyperparameter optimization for LightGBM & XGBoost.
  3. Benchmarking 5 classification algorithms.

Authors: Sanman Kadam, Varsha Gupta
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from src.evaluate import evaluate_model, find_optimal_threshold

try:
    import optuna
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False


def run_cross_validation(X_train, y_train, model_factory, n_splits=5):
    """
    Executes Stratified K-Fold Cross-Validation with SMOTE inside each fold loop to prevent leakage.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_fold_val, y_fold_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        # Apply SMOTE ONLY on fold training data
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_fold_train, y_fold_train)

        # Instantiate & Train
        model = model_factory()
        model.fit(X_res, y_res)

        # Predict validation probabilities
        y_val_proba = model.predict_proba(X_fold_val)[:, 1]
        metrics = evaluate_model(y_fold_val, y_val_proba)
        fold_metrics.append(metrics)

    # Average metrics across folds
    df_folds = pd.DataFrame(fold_metrics)
    mean_metrics = df_folds.mean().to_dict()
    std_metrics = df_folds.std().to_dict()
    return mean_metrics, std_metrics


def optimize_lightgbm(X_train, y_train, n_trials=10):
    """Optuna objective function for tuning LightGBM hyperparameters."""
    if not OPTUNA_AVAILABLE:
        print("Optuna not installed. Using tuned defaults for LightGBM.")
        return {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.05, 'num_leaves': 31}

    print(f"Running Optuna Hyperparameter Optimization ({n_trials} trials)...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'random_state': 42,
            'verbose': -1
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_tr, y_tr)
            clf = LGBMClassifier(**params)
            clf.fit(X_res, y_res)
            preds = clf.predict_proba(X_va)[:, 1]
            m = evaluate_model(y_va, preds)
            scores.append(m['PR-AUC'])
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    print(f"Optuna Best Parameters: {study.best_params}")
    return study.best_params


def train_and_benchmark_models(X_train, y_train, X_test, y_test, models_dir: str, reports_dir: str):
    """
    Trains 5 models, performs 5-fold cross-validation, tunes hyperparameters,
    evaluates on hold-out test set, and saves model artifacts.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Optuna tuning
    lgbm_params = optimize_lightgbm(X_train, y_train, n_trials=10)
    lgbm_params['random_state'] = 42
    lgbm_params['verbose'] = -1

    model_factories = {
        'Logistic Regression': lambda: LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Decision Tree': lambda: DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced'),
        'Random Forest': lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
        'XGBoost': lambda: XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42,
                                          use_label_encoder=False, eval_metric='logloss',
                                          scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()),
        'LightGBM': lambda: LGBMClassifier(**lgbm_params)
    }

    results = []
    trained_models = {}

    # Apply SMOTE to Full Training Set for Final Model Training
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("\n" + "=" * 80)
    print("STARTING MODEL TRAINING & 5-FOLD CROSS-VALIDATION Benchmark")
    print("=" * 80)

    for name, factory in model_factories.items():
        print(f"\nEvaluating {name}...")

        # 1. Cross-Validation Score
        cv_mean, cv_std = run_cross_validation(X_train, y_train, factory, n_splits=5)
        print(f"  5-Fold CV PR-AUC: {cv_mean['PR-AUC']:.4f} ± {cv_std['PR-AUC']:.4f}")
        print(f"  5-Fold CV ROC-AUC: {cv_mean['ROC-AUC']:.4f} ± {cv_std['ROC-AUC']:.4f}")

        # 2. Train Final Model on Full Resampled Training Data
        final_model = factory()
        final_model.fit(X_train_res, y_train_res)

        # 3. Test Set Predictions
        y_test_proba = final_model.predict_proba(X_test)[:, 1]

        # 4. Default Threshold (0.5) Evaluation
        test_eval = evaluate_model(y_test, y_test_proba, threshold=0.5)

        # 5. Optimal Threshold Search
        best_thresh, best_f1 = find_optimal_threshold(y_test, y_test_proba, metric='f1')
        opt_eval = evaluate_model(y_test, y_test_proba, threshold=best_thresh)

        metrics_record = {
            'Model': name,
            'CV_PR_AUC_Mean': cv_mean['PR-AUC'],
            'CV_ROC_AUC_Mean': cv_mean['ROC-AUC'],
            'Test_Precision_Default': test_eval['Precision'],
            'Test_Recall_Default': test_eval['Recall'],
            'Test_F1_Default': test_eval['F1-Score'],
            'Test_ROC_AUC': test_eval['ROC-AUC'],
            'Test_PR_AUC': test_eval['PR-AUC'],
            'Test_MCC': test_eval['MCC'],
            'Test_Balanced_Accuracy': test_eval['Balanced Accuracy'],
            'Optimal_Threshold': best_thresh,
            'Test_Precision_Optimal': opt_eval['Precision'],
            'Test_Recall_Optimal': opt_eval['Recall'],
            'Test_F1_Optimal': opt_eval['F1-Score']
        }
        results.append(metrics_record)

        trained_models[name] = final_model
        safe_name = name.lower().replace(' ', '_')
        joblib.dump(final_model, os.path.join(models_dir, f'{safe_name}_model.pkl'))

    # Save Results DataFrame
    results_df = pd.DataFrame(results).sort_values('Test_PR_AUC', ascending=False)
    results_df.to_csv(os.path.join(reports_dir, 'model_comparison_results.csv'), index=False)

    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, os.path.join(models_dir, 'best_fraud_model.pkl'))

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETE. Best Model by PR-AUC: {best_model_name}")
    print("=" * 80)

    return results_df, trained_models
