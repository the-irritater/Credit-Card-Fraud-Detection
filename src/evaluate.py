"""
Model Evaluation, Threshold Optimization & Calibration Module
==============================================================
Calculates comprehensive statistical metrics for imbalanced fraud classification:
  - ROC-AUC & PR-AUC (Average Precision Score)
  - Precision, Recall, F1-Score, F2-Score (F-beta with β=2)
  - Matthews Correlation Coefficient (MCC)
  - Balanced Accuracy & Cohen's Kappa
  - Cost-Sensitive Threshold Optimization
  - Confidence Interval Computation from Cross-Validation Folds

Authors: Sanman Kadam, Varsha Gupta
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score, confusion_matrix
)

from src.config import (
    THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS,
    COST_FN, COST_FP
)


def evaluate_model(y_true, y_pred_proba, threshold: float = 0.5):
    """
    Evaluates predictions at a specified probability threshold.

    Parameters:
        y_true: True binary target labels.
        y_pred_proba: Predicted probabilities for the positive class (fraud).
        threshold: Probability threshold for positive classification.

    Returns:
        Dictionary of comprehensive evaluation metrics including F2-Score.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    mcc = matthews_corrcoef(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'F2-Score': f2,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'MCC': mcc,
        'Balanced Accuracy': bal_acc,
        'Cohen Kappa': kappa,
        'Threshold': threshold,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }


def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """
    Finds the classification threshold that maximizes the specified metric.

    Supported metrics:
        - 'f1': F1-Score (harmonic mean of precision & recall)
        - 'f2': F2-Score (β=2, emphasizes recall for fraud detection)
        - 'mcc': Matthews Correlation Coefficient
        - 'recall': Recall (sensitivity)
        - 'cost': Cost-sensitive optimization using FN/FP cost ratio

    Parameters:
        y_true: True binary target labels.
        y_pred_proba: Predicted probabilities for the positive class.
        metric: Target metric to maximize.

    Returns:
        Tuple of (best_threshold, best_metric_score).
    """
    thresholds = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS)
    best_thresh = 0.5
    best_score = -np.inf

    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'f2':
            score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        elif metric == 'mcc':
            score = matthews_corrcoef(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'cost':
            # Cost-sensitive: minimize total cost → maximize negative cost
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            total_cost = (fn * COST_FN) + (fp * COST_FP)
            score = -total_cost  # Negate so maximization works
        else:
            score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_thresh = t

    # For cost metric, return the actual cost (positive) for display
    if metric == 'cost':
        best_score = -best_score

    return best_thresh, best_score


def compute_threshold_curve(y_true, y_pred_proba):
    """
    Computes Precision, Recall, F1, and F2 across all thresholds.
    Used for threshold optimization visualization.

    Returns:
        DataFrame with columns: Threshold, Precision, Recall, F1, F2
    """
    thresholds = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEPS)
    records = []
    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        records.append({
            'Threshold': t,
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, zero_division=0),
            'F2-Score': fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        })
    return pd.DataFrame(records)


def compute_confidence_intervals(fold_metrics_df):
    """
    Computes mean ± standard deviation from cross-validation fold metrics.

    Parameters:
        fold_metrics_df: DataFrame where each row is one fold's metrics.

    Returns:
        Dictionary with keys: {metric}_mean, {metric}_std, {metric}_ci
        where _ci is a formatted string like '0.991 ± 0.004'.
    """
    result = {}
    for col in fold_metrics_df.columns:
        if col in ('Threshold', 'TN', 'FP', 'FN', 'TP'):
            continue
        mean_val = fold_metrics_df[col].mean()
        std_val = fold_metrics_df[col].std()
        result[f'{col}_mean'] = mean_val
        result[f'{col}_std'] = std_val
        result[f'{col}_ci'] = f'{mean_val:.4f} ± {std_val:.4f}'
    return result
