"""
Model Evaluation & Threshold Optimization Module
=================================================
Calculates statistical metrics for imbalanced fraud classification:
  - ROC-AUC
  - PR-AUC (Average Precision Score)
  - Precision, Recall, F1-Score
  - Matthews Correlation Coefficient (MCC)
  - Balanced Accuracy & Cohen's Kappa
  - Threshold Optimization for Optimal F1-Score

Authors: Sanman Kadam, Varsha Gupta
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score, confusion_matrix
)


def evaluate_model(y_true, y_pred_proba, threshold: float = 0.5):
    """
    Evaluates predictions at a specified probability threshold.

    Parameters:
        y_true: True binary target labels.
        y_pred_proba: Predicted probabilities for the positive class (fraud).
        threshold: Probability threshold for positive classification.

    Returns:
        Dictionary of comprehensive evaluation metrics.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
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
    Finds the classification threshold that maximizes the specified metric (default F1).

    Parameters:
        y_true: True binary target labels.
        y_pred_proba: Predicted probabilities for the positive class.
        metric: Target metric to maximize ('f1', 'mcc', or 'recall').

    Returns:
        Best threshold float and best metric score float.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_score = -1.0

    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'mcc':
            score = matthews_corrcoef(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_thresh = t

    return best_thresh, best_score
