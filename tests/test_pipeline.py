"""
Unit Tests for Credit Card Fraud Detection Pipeline
=====================================================
Tests core functions without requiring the full dataset.

Run: python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest


# ── Config Tests ─────────────────────────────────────────────────────────────

class TestConfig:
    def test_random_state_is_integer(self):
        from src.config import RANDOM_STATE
        assert isinstance(RANDOM_STATE, int)
        assert RANDOM_STATE >= 0

    def test_cv_folds_valid(self):
        from src.config import CV_FOLDS, CV_REPEATS
        assert CV_FOLDS >= 2
        assert CV_REPEATS >= 1

    def test_test_size_valid(self):
        from src.config import TEST_SIZE
        assert 0 < TEST_SIZE < 1

    def test_cost_ratio_valid(self):
        from src.config import COST_FN, COST_FP
        assert COST_FN > 0
        assert COST_FP > 0
        assert COST_FN > COST_FP  # FN should cost more than FP in fraud

    def test_paths_defined(self):
        from src.config import MODELS_DIR, REPORTS_DIR, IMAGES_DIR
        assert MODELS_DIR is not None
        assert REPORTS_DIR is not None
        assert IMAGES_DIR is not None


# ── Constants Tests ──────────────────────────────────────────────────────────

class TestConstants:
    def test_amount_bins_and_labels_match(self):
        from src.constants import AMOUNT_BINS, AMOUNT_LABELS
        assert len(AMOUNT_LABELS) == len(AMOUNT_BINS) - 1

    def test_risk_thresholds_ordered(self):
        from src.constants import RISK_THRESHOLDS
        assert RISK_THRESHOLDS['HIGH'] > RISK_THRESHOLDS['MEDIUM']
        assert RISK_THRESHOLDS['MEDIUM'] > RISK_THRESHOLDS['LOW']
        assert RISK_THRESHOLDS['LOW'] > RISK_THRESHOLDS['SAFE']

    def test_model_display_names(self):
        from src.constants import MODEL_DISPLAY_NAMES
        assert 'random_forest' in MODEL_DISPLAY_NAMES
        assert 'xgboost' in MODEL_DISPLAY_NAMES


# ── Evaluate Tests ───────────────────────────────────────────────────────────

class TestEvaluate:
    def setup_method(self):
        """Create synthetic test data."""
        np.random.seed(42)
        self.y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        self.y_proba = np.array([0.1, 0.2, 0.15, 0.3, 0.05, 0.1, 0.25, 0.85, 0.9, 0.7])

    def test_evaluate_model_returns_all_keys(self):
        from src.evaluate import evaluate_model
        result = evaluate_model(self.y_true, self.y_proba, threshold=0.5)

        expected_keys = [
            'Accuracy', 'Precision', 'Recall', 'F1-Score', 'F2-Score',
            'ROC-AUC', 'PR-AUC', 'MCC', 'Balanced Accuracy', 'Cohen Kappa',
            'Threshold', 'TN', 'FP', 'FN', 'TP'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_evaluate_model_f2_score_present(self):
        from src.evaluate import evaluate_model
        result = evaluate_model(self.y_true, self.y_proba)
        assert 'F2-Score' in result
        assert 0 <= result['F2-Score'] <= 1

    def test_evaluate_model_metrics_range(self):
        from src.evaluate import evaluate_model
        result = evaluate_model(self.y_true, self.y_proba)
        for key in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']:
            assert 0 <= result[key] <= 1, f"{key} out of range: {result[key]}"

    def test_confusion_matrix_sums(self):
        from src.evaluate import evaluate_model
        result = evaluate_model(self.y_true, self.y_proba)
        total = result['TN'] + result['FP'] + result['FN'] + result['TP']
        assert total == len(self.y_true)

    def test_find_optimal_threshold_returns_valid(self):
        from src.evaluate import find_optimal_threshold
        thresh, score = find_optimal_threshold(self.y_true, self.y_proba, metric='f1')
        assert 0 < thresh < 1
        assert score >= 0

    def test_find_optimal_threshold_f2(self):
        from src.evaluate import find_optimal_threshold
        thresh, score = find_optimal_threshold(self.y_true, self.y_proba, metric='f2')
        assert 0 < thresh < 1

    def test_find_optimal_threshold_cost(self):
        from src.evaluate import find_optimal_threshold
        thresh, cost = find_optimal_threshold(self.y_true, self.y_proba, metric='cost')
        assert 0 < thresh < 1
        assert cost >= 0  # Cost should be non-negative

    def test_compute_threshold_curve(self):
        from src.evaluate import compute_threshold_curve
        df = compute_threshold_curve(self.y_true, self.y_proba)
        assert 'Threshold' in df.columns
        assert 'Precision' in df.columns
        assert 'Recall' in df.columns
        assert 'F1-Score' in df.columns
        assert 'F2-Score' in df.columns
        assert len(df) > 0

    def test_compute_confidence_intervals(self):
        from src.evaluate import compute_confidence_intervals
        fold_data = pd.DataFrame({
            'PR-AUC': [0.85, 0.87, 0.83, 0.86, 0.84],
            'ROC-AUC': [0.97, 0.98, 0.96, 0.97, 0.97]
        })
        result = compute_confidence_intervals(fold_data)
        assert 'PR-AUC_mean' in result
        assert 'PR-AUC_std' in result
        assert 'PR-AUC_ci' in result
        assert '±' in result['PR-AUC_ci']


# ── Feature Engineering Tests ────────────────────────────────────────────────

class TestFeatureEngineering:
    def setup_method(self):
        """Create minimal synthetic dataset mimicking creditcard.csv structure."""
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame({
            'Time': np.random.uniform(0, 172800, n),
            'Amount': np.random.exponential(100, n),
            'Class': np.random.choice([0, 1], n, p=[0.95, 0.05]),
            **{f'V{i}': np.random.randn(n) for i in range(1, 29)}
        })

    def test_engineer_features_adds_expected_columns(self):
        from src.feature_engineering import engineer_features
        result = engineer_features(self.df)

        expected_new = ['Hour', 'Is_Night', 'Hour_Of_Week', 'Is_Weekend',
                       'Amount_Log', 'Amount_Zscore', 'Amount_Category',
                       'V1_V2_Interaction', 'V14_Amount']
        for col in expected_new:
            assert col in result.columns, f"Missing column: {col}"

    def test_hour_range(self):
        from src.feature_engineering import engineer_features
        result = engineer_features(self.df)
        assert result['Hour'].min() >= 0
        assert result['Hour'].max() < 24

    def test_is_night_binary(self):
        from src.feature_engineering import engineer_features
        result = engineer_features(self.df)
        assert set(result['Is_Night'].unique()).issubset({0, 1})

    def test_is_weekend_binary(self):
        from src.feature_engineering import engineer_features
        result = engineer_features(self.df)
        assert set(result['Is_Weekend'].unique()).issubset({0, 1})

    def test_amount_zscore_centered(self):
        from src.feature_engineering import engineer_features
        result = engineer_features(self.df)
        # Z-score should have mean ≈ 0
        assert abs(result['Amount_Zscore'].mean()) < 0.01

    def test_preserves_original_columns(self):
        from src.feature_engineering import engineer_features
        result = engineer_features(self.df)
        assert 'Class' in result.columns
        assert 'Amount' in result.columns
        assert 'Time' in result.columns
