"""
Preprocessing & Data Splitting Module
====================================
Prevents statistical data leakage by:
  1. Splitting data into Train and Test partitions FIRST.
  2. Fitting separate RobustScalers for Amount and Time EXCLUSIVELY on X_train.
  3. Transforming X_test using the scalers fitted on X_train.
  4. Fitting Isolation Forest anomaly detector on X_train and scoring both splits.

Authors: Sanman Kadam, Varsha Gupta
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
import joblib
import os

from src.config import (
    TEST_SIZE, RANDOM_STATE,
    ISOLATION_FOREST_CONTAMINATION, ISOLATION_FOREST_N_ESTIMATORS,
    MODELS_DIR
)
from src.logging_config import get_logger

logger = get_logger('preprocessing')


def prepare_data_splits(df: pd.DataFrame, test_size: float = None, random_state: int = None):
    """
    Performs stratified train/test split, applies scaling strictly post-split,
    and computes Isolation Forest anomaly scores.

    Parameters:
        df: DataFrame containing all features and target Class.
        test_size: Proportion of dataset to hold out for testing.
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test, amount_scaler, time_scaler
    """
    test_size = test_size or TEST_SIZE
    random_state = random_state or RANDOM_STATE

    X = df.drop(columns=['Class'])
    y = df['Class']

    # 1. Stratified Train-Test Split BEFORE scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train split: {X_train.shape[0]:,} samples | Test split: {X_test.shape[0]:,} samples")

    # 2. Separate Scalers for Amount and Time
    amount_scaler = RobustScaler()
    time_scaler = RobustScaler()

    # Fit ONLY on X_train to prevent leakage
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train['Scaled_Amount'] = amount_scaler.fit_transform(X_train[['Amount']])
    X_train['Scaled_Time'] = time_scaler.fit_transform(X_train[['Time']])

    # Transform X_test using X_train parameters
    X_test['Scaled_Amount'] = amount_scaler.transform(X_test[['Amount']])
    X_test['Scaled_Time'] = time_scaler.transform(X_test[['Time']])

    # 3. Isolation Forest Anomaly Scores (fitted on X_train only)
    logger.info("Fitting Isolation Forest for anomaly scoring (on X_train only)...")
    # Use PCA features (V1-V28) + Amount for anomaly detection
    iso_features = [c for c in X_train.columns if c.startswith('V') and c[1:].isdigit()]
    iso_features.append('Scaled_Amount')

    iso_forest = IsolationForest(
        n_estimators=ISOLATION_FOREST_N_ESTIMATORS,
        contamination=ISOLATION_FOREST_CONTAMINATION,
        random_state=random_state,
        n_jobs=-1
    )
    iso_forest.fit(X_train[iso_features])

    # Score both splits (decision_function: lower = more anomalous)
    X_train['Isolation_Score'] = iso_forest.decision_function(X_train[iso_features])
    X_test['Isolation_Score'] = iso_forest.decision_function(X_test[iso_features])
    logger.info("Isolation Forest anomaly scores computed for both splits.")

    # Drop raw Amount and Time for modeling
    X_train.drop(columns=['Amount', 'Time'], inplace=True)
    X_test.drop(columns=['Amount', 'Time'], inplace=True)

    return X_train, X_test, y_train, y_test, amount_scaler, time_scaler, iso_forest


def save_scalers_and_features(amount_scaler, time_scaler, feature_names, models_dir: str,
                               iso_forest=None):
    """Saves fitted scalers, Isolation Forest, and feature list for production inference."""
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(amount_scaler, os.path.join(models_dir, 'amount_scaler.pkl'))
    joblib.dump(time_scaler, os.path.join(models_dir, 'time_scaler.pkl'))
    joblib.dump(feature_names, os.path.join(models_dir, 'feature_names.pkl'))

    if iso_forest is not None:
        joblib.dump(iso_forest, os.path.join(models_dir, 'isolation_forest.pkl'))
        logger.info(f"Saved isolation_forest.pkl to {models_dir}")

    logger.info(f"Saved amount_scaler.pkl, time_scaler.pkl, and feature_names.pkl to {models_dir}")
