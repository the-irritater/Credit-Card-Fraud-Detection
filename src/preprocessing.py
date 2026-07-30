"""
Preprocessing & Data Splitting Module
====================================
Prevents statistical data leakage by:
  1. Splitting data into Train and Test partitions FIRST.
  2. Fitting separate RobustScalers for Amount and Time EXCLUSIVELY on X_train.
  3. Transforming X_test using the scalers fitted on X_train.

Authors: Sanman Kadam, Varsha Gupta
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib
import os


def prepare_data_splits(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Performs stratified train/test split and applies scaling strictly post-split.

    Parameters:
        df: DataFrame containing all features and target Class.
        test_size: Proportion of dataset to hold out for testing.
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test, amount_scaler, time_scaler
    """
    X = df.drop(columns=['Class'])
    y = df['Class']

    # 1. Stratified Train-Test Split BEFORE scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train split: {X_train.shape[0]:,} samples | Test split: {X_test.shape[0]:,} samples")

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

    # Drop raw Amount and Time for modeling
    X_train.drop(columns=['Amount', 'Time'], inplace=True)
    X_test.drop(columns=['Amount', 'Time'], inplace=True)

    return X_train, X_test, y_train, y_test, amount_scaler, time_scaler


def save_scalers_and_features(amount_scaler, time_scaler, feature_names, models_dir: str):
    """Saves fitted scalers and feature list for production inference."""
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(amount_scaler, os.path.join(models_dir, 'amount_scaler.pkl'))
    joblib.dump(time_scaler, os.path.join(models_dir, 'time_scaler.pkl'))
    joblib.dump(feature_names, os.path.join(models_dir, 'feature_names.pkl'))
    print(f"Saved amount_scaler.pkl, time_scaler.pkl, and feature_names.pkl to {models_dir}")
