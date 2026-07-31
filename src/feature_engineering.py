"""
Feature Engineering Module
==========================
Constructs deterministic domain-specific features:
  - Temporal features (Hour, Is_Night, Hour_Of_Week, Is_Weekend)
  - Log-transformed transaction amounts
  - Amount categorization buckets & Z-scores
  - PCA interaction terms

Note: Leakage-sensitive features (Isolation Forest scores) are computed
in preprocessing.py AFTER train/test split.

Authors: Sanman Kadam, Varsha Gupta
"""

import pandas as pd
import numpy as np

from src.constants import (
    AMOUNT_BINS, AMOUNT_LABELS,
    NIGHT_START_HOUR, NIGHT_END_HOUR,
    SECONDS_PER_HOUR, SECONDS_PER_DAY, HOURS_PER_WEEK
)
from src.logging_config import get_logger

logger = get_logger('feature_engineering')


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies deterministic feature transformations to the dataset.
    Note: Fitting scalers is deferred to preprocessing AFTER train/test split.

    Parameters:
        df: Cleaned credit card DataFrame.

    Returns:
        DataFrame enriched with engineered features.
    """
    df_feat = df.copy()

    # 1. Temporal Features
    df_feat['Hour'] = (df_feat['Time'] / SECONDS_PER_HOUR) % 24
    df_feat['Is_Night'] = (
        (df_feat['Hour'] >= NIGHT_START_HOUR) | (df_feat['Hour'] <= NIGHT_END_HOUR)
    ).astype(int)

    # 2. Hour of Week — maps transaction time to a 168-hour weekly cycle.
    # Since the dataset spans ~48 hours, this captures daily periodicity.
    df_feat['Hour_Of_Week'] = (df_feat['Time'] / SECONDS_PER_HOUR) % HOURS_PER_WEEK

    # 3. Weekend Indicator — Heuristic: the dataset starts at Time=0 (unknown day).
    # Transactions beyond 24h are on "day 2". This is an approximation since
    # the actual day of week is anonymized, but captures behavioral shift patterns.
    df_feat['Is_Weekend'] = (df_feat['Time'] >= SECONDS_PER_DAY).astype(int)

    # 4. Amount Log-Transformation
    df_feat['Amount_Log'] = np.log1p(df_feat['Amount'])

    # 5. Amount Z-Score (population-level, pre-split)
    # Note: This uses global statistics intentionally for feature creation.
    # Post-split scaling (RobustScaler) handles leakage-free normalization.
    amount_mean = df_feat['Amount'].mean()
    amount_std = df_feat['Amount'].std()
    df_feat['Amount_Zscore'] = (df_feat['Amount'] - amount_mean) / (amount_std + 1e-8)

    # 6. Amount Ordinal Categories
    df_feat['Amount_Category'] = pd.cut(
        df_feat['Amount'], bins=AMOUNT_BINS, labels=AMOUNT_LABELS
    ).astype(int)

    # 7. Interaction Features (Using raw V1, V2, V14)
    df_feat['V1_V2_Interaction'] = df_feat['V1'] * df_feat['V2']
    df_feat['V14_Amount'] = df_feat['V14'] * np.log1p(df_feat['Amount'])

    n_features = df_feat.shape[1] - 1  # Exclude target 'Class'
    logger.info(f"Feature engineering complete. Total features: {n_features} (excluding target Class)")

    return df_feat
