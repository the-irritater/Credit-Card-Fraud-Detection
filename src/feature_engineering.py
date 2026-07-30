"""
Feature Engineering Module
==========================
Constructs deterministic domain-specific features:
  - Temporal features (Hour, Is_Night)
  - Log-transformed transaction amounts
  - Amount categorization buckets & Z-scores
  - PCA interaction terms

Authors: Sanman Kadam, Varsha Gupta
"""

import pandas as pd
import numpy as np


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
    df_feat['Hour'] = (df_feat['Time'] / 3600) % 24
    df_feat['Is_Night'] = ((df_feat['Hour'] >= 22) | (df_feat['Hour'] <= 5)).astype(int)

    # 2. Amount Log-Transformation
    df_feat['Amount_Log'] = np.log1p(df_feat['Amount'])

    # 3. Amount Ordinal Categories
    bins = [-np.inf, 10, 50, 100, 500, 1000, np.inf]
    labels = [0, 1, 2, 3, 4, 5]
    df_feat['Amount_Category'] = pd.cut(df_feat['Amount'], bins=bins, labels=labels).astype(int)

    # 4. Interaction Features (Using raw V1, V2, V14)
    df_feat['V1_V2_Interaction'] = df_feat['V1'] * df_feat['V2']
    df_feat['V14_Amount'] = df_feat['V14'] * np.log1p(df_feat['Amount'])

    print(f"Feature engineering complete. Total features: {df_feat.shape[1] - 1} (excluding target Class)")
    return df_feat
