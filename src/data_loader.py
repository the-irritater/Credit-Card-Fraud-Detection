"""
Data Loading Module
===================
Handles loading, inspecting, deduplication, and initial missing value checks.
Authors: Sanman Kadam, Varsha Gupta
"""

import pandas as pd
import numpy as np
import os


def load_dataset(data_path: str) -> pd.DataFrame:
    """
    Loads credit card transaction dataset and removes duplicate records.

    Parameters:
        data_path: Path to the raw creditcard.csv file.

    Returns:
        Cleaned Pandas DataFrame without duplicates.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"Loading raw dataset from {data_path}...")
    df = pd.read_csv(data_path)
    initial_rows = len(df)

    # Deduplication
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"Removed {duplicates:,} duplicate rows ({initial_rows:,} -> {len(df):,})")

    # Verify no missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"Warning: Found {missing} missing values. Imputing with median...")
        df.fillna(df.median(), inplace=True)
    else:
        print("Verified zero missing values in dataset.")

    fraud_count = (df['Class'] == 1).sum()
    genuine_count = (df['Class'] == 0).sum()
    print(f"Genuine: {genuine_count:,} ({genuine_count/len(df)*100:.3f}%) | "
          f"Fraud: {fraud_count:,} ({fraud_count/len(df)*100:.3f}%)")

    return df
