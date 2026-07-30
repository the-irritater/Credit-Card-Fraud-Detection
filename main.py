"""
Credit Card Fraud Detection — Main Pipeline Script
====================================================
Authors: Sanman Kadam, Varsha Gupta

Run the complete pipeline from command line:
    python main.py

This script executes all steps sequentially:
  1. Load & clean data
  2. Feature engineering
  3. Handle class imbalance (SMOTE)
  4. Train & evaluate models
  5. Save best model + artifacts
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score
)
import joblib

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, 'creditcard.csv')
DATA_CLEANED = os.path.join(BASE_DIR, 'data', 'cleaned', 'creditcard_cleaned.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'data', 'cleaned'), exist_ok=True)


def banner(text):
    print(f'\n{"=" * 60}')
    print(f'  {text}')
    print(f'{"=" * 60}')


def load_and_clean(path):
    """Step 1-3: Load, inspect, and clean the dataset."""
    banner('STEP 1 — Loading Dataset')
    df = pd.read_csv(path)
    print(f'  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns')

    # Remove duplicates
    dups = df.duplicated().sum()
    if dups > 0:
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f'  Removed {dups:,} duplicate rows')

    # Missing values
    missing = df.isnull().sum().sum()
    print(f'  Missing values: {missing}')

    # Class distribution
    fraud = (df['Class'] == 1).sum()
    genuine = (df['Class'] == 0).sum()
    print(f'  Genuine: {genuine:,} | Fraud: {fraud:,} ({fraud / len(df) * 100:.3f}%)')

    return df


def scale_features(df):
    """Step 3: Scale Amount and Time using RobustScaler."""
    banner('STEP 2 — Scaling Features')
    scaler = RobustScaler()
    df['Scaled_Amount'] = scaler.fit_transform(df[['Amount']])
    df['Scaled_Time'] = scaler.fit_transform(df[['Time']])

    # Save scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'robust_scaler.pkl'))
    print('  RobustScaler applied to Amount and Time')
    return df, scaler


def engineer_features(df):
    """Step 6: Create additional features."""
    banner('STEP 3 — Feature Engineering')
    df_model = df.drop(['Amount', 'Time'], axis=1)

    df_model['Hour'] = (df['Time'] / 3600) % 24
    df_model['Is_Night'] = ((df_model['Hour'] >= 22) | (df_model['Hour'] <= 5)).astype(int)
    df_model['Amount_Log'] = np.log1p(df['Amount'])

    bins = [0, 10, 50, 100, 500, 1000, float('inf')]
    labels = ['0-10', '10-50', '50-100', '100-500', '500-1000', '1000+']
    df_model['Amount_Category'] = pd.cut(df['Amount'], bins=bins, labels=labels).cat.codes

    df_model['V1_V2_Interaction'] = df_model['V1'] * df_model['V2']
    df_model['V14_Amount'] = df_model['V14'] * df_model['Scaled_Amount']

    print(f'  {df_model.shape[1] - 1} features ready (including engineered)')
    return df_model


def split_and_resample(df_model):
    """Step 7-8: Train-test split, then SMOTE on training data."""
    banner('STEP 4 — Split & SMOTE Resampling')
    X = df_model.drop('Class', axis=1)
    y = df_model['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}')

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f'  After SMOTE — Genuine: {(y_res == 0).sum():,} | Fraud: {(y_res == 1).sum():,}')

    # Save feature names
    joblib.dump(X.columns.tolist(), os.path.join(MODELS_DIR, 'feature_names.pkl'))

    return X_res, y_res, X_test, y_test


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Step 9-10: Train 5 models and evaluate."""
    banner('STEP 5 — Training & Evaluating Models')

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42,
                                  use_label_encoder=False, eval_metric='logloss',
                                  scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()),
        'LightGBM': LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                    random_state=42, is_unbalance=True, verbose=-1),
    }

    results = {}
    best_auc = 0
    best_name = None
    best_model = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1-Score': f1, 'ROC-AUC': auc
        }

        status = ''
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = model
            status = ' <- BEST'

        print(f'  {name:25s} | Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} AUC={auc:.4f}{status}')

        # Save every model
        safe = name.lower().replace(' ', '_')
        joblib.dump(model, os.path.join(MODELS_DIR, f'{safe}_model.pkl'))

    # Save best model separately
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_fraud_model.pkl'))

    # Save results CSV
    results_df = pd.DataFrame(results).T.sort_values('ROC-AUC', ascending=False)
    results_df.to_csv(os.path.join(REPORTS_DIR, 'model_comparison_results.csv'))

    print(f'\n  Best Model: {best_name} (ROC-AUC = {best_auc:.4f})')
    return results_df, best_name, best_model


def main():
    print('\nCREDIT CARD FRAUD DETECTION PIPELINE')
    print('Authors: Sanman Kadam, Varsha Gupta')
    print('=' * 60)

    # 1. Load & Clean
    df = load_and_clean(DATA_RAW)

    # 2. Scale
    df, scaler = scale_features(df)

    # 3. Save cleaned data
    df.to_csv(DATA_CLEANED, index=False)

    # 4. Feature engineering
    df_model = engineer_features(df)

    # 5. Split & SMOTE
    X_train, y_train, X_test, y_test = split_and_resample(df_model)

    # 6. Train & Evaluate
    results_df, best_name, best_model = train_and_evaluate(
        X_train, y_train, X_test, y_test
    )

    # 7. Summary
    banner('PIPELINE COMPLETE')
    print(f'  Best Model:    {best_name}')
    print(f'  Models saved:  {MODELS_DIR}/')
    print(f'  Results saved: {REPORTS_DIR}/model_comparison_results.csv')
    print(f'  Cleaned data:  {DATA_CLEANED}')
    print()


if __name__ == '__main__':
    main()
