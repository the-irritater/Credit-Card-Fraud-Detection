# AI-Based Credit Card Fraud Detection - Step-by-Step Project Report

---

## Authors

| Author | Role |
|--------|------|
| **Varsha Gupta** | Project Lead / Data Analyst |
| **Sanman Kadam** | Data Analyst |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Problem & Objective](#2-business-problem--objective)
3. [Dataset Description](#3-dataset-description)
4. [Project Architecture](#4-project-architecture)
5. [Step 1 - Project Setup & Libraries](#step-1---project-setup--libraries)
6. [Step 2 - Data Loading & Inspection](#step-2---data-loading--inspection)
7. [Step 3 - Data Cleaning](#step-3---data-cleaning)
8. [Step 4 - Exploratory Data Analysis (EDA)](#step-4---exploratory-data-analysis-eda)
9. [Step 5 - SQL Analysis](#step-5---sql-analysis)
10. [Step 6 - Feature Engineering](#step-6---feature-engineering)
11. [Step 7 - Handling Class Imbalance](#step-7---handling-class-imbalance)
12. [Step 8 - Train-Test Split](#step-8---train-test-split)
13. [Step 9 - Model Training](#step-9---model-training)
14. [Step 10 - Model Evaluation](#step-10---model-evaluation)
15. [Step 11 - Feature Importance](#step-11---feature-importance)
16. [Step 12 - Model Saving & Deployment](#step-12---model-saving--deployment)
17. [Step 13 - Power BI Dashboard Design](#step-13---power-bi-dashboard-design)
18. [Step 14 - Streamlit App Deployment](#step-14---streamlit-app-deployment)
19. [Key Findings & Business Impact](#key-findings--business-impact)
20. [Future Improvements](#future-improvements)

---

## 1. Project Overview

| Item | Detail |
|------|--------|
| **Project Title** | AI-Based Credit Card Fraud Detection and Transaction Analytics System |
| **Domain** | Financial Services / Banking |
| **Type** | Binary Classification (Fraud vs Genuine) |
| **Dataset** | Kaggle - Credit Card Fraud Detection (284,807 transactions) |
| **Goal** | Detect fraudulent credit card transactions with high recall and minimize false positives |

---

## 2. Business Problem & Objective

### The Problem

Banks process **millions of transactions daily**. Among these, a very small fraction (~0.17%) are fraudulent. However, even this tiny percentage translates to **millions of dollars in losses** annually. Manual review of every transaction is physically and economically impossible.

### Why This Matters

| Scenario | Impact |
|----------|--------|
| **Missed Fraud (False Negative)** | Direct financial loss to the bank and customer |
| **False Alarm (False Positive)** | Genuine transactions blocked -> customer frustration, operational cost |
| **No System** | Entire fraud goes undetected until customer reports |

### Our Objective

Build an **automated fraud detection system** that:

1. **Detects fraudulent transactions** with high recall (catch as many frauds as possible)
2. **Minimizes false alarms** to reduce unnecessary blocking of genuine transactions
3. **Provides business intelligence** through dashboards and reports
4. **Enables real-time prediction** via a deployable web application

---

## 3. Dataset Description

**Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains transactions made by European cardholders in September 2013 over **2 days**.

| Column | Description | Type |
|--------|-------------|------|
| `Time` | Seconds elapsed since the first transaction in the dataset | Numeric |
| `V1` - `V28` | PCA-transformed features (anonymized for privacy) | Numeric |
| `Amount` | Transaction amount in Euros | Numeric |
| `Class` | Target variable: **0** = Genuine, **1** = Fraud | Binary |

### Key Statistics

| Metric | Value |
|--------|-------|
| Total Transactions | 284,807 |
| Genuine Transactions | 284,315 (99.83%) |
| Fraudulent Transactions | 492 (0.17%) |
| Number of Features | 31 (including target) |
| Missing Values | 0 |
| Imbalance Ratio | ~578 : 1 |

### Why PCA Features?

The original features (e.g., merchant name, card number, location) have been transformed using **Principal Component Analysis (PCA)** to protect customer privacy. This means:
- We **cannot** know the real-world meaning of V1-V28
- The features are already **numerically scaled** (mean approx 0, std approx 1)
- Only `Time` and `Amount` are in their original form

---

## 4. Project Architecture

```
Credit Card Dataset (CSV)
        |
        v
+-----------------+
|  Data Cleaning  |  Remove duplicates, handle nulls, scale features
+--------+--------+
         |
         v
+-----------------+
|       EDA       |  Visualize distributions, correlations, patterns
+--------+--------+
         |
         v
+-----------------+
|  SQL Analysis   |  Business queries on transaction data
+--------+--------+
         |
         v
+---------------------+
| Feature Engineering |  Create Hour, Is_Night, Amount_Log, interactions
+--------+------------+
         |
         v
+---------------------+
| Handle Imbalance    |  SMOTE (Synthetic Minority Oversampling)
+--------+------------+
         |
         v
+-----------------+
| Model Training  |  LR, DT, RF, XGBoost, LightGBM
+--------+--------+
         |
         v
+-----------------+
|   Evaluation    |  Precision, Recall, F1, ROC-AUC, Confusion Matrix
+--------+--------+
         |
         v
+-----------------+        +-----------------+
| Power BI        |        | Streamlit App   |
| Dashboard       |        | (Deployment)    |
+-----------------+        +-----------------+
```

---

## Step 1 - Project Setup & Libraries

### What We Do

Install and import all required Python libraries organized by purpose.

### Why We Use These Libraries

| Library | Purpose | Why This Library? |
|---------|---------|-------------------|
| **Pandas** | Data manipulation | Industry standard for tabular data; DataFrame operations |
| **NumPy** | Numerical computing | Fast array operations, mathematical functions |
| **Matplotlib** | Static plots | Fine-grained control over publication-quality charts |
| **Seaborn** | Statistical plots | Built on Matplotlib; cleaner syntax for statistical visuals |
| **Plotly** | Interactive plots | Hover tooltips, zoom, pan; great for dashboards |
| **Scikit-learn** | ML framework | Comprehensive toolkit for preprocessing, models, metrics |
| **XGBoost** | Gradient boosting | State-of-the-art for tabular data; handles imbalance |
| **LightGBM** | Gradient boosting | Faster than XGBoost; memory efficient; great accuracy |
| **imbalanced-learn** | Class imbalance | SMOTE implementation and resampling strategies |
| **joblib** | Model persistence | Save/load trained models efficiently |

---

## Step 2 - Data Loading & Inspection

### What We Do

Load the CSV file into a Pandas DataFrame and perform initial inspection.

### Why This Step?

Before any analysis, we must understand:
- **Shape**: How many rows and columns?
- **Data Types**: Are features numeric? Is the target binary?
- **Statistics**: What are the min, max, mean of each feature?
- **Class Distribution**: How imbalanced is the target?

### What We Find

- **No missing values** - the dataset is complete
- **All columns are numeric** - PCA already handled encoding
- **Extreme imbalance** - Only 0.17% of transactions are fraud
- **Amount is right-skewed** - Most transactions are small, a few are very large

---

## Step 3 - Data Cleaning

### What We Do

1. Check for and remove **duplicate records**
2. Verify no **missing values** exist
3. **Scale** the `Amount` and `Time` columns to match V1-V28

### Why This Step?

| Task | Why? |
|------|------|
| **Remove Duplicates** | Duplicate records inflate class counts and can cause the model to memorize specific transactions instead of learning patterns |
| **Handle Missing Values** | Many ML algorithms cannot handle null values; missing data can introduce bias |
| **Scale Amount & Time** | V1-V28 are already PCA-transformed, but Amount ranges from 0 to 25,691 and Time from 0 to 172,792. Without scaling, these features would dominate distance-based calculations |

### Why RobustScaler Instead of StandardScaler?

| Scaler | Method | When to Use |
|--------|--------|-------------|
| **StandardScaler** | Uses mean and standard deviation | When data is roughly normally distributed |
| **RobustScaler** | Uses median and interquartile range (IQR) | When data has **outliers** (our case - Amount has extreme outliers) |

The `Amount` column has extreme outliers (max approx $25,691 vs median approx $22). `RobustScaler` is **resistant to outliers** because it uses the median and IQR instead of the mean and standard deviation.

---

## Step 4 - Exploratory Data Analysis (EDA)

### What We Do

Visualize and analyze the data to uncover patterns, distributions, and relationships.

### Analysis 1: Class Distribution

**What we find**: Only 492 out of 284,807 transactions are fraudulent (0.17%). This extreme imbalance means:
- A model predicting "Genuine" for everything achieves **99.83% accuracy** but catches **zero fraud**
- We need specialized techniques (SMOTE, class weights)
- Accuracy is a **misleading metric** - we must use Precision, Recall, F1, and ROC-AUC

### Analysis 2: Transaction Amount Distribution

**What we find**:
- Genuine transactions have a **wide range** of amounts ($0 to $25,691)
- Fraudulent transactions tend to be **smaller** in amount (average approx $122 vs $88 for genuine)
- Smaller fraud transactions are **harder to detect** and less likely to trigger manual review

### Analysis 3: Time Analysis

**What we find**:
- Normal transaction volume drops significantly during **nighttime hours** (roughly 11PM - 6AM)
- Fraud transactions are more **evenly distributed** across all hours, including late night
- Fraud-to-genuine ratio is **higher during low-volume periods** (late night / early morning)

### Analysis 4: Feature Correlations with Fraud

**What we find**:
- **Negatively correlated** with fraud: V14, V12, V10, V3 (lower values -> higher fraud likelihood)
- **Positively correlated** with fraud: V4, V11, V2, V19 (higher values -> higher fraud likelihood)

---

## Step 5 - SQL Analysis

### What We Do

Write SQL queries to extract business insights from the transaction data.

### Key SQL Queries & Business Questions

#### Q1: How many fraudulent transactions exist?

```sql
SELECT COUNT(*) AS fraud_count
FROM transactions
WHERE Class = 1;
```

#### Q2: What is the total financial loss from fraud?

```sql
SELECT SUM(Amount) AS total_fraud_loss
FROM transactions
WHERE Class = 1;
```

#### Q3: What is the average fraud vs genuine transaction amount?

```sql
SELECT Class,
       AVG(Amount) AS avg_amount,
       MIN(Amount) AS min_amount,
       MAX(Amount) AS max_amount
FROM transactions
GROUP BY Class;
```

#### Q4: Which hours have the highest fraud rate?

```sql
SELECT FLOOR(Time / 3600) % 24 AS hour_of_day,
       COUNT(*) AS fraud_count
FROM transactions
WHERE Class = 1
GROUP BY hour_of_day
ORDER BY fraud_count DESC;
```

#### Q5: Top 10 highest fraud amounts

```sql
SELECT *
FROM transactions
WHERE Class = 1
ORDER BY Amount DESC
LIMIT 10;
```

---

## Step 6 - Feature Engineering

### What We Do

Create new features from existing data to improve model performance.

### Features We Create

| Feature | Formula | Why? |
|---------|---------|------|
| `Hour` | `(Time / 3600) % 24` | Captures time-of-day patterns (fraud spikes at night) |
| `Is_Night` | `1 if Hour in [22, 5] else 0` | Binary indicator for high-risk nighttime window |
| `Amount_Log` | `log(1 + Amount)` | Reduces right-skewness of Amount |
| `Amount_Category` | Binned into ranges | Captures amount-based risk levels |
| `V1_V2_Interaction` | `V1 x V2` | Captures interaction effects between top PCA components |
| `V14_Amount` | `V14 x Scaled_Amount` | V14 is the most predictive feature; interaction with amount reveals patterns |

---

## Step 7 - Handling Class Imbalance

### What We Do

Balance the training data so the model learns to recognize fraud patterns despite having very few fraud examples.

### Techniques Compared

| Technique | How It Works | Pros | Cons |
|-----------|-------------|------|------|
| **Random Under-Sampling** | Randomly removes majority class samples | Simple, fast | Loses valuable data |
| **Random Over-Sampling** | Duplicates minority class samples | Simple, no data loss | Overfitting risk |
| **SMOTE** | Creates synthetic minority samples by interpolating between existing fraud points | Realistic new samples, preserves all data | Can create noisy samples in overlapping regions |
| **Class Weights** | Penalizes fraud misclassification more heavily | No data modification needed | May not be sufficient for extreme imbalance |

### Critical Rule: Split BEFORE SMOTE!

```
WRONG: SMOTE -> Split (data leakage!)
RIGHT: Split -> SMOTE on training set only
```

---

## Step 8 - Train-Test Split

Divide the dataset into **80% training** and **20% testing** sets with stratified splitting to preserve fraud ratios.

---

## Step 9 - Model Training

Train five classification algorithms:

1. **Logistic Regression** (Baseline)
2. **Decision Tree**
3. **Random Forest**
4. **XGBoost**
5. **LightGBM**

---

## Step 10 - Model Evaluation

### Metrics Used

| Metric | Importance |
|--------|------------|
| **Recall** | **Most Critical** - Catch as much fraud as possible |
| **Precision** | Avoid unnecessary false alarms |
| **F1-Score** | Harmonic balance between Precision and Recall |
| **ROC-AUC** | Overall model discrimination ability |

### Confusion Matrix Interpretation

| Cell | Meaning | Business Impact |
|------|---------|----------------|
| **TN** (True Negative) | Genuine correctly classified | Normal operation |
| **FP** (False Positive) | Genuine flagged as fraud | Unnecessary investigation |
| **FN** (False Negative) | Fraud missed by model | **Financial loss!** Most dangerous error |
| **TP** (True Positive) | Fraud correctly caught | Fraud prevented |

---

## Step 11 - Feature Importance

Identified top features contributing to fraud detection:
- **V14**: Highest predictive power
- **V12, V10, V17**: Strong secondary indicators
- **Scaled_Amount**: Amount contributes significantly to risk scoring

---

## Step 12 - Model Saving & Deployment

Saved artifacts under `models/`:
- `best_fraud_model.pkl` (LightGBM / Best model)
- `robust_scaler.pkl`
- `feature_names.pkl`

---

## Step 13 - Power BI Dashboard Design

Designed a 3-page business analytics dashboard:
1. Executive Summary
2. Fraud Deep Dive
3. Model Performance

---

## Step 14 - Streamlit App Deployment

Built a real-time Streamlit web app allowing users to input transaction data, compute fraud risk scores, and view interactive model dashboards.

---

## Key Findings & Business Impact

1. **High Recall**: Missing fraud costs significantly more than reviewing false alarms.
2. **Night Risk**: Fraud probability spikes during low-volume hours (10PM-5AM).
3. **Imbalance Strategy**: SMOTE on training data prevents baseline bias without data leakage.

---

## Future Improvements

1. Hyperparameter tuning using Optuna.
2. Model explainability using SHAP values.
3. Real-time streaming using Apache Kafka.

---

*Report prepared as part of the AI-Based Credit Card Fraud Detection portfolio project.*
