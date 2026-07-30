-- ============================================================
-- Credit Card Fraud Detection - SQL Schema & Analytical Queries
-- Authors: Sanman Kadam, Varsha Gupta
-- ============================================================

-- 1. Create Database
CREATE DATABASE IF NOT EXISTS credit_card_fraud;
USE credit_card_fraud;

-- 2. Create Transactions Table
CREATE TABLE IF NOT EXISTS transactions (
    TransactionID   INT AUTO_INCREMENT PRIMARY KEY,
    Time            FLOAT NOT NULL,
    V1              FLOAT, V2  FLOAT, V3  FLOAT, V4  FLOAT, V5  FLOAT,
    V6              FLOAT, V7  FLOAT, V8  FLOAT, V9  FLOAT, V10 FLOAT,
    V11             FLOAT, V12 FLOAT, V13 FLOAT, V14 FLOAT, V15 FLOAT,
    V16             FLOAT, V17 FLOAT, V18 FLOAT, V19 FLOAT, V20 FLOAT,
    V21             FLOAT, V22 FLOAT, V23 FLOAT, V24 FLOAT, V25 FLOAT,
    V26             FLOAT, V27 FLOAT, V28 FLOAT,
    Amount          FLOAT NOT NULL,
    Class           INT NOT NULL DEFAULT 0,
    INDEX idx_class (Class),
    INDEX idx_amount (Amount),
    INDEX idx_time (Time)
);

-- ============================================================
-- ANALYTICAL QUERIES
-- ============================================================

-- Q1: Total transactions by class
SELECT
    CASE WHEN Class = 0 THEN 'Genuine' ELSE 'Fraud' END AS Transaction_Type,
    COUNT(*) AS Total_Count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions), 3) AS Percentage
FROM transactions
GROUP BY Class;

-- Q2: Total fraud financial loss
SELECT
    COUNT(*) AS Fraud_Count,
    SUM(Amount) AS Total_Fraud_Loss,
    AVG(Amount) AS Avg_Fraud_Amount,
    MAX(Amount) AS Max_Fraud_Amount,
    MIN(Amount) AS Min_Fraud_Amount
FROM transactions
WHERE Class = 1;

-- Q3: Amount statistics by class
SELECT
    CASE WHEN Class = 0 THEN 'Genuine' ELSE 'Fraud' END AS Transaction_Type,
    COUNT(*) AS Count,
    ROUND(AVG(Amount), 2) AS Avg_Amount,
    ROUND(MIN(Amount), 2) AS Min_Amount,
    ROUND(MAX(Amount), 2) AS Max_Amount,
    ROUND(STDDEV(Amount), 2) AS Std_Amount
FROM transactions
GROUP BY Class;

-- Q4: Fraud count by hour of day
SELECT
    FLOOR((Time / 3600) % 24) AS Hour_of_Day,
    COUNT(*) AS Fraud_Count,
    ROUND(SUM(Amount), 2) AS Total_Fraud_Amount
FROM transactions
WHERE Class = 1
GROUP BY Hour_of_Day
ORDER BY Fraud_Count DESC;

-- Q5: Top 10 highest fraud transaction amounts
SELECT
    TransactionID,
    Time,
    Amount,
    Class
FROM transactions
WHERE Class = 1
ORDER BY Amount DESC
LIMIT 10;

-- Q6: Amount range analysis for fraud
SELECT
    CASE
        WHEN Amount BETWEEN 0 AND 10 THEN '0-10'
        WHEN Amount BETWEEN 10 AND 50 THEN '10-50'
        WHEN Amount BETWEEN 50 AND 100 THEN '50-100'
        WHEN Amount BETWEEN 100 AND 500 THEN '100-500'
        WHEN Amount BETWEEN 500 AND 1000 THEN '500-1000'
        ELSE '1000+'
    END AS Amount_Range,
    COUNT(*) AS Fraud_Count,
    ROUND(SUM(Amount), 2) AS Total_Loss
FROM transactions
WHERE Class = 1
GROUP BY Amount_Range
ORDER BY Fraud_Count DESC;

-- Q7: Fraud rate by time period (day vs night)
SELECT
    CASE
        WHEN FLOOR((Time / 3600) % 24) BETWEEN 6 AND 21 THEN 'Day (6AM-9PM)'
        ELSE 'Night (9PM-6AM)'
    END AS Time_Period,
    COUNT(*) AS Total_Transactions,
    SUM(Class) AS Fraud_Count,
    ROUND(SUM(Class) * 100.0 / COUNT(*), 4) AS Fraud_Rate_Pct
FROM transactions
GROUP BY Time_Period;

-- Q8: Cumulative fraud loss over time
SELECT
    FLOOR(Time / 3600) AS Hour_Elapsed,
    COUNT(*) AS Fraud_Count,
    SUM(Amount) AS Hourly_Loss,
    SUM(SUM(Amount)) OVER (ORDER BY FLOOR(Time / 3600)) AS Cumulative_Loss
FROM transactions
WHERE Class = 1
GROUP BY Hour_Elapsed
ORDER BY Hour_Elapsed;
