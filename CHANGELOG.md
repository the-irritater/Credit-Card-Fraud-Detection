# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-07-31

### Added
- **Repeated Stratified KFold Cross-Validation** (5 folds × 3 repeats = 15 evaluations) for statistically stable estimates
- **Confidence intervals** for all cross-validation metrics (mean ± std format)
- **4 new models**: CatBoost (optional), HistGradientBoosting, BalancedRandomForest, EasyEnsemble — total 9 models benchmarked
- **F2-Score** (β=2) metric emphasizing recall for fraud detection
- **Cost-sensitive threshold optimization** with configurable FN/FP cost ratio
- **Probability calibration** (CalibratedClassifierCV with isotonic regression) for top 3 models
- **Isolation Forest anomaly scores** as a feature (fitted on training set only to prevent leakage)
- **Amount Z-score**, **Hour of Week**, **Is_Weekend** features
- **Calibration curve** visualization (raw vs. calibrated probabilities)
- **Threshold optimization plot** (Precision, Recall, F1, F2 vs. threshold)
- **Model comparison radar chart**
- **Confidence interval bar charts** with error bars
- **SHAP waterfall plot** for individual fraud prediction explanation
- **SHAP bar plot** for mean absolute feature importance
- **Feature importance comparison** across multiple tree models
- Centralized `src/config.py` (all hyperparameters and paths)
- Static `src/constants.py` (domain knowledge and enumerations)
- Structured Python `logging` replacing all `print()` statements
- Pipeline log output to `logs/pipeline.log`
- CLI argument parsing (`--quick`, `--data`)
- Step-by-step execution timing
- `LICENSE` (MIT)
- `CONTRIBUTING.md`
- `CHANGELOG.md`
- `.github/workflows/ci.yml` (GitHub Actions CI)
- `tests/test_pipeline.py` (pytest unit tests)

### Changed
- `StratifiedKFold` → `RepeatedStratifiedKFold` for more robust CV estimates
- `train_and_benchmark_models()` now returns 3 values: `results_df`, `trained_models`, `calibration_data`
- `prepare_data_splits()` now returns 7 values (added `iso_forest`)
- `model_comparison_results.csv` schema expanded with CI columns and new models
- Version bumped from 2.0.0 to 3.0.0

### Improved
- De-emphasized accuracy in favor of PR-AUC, Recall, MCC, F2-Score
- All visualizations use configurable DPI and style from config
- Feature engineering uses constants from `constants.py`

---

## [2.0.0] - 2026-07-31

### Added
- Modular `src/` Python package structure
- Streamlit web application (`app/streamlit_app.py`)
- SQL schema and analytical queries (`sql/schema_and_queries.sql`)
- Docker container configuration (`Dockerfile`)
- SHAP explainability analysis
- Optuna hyperparameter optimization for LightGBM
- 5-Fold Stratified Cross-Validation with SMOTE inside folds
- Threshold optimization (F1-maximizing search)
- Dual RobustScaler normalization (leakage-free)
- Model serialization to `models/` directory
- Interactive HTML dashboard (`dashboard/index.html`)
- Comprehensive README with metrics, architecture, and reproduction steps

---

## [1.0.0] - 2026-07-30

### Added
- Initial Jupyter notebook implementation
- Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM
- Basic SMOTE resampling
- ROC-AUC and accuracy evaluation
- Correlation and distribution plots
