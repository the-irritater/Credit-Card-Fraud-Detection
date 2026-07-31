# Contributing to Credit Card Fraud Detection

Thank you for your interest in contributing to this project! This guide will help you get started.

## Development Environment Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/<your-username>/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install development tools
pip install flake8 pytest
```

## Code Style

This project follows **PEP 8** with the following adjustments:

- Maximum line length: **120 characters**
- Use **type hints** for function signatures
- Use **docstrings** (Google-style) for all public functions
- Use `logging` instead of `print()` for output

## Project Structure

All source code lives in `src/`:

| Module | Purpose |
|--------|---------|
| `config.py` | Centralized configuration and hyperparameters |
| `constants.py` | Static constants and domain knowledge |
| `logging_config.py` | Structured logging setup |
| `data_loader.py` | Dataset loading and deduplication |
| `feature_engineering.py` | Deterministic feature transformations |
| `preprocessing.py` | Train/test split, scaling, anomaly scoring |
| `train.py` | Model training, cross-validation, calibration |
| `evaluate.py` | Metric computation and threshold optimization |
| `utils.py` | Visualization and SHAP analysis |

## Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style above.

3. **Run tests** to ensure nothing is broken:
   ```bash
   python -m pytest tests/ -v
   flake8 src/ --max-line-length=120 --ignore=E501,W503
   ```

4. **Update documentation** if your changes affect the README, docstrings, or report.

5. **Submit a Pull Request** with:
   - Clear description of what changed
   - Reference to related issue (if applicable)
   - Test results summary

## Reporting Issues

When reporting bugs, please include:

- Python version
- Operating system
- Full error traceback
- Steps to reproduce

## Areas for Contribution

- Additional ML models or ensemble methods
- Real-time streaming pipeline (Kafka integration)
- Expanded SHAP analysis and interpretability
- Unit test coverage improvements
- Documentation and tutorial improvements

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
