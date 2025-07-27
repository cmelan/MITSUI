# MITSUI&CO. Commodity Prediction Challenge

## Overview
This project is for the Kaggle MITSUI&CO. Commodity Prediction Challenge. The goal is to develop a robust model for accurate and stable prediction of commodity prices using historical data from LME, JPX, US Stock, and Forex markets.

## Project Structure
```
├── data/                    # Raw and processed data files
├── notebooks/               # Jupyter notebooks for EDA, modeling, and experiments
│   ├── eda.ipynb           # Exploratory Data Analysis
│   ├── baseline_model.ipynb # Baseline model implementation
│   └── ...                 # Other modeling notebooks
├── src/                    # Source code for data processing, feature engineering, and modeling
│   ├── data_processing.py  # Data loading and preprocessing functions
│   ├── models.py           # Model wrappers and factory functions
│   ├── ensemble.py         # Ensemble methods (averaging, stacking, etc.)
│   ├── feature_selection.py # Feature importance and selection methods
│   ├── cv.py              # Time-series cross-validation
│   └── tuning.py          # Hyperparameter optimization
├── models/                 # Saved model files
├── outputs/                # Submission files and results
├── requirements.txt        # Python dependencies
├── setup_project.py        # Project setup and dependency checker
├── run_baseline.py         # Baseline model runner
└── README.md              # This file
```

## Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Run setup script to check environment and generate sample data
python setup_project.py
```

### 2. Run Competition Pipeline
```bash
# Run full competition pipeline with sample data
python run_competition.py

# Or run baseline model
python run_baseline.py
```

### 3. Explore with Jupyter
```bash
# Start Jupyter notebook
jupyter notebook notebooks/eda.ipynb
```

## Detailed Setup Instructions

### Prerequisites
- Python 3.7+
- pip package manager

### Installation Steps
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "MITSUI&CO. Commodity Prediction Challenge"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run setup script**
   ```bash
   python setup_project.py
   ```
   This will:
   - Check Python version and dependencies
   - Create necessary directories
   - Generate sample data for testing
   - Verify custom modules can be imported

4. **Download competition data** (optional)
   - Download from [Kaggle Competition](https://kaggle.com/competitions/mitsui-commodity-prediction-challenge)
   - Place files in `data/` directory:
     - `train.csv`
     - `test.csv`
     - `train_labels.csv`
     - `target_pairs.csv`

## Usage

### Running the Competition Pipeline
```bash
python run_competition.py
```
This will:
- Load and preprocess competition data
- Parse 424 target configurations
- Create target-specific features
- Train multi-target models (LightGBM, XGBoost)
- Generate predictions for all targets
- Create competition submission format
- Save results to `outputs/` directory

### Running the Baseline Model
```bash
python run_baseline.py
```
This will:
- Load and preprocess data
- Perform feature engineering
- Train baseline models (LightGBM, XGBoost, Ridge)
- Generate ensemble predictions
- Save results to `outputs/` directory

### Exploring Data with Jupyter
```bash
jupyter notebook notebooks/eda.ipynb
```
The EDA notebook provides:
- Data overview and quality analysis
- Time series analysis
- Feature correlation analysis
- Target variable analysis

### Using Custom Modules
```python
import sys
sys.path.append('src')

from data_processing import load_data, add_lag_features
from models import get_model
from ensemble import simple_average
from cv import time_series_cv_split
```

## Features

### Data Processing (`src/data_processing.py`)
- **Data loading**: Load competition data files
- **Missing value handling**: Forward fill, backward fill, mean imputation
- **Feature engineering**:
  - Lag features (1, 2, 3 periods)
  - Rolling statistics (mean, std, min, max)
  - Exponentially weighted moving averages
  - Calendar features (day of week, month)
  - Cross-asset features (spreads, ratios, correlations)
  - Extended rolling features (skew, kurtosis, volatility, momentum)
  - Interaction features (pairwise products)

### Multi-Target Modeling (`src/multi_target.py`)
- **Target parsing**: Parse 424 target configurations from target_pairs.csv
- **Target-specific features**: Create features for single assets and asset pairs
- **Multi-target models**: Handle all 424 targets simultaneously
- **Competition evaluation**: Sharpe ratio variant metric implementation
- **Submission format**: Generate competition-ready submission files

### Models (`src/models.py`)
- **LightGBM**: Gradient boosting with LightGBM
- **XGBoost**: Gradient boosting with XGBoost
- **CatBoost**: Gradient boosting with CatBoost
- **Random Forest**: Ensemble of decision trees
- **Ridge Regression**: Linear regression with L2 regularization

### Ensemble Methods (`src/ensemble.py`)
- **Simple averaging**: Equal weights for all models
- **Weighted averaging**: Custom weights for models
- **Stacking**: Meta-model on base model predictions
- **Out-of-fold stacking**: Stacking with cross-validation
- **Weight optimization**: Optimize weights using scipy

### Feature Selection (`src/feature_selection.py`)
- **SHAP importance**: SHAP-based feature importance
- **Permutation importance**: Permutation-based importance
- **Recursive feature elimination**: RFE for feature selection
- **Correlation filtering**: Remove highly correlated features

### Cross-Validation (`src/cv.py`)
- **Time-series CV**: Expanding and rolling window splits
- **Configurable parameters**: Number of splits, test size

### Hyperparameter Tuning (`src/tuning.py`)
- **Optuna integration**: Bayesian optimization
- **LightGBM tuning**: Comprehensive parameter search
- **Spearman correlation**: Competition metric optimization

## Competition Information

### Objective
Predict **424 different targets** consisting of:
- **Single asset returns**: Log returns of individual financial instruments
- **Asset pair differences**: Price differences between pairs of assets
- **Multi-market data**: LME (metals), JPX (Japanese), US (stocks), FX (forex)

### Key Challenge Details
- **424 targets**: Each with different lags (1-4 days) and asset combinations
- **Price-difference series**: Many targets are differences between asset pairs
- **Multi-asset relationships**: Cross-market correlations and spreads
- **Stability focus**: Consistent performance across different market conditions

### Evaluation Metric
- **Sharpe Ratio Variant**: Mean Spearman correlation / Standard deviation
- **Stability emphasis**: Lower variance in predictions is rewarded
- **Multi-target**: Performance across all 424 targets

### Data Structure
- **1977 columns**: Massive feature space from multiple markets
- **Time series**: Historical data with UTC timestamps
- **Market-specific**: Different trading days and time zones
- **Target pairs**: Detailed configuration for each of 424 targets

## Next Steps

### Immediate Actions
1. **Download real competition data** and replace sample files
2. **Run EDA notebook** to understand data characteristics
3. **Execute baseline model** to establish performance benchmark
4. **Analyze results** and identify improvement opportunities

### Model Development
1. **Feature engineering**: Implement domain-specific features
2. **Hyperparameter tuning**: Optimize model parameters
3. **Ensemble methods**: Combine multiple models effectively
4. **Cross-validation**: Ensure robust evaluation
5. **Feature selection**: Identify most important features

### Advanced Techniques
1. **Multi-target modeling**: Predict multiple commodities simultaneously
2. **Time series specific models**: LSTM, GRU, Transformer models
3. **Advanced ensembles**: Stacking, blending, weighted combinations
4. **Domain knowledge**: Incorporate commodity market insights

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure all dependencies are installed
2. **Data not found**: Run `setup_project.py` to generate sample data
3. **Memory issues**: Reduce feature set or use data sampling
4. **Model convergence**: Adjust learning rates or regularization

### Getting Help
- Check the setup script output for specific error messages
- Review the competition documentation on Kaggle
- Examine the sample code in notebooks and scripts

## References
- [Competition Link](https://kaggle.com/competitions/mitsui-commodity-prediction-challenge)
- [Kaggle Documentation](https://www.kaggle.com/docs)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)

## License
This project is for educational and competition purposes. Please refer to the Kaggle competition terms and conditions.
