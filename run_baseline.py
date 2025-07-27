#!/usr/bin/env python3
"""
MITSUI&CO. Commodity Prediction Challenge - Baseline Model Runner

This script runs a baseline model to test the project setup and establish performance benchmarks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
import sys
sys.path.append('src')
from data_processing import (
    load_data, fill_missing, add_lag_features, 
    add_rolling_features, add_calendar_features
)
from models import get_model
from cv import time_series_cv_split
from ensemble import simple_average, weighted_average

def main():
    """Run baseline model."""
    print("🚀 Running Baseline Model")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    try:
        train, test, train_labels, target_pairs = load_data(
            'data/train.csv', 'data/test.csv', 
            'data/train_labels.csv', 'data/target_pairs.csv'
        )
        print("✅ Data loaded successfully")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Please run setup_project.py first to generate sample data")
        return False
    
    # Basic preprocessing
    print("\nPreprocessing data...")
    
    # Convert date_id to datetime
    train['date_id'] = pd.to_datetime(train['date_id'])
    test['date_id'] = pd.to_datetime(test['date_id'])
    train_labels['date_id'] = pd.to_datetime(train_labels['date_id'])
    
    # Sort by date
    train = train.sort_values('date_id').reset_index(drop=True)
    test = test.sort_values('date_id').reset_index(drop=True)
    train_labels = train_labels.sort_values('date_id').reset_index(drop=True)
    
    # Fill missing values
    train = fill_missing(train, method='ffill')
    test = fill_missing(test, method='ffill')
    
    print(f"✅ Train shape: {train.shape}")
    print(f"✅ Test shape: {test.shape}")
    
    # Feature engineering
    print("\nFeature engineering...")
    
    # Identify numerical features
    feature_cols = [col for col in train.columns if col != 'date_id']
    
    # Add calendar features
    train = add_calendar_features(train)
    test = add_calendar_features(test)
    
    # Add lag features
    if len(feature_cols) > 0:
        train = add_lag_features(train, feature_cols, lags=[1, 2, 3])
        test = add_lag_features(test, feature_cols, lags=[1, 2, 3])
    
    # Add rolling features
    if len(feature_cols) > 0:
        train = add_rolling_features(train, feature_cols, windows=[3, 7])
        test = add_rolling_features(test, feature_cols, windows=[3, 7])
    
    print(f"✅ Final train shape: {train.shape}")
    print(f"✅ Final test shape: {test.shape}")
    
    # Prepare features and target
    feature_cols = [col for col in train.columns if col not in ['date_id', 'year', 'month', 'dayofweek']]
    
    # Remove features with too many missing values
    missing_pct = train[feature_cols].isnull().sum() / len(train)
    feature_cols = [col for col in feature_cols if missing_pct[col] < 0.5]
    
    print(f"✅ Selected {len(feature_cols)} features")
    
    # Prepare X and y
    X_train = train[feature_cols].fillna(0)
    X_test = test[feature_cols].fillna(0)
    y_train = train_labels['target'].values
    
    # Define baseline models
    print("\nTraining baseline models...")
    
    models = {
        'LightGBM': get_model('lgbm', {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': 42
        }),
        'XGBoost': get_model('xgb', {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': 42
        }),
        'Ridge': get_model('ridge', {
            'alpha': 1.0,
            'random_state': 42
        })
    }
    
    # Train models on full training data
    final_models = {}
    predictions = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        final_models[model_name] = model
        
        # Generate predictions
        predictions[model_name] = model.predict(X_test)
        print(f"✅ {model_name} trained and predictions generated")
    
    # Create ensemble predictions
    print("\nCreating ensemble predictions...")
    
    ensemble_simple = simple_average(list(predictions.values()))
    ensemble_weighted = weighted_average(list(predictions.values()))
    
    print("✅ Ensemble predictions created")
    
    # Save results
    print("\nSaving results...")
    
    # Create submission format
    submission = pd.DataFrame({
        'date_id': test['date_id'],
        'target': ensemble_simple
    })
    
    # Save to outputs directory
    submission.to_csv('outputs/baseline_submission.csv', index=False)
    print("✅ Baseline submission saved to outputs/baseline_submission.csv")
    
    # Save detailed results
    detailed_results = pd.DataFrame({
        'date_id': test['date_id'],
        'target': ensemble_simple,
        'lightgbm_pred': predictions['LightGBM'],
        'xgboost_pred': predictions['XGBoost'],
        'ridge_pred': predictions['Ridge'],
        'ensemble_simple': ensemble_simple,
        'ensemble_weighted': ensemble_weighted
    })
    
    detailed_results.to_csv('outputs/baseline_detailed_results.csv', index=False)
    print("✅ Detailed results saved to outputs/baseline_detailed_results.csv")
    
    # Print summary
    print("\n" + "=" * 50)
    print("✅ Baseline model completed successfully!")
    print(f"📊 Number of features: {len(feature_cols)}")
    print(f"📊 Training samples: {len(X_train)}")
    print(f"📊 Test samples: {len(X_test)}")
    print(f"📊 Models trained: {len(models)}")
    
    print("\n📋 Next Steps:")
    print("1. Check the generated files in outputs/ directory")
    print("2. Run cross-validation for proper evaluation")
    print("3. Implement more sophisticated feature engineering")
    print("4. Add hyperparameter tuning")
    print("5. Create multi-target modeling pipeline")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 