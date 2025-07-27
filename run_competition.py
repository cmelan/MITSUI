#!/usr/bin/env python3
"""
MITSUI&CO. Commodity Prediction Challenge - Competition Runner

This script implements the full competition pipeline for the 424-target prediction challenge.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
import sys
sys.path.append('src')
from data_processing import (
    load_data, fill_missing, add_lag_features, 
    add_rolling_features, add_calendar_features
)
from multi_target import (
    prepare_multi_target_data, MultiTargetModel, 
    evaluate_multi_target, create_submission_format
)
from ensemble import simple_average, weighted_average

def main():
    """Run the full competition pipeline."""
    print("🚀 MITSUI&CO. Commodity Prediction Challenge - Competition Runner")
    print("=" * 70)
    
    # Load competition data
    print("Loading competition data...")
    try:
        train, test, train_labels, target_pairs = load_data(
            'data/train.csv', 'data/test.csv', 
            'data/train_labels.csv', 'data/target_pairs.csv'
        )
        print("✅ Competition data loaded successfully")
    except Exception as e:
        print(f"❌ Error loading competition data: {e}")
        print("Please download the competition data from Kaggle and place it in the data/ directory")
        return False
    
    # Display data overview
    print(f"\n📊 Data Overview:")
    print(f"   Train shape: {train.shape}")
    print(f"   Test shape: {test.shape}")
    print(f"   Train labels shape: {train_labels.shape}")
    print(f"   Target pairs: {len(target_pairs)} targets")
    
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
    
    print("✅ Data preprocessing completed")
    
    # Prepare multi-target data
    print("\nPreparing multi-target data...")
    
    X_train, y_train, X_test, target_cols, target_configs = prepare_multi_target_data(
        train, train_labels, target_pairs, test
    )
    
    print(f"✅ Multi-target data prepared:")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Targets: {len(target_cols)}")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0] if X_test is not None else 0}")
    
    # Analyze target configurations
    print(f"\n📈 Target Analysis:")
    single_targets = sum(1 for config in target_configs.values() if config['type'] == 'single')
    difference_targets = sum(1 for config in target_configs.values() if config['type'] == 'difference')
    
    print(f"   Single asset targets: {single_targets}")
    print(f"   Asset pair differences: {difference_targets}")
    
    lag_counts = {}
    for config in target_configs.values():
        lag = config['lag']
        lag_counts[lag] = lag_counts.get(lag, 0) + 1
    
    print(f"   Lag distribution: {lag_counts}")
    
    # Train multiple models
    print("\nTraining multi-target models...")
    
    models = {
        'LightGBM': MultiTargetModel(base_model='lgbm'),
        'XGBoost': MultiTargetModel(base_model='xgb')
    }
    
    predictions = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train, target_configs)
        
        # Generate predictions
        if X_test is not None:
            pred = model.predict(X_test)
            predictions[model_name] = pred
            print(f"✅ {model_name} trained and predictions generated")
        else:
            print(f"✅ {model_name} trained (no test data for predictions)")
    
    # Create ensemble predictions
    if len(predictions) > 1 and X_test is not None:
        print("\nCreating ensemble predictions...")
        
        # Simple average ensemble
        ensemble_simple = simple_average(list(predictions.values()))
        predictions['Ensemble_Simple'] = pd.DataFrame(
            ensemble_simple, 
            columns=target_cols, 
            index=X_test.index
        )
        
        # Weighted average ensemble
        ensemble_weighted = weighted_average(list(predictions.values()))
        predictions['Ensemble_Weighted'] = pd.DataFrame(
            ensemble_weighted, 
            columns=target_cols, 
            index=X_test.index
        )
        
        print("✅ Ensemble predictions created")
    
    # Evaluate on training data (if we have validation split)
    if len(train) > 1000:  # Only if we have enough data
        print("\nEvaluating models on validation split...")
        
        # Use last 20% of data for validation
        split_idx = int(len(X_train) * 0.8)
        X_val = X_train.iloc[split_idx:]
        y_val = y_train.iloc[split_idx:]
        
        for model_name, model in models.items():
            val_pred = model.predict(X_val)
            results = evaluate_multi_target(y_val, val_pred, target_configs)
            
            if results:
                print(f"   {model_name}:")
                print(f"     Mean Spearman: {results['mean_spearman']:.4f}")
                print(f"     Std Spearman: {results['std_spearman']:.4f}")
                print(f"     Competition Score: {results['competition_score']:.4f}")
    
    # Save results
    if X_test is not None:
        print("\nSaving competition results...")
        
        # Save detailed predictions
        for model_name, pred_df in predictions.items():
            detailed_path = f'outputs/{model_name.lower()}_predictions.csv'
            pred_df.to_csv(detailed_path, index=False)
            print(f"✅ {model_name} predictions saved to {detailed_path}")
        
        # Create submission format (using ensemble)
        if 'Ensemble_Simple' in predictions:
            submission = create_submission_format(
                predictions['Ensemble_Simple'], test, target_cols
            )
        else:
            # Use first available model
            first_model = list(predictions.keys())[0]
            submission = create_submission_format(
                predictions[first_model], test, target_cols
            )
        
        submission_path = 'outputs/competition_submission.csv'
        submission.to_csv(submission_path, index=False)
        print(f"✅ Competition submission saved to {submission_path}")
        
        # Save target analysis
        target_analysis = pd.DataFrame([
            {
                'target': target_name,
                'type': config['type'],
                'lag': config['lag'],
                'asset1': config['asset1'],
                'asset2': config['asset2']
            }
            for target_name, config in target_configs.items()
        ])
        
        target_analysis.to_csv('outputs/target_analysis.csv', index=False)
        print("✅ Target analysis saved to outputs/target_analysis.csv")
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ Competition pipeline completed successfully!")
    print(f"📊 Models trained: {len(models)}")
    print(f"📊 Targets handled: {len(target_cols)}")
    print(f"📊 Features generated: {X_train.shape[1]}")
    
    if X_test is not None:
        print(f"📊 Test predictions generated: {X_test.shape[0]} samples")
    
    print("\n📋 Next Steps:")
    print("1. Download real competition data from Kaggle")
    print("2. Run this script with actual data")
    print("3. Optimize hyperparameters for each model")
    print("4. Implement advanced feature engineering")
    print("5. Add cross-validation for robust evaluation")
    print("6. Submit to Kaggle competition")
    
    print("\n🔗 Competition Link:")
    print("https://kaggle.com/competitions/mitsui-commodity-prediction-challenge")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 