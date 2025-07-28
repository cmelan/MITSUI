# MITSUI&CO. Commodity Prediction Challenge - Advanced Submission
# This version includes feature engineering and advanced modeling

# Install required packages
!pip install polars kaggle-evaluation lightgbm xgboost scikit-learn

import os
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import gc

import kaggle_evaluation.core.base_gateway
import kaggle_evaluation.core.templates

class MitsuiAdvancedGateway(kaggle_evaluation.core.templates.Gateway):
    def __init__(self, data_paths: tuple[str] | None = None):
        super().__init__(data_paths, file_share_dir=None)
        self.data_paths = data_paths
        self.row_id_column_name = 'date_id'
        self.set_response_timeout_seconds(60 * 5)
        self.models = {}
        self.scalers = {}
        self.feature_columns = None

    def unpack_data_paths(self):
        if not self.data_paths:
            self.competition_data_dir = '/kaggle/input/mitsui-commodity-prediction-challenge/'
        else:
            self.competition_data_dir = self.data_paths[0]
        self.competition_data_dir = Path(self.competition_data_dir)

    def create_features(self, df):
        """Create advanced features"""
        features = df.copy()
        
        # Basic statistical features
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'date_id']
        
        if len(numeric_cols) > 0:
            # Rolling statistics
            for col in numeric_cols[:10]:  # Limit to first 10 columns to avoid memory issues
                try:
                    features[f'{col}_rolling_mean'] = features[col].rolling(window=3, min_periods=1).mean()
                    features[f'{col}_rolling_std'] = features[col].rolling(window=3, min_periods=1).std()
                except:
                    pass
            
            # Lag features
            for col in numeric_cols[:5]:  # Limit to first 5 columns
                try:
                    features[f'{col}_lag1'] = features[col].shift(1)
                    features[f'{col}_lag2'] = features[col].shift(2)
                except:
                    pass
        
        # Fill NaN values
        features = features.fillna(0)
        
        return features

    def train_advanced_models(self):
        """Train advanced models with feature engineering"""
        print("Training advanced models with feature engineering...")
        
        # Load training data
        train_df = pd.read_csv(self.competition_data_dir / 'train.csv')
        train_labels = pd.read_csv(self.competition_data_dir / 'train_labels.csv')
        
        # Create advanced features
        train_df_enhanced = self.create_features(train_df)
        
        # Get feature columns
        self.feature_columns = [col for col in train_df_enhanced.columns if col != 'date_id']
        target_columns = [col for col in train_labels.columns if col != 'date_id']
        
        print(f"Training on {len(self.feature_columns)} enhanced features for {len(target_columns)} targets")
        
        # Prepare training data
        X_train = train_df_enhanced[self.feature_columns]
        
        # Train advanced model for each target
        for target_col in target_columns:
            if target_col in train_labels.columns:
                y_train = train_labels[target_col].fillna(0)
                
                # Skip if all values are the same
                if y_train.nunique() <= 1:
                    continue
                
                # Create and fit scaler
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                self.scalers[target_col] = scaler
                
                # Train LightGBM with optimized parameters
                model = lgb.LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=8,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1,
                    early_stopping_rounds=50
                )
                
                try:
                    model.fit(X_train_scaled, y_train)
                    self.models[target_col] = model
                except Exception as e:
                    print(f"Error training model for {target_col}: {e}")
                    # Fallback to simple model
                    self.models[target_col] = Ridge(alpha=1.0)
                    self.models[target_col].fit(X_train_scaled, y_train)
        
        print(f"Trained {len(self.models)} advanced models")
        del train_df, train_labels, train_df_enhanced, X_train
        gc.collect()

    def generate_data_batches(self):
        test = pl.read_csv(self.competition_data_dir / 'test.csv')

        label_lag_dir = self.competition_data_dir / 'lagged_test_labels'
        label_lags_1 = pl.read_csv(label_lag_dir / 'test_labels_lag_1.csv')
        label_lags_2 = pl.read_csv(label_lag_dir / 'test_labels_lag_2.csv')
        label_lags_3 = pl.read_csv(label_lag_dir / 'test_labels_lag_3.csv')
        label_lags_4 = pl.read_csv(label_lag_dir / 'test_labels_lag_4.csv')

        date_ids = test['date_id'].unique(maintain_order=True).to_list()
        for date_id in date_ids:
            test_batch = test.filter(pl.col('date_id') == date_id)
            label_lags_1_batch = label_lags_1.filter(pl.col('date_id') == date_id)
            label_lags_2_batch = label_lags_2.filter(pl.col('date_id') == date_id)
            label_lags_3_batch = label_lags_3.filter(pl.col('date_id') == date_id)
            label_lags_4_batch = label_lags_4.filter(pl.col('date_id') == date_id)

            yield (
                (test_batch, label_lags_1_batch, label_lags_2_batch, label_lags_3_batch, label_lags_4_batch),
                date_id,
            )

    def competition_specific_validation(self, prediction, row_ids, data_batch) -> None:
        assert isinstance(prediction, (pd.DataFrame, pl.DataFrame))
        assert len(prediction) == 1
        assert 'date_id' not in prediction.columns
        provided_label_lags = pl.concat([i.drop(['date_id', 'label_date_id']) for i in data_batch[1:]], how='horizontal')
        assert len(prediction.columns) == len(provided_label_lags.columns)

    def predict(self, data_batch):
        """
        Generate advanced predictions with feature engineering
        """
        test_batch, label_lags_1_batch, label_lags_2_batch, label_lags_3_batch, label_lags_4_batch = data_batch
        
        # Get the target columns from lag data
        lag_1_values = label_lags_1_batch.drop(['date_id', 'label_date_id']).to_pandas()
        target_columns = lag_1_values.columns.tolist()
        
        # Prepare test features with enhancement
        test_features = test_batch.to_pandas()
        test_features_enhanced = self.create_features(test_features)
        
        if self.feature_columns:
            X_test = test_features_enhanced[self.feature_columns]
        else:
            feature_cols = [col for col in test_features_enhanced.columns if col != 'date_id']
            X_test = test_features_enhanced[feature_cols]
        
        # Generate predictions
        predictions = pd.DataFrame()
        
        for target_col in target_columns:
            if target_col in self.models and target_col in self.scalers:
                # Use trained model with scaling
                X_test_scaled = self.scalers[target_col].transform(X_test)
                pred = self.models[target_col].predict(X_test_scaled)[0]
                predictions[target_col] = [pred]
            else:
                # Fallback to lag-based prediction
                lag_val = lag_1_values[target_col].iloc[0] if len(lag_1_values) > 0 else 0.0
                predictions[target_col] = [lag_val + np.random.normal(0, 0.001)]
        
        # Ensure all values are finite
        predictions = predictions.fillna(0.0)
        predictions = predictions.replace([np.inf, -np.inf], 0.0)
        
        return predictions

# Run the gateway
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print("Running advanced competition evaluation...")
    gateway = MitsuiAdvancedGateway()
    gateway.train_advanced_models()  # Train advanced models first
    gateway.run()
else:
    print("This is a development run - skipping competition evaluation")
    print("To test the submission, this notebook should be submitted to Kaggle") 