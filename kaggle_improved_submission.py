# MITSUI&CO. Commodity Prediction Challenge - Improved Submission
# This version uses actual machine learning models

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
import gc

import kaggle_evaluation.core.base_gateway
import kaggle_evaluation.core.templates

class MitsuiImprovedGateway(kaggle_evaluation.core.templates.Gateway):
    def __init__(self, data_paths: tuple[str] | None = None):
        super().__init__(data_paths, file_share_dir=None)
        self.data_paths = data_paths
        self.row_id_column_name = 'date_id'
        self.set_response_timeout_seconds(60 * 5)
        self.models = {}  # Store trained models
        self.feature_columns = None

    def unpack_data_paths(self):
        if not self.data_paths:
            self.competition_data_dir = '/kaggle/input/mitsui-commodity-prediction-challenge/'
        else:
            self.competition_data_dir = self.data_paths[0]
        self.competition_data_dir = Path(self.competition_data_dir)

    def train_models(self):
        """Train models on the training data"""
        print("Training models...")
        
        # Load training data
        train_df = pd.read_csv(self.competition_data_dir / 'train.csv')
        train_labels = pd.read_csv(self.competition_data_dir / 'train_labels.csv')
        
        # Get feature columns (exclude date_id and target columns)
        self.feature_columns = [col for col in train_df.columns if col != 'date_id']
        target_columns = [col for col in train_labels.columns if col != 'date_id']
        
        print(f"Training on {len(self.feature_columns)} features for {len(target_columns)} targets")
        
        # Prepare training data
        X_train = train_df[self.feature_columns].fillna(0)
        
        # Train a model for each target
        for target_col in target_columns:
            if target_col in train_labels.columns:
                y_train = train_labels[target_col].fillna(0)
                
                # Skip if all values are the same
                if y_train.nunique() <= 1:
                    continue
                
                # Train LightGBM model
                model = lgb.LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    verbose=-1
                )
                
                try:
                    model.fit(X_train, y_train)
                    self.models[target_col] = model
                except Exception as e:
                    print(f"Error training model for {target_col}: {e}")
                    # Fallback to simple model
                    self.models[target_col] = Ridge(alpha=1.0)
                    self.models[target_col].fit(X_train, y_train)
        
        print(f"Trained {len(self.models)} models")
        del train_df, train_labels, X_train
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
        Generate predictions using trained models
        """
        test_batch, label_lags_1_batch, label_lags_2_batch, label_lags_3_batch, label_lags_4_batch = data_batch
        
        # Get the target columns from lag data
        lag_1_values = label_lags_1_batch.drop(['date_id', 'label_date_id']).to_pandas()
        target_columns = lag_1_values.columns.tolist()
        
        # Prepare test features
        test_features = test_batch.to_pandas()
        if self.feature_columns:
            X_test = test_features[self.feature_columns].fillna(0)
        else:
            # Fallback: use all columns except date_id
            feature_cols = [col for col in test_features.columns if col != 'date_id']
            X_test = test_features[feature_cols].fillna(0)
        
        # Generate predictions
        predictions = pd.DataFrame()
        
        for target_col in target_columns:
            if target_col in self.models:
                # Use trained model
                pred = self.models[target_col].predict(X_test)[0]
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
    print("Running improved competition evaluation...")
    gateway = MitsuiImprovedGateway()
    gateway.train_models()  # Train models first
    gateway.run()
else:
    print("This is a development run - skipping competition evaluation")
    print("To test the submission, this notebook should be submitted to Kaggle") 