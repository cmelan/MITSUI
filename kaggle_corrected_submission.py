# MITSUI&CO. Commodity Prediction Challenge - Corrected Submission
# Copy this entire code to a Kaggle notebook and run

print("Starting MITSUI submission...")

# Install only polars (kaggle_evaluation is already available in competition data)
!pip install polars

import os
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path

# Import kaggle_evaluation (already available in competition data)
import kaggle_evaluation.core.base_gateway
import kaggle_evaluation.core.templates

class MitsuiGateway(kaggle_evaluation.core.templates.Gateway):
    def __init__(self, data_paths: tuple[str] | None = None):
        super().__init__(data_paths, file_share_dir=None)
        self.data_paths = data_paths
        self.row_id_column_name = 'date_id'
        self.set_response_timeout_seconds(60 * 5)

    def unpack_data_paths(self):
        if not self.data_paths:
            self.competition_data_dir = '/kaggle/input/mitsui-commodity-prediction-challenge/'
        else:
            self.competition_data_dir = self.data_paths[0]
        self.competition_data_dir = Path(self.competition_data_dir)

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
        test_batch, label_lags_1_batch, label_lags_2_batch, label_lags_3_batch, label_lags_4_batch = data_batch
        
        # Get the lag values
        lag_1_values = label_lags_1_batch.drop(['date_id', 'label_date_id']).to_pandas()
        
        # Simple prediction: use lag_1 values with small noise
        predictions = lag_1_values.copy()
        noise = np.random.normal(0, 0.001, predictions.shape)
        predictions = predictions + noise
        
        # Ensure all values are finite
        predictions = predictions.fillna(0.0)
        predictions = predictions.replace([np.inf, -np.inf], 0.0)
        
        return predictions

# Run the gateway
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print("Running competition evaluation...")
    gateway = MitsuiGateway()
    gateway.run()
    print("Submission complete!")
else:
    print("Development mode - skipping evaluation")
    print("Submit this notebook to Kaggle for evaluation") 