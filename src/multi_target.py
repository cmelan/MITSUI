import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import xgboost as xgb


def parse_target_pairs(target_pairs_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Parse target_pairs.csv to understand each target's structure.
    
    Returns:
        Dictionary mapping target names to their configuration
    """
    target_configs = {}
    
    # Check if we have the expected columns
    expected_columns = ['target', 'lag', 'pair']
    if not all(col in target_pairs_df.columns for col in expected_columns):
        # Create sample configurations for testing
        print("⚠️  Using sample target configurations (real data will have proper structure)")
        
        # Create sample targets based on available features
        sample_targets = [
            ('target_0', 1, 'feature_1', 'single'),
            ('target_1', 1, 'feature_1 - feature_2', 'difference'),
            ('target_2', 2, 'feature_2 - feature_3', 'difference'),
            ('target_3', 1, 'feature_3', 'single'),
        ]
        
        for target_name, lag, pair, target_type in sample_targets:
            if target_type == 'difference':
                asset1, asset2 = pair.split(' - ')
            else:
                asset1 = pair
                asset2 = None
            
            target_configs[target_name] = {
                'lag': lag,
                'type': target_type,
                'asset1': asset1,
                'asset2': asset2,
                'pair': pair
            }
        
        return target_configs
    
    # Parse real target pairs data
    for _, row in target_pairs_df.iterrows():
        target_name = row['target']
        lag = row['lag']
        pair = row['pair']
        
        # Parse the pair to understand the target structure
        if ' - ' in pair:
            # Asset pair difference
            asset1, asset2 = pair.split(' - ')
            target_type = 'difference'
        else:
            # Single asset return
            asset1 = pair
            asset2 = None
            target_type = 'single'
        
        target_configs[target_name] = {
            'lag': lag,
            'type': target_type,
            'asset1': asset1,
            'asset2': asset2,
            'pair': pair
        }
    
    return target_configs


def create_target_features(
    train_df: pd.DataFrame, 
    target_configs: Dict[str, Dict],
    target_cols: List[str]
) -> pd.DataFrame:
    """
    Create features specific to each target based on its configuration.
    
    Args:
        train_df: Training data with price columns
        target_configs: Parsed target configurations
        target_cols: List of target column names
    
    Returns:
        DataFrame with target-specific features
    """
    df = train_df.copy()
    
    for target_name in target_cols:
        if target_name not in target_configs:
            continue
            
        config = target_configs[target_name]
        lag = config['lag']
        target_type = config['type']
        
        if target_type == 'single':
            # Single asset return features
            asset = config['asset1']
            if asset in df.columns:
                # Log returns
                df[f'{target_name}_log_return'] = np.log(df[asset] / df[asset].shift(1))
                # Price momentum
                df[f'{target_name}_momentum_5'] = df[asset] / df[asset].shift(5) - 1
                df[f'{target_name}_momentum_10'] = df[asset] / df[asset].shift(10) - 1
                # Volatility
                df[f'{target_name}_volatility_5'] = df[asset].rolling(5).std()
                df[f'{target_name}_volatility_10'] = df[asset].rolling(10).std()
                
        elif target_type == 'difference':
            # Asset pair difference features
            asset1, asset2 = config['asset1'], config['asset2']
            
            if asset1 in df.columns and asset2 in df.columns:
                # Price difference
                df[f'{target_name}_price_diff'] = df[asset1] - df[asset2]
                # Price ratio
                df[f'{target_name}_price_ratio'] = df[asset1] / df[asset2]
                # Spread percentage
                df[f'{target_name}_spread_pct'] = (df[asset1] - df[asset2]) / df[asset2]
                
                # Rolling statistics of the spread
                df[f'{target_name}_spread_mean_5'] = df[f'{target_name}_price_diff'].rolling(5).mean()
                df[f'{target_name}_spread_std_5'] = df[f'{target_name}_price_diff'].rolling(5).std()
                df[f'{target_name}_spread_mean_10'] = df[f'{target_name}_price_diff'].rolling(10).mean()
                df[f'{target_name}_spread_std_10'] = df[f'{target_name}_price_diff'].rolling(10).std()
                
                # Z-score of spread
                df[f'{target_name}_spread_zscore'] = (
                    df[f'{target_name}_price_diff'] - df[f'{target_name}_spread_mean_10']
                ) / df[f'{target_name}_spread_std_10']
    
    return df


def prepare_multi_target_data(
    train_df: pd.DataFrame,
    train_labels_df: pd.DataFrame,
    target_pairs_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict]:
    """
    Prepare data for multi-target prediction.
    
    Returns:
        X_train, y_train, target_columns, target_configs
    """
    # Parse target configurations
    target_configs = parse_target_pairs(target_pairs_df)
    
    # Get target column names
    target_cols = [col for col in train_labels_df.columns if col.startswith('target_')]
    
    # Create target-specific features
    train_with_features = create_target_features(train_df, target_configs, target_cols)
    
    if test_df is not None:
        test_with_features = create_target_features(test_df, target_configs, target_cols)
    else:
        test_with_features = None
    
    # Prepare X and y
    feature_cols = [col for col in train_with_features.columns 
                   if col not in ['date_id'] + target_cols]
    
    X_train = train_with_features[feature_cols].fillna(0)
    y_train = train_labels_df[target_cols].fillna(0)
    
    if test_with_features is not None:
        X_test = test_with_features[feature_cols].fillna(0)
    else:
        X_test = None
    
    return X_train, y_train, X_test, target_cols, target_configs


class MultiTargetModel:
    """
    Multi-target prediction model optimized for the competition.
    """
    
    def __init__(self, base_model='lgbm', **kwargs):
        self.base_model = base_model
        self.model = None
        self.target_configs = None
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, target_configs: Dict):
        """Fit multi-target model."""
        self.target_configs = target_configs
        
        if self.base_model == 'lgbm':
            base = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
        elif self.base_model == 'xgb':
            base = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown base model: {self.base_model}")
        
        self.model = MultiOutputRegressor(base)
        self.model.fit(X, y)
        self.feature_names = X.columns.tolist()
        
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict for all targets."""
        predictions = self.model.predict(X)
        target_cols = [f'target_{i}' for i in range(predictions.shape[1])]
        return pd.DataFrame(predictions, columns=target_cols, index=X.index)


def evaluate_multi_target(
    y_true: pd.DataFrame, 
    y_pred: pd.DataFrame,
    target_configs: Dict
) -> Dict:
    """
    Evaluate multi-target predictions using competition metric.
    """
    from scipy.stats import spearmanr
    
    results = {}
    
    # Calculate Spearman correlation for each target
    correlations = []
    for col in y_true.columns:
        if col in y_pred.columns:
            mask = ~(y_true[col].isna() | y_pred[col].isna())
            if mask.sum() > 10:  # Need sufficient data
                corr, _ = spearmanr(y_true[col][mask], y_pred[col][mask])
                if not np.isnan(corr):
                    correlations.append(corr)
    
    if correlations:
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        
        # Competition metric: mean / std (Sharpe ratio variant)
        competition_score = mean_corr / (std_corr + 1e-8)
        
        results = {
            'mean_spearman': mean_corr,
            'std_spearman': std_corr,
            'competition_score': competition_score,
            'num_targets': len(correlations),
            'target_correlations': correlations
        }
    
    return results


def create_submission_format(
    predictions: pd.DataFrame,
    test_df: pd.DataFrame,
    target_cols: List[str]
) -> pd.DataFrame:
    """
    Create submission format for the competition.
    """
    submission = test_df[['date_id']].copy()
    
    # Add predictions for each target
    for col in target_cols:
        if col in predictions.columns:
            submission[col] = predictions[col].values
    
    return submission 