# MITSUI&CO. Commodity Prediction Challenge - Enhanced ML Template
# This template implements sophisticated ML models with feature engineering and ensemble methods

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

print("🚀 Starting MITSUI Commodity Prediction Challenge")
print(f"⏰ Started at: {datetime.now()}")

# ============================================================================
# PATH SETUP AND DATA LOADING
# ============================================================================

# Detect competition data path
competition_path = None
possible_paths = [
    '/kaggle/input/mitsui-commodity-prediction-challenge',
    '/kaggle/input/mitsui-commodity-prediction-challenge/data',
    './data'
]

for path in possible_paths:
    if os.path.exists(path):
        competition_path = path
        print(f"✅ Found competition data at: {path}")
        break

if not competition_path:
    print("❌ Competition data not found in expected locations")
    raise FileNotFoundError("Competition data not found")

# Setup evaluation API path
evaluation_path = f'{competition_path}/kaggle_evaluation'
if os.path.exists(evaluation_path):
    sys.path.append(evaluation_path)
    print(f"✅ Added evaluation path: {evaluation_path}")
else:
    print(f"❌ Evaluation path not found: {evaluation_path}")
    # Try alternative paths
    alt_paths = [
        '/kaggle/input/mitsui-commodity-prediction-challenge/kaggle_evaluation',
        '/kaggle/input/mitsui-commodity-prediction-challenge/data/kaggle_evaluation',
        './kaggle_evaluation'
    ]
    for alt_path in alt_paths:
        if os.path.exists(alt_path):
            sys.path.append(alt_path)
            print(f"✅ Added alternative evaluation path: {alt_path}")
            break
    else:
        print("❌ No evaluation API found in any expected location")

try:
    # Import the evaluation API
    from mitsui_gateway import MitsuiGateway
    from kaggle_evaluation.core.templates import InferenceServer
    print("✅ Successfully imported evaluation API")
    
    # Load the target pairs information
    target_pairs_path = f'{competition_path}/target_pairs.csv'
    target_pairs = pd.read_csv(target_pairs_path)
    print(f"📊 Loaded target pairs: {len(target_pairs)} targets")
    
    # Load training data for feature engineering
    train_data_path = f'{competition_path}/train.csv'
    print("📈 Loading training data...")
    train_data = pd.read_csv(train_data_path)
    print(f"📈 Training data shape: {train_data.shape}")
    
    # Load training labels
    train_labels_path = f'{competition_path}/train_labels.csv'
    print("🎯 Loading training labels...")
    train_labels = pd.read_csv(train_labels_path)
    print(f"🎯 Training labels shape: {train_labels.shape}")
    
except ImportError as e:
    print(f"❌ Import error: {str(e)}")
    print("💡 The evaluation API is not available in this environment.")
    print("📋 This notebook needs to be run on Kaggle's competition platform.")
    raise
    
except Exception as e:
    print(f"❌ Error loading data: {str(e)}")
    print("💡 Check that the competition data is properly loaded.")
    raise

# ============================================================================
# FEATURE ENGINEERING CLASS
# ============================================================================

class FeatureEngineer:
    """Advanced feature engineering for financial time series data."""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_columns = []
        
    def create_technical_indicators(self, df):
        """Create technical indicators for financial data."""
        features = df.copy()
        
        # Get price columns (close prices)
        price_cols = [col for col in df.columns if 'Close' in col or 'close' in col]
        
        for col in price_cols:
            if col in df.columns:
                # Price-based features
                features[f'{col}_pct_change'] = df[col].pct_change()
                features[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
                
                # Moving averages
                for window in [5, 10, 20, 50]:
                    features[f'{col}_ma_{window}'] = df[col].rolling(window=window).mean()
                    features[f'{col}_ma_ratio_{window}'] = df[col] / features[f'{col}_ma_{window}']
                
                # Volatility features
                for window in [5, 10, 20]:
                    features[f'{col}_volatility_{window}'] = df[col].rolling(window=window).std()
                
                # Momentum features
                for period in [5, 10, 20]:
                    features[f'{col}_momentum_{period}'] = df[col] - df[col].shift(period)
                    features[f'{col}_momentum_pct_{period}'] = (df[col] - df[col].shift(period)) / df[col].shift(period)
        
        # Volume-based features (if available)
        volume_cols = [col for col in df.columns if 'Volume' in col or 'volume' in col]
        for col in volume_cols:
            if col in df.columns:
                features[f'{col}_pct_change'] = df[col].pct_change()
                for window in [5, 10, 20]:
                    features[f'{col}_ma_{window}'] = df[col].rolling(window=window).mean()
        
        # FX-specific features
        fx_cols = [col for col in df.columns if col.startswith('FX_')]
        for col in fx_cols:
            if col in df.columns:
                features[f'{col}_pct_change'] = df[col].pct_change()
                features[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
        
        return features
    
    def create_cross_asset_features(self, df):
        """Create features based on relationships between different assets."""
        features = df.copy()
        
        # Commodity spreads
        if 'LME_AH_Close' in df.columns and 'LME_ZS_Close' in df.columns:
            features['LME_AH_ZS_spread'] = df['LME_AH_Close'] - df['LME_ZS_Close']
            features['LME_AH_ZS_ratio'] = df['LME_AH_Close'] / df['LME_ZS_Close']
        
        if 'LME_PB_Close' in df.columns and 'LME_CA_Close' in df.columns:
            features['LME_PB_CA_spread'] = df['LME_PB_Close'] - df['LME_CA_Close']
            features['LME_PB_CA_ratio'] = df['LME_PB_Close'] / df['LME_CA_Close']
        
        # Gold-Platinum spread
        if 'JPX_Gold_Standard_Futures_Close' in df.columns and 'JPX_Platinum_Standard_Futures_Close' in df.columns:
            features['Gold_Platinum_spread'] = df['JPX_Gold_Standard_Futures_Close'] - df['JPX_Platinum_Standard_Futures_Close']
            features['Gold_Platinum_ratio'] = df['JPX_Gold_Standard_Futures_Close'] / df['JPX_Platinum_Standard_Futures_Close']
        
        # Currency strength indices
        fx_cols = [col for col in df.columns if col.startswith('FX_')]
        if len(fx_cols) > 0:
            # USD strength index (average of USD pairs)
            usd_pairs = [col for col in fx_cols if 'USD' in col and col != 'FX_USDJPY']
            if usd_pairs:
                features['USD_strength'] = df[usd_pairs].mean(axis=1)
        
        return features
    
    def create_lag_features(self, df, max_lag=5):
        """Create lagged features for time series prediction."""
        features = df.copy()
        
        # Create lags for key price columns
        key_cols = [col for col in df.columns if 'Close' in col or 'close' in col]
        key_cols.extend([col for col in df.columns if col.startswith('FX_')])
        
        for col in key_cols:
            if col in df.columns:
                for lag in range(1, max_lag + 1):
                    features[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return features
    
    def engineer_features(self, df):
        """Main feature engineering pipeline."""
        print("🔧 Starting feature engineering...")
        
        # Apply all feature engineering steps
        features = self.create_technical_indicators(df)
        features = self.create_cross_asset_features(features)
        features = self.create_lag_features(features)
        
        # Remove infinite and NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Store feature columns for later use
        self.feature_columns = [col for col in features.columns if col != 'date_id']
        
        print(f"✅ Feature engineering completed. Total features: {len(self.feature_columns)}")
        return features

# ============================================================================
# ENSEMBLE MODEL CLASS
# ============================================================================

class EnsembleModel:
    """Ensemble of multiple ML models for robust predictions."""
    
    def __init__(self, n_models=5):
        self.n_models = n_models
        self.models = []
        self.scalers = []
        self.is_trained = False
        
    def create_models(self):
        """Create diverse set of models for ensemble."""
        self.models = [
            # Linear models
            Ridge(alpha=1.0, random_state=42),
            Lasso(alpha=0.01, random_state=42),
            
            # Tree-based models
            RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
            
            # Gradient boosting
            xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
            lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        ]
        
        # Create scalers for each model
        self.scalers = [StandardScaler() for _ in range(len(self.models))]
        
        print(f"✅ Created {len(self.models)} models for ensemble")
    
    def train_models(self, X, y):
        """Train all models in the ensemble."""
        print("🏋️ Training ensemble models...")
        
        # Create models if not already created
        if not self.models:
            self.create_models()
        
        # Train each model
        for i, (model, scaler) in enumerate(zip(self.models, self.scalers)):
            print(f"  Training model {i+1}/{len(self.models)}: {type(model).__name__}")
            
            # Scale features
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model.fit(X_scaled, y)
        
        self.is_trained = True
        print("✅ All models trained successfully")
    
    def predict(self, X):
        """Generate ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        predictions = []
        
        # Get predictions from each model
        for model, scaler in zip(self.models, self.scalers):
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)
            predictions.append(pred)
        
        # Average predictions (simple ensemble)
        ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred

# ============================================================================
# TARGET-SPECIFIC MODELING
# ============================================================================

class TargetModelManager:
    """Manages models for each target, using target_pairs information."""
    
    def __init__(self, target_pairs):
        self.target_pairs = target_pairs
        self.feature_engineer = FeatureEngineer()
        self.target_models = {}
        self.is_trained = False
        
    def get_target_features(self, target_name, df):
        """Get relevant features for a specific target based on target_pairs."""
        target_info = self.target_pairs[self.target_pairs['target'] == target_name].iloc[0]
        pair = target_info['pair']
        
        # Extract relevant features based on the pair
        relevant_features = []
        
        # Add features for each asset in the pair
        if ' - ' in pair:
            # This is a spread target
            asset1, asset2 = pair.split(' - ')
            relevant_features.extend([col for col in df.columns if asset1 in col])
            relevant_features.extend([col for col in df.columns if asset2 in col])
        else:
            # This is a single asset target
            relevant_features.extend([col for col in df.columns if pair in col])
        
        # Add general market features
        relevant_features.extend([col for col in df.columns if 'FX_' in col])
        relevant_features.extend([col for col in df.columns if 'LME_' in col])
        relevant_features.extend([col for col in df.columns if 'JPX_' in col])
        
        # Remove duplicates and ensure date_id is not included
        relevant_features = list(set(relevant_features))
        if 'date_id' in relevant_features:
            relevant_features.remove('date_id')
        
        return relevant_features
    
    def train_target_model(self, target_name, train_data, train_labels):
        """Train a model for a specific target."""
        print(f"🎯 Training model for {target_name}")
        
        # Get target values
        target_values = train_labels[target_name].values
        
        # Get relevant features for this target
        relevant_features = self.get_target_features(target_name, train_data)
        
        if len(relevant_features) == 0:
            print(f"⚠️ No relevant features found for {target_name}, using all features")
            relevant_features = [col for col in train_data.columns if col != 'date_id']
        
        # Prepare training data
        X = train_data[relevant_features].values
        y = target_values
        
        # Remove rows with NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            print(f"⚠️ No valid training data for {target_name}, using simple baseline")
            self.target_models[target_name] = None
            return
        
        # Create and train ensemble model
        ensemble = EnsembleModel()
        ensemble.train_models(X, y)
        
        # Store model and feature list
        self.target_models[target_name] = {
            'model': ensemble,
            'features': relevant_features
        }
    
    def train_all_models(self, train_data, train_labels):
        """Train models for all targets."""
        print("🏋️ Training models for all targets...")
        
        # Engineer features for training data
        engineered_data = self.feature_engineer.engineer_features(train_data)
        
        # Train models for each target
        target_columns = [col for col in train_labels.columns if col.startswith('target_')]
        
        for target_name in target_columns:
            try:
                self.train_target_model(target_name, engineered_data, train_labels)
            except Exception as e:
                print(f"❌ Error training model for {target_name}: {str(e)}")
                self.target_models[target_name] = None
        
        self.is_trained = True
        print(f"✅ Training completed for {len(self.target_models)} targets")
    
    def predict_target(self, target_name, test_data):
        """Predict for a specific target."""
        if target_name not in self.target_models or self.target_models[target_name] is None:
            # Return baseline prediction
            return np.random.normal(0, 0.01)
        
        model_info = self.target_models[target_name]
        model = model_info['model']
        features = model_info['features']
        
        # Get relevant features from test data
        X = test_data[features].values
        
        # Handle NaN values
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)
        
        # Make prediction
        prediction = model.predict(X.reshape(1, -1))[0]
        
        return prediction

# ============================================================================
# MAIN PREDICTION FUNCTION
# ============================================================================

# Initialize the target model manager
print("🔧 Initializing target model manager...")
target_manager = TargetModelManager(target_pairs)

# Train models on historical data
print("📚 Training models on historical data...")
target_manager.train_all_models(train_data, train_labels)

def predict(data_batch):
    """
    Generate predictions for all 424 targets for a given data batch.
    Uses sophisticated ML models with feature engineering and ensemble methods.
    
    Args:
        data_batch: Tuple containing (test_batch, label_lags_1_batch, label_lags_2_batch, label_lags_3_batch, label_lags_4_batch)
    
    Returns:
        DataFrame with predictions for all 424 targets
    """
    # Extract the test data from the batch
    test_batch = data_batch[0]
    
    print(f"🔮 Making predictions for batch with {len(test_batch)} samples")
    
    # Engineer features for test data
    engineered_test_data = target_manager.feature_engineer.engineer_features(test_batch)
    
    # Generate predictions for all 424 targets
    predictions = {}
    
    for i in range(424):
        target_name = f'target_{i}'
        try:
            # Use trained model for prediction
            pred_value = target_manager.predict_target(target_name, engineered_test_data)
            predictions[target_name] = pred_value
        except Exception as e:
            print(f"⚠️ Error predicting {target_name}: {str(e)}, using baseline")
            # Fallback to baseline prediction
            predictions[target_name] = np.random.normal(0, 0.01)
    
    # Convert to DataFrame (required format)
    pred_df = pd.DataFrame([predictions])
    
    print(f"✅ Generated predictions for {len(predictions)} targets")
    return pred_df

print("🔧 Prediction function defined")

# ============================================================================
# INFERENCE SERVER SETUP
# ============================================================================

# Create a proper implementation of InferenceServer
class MitsuiInferenceServer(InferenceServer):
    def _get_gateway_for_test(self, data_paths, file_share_dir=None, *args, **kwargs):
        """Return a gateway instance for testing."""
        return MitsuiGateway(data_paths)

# Create the inference server
print("🔧 Creating inference server...")
inference_server = MitsuiInferenceServer(predict)
print("✅ Inference server created successfully")

# Start the server
print("🚀 Starting inference server...")
inference_server.serve()

print("🎉 Inference server completed!")
print(f"⏰ Completed at: {datetime.now()}") 