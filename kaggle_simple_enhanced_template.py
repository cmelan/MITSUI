# MITSUI&CO. Commodity Prediction Challenge - Simple Enhanced Template
# This template implements key improvements while being memory-efficient

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Essential ML Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

print("🚀 Starting MITSUI Commodity Prediction Challenge - Enhanced")
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
# SIMPLE FEATURE ENGINEERING
# ============================================================================

def create_basic_features(df):
    """Create basic technical indicators and features."""
    features = df.copy()
    
    # Get key price columns
    price_cols = [col for col in df.columns if 'Close' in col or 'close' in col]
    fx_cols = [col for col in df.columns if col.startswith('FX_')]
    
    # Create features for price columns
    for col in price_cols + fx_cols:
        if col in df.columns:
            # Basic price changes
            features[f'{col}_pct_change'] = df[col].pct_change()
            features[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
            
            # Simple moving averages
            features[f'{col}_ma_5'] = df[col].rolling(window=5).mean()
            features[f'{col}_ma_20'] = df[col].rolling(window=20).mean()
            
            # Price to MA ratios
            features[f'{col}_ma_5_ratio'] = df[col] / features[f'{col}_ma_5']
            features[f'{col}_ma_20_ratio'] = df[col] / features[f'{col}_ma_20']
    
    # Create cross-asset features
    if 'LME_AH_Close' in df.columns and 'LME_ZS_Close' in df.columns:
        features['LME_AH_ZS_spread'] = df['LME_AH_Close'] - df['LME_ZS_Close']
    
    if 'JPX_Gold_Standard_Futures_Close' in df.columns and 'JPX_Platinum_Standard_Futures_Close' in df.columns:
        features['Gold_Platinum_spread'] = df['JPX_Gold_Standard_Futures_Close'] - df['JPX_Platinum_Standard_Futures_Close']
    
    # Remove infinite and NaN values
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return features

# ============================================================================
# SIMPLE MODEL TRAINING
# ============================================================================

def train_simple_model(X, y):
    """Train a simple ensemble model."""
    # Create models
    models = [
        Ridge(alpha=1.0, random_state=42),
        RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42),
        xgb.XGBRegressor(n_estimators=50, max_depth=6, learning_rate=0.1, random_state=42)
    ]
    
    # Train models
    trained_models = []
    for model in models:
        model.fit(X, y)
        trained_models.append(model)
    
    return trained_models

def predict_ensemble(models, X):
    """Make ensemble predictions."""
    predictions = []
    for model in models:
        pred = model.predict(X)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred

# ============================================================================
# TARGET-SPECIFIC MODELING
# ============================================================================

class SimpleTargetManager:
    """Simple target-specific model manager."""
    
    def __init__(self, target_pairs):
        self.target_pairs = target_pairs
        self.target_models = {}
        self.is_trained = False
        
    def get_target_features(self, target_name, df):
        """Get relevant features for a specific target."""
        target_info = self.target_pairs[self.target_pairs['target'] == target_name].iloc[0]
        pair = target_info['pair']
        
        # Get relevant features based on the pair
        relevant_features = []
        
        if ' - ' in pair:
            # Spread target
            asset1, asset2 = pair.split(' - ')
            relevant_features.extend([col for col in df.columns if asset1 in col])
            relevant_features.extend([col for col in df.columns if asset2 in col])
        else:
            # Single asset target
            relevant_features.extend([col for col in df.columns if pair in col])
        
        # Add some general features
        relevant_features.extend([col for col in df.columns if 'FX_' in col][:10])  # Limit FX features
        relevant_features.extend([col for col in df.columns if 'LME_' in col][:10])  # Limit LME features
        
        # Remove duplicates and date_id
        relevant_features = list(set(relevant_features))
        if 'date_id' in relevant_features:
            relevant_features.remove('date_id')
        
        return relevant_features
    
    def train_all_models(self, train_data, train_labels):
        """Train models for all targets."""
        print("🏋️ Training simple models for all targets...")
        
        # Create features
        engineered_data = create_basic_features(train_data)
        
        # Train models for each target
        target_columns = [col for col in train_labels.columns if col.startswith('target_')]
        
        for i, target_name in enumerate(target_columns):
            if i % 50 == 0:  # Progress update every 50 targets
                print(f"  Training target {i+1}/{len(target_columns)}")
            
            try:
                # Get target values
                target_values = train_labels[target_name].values
                
                # Get relevant features
                relevant_features = self.get_target_features(target_name, engineered_data)
                
                if len(relevant_features) == 0:
                    # Use a subset of all features if no specific features found
                    relevant_features = [col for col in engineered_data.columns if col != 'date_id'][:50]
                
                # Prepare training data
                X = engineered_data[relevant_features].values
                y = target_values
                
                # Remove NaN values
                valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X = X[valid_mask]
                y = y[valid_mask]
                
                if len(X) > 100:  # Only train if we have enough data
                    # Train model
                    models = train_simple_model(X, y)
                    
                    # Store model and features
                    self.target_models[target_name] = {
                        'models': models,
                        'features': relevant_features
                    }
                else:
                    self.target_models[target_name] = None
                    
            except Exception as e:
                print(f"⚠️ Error training {target_name}: {str(e)}")
                self.target_models[target_name] = None
        
        self.is_trained = True
        print(f"✅ Training completed for {len(self.target_models)} targets")
    
    def predict_target(self, target_name, test_data):
        """Predict for a specific target."""
        if target_name not in self.target_models or self.target_models[target_name] is None:
            # Return baseline prediction
            return np.random.normal(0, 0.01)
        
        model_info = self.target_models[target_name]
        models = model_info['models']
        features = model_info['features']
        
        # Get features from test data
        available_features = [f for f in features if f in test_data.columns]
        
        if len(available_features) == 0:
            return np.random.normal(0, 0.01)
        
        X = test_data[available_features].values
        
        # Handle NaN values
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)
        
        # Make prediction
        prediction = predict_ensemble(models, X.reshape(1, -1))[0]
        
        return prediction

# ============================================================================
# MAIN PREDICTION FUNCTION
# ============================================================================

# Initialize the target manager
print("🔧 Initializing simple target manager...")
target_manager = SimpleTargetManager(target_pairs)

# Train models on historical data
print("📚 Training models on historical data...")
target_manager.train_all_models(train_data, train_labels)

def predict(data_batch):
    """
    Generate predictions for all 424 targets for a given data batch.
    Uses simple but effective ML models with basic feature engineering.
    
    Args:
        data_batch: Tuple containing (test_batch, label_lags_1_batch, label_lags_2_batch, label_lags_3_batch, label_lags_4_batch)
    
    Returns:
        DataFrame with predictions for all 424 targets
    """
    # Extract the test data from the batch
    test_batch = data_batch[0]
    
    print(f"🔮 Making predictions for batch with {len(test_batch)} samples")
    
    # Create features for test data
    engineered_test_data = create_basic_features(test_batch)
    
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