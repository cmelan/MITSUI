import pandas as pd
import numpy as np
from typing import List, Optional
import itertools
import holidays
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --- Cross-Asset Features ---
def add_cross_asset_features(df: pd.DataFrame, asset_cols: List[str]) -> pd.DataFrame:
    """
    Add cross-asset spreads, ratios, and correlations between asset columns.
    """
    for a1, a2 in itertools.combinations(asset_cols, 2):
        df[f'{a1}_minus_{a2}'] = df[a1] - df[a2]
        df[f'{a1}_div_{a2}'] = df[a1] / (df[a2] + 1e-8)
        # Rolling correlation (window=7)
        df[f'{a1}_corr_{a2}_7'] = df[a1].rolling(7, min_periods=1).corr(df[a2])
    return df

# --- Calendar Features ---
def add_calendar_features(df: pd.DataFrame, date_col: str = 'date_id', country: str = 'JP') -> pd.DataFrame:
    """
    Add day of week, month, and holiday indicator features.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    # Holiday indicator
    jp_holidays = holidays.country_holidays(country)
    df['is_holiday'] = df[date_col].isin(jp_holidays).astype(int)
    return df

# --- Interaction Features ---
def add_interaction_features(df: pd.DataFrame, feature_cols: List[str], max_pairs: int = 20) -> pd.DataFrame:
    """
    Add pairwise product interaction features for top N features.
    """
    pairs = list(itertools.combinations(feature_cols[:max_pairs], 2))
    for f1, f2 in pairs:
        df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
    return df

# --- Z-score Features ---
def add_zscore_features(df: pd.DataFrame, feature_cols: List[str], window: int = 14) -> pd.DataFrame:
    """
    Add rolling Z-score features for each column.
    """
    for col in feature_cols:
        roll_mean = df[col].rolling(window, min_periods=1).mean()
        roll_std = df[col].rolling(window, min_periods=1).std()
        df[f'{col}_zscore_{window}'] = (df[col] - roll_mean) / (roll_std + 1e-8)
    return df

# --- Volatility Regime Detection ---
def add_volatility_features(df: pd.DataFrame, feature_cols: List[str], window: int = 21) -> pd.DataFrame:
    """
    Add volatility regime features and volatility clustering.
    """
    for col in feature_cols:
        # Rolling volatility
        df[f'{col}_volatility_{window}'] = df[col].rolling(window, min_periods=1).std()
        # Volatility of volatility
        df[f'{col}_vol_of_vol_{window}'] = df[f'{col}_volatility_{window}'].rolling(window//2, min_periods=1).std()
        # Volatility regime (high/low)
        vol_median = df[f'{col}_volatility_{window}'].median()
        df[f'{col}_high_vol_regime'] = (df[f'{col}_volatility_{window}'] > vol_median).astype(int)
    return df

# --- Advanced Time-Series Features ---
def add_advanced_ts_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Add advanced time-series features: momentum, mean reversion, trend strength.
    """
    for col in feature_cols:
        # Momentum features
        df[f'{col}_momentum_5'] = df[col] / df[col].shift(5) - 1
        df[f'{col}_momentum_10'] = df[col] / df[col].shift(10) - 1
        
        # Mean reversion features
        roll_mean_20 = df[col].rolling(20, min_periods=1).mean()
        df[f'{col}_mean_reversion'] = (df[col] - roll_mean_20) / roll_mean_20
        
        # Trend strength (ADX-like)
        high_20 = df[col].rolling(20, min_periods=1).max()
        low_20 = df[col].rolling(20, min_periods=1).min()
        df[f'{col}_trend_strength'] = (high_20 - low_20) / roll_mean_20
        
        # Price position within range
        df[f'{col}_price_position'] = (df[col] - low_20) / (high_20 - low_20 + 1e-8)
    return df

# --- Market Microstructure Features ---
def add_microstructure_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Add market microstructure features: bid-ask spreads, volume, liquidity.
    """
    for col in feature_cols:
        # Bid-ask spread proxy (using rolling min/max)
        roll_min = df[col].rolling(5, min_periods=1).min()
        roll_max = df[col].rolling(5, min_periods=1).max()
        df[f'{col}_spread_proxy'] = (roll_max - roll_min) / roll_min
        
        # Liquidity proxy (inverse of volatility)
        vol = df[col].rolling(10, min_periods=1).std()
        df[f'{col}_liquidity'] = 1 / (vol + 1e-8)
        
        # Price impact (change in price vs change in feature)
        df[f'{col}_price_impact'] = df[col].diff() / df[col].shift(1)
    return df

# --- Neural Network Meta-Model ---
def create_nn_meta_model(X: pd.DataFrame, y: pd.DataFrame, hidden_sizes: List[int] = [100, 50]) -> MLPRegressor:
    """
    Create a neural network meta-model for stacking.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    nn_model = MLPRegressor(
        hidden_layer_sizes=hidden_sizes,
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    return nn_model, scaler

# --- Automated Feature Selection ---
def shap_feature_importance(X: pd.DataFrame, y: pd.Series, model=None, max_features: int = 100) -> List[str]:
    """
    Use SHAP values to select top features.
    """
    if model is None:
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:max_features]
    return X.columns[top_idx].tolist()

def permutation_feature_importance(X: pd.DataFrame, y: pd.Series, model=None, max_features: int = 100) -> List[str]:
    """
    Use permutation importance to select top features.
    """
    if model is None:
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    result = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=1)
    top_idx = np.argsort(result.importances_mean)[::-1][:max_features]
    return X.columns[top_idx].tolist()

# --- Comprehensive Feature Engineering Pipeline ---
def create_comprehensive_features(df: pd.DataFrame, feature_cols: List[str], 
                                asset_cols: List[str] = None, date_col: str = 'date_id') -> pd.DataFrame:
    """
    Create all advanced features in one pipeline.
    """
    print("🔧 Creating comprehensive feature set...")
    
    # Basic features
    df = add_calendar_features(df, date_col)
    
    # Cross-asset features (if asset columns provided)
    if asset_cols:
        df = add_cross_asset_features(df, asset_cols)
    
    # Advanced time-series features
    df = add_advanced_ts_features(df, feature_cols)
    
    # Volatility features
    df = add_volatility_features(df, feature_cols)
    
    # Microstructure features
    df = add_microstructure_features(df, feature_cols)
    
    # Z-score features
    df = add_zscore_features(df, feature_cols)
    
    # Interaction features (limited for memory)
    df = add_interaction_features(df, feature_cols, max_pairs=15)
    
    # Fill missing values
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    print(f"✅ Created {df.shape[1]} features total")
    return df 