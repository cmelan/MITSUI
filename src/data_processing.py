import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from scipy.stats import skew, kurtosis


def load_data(
    train_path: str, 
    test_path: str, 
    train_labels_path: str, 
    target_pairs_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load main competition data files.
    Returns: train, test, train_labels, target_pairs DataFrames
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train_labels = pd.read_csv(train_labels_path)
    target_pairs = pd.read_csv(target_pairs_path)
    return train, test, train_labels, target_pairs


def fill_missing(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    Fill missing values in a DataFrame using the specified method.
    method: 'ffill', 'bfill', or 'mean'
    """
    if method == 'ffill':
        return df.fillna(method='ffill')
    elif method == 'bfill':
        return df.fillna(method='bfill')
    elif method == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    else:
        raise ValueError("Unknown fill method")


def add_lag_features(
    df: pd.DataFrame, 
    cols: List[str], 
    lags: List[int], 
    date_col: str = 'date_id'
) -> pd.DataFrame:
    """
    Add lag features for specified columns and lags.
    """
    df = df.sort_values(date_col)
    for col in cols:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df


def add_price_diff_features(
    df: pd.DataFrame, 
    cols: List[str], 
    date_col: str = 'date_id'
) -> pd.DataFrame:
    """
    Add price difference features (first difference) for specified columns.
    """
    df = df.sort_values(date_col)
    for col in cols:
        df[f'{col}_diff1'] = df[col].diff(1)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    cols: List[str],
    windows: List[int] = [3, 7, 14],
    date_col: str = 'date_id'
) -> pd.DataFrame:
    """
    Add rolling mean, std, min, max for specified columns and windows.
    """
    df = df.sort_values(date_col)
    for col in cols:
        for window in windows:
            df[f'{col}_rollmean{window}'] = df[col].rolling(window).mean()
            df[f'{col}_rollstd{window}'] = df[col].rolling(window).std()
            df[f'{col}_rollmin{window}'] = df[col].rolling(window).min()
            df[f'{col}_rollmax{window}'] = df[col].rolling(window).max()
    return df

def add_ewm_features(
    df: pd.DataFrame,
    cols: List[str],
    spans: List[int] = [3, 7, 14],
    date_col: str = 'date_id'
) -> pd.DataFrame:
    """
    Add exponentially weighted mean features for specified columns and spans.
    """
    df = df.sort_values(date_col)
    for col in cols:
        for span in spans:
            df[f'{col}_ewm{span}'] = df[col].ewm(span=span, adjust=False).mean()
    return df

def add_calendar_features(df: pd.DataFrame, date_col: str = 'date_id') -> pd.DataFrame:
    """
    Add calendar features: day of week, month.
    Assumes date_col is in YYYY-MM-DD or similar format.
    """
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    return df

def add_cross_asset_features(
    df: pd.DataFrame,
    asset_groups: List[List[str]],
    date_col: str = 'date_id'
) -> pd.DataFrame:
    """
    Add cross-asset features: spreads, ratios, rolling correlations between asset columns.
    asset_groups: list of lists, each sublist contains asset columns to compare.
    """
    df = df.sort_values(date_col)
    for group in asset_groups:
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                a, b = group[i], group[j]
                df[f'{a}_minus_{b}'] = df[a] - df[b]
                df[f'{a}_div_{b}'] = df[a] / (df[b] + 1e-9)
                # Rolling correlation
                df[f'{a}_corr_{b}_7'] = df[a].rolling(7).corr(df[b])
    return df

def add_extended_rolling_features(
    df: pd.DataFrame,
    cols: List[str],
    windows: List[int] = [3, 7, 14],
    date_col: str = 'date_id'
) -> pd.DataFrame:
    """
    Add rolling skew, kurtosis, volatility, and momentum for specified columns and windows.
    """
    df = df.sort_values(date_col)
    for col in cols:
        for window in windows:
            df[f'{col}_rollskew{window}'] = df[col].rolling(window).apply(skew, raw=True)
            df[f'{col}_rollkurt{window}'] = df[col].rolling(window).apply(kurtosis, raw=True)
            # Volatility (std)
            df[f'{col}_vol{window}'] = df[col].rolling(window).std()
            # Momentum (current - mean of window)
            df[f'{col}_momentum{window}'] = df[col] - df[col].rolling(window).mean()
    return df

def add_interaction_features(
    df: pd.DataFrame,
    cols: List[str],
    degree: int = 2
) -> pd.DataFrame:
    """
    Add interaction features (pairwise products) for specified columns.
    """
    from itertools import combinations
    for comb in combinations(cols, degree):
        name = '_x_'.join(comb)
        df[f'inter_{name}'] = df[comb[0]] * df[comb[1]]
    return df

# Example usage (uncomment for scripts or notebooks):
# train, test, train_labels, target_pairs = load_data(
#     'data/train.csv', 'data/test.csv', 'data/train_labels.csv', 'data/target_pairs.csv')
# train = fill_missing(train, method='ffill')
# time_series_cols = [col for col in train.columns if col != 'date_id']
# train = add_lag_features(train, time_series_cols, lags=[1,2,3])
# train = add_price_diff_features(train, time_series_cols) 
# asset_groups = [['LME_A', 'LME_B', 'LME_C'], ['JPX_X', 'JPX_Y']]
# train = add_cross_asset_features(train, asset_groups)
# train = add_extended_rolling_features(train, time_series_cols)
# train = add_interaction_features(train, time_series_cols) 