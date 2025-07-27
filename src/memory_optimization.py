import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import gc
import psutil
import os


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage of a dataframe by setting data types.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage of dataframe is {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage after optimization is: {end_mem:.2f} MB')
        print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df


def chunk_processing(
    df: pd.DataFrame, 
    chunk_size: int = 10000,
    func: callable = None,
    **kwargs
) -> pd.DataFrame:
    """
    Process large dataframe in chunks to reduce memory usage.
    """
    if func is None:
        return df
    
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size].copy()
        processed_chunk = func(chunk, **kwargs)
        chunks.append(processed_chunk)
        
        # Force garbage collection
        del chunk
        gc.collect()
    
    return pd.concat(chunks, ignore_index=True)


def select_features_by_importance(
    X: pd.DataFrame,
    y: pd.DataFrame,
    max_features: int = 500,
    method: str = 'correlation'
) -> List[str]:
    """
    Select most important features to reduce dimensionality.
    """
    if method == 'correlation':
        # Calculate correlation with targets
        correlations = []
        for col in X.columns:
            if col.startswith('date_id'):
                continue
            corr_sum = 0
            for target_col in y.columns:
                corr = abs(X[col].corr(y[target_col]))
                if not np.isnan(corr):
                    corr_sum += corr
            correlations.append((col, corr_sum))
        
        # Sort by correlation and select top features
        correlations.sort(key=lambda x: x[1], reverse=True)
        selected_features = [col for col, _ in correlations[:max_features]]
        
    elif method == 'variance':
        # Select features with highest variance
        variances = X.var().sort_values(ascending=False)
        selected_features = variances.head(max_features).index.tolist()
    
    else:
        # Default: select first max_features
        selected_features = X.columns[:max_features].tolist()
    
    return selected_features


def optimize_target_processing(
    train_labels: pd.DataFrame,
    target_pairs: pd.DataFrame,
    max_targets: Optional[int] = None
) -> pd.DataFrame:
    """
    Optimize target processing by selecting most important targets.
    """
    if max_targets is None:
        return train_labels
    
    # Calculate target importance (e.g., variance, correlation with features)
    target_importance = []
    for col in train_labels.columns:
        if col.startswith('target_'):
            importance = train_labels[col].var()  # Use variance as importance
            target_importance.append((col, importance))
    
    # Select top targets
    target_importance.sort(key=lambda x: x[1], reverse=True)
    selected_targets = [col for col, _ in target_importance[:max_targets]]
    
    # Add date_id back
    if 'date_id' in train_labels.columns:
        selected_targets = ['date_id'] + selected_targets
    
    return train_labels[selected_targets]


def monitor_memory_usage():
    """
    Monitor current memory usage.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.2f} MB")
    return memory_mb


def create_memory_efficient_pipeline(
    train_df: pd.DataFrame,
    train_labels: pd.DataFrame,
    target_pairs: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    max_features: int = 500,
    max_targets: Optional[int] = None,
    chunk_size: int = 10000
) -> tuple:
    """
    Create a memory-efficient pipeline for the competition.
    """
    print("🔧 Starting memory-efficient pipeline...")
    monitor_memory_usage()
    
    # 1. Reduce memory usage of input data
    print("📊 Optimizing memory usage...")
    train_df = reduce_mem_usage(train_df)
    train_labels = reduce_mem_usage(train_labels)
    if test_df is not None:
        test_df = reduce_mem_usage(test_df)
    
    monitor_memory_usage()
    
    # 2. Select important features
    print(f"🎯 Selecting top {max_features} features...")
    feature_cols = [col for col in train_df.columns if col != 'date_id']
    selected_features = select_features_by_importance(
        train_df[feature_cols], 
        train_labels.drop('date_id', axis=1, errors='ignore'),
        max_features=max_features
    )
    
    train_df_selected = train_df[['date_id'] + selected_features]
    if test_df is not None:
        test_df_selected = test_df[['date_id'] + selected_features]
    else:
        test_df_selected = None
    
    monitor_memory_usage()
    
    # 3. Optimize targets
    if max_targets is not None:
        print(f"🎯 Selecting top {max_targets} targets...")
        train_labels = optimize_target_processing(train_labels, target_pairs, max_targets)
    
    monitor_memory_usage()
    
    # 4. Force garbage collection
    gc.collect()
    
    print("✅ Memory-efficient pipeline created!")
    return train_df_selected, train_labels, test_df_selected, selected_features


def batch_predict(
    model,
    X: pd.DataFrame,
    batch_size: int = 1000
) -> np.ndarray:
    """
    Make predictions in batches to reduce memory usage.
    """
    predictions = []
    
    for i in range(0, len(X), batch_size):
        batch = X.iloc[i:i+batch_size]
        batch_pred = model.predict(batch)
        predictions.append(batch_pred)
        
        # Clear batch from memory
        del batch
        gc.collect()
    
    return np.vstack(predictions) 