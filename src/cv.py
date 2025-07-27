import numpy as np
from typing import Iterator, Tuple


def time_series_cv_split(
    n_samples: int,
    n_splits: int = 5,
    test_size: int = 90,
    expanding: bool = True
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate train/validation indices for time-series cross-validation.
    - expanding: if True, use expanding window; else, use rolling window.
    Yields (train_idx, val_idx) for each split.
    """
    indices = np.arange(n_samples)
    split_starts = range(n_samples - n_splits * test_size, n_samples - test_size + 1, test_size)
    for start in split_starts:
        if expanding:
            train_idx = indices[:start]
        else:
            train_idx = indices[start - (n_splits * test_size):start]
        val_idx = indices[start:start + test_size]
        yield train_idx, val_idx

# Example usage:
# for train_idx, val_idx in time_series_cv_split(len(X), n_splits=5, test_size=90):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx] 