import numpy as np
import pandas as pd
from typing import List, Optional, Callable
from sklearn.base import RegressorMixin
from scipy.optimize import minimize


def simple_average(predictions: List[np.ndarray]) -> np.ndarray:
    """
    Average predictions from multiple models (equal weights).
    """
    return np.mean(predictions, axis=0)


def weighted_average(predictions: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Weighted average of predictions from multiple models.
    """
    if weights is None:
        weights = [1.0 / len(predictions)] * len(predictions)
    return np.average(predictions, axis=0, weights=weights)


def stacking(
    base_preds: List[np.ndarray],
    y_true: np.ndarray,
    meta_model: RegressorMixin
) -> np.ndarray:
    """
    Fit a meta-model (stacker) on base model predictions.
    Returns meta-model predictions.
    """
    X_meta = np.column_stack(base_preds)
    meta_model.fit(X_meta, y_true)
    return meta_model.predict(X_meta)


def optimize_weights(
    preds_list: List[np.ndarray],
    y_true: np.ndarray,
    scorer: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """
    Find optimal weights for ensembling by maximizing the scorer (e.g., Spearman correlation).
    Returns optimal weights (sum to 1).
    """
    n_models = len(preds_list)
    x0 = np.ones(n_models) / n_models
    bounds = [(0, 1)] * n_models
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    def loss(w):
        blended = np.sum([w[i] * preds_list[i] for i in range(n_models)], axis=0)
        return -scorer(y_true, blended)
    res = minimize(loss, x0, bounds=bounds, constraints=constraints)
    return res.x

def stacking_oof(
    base_models: List[RegressorMixin],
    meta_model: RegressorMixin,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: List[tuple]
) -> np.ndarray:
    """
    Out-of-fold stacking: train base models on each fold, use OOF preds as meta-model input.
    Returns meta-model predictions for X.
    """
    oof_preds = np.zeros((X.shape[0], len(base_models)))
    for i, model in enumerate(base_models):
        for train_idx, val_idx in cv_splits:
            model.fit(X[train_idx], y[train_idx])
            oof_preds[val_idx, i] = model.predict(X[val_idx])
    meta_model.fit(oof_preds, y)
    return meta_model.predict(oof_preds)

# Example usage:
# preds1 = model1.predict(X_val)
# preds2 = model2.predict(X_val)
# avg_preds = simple_average([preds1, preds2])
# weighted_preds = weighted_average([preds1, preds2], weights=[0.7, 0.3])
# from sklearn.linear_model import Ridge
# stack_preds = stacking([preds1, preds2], y_val, Ridge()) 
# weights = optimize_weights([preds1, preds2], y_val, lambda y, p: spearmanr(y, p).correlation)
# stack_preds = stacking_oof([model1, model2], Ridge(), X, y, list_of_cv_splits) 