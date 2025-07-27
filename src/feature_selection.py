import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.base import RegressorMixin
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
import shap


def shap_feature_importance(model: RegressorMixin, X: pd.DataFrame) -> pd.Series:
    """
    Compute SHAP feature importances for a fitted model.
    Returns a Series indexed by feature name.
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    importances = np.abs(shap_values.values).mean(axis=0)
    return pd.Series(importances, index=X.columns).sort_values(ascending=False)


def permutation_feature_importance(model: RegressorMixin, X: pd.DataFrame, y: np.ndarray, scoring: str = 'neg_mean_squared_error') -> pd.Series:
    """
    Compute permutation feature importances for a fitted model.
    Returns a Series indexed by feature name.
    """
    result = permutation_importance(model, X, y, scoring=scoring, n_repeats=5, random_state=42)
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)


def recursive_feature_elimination(model: RegressorMixin, X: pd.DataFrame, y: np.ndarray, n_features_to_select: int = 20) -> List[str]:
    """
    Perform recursive feature elimination (RFE) to select top features.
    Returns a list of selected feature names.
    """
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)
    return list(X.columns[rfe.support_])


def correlation_filter(X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """
    Remove features with correlation above the threshold.
    Returns a list of selected feature names.
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return [col for col in X.columns if col not in to_drop]

# Example usage:
# import lightgbm as lgb
# model = lgb.LGBMRegressor().fit(X, y)
# shap_imp = shap_feature_importance(model, X)
# perm_imp = permutation_feature_importance(model, X, y)
# top_features = recursive_feature_elimination(model, X, y, n_features_to_select=30)
# filtered_features = correlation_filter(X, threshold=0.9) 