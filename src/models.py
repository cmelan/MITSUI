from typing import Any, Dict
import numpy as np

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

class LGBMWrapper:
    def __init__(self, params: Dict[str, Any]):
        self.model = lgb.LGBMRegressor(**params)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)

class XGBWrapper:
    def __init__(self, params: Dict[str, Any]):
        self.model = xgb.XGBRegressor(**params, use_label_encoder=False, eval_metric='rmse')
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)

class CatBoostWrapper:
    def __init__(self, params: Dict[str, Any]):
        self.model = cb.CatBoostRegressor(**params, verbose=0)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)

class RFWrapper:
    def __init__(self, params: Dict[str, Any]):
        self.model = RandomForestRegressor(**params)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)

class RidgeWrapper:
    def __init__(self, params: Dict[str, Any]):
        self.model = Ridge(**params)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)

def get_model(name: str, params: Dict[str, Any]):
    """
    Instantiate a model wrapper by name.
    Supported: 'lgbm', 'xgb', 'catboost', 'rf', 'ridge'
    """
    if name == 'lgbm':
        return LGBMWrapper(params)
    elif name == 'xgb':
        return XGBWrapper(params)
    elif name == 'catboost':
        return CatBoostWrapper(params)
    elif name == 'rf':
        return RFWrapper(params)
    elif name == 'ridge':
        return RidgeWrapper(params)
    else:
        raise ValueError(f'Unknown model name: {name}')

# Example usage:
# model = get_model('lgbm', {'n_estimators': 100})
# model.fit(X_train, y_train)
# y_pred = model.predict(X_val) 