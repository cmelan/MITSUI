import optuna
import lightgbm as lgb
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr


def spearman_scorer(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation


def tune_lgbm(
    X_train, y_train, X_val, y_val,
    n_trials: int = 30,
    random_state: int = 42
):
    """
    Run Optuna hyperparameter tuning for LightGBM.
    Returns best params and best score.
    """
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': random_state,
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 16, 128),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        }
        model = lgb.LGBMRegressor(**params, n_estimators=100)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
        y_pred = model.predict(X_val)
        score = spearman_scorer(y_val, y_pred)
        return -score  # Optuna minimizes

    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_score = -study.best_value
    return best_params, best_score

# Example usage:
# best_params, best_score = tune_lgbm(X_train, y_train, X_val, y_val, n_trials=30) 