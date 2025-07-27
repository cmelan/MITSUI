import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')


def time_series_cv_robust(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int = 5,
    test_size: int = 90,
    gap: int = 30
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Robust time-series cross-validation with gap to prevent data leakage.
    """
    splits = []
    n_samples = len(X)
    
    for i in range(n_splits):
        # Calculate split indices
        test_start = n_samples - (n_splits - i) * test_size
        test_end = test_start + test_size
        
        # Add gap between train and test
        train_end = test_start - gap
        
        if train_end > 0:
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((
                X.iloc[train_indices], 
                y.iloc[train_indices],
                X.iloc[test_indices], 
                y.iloc[test_indices]
            ))
    
    return splits


def evaluate_stability(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    target_configs: Dict
) -> Dict:
    """
    Evaluate model stability across different time periods.
    """
    results = {}
    
    # Calculate Spearman correlation for each target
    correlations = []
    for col in y_true.columns:
        if col in y_pred.columns:
            mask = ~(y_true[col].isna() | y_pred[col].isna())
            if mask.sum() > 10:
                corr, _ = spearmanr(y_true[col][mask], y_pred[col][mask])
                if not np.isnan(corr):
                    correlations.append(corr)
    
    if correlations:
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        
        # Competition metric: mean / std (Sharpe ratio variant)
        competition_score = mean_corr / (std_corr + 1e-8)
        
        # Stability metrics
        cv_score = mean_corr / (std_corr + 1e-8)  # Coefficient of variation
        
        results = {
            'mean_spearman': mean_corr,
            'std_spearman': std_corr,
            'competition_score': competition_score,
            'stability_cv': cv_score,
            'num_targets': len(correlations),
            'target_correlations': correlations
        }
    
    return results


def detect_overfitting(
    train_scores: List[float],
    val_scores: List[float],
    threshold: float = 0.1
) -> Dict:
    """
    Detect overfitting by comparing train and validation scores.
    """
    train_mean = np.mean(train_scores)
    val_mean = np.mean(val_scores)
    train_std = np.std(train_scores)
    val_std = np.std(val_scores)
    
    # Calculate overfitting metrics
    score_gap = train_mean - val_mean
    gap_ratio = score_gap / (val_mean + 1e-8)
    
    # Stability comparison
    stability_gap = train_std - val_std
    
    is_overfitting = gap_ratio > threshold
    
    return {
        'train_mean': train_mean,
        'val_mean': val_mean,
        'score_gap': score_gap,
        'gap_ratio': gap_ratio,
        'train_std': train_std,
        'val_std': val_std,
        'stability_gap': stability_gap,
        'is_overfitting': is_overfitting,
        'severity': 'High' if gap_ratio > 0.2 else 'Medium' if gap_ratio > 0.1 else 'Low'
    }


def robust_model_selection(
    models: Dict,
    X: pd.DataFrame,
    y: pd.DataFrame,
    cv_splits: List[Tuple],
    target_configs: Dict
) -> Tuple[str, Dict]:
    """
    Select the most robust model based on stability and generalization.
    """
    model_results = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        
        train_scores = []
        val_scores = []
        val_predictions = []
        
        for i, (X_train, y_train, X_val, y_val) in enumerate(cv_splits):
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            
            # Evaluate
            train_eval = evaluate_stability(y_train, pd.DataFrame(y_pred_train, columns=y_train.columns), target_configs)
            val_eval = evaluate_stability(y_val, pd.DataFrame(y_pred_val, columns=y_val.columns), target_configs)
            
            if train_eval and val_eval:
                train_scores.append(train_eval['competition_score'])
                val_scores.append(val_eval['competition_score'])
                val_predictions.append(y_pred_val)
        
        if train_scores and val_scores:
            # Check for overfitting
            overfitting_analysis = detect_overfitting(train_scores, val_scores)
            
            model_results[model_name] = {
                'train_scores': train_scores,
                'val_scores': val_scores,
                'train_mean': np.mean(train_scores),
                'val_mean': np.mean(val_scores),
                'train_std': np.std(train_scores),
                'val_std': np.std(val_scores),
                'overfitting': overfitting_analysis,
                'val_predictions': val_predictions
            }
    
    # Select best model based on validation score and stability
    best_model = None
    best_score = -np.inf
    
    for model_name, results in model_results.items():
        # Penalize overfitting
        overfitting_penalty = 0
        if results['overfitting']['is_overfitting']:
            overfitting_penalty = results['overfitting']['gap_ratio'] * 0.5
        
        # Final score: validation score - overfitting penalty
        final_score = results['val_mean'] - overfitting_penalty
        
        if final_score > best_score:
            best_score = final_score
            best_model = model_name
    
    return best_model, model_results


def cross_validate_stability(
    model,
    X: pd.DataFrame,
    y: pd.DataFrame,
    target_configs: Dict,
    n_splits: int = 5
) -> Dict:
    """
    Comprehensive cross-validation focusing on stability.
    """
    cv_splits = time_series_cv_robust(X, y, n_splits=n_splits)
    
    all_scores = []
    all_predictions = []
    
    for i, (X_train, y_train, X_val, y_val) in enumerate(cv_splits):
        print(f"Fold {i+1}/{len(cv_splits)}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_val)
        
        # Evaluate
        eval_results = evaluate_stability(
            y_val, 
            pd.DataFrame(y_pred, columns=y_val.columns), 
            target_configs
        )
        
        if eval_results:
            all_scores.append(eval_results['competition_score'])
            all_predictions.append(y_pred)
    
    if all_scores:
        return {
            'mean_score': np.mean(all_scores),
            'std_score': np.std(all_scores),
            'cv_score': np.mean(all_scores) / (np.std(all_scores) + 1e-8),
            'scores': all_scores,
            'predictions': all_predictions,
            'stability': 1 / (np.std(all_scores) + 1e-8)  # Higher is more stable
        }
    
    return {}


def prevent_overfitting_recommendations(
    overfitting_analysis: Dict
) -> List[str]:
    """
    Generate recommendations to prevent overfitting.
    """
    recommendations = []
    
    if overfitting_analysis['is_overfitting']:
        severity = overfitting_analysis['severity']
        gap_ratio = overfitting_analysis['gap_ratio']
        
        recommendations.append(f"⚠️  {severity} overfitting detected (gap ratio: {gap_ratio:.3f})")
        
        if gap_ratio > 0.2:
            recommendations.extend([
                "🔧 Increase regularization (alpha, lambda)",
                "🔧 Reduce model complexity (max_depth, num_leaves)",
                "🔧 Use early stopping",
                "🔧 Increase minimum samples per leaf",
                "🔧 Add dropout or feature sampling"
            ])
        elif gap_ratio > 0.1:
            recommendations.extend([
                "🔧 Moderate regularization increase",
                "🔧 Reduce feature set",
                "🔧 Use cross-validation for hyperparameter tuning"
            ])
        else:
            recommendations.extend([
                "🔧 Slight regularization adjustment",
                "🔧 Monitor validation performance"
            ])
    else:
        recommendations.append("✅ No significant overfitting detected")
    
    return recommendations 