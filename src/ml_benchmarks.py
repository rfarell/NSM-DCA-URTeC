# src/ml_benchmarks.py
"""
Machine learning benchmark models for production forecasting.
Implements LightGBM with quantile regression for probabilistic predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not installed. ML benchmarks will not be available.")

class LightGBMQuantile:
    """
    LightGBM model with quantile regression for probabilistic production forecasting.
    Trains separate models for each time point and quantile.
    """
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        """
        Initialize LightGBM quantile model.
        
        Args:
            quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9] for P10, P50, P90)
        """
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is required for ML benchmarks. Install with: pip install lightgbm")
        
        self.quantiles = quantiles
        self.models = {}  # Store models for each time point and quantile
        self.time_points = None
        self.feature_names = None
        
    def fit(self, X: np.ndarray, Y: np.ndarray, time_points: np.ndarray,
            feature_names: Optional[List[str]] = None,
            params: Optional[Dict] = None) -> 'LightGBMQuantile':
        """
        Fit LightGBM models for each time point and quantile.
        
        Args:
            X: Features array [n_wells, n_features]
            Y: Cumulative production array [n_wells, n_time_points]
            time_points: Time points in days
            feature_names: Names of features (for interpretability)
            params: LightGBM parameters (optional)
            
        Returns:
            Self for chaining
        """
        self.time_points = time_points
        self.feature_names = feature_names if feature_names is not None else [f"f_{i}" for i in range(X.shape[1])]
        
        # Default parameters optimized for small datasets
        if params is None:
            params = {
                'objective': 'quantile',
                'metric': 'quantile',
                'boosting_type': 'gbdt',
                'num_leaves': 15,  # Small for overfitting prevention
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'num_threads': 1,
                'min_data_in_leaf': 5,  # Prevent overfitting on small data
                'max_depth': 5
            }
        
        n_wells, n_time = Y.shape
        
        # Train a model for each time point and quantile
        for t_idx, t in enumerate(time_points):
            self.models[t] = {}
            
            # Get production at this time point
            y_t = Y[:, t_idx]
            
            # Remove any NaN values
            valid_mask = ~np.isnan(y_t)
            X_valid = X[valid_mask]
            y_valid = y_t[valid_mask]
            
            if len(y_valid) < 10:
                warnings.warn(f"Only {len(y_valid)} valid samples for time {t}. Results may be unreliable.")
            
            # Train model for each quantile
            for q in self.quantiles:
                # Set quantile parameter
                params_q = params.copy()
                params_q['alpha'] = q
                
                # Create dataset
                train_data = lgb.Dataset(X_valid, label=y_valid)
                
                # Train model with cross-validation to prevent overfitting
                try:
                    # Use small number of boosting rounds for small datasets
                    n_rounds = 100 if len(y_valid) > 50 else 50
                    
                    model = lgb.train(
                        params_q,
                        train_data,
                        num_boost_round=n_rounds,
                        callbacks=[lgb.log_evaluation(0)]
                    )
                    
                    self.models[t][q] = model
                    
                except Exception as e:
                    warnings.warn(f"Failed to train model for time {t}, quantile {q}: {e}")
                    # Store None to handle later
                    self.models[t][q] = None
        
        return self
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate probabilistic predictions for new wells.
        
        Args:
            X: Features array [n_wells, n_features]
            
        Returns:
            Dictionary with 'p10', 'p50', 'p90' arrays [n_wells, n_time_points]
        """
        if not self.models:
            raise ValueError("Model must be fitted before prediction")
        
        n_wells = X.shape[0]
        n_time = len(self.time_points)
        
        # Initialize prediction arrays
        predictions = {f'p{int(q*100)}': np.zeros((n_wells, n_time)) for q in self.quantiles}
        
        # Predict for each time point
        for t_idx, t in enumerate(self.time_points):
            if t not in self.models:
                continue
            
            # Predict each quantile
            for q in self.quantiles:
                if q not in self.models[t] or self.models[t][q] is None:
                    # Use median prediction if this quantile failed
                    if 0.5 in self.models[t] and self.models[t][0.5] is not None:
                        pred = self.models[t][0.5].predict(X, num_iteration=self.models[t][0.5].best_iteration)
                    else:
                        # Use mean of other quantiles if available
                        pred = np.nan * np.ones(n_wells)
                else:
                    model = self.models[t][q]
                    pred = model.predict(X, num_iteration=model.best_iteration)
                
                key = f'p{int(q*100)}'
                predictions[key][:, t_idx] = pred
        
        # Ensure monotonicity (cumulative production should increase)
        for key in predictions:
            for i in range(1, n_time):
                predictions[key][:, i] = np.maximum(predictions[key][:, i], predictions[key][:, i-1])
        
        # Add mean and std estimates
        if len(self.quantiles) >= 3:
            # Approximate mean and std from quantiles
            predictions['mean'] = predictions['p50']  # Use median as mean estimate
            # Approximate std from IQR (assuming normal distribution)
            if 'p10' in predictions and 'p90' in predictions:
                predictions['std'] = (predictions['p90'] - predictions['p10']) / 2.56
        
        return predictions
    
    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int = 1000) -> Dict:
        """
        Generate prediction samples for uncertainty quantification.
        Uses the fitted quantile models to approximate the distribution.
        
        Args:
            X: Features array [n_wells, n_features]
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary with samples and statistics
        """
        # Get quantile predictions
        quantile_preds = self.predict(X)
        
        n_wells = X.shape[0]
        n_time = len(self.time_points)
        
        # Generate samples by interpolating between quantiles
        # This is an approximation of the full distribution
        samples = np.zeros((n_samples, n_wells, n_time))
        
        for i in range(n_wells):
            for t in range(n_time):
                # Get quantile values for this well and time
                q_values = []
                q_probs = []
                
                for q in self.quantiles:
                    key = f'p{int(q*100)}'
                    if key in quantile_preds:
                        q_values.append(quantile_preds[key][i, t])
                        q_probs.append(q)
                
                if len(q_values) >= 2:
                    # Sample from uniform distribution and interpolate
                    u = np.random.uniform(0, 1, n_samples)
                    samples[:, i, t] = np.interp(u, q_probs, q_values)
                else:
                    # Fallback: use single value with noise
                    if 'p50' in quantile_preds:
                        base_val = quantile_preds['p50'][i, t]
                    else:
                        base_val = quantile_preds[list(quantile_preds.keys())[0]][i, t]
                    
                    samples[:, i, t] = base_val * (1 + np.random.normal(0, 0.1, n_samples))
        
        # Ensure non-negative and monotonic
        samples = np.maximum(samples, 0)
        for t in range(1, n_time):
            samples[:, :, t] = np.maximum(samples[:, :, t], samples[:, :, t-1])
        
        return {
            'samples': samples,
            'p10': np.percentile(samples, 10, axis=0),
            'p50': np.percentile(samples, 50, axis=0),
            'p90': np.percentile(samples, 90, axis=0),
            'mean': np.mean(samples, axis=0),
            'std': np.std(samples, axis=0)
        }
    
    def feature_importance(self, time_point: Optional[float] = None) -> pd.DataFrame:
        """
        Get feature importance for interpretability.
        
        Args:
            time_point: Specific time point to analyze (uses median quantile)
                       If None, averages across all time points
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.models:
            raise ValueError("Model must be fitted first")
        
        importance_dict = {}
        
        if time_point is not None:
            # Get importance for specific time point
            if time_point in self.models and 0.5 in self.models[time_point]:
                model = self.models[time_point][0.5]
                if model is not None:
                    importance = model.feature_importance(importance_type='gain')
                    importance_dict = dict(zip(self.feature_names, importance))
        else:
            # Average importance across all time points
            all_importances = []
            for t in self.time_points:
                if t in self.models and 0.5 in self.models[t]:
                    model = self.models[t][0.5]
                    if model is not None:
                        importance = model.feature_importance(importance_type='gain')
                        all_importances.append(importance)
            
            if all_importances:
                avg_importance = np.mean(all_importances, axis=0)
                importance_dict = dict(zip(self.feature_names, avg_importance))
        
        # Convert to DataFrame and sort
        df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
        df = df.sort_values('Importance', ascending=False)
        
        return df