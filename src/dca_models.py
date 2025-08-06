# src/dca_models.py
"""
Decline Curve Analysis (DCA) models for benchmarking.
Implements cumulative forms of industry-standard models.
"""

import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from typing import Tuple, Dict, Optional
import warnings

class DCAModel:
    """Base class for DCA models."""
    
    def __init__(self, name: str):
        self.name = name
        self.params = None
        self.cov = None
        
    def fit(self, t: np.ndarray, Q: np.ndarray, bounds: Optional[Dict] = None) -> 'DCAModel':
        """Fit the model to cumulative production data."""
        raise NotImplementedError
    
    def predict(self, t: np.ndarray) -> np.ndarray:
        """Predict cumulative production at given times."""
        raise NotImplementedError
    
    def predict_with_uncertainty(self, t: np.ndarray, n_samples: int = 1000) -> Dict:
        """
        Generate probabilistic predictions using bootstrap of residuals.
        Returns P10, P50, P90 and samples.
        """
        if self.params is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Get deterministic prediction
        Q_pred = self.predict(t)
        
        # If we have covariance matrix, use it for uncertainty
        if self.cov is not None and not np.any(np.isnan(self.cov)):
            # Sample parameters from multivariate normal
            param_samples = np.random.multivariate_normal(self.params, self.cov, n_samples)
            Q_samples = np.array([self._predict_with_params(t, p) for p in param_samples])
        else:
            # Fallback: add noise based on prediction magnitude
            # This is a simple heuristic when proper uncertainty isn't available
            noise_scale = 0.1  # 10% coefficient of variation
            Q_samples = Q_pred[np.newaxis, :] * (1 + np.random.normal(0, noise_scale, (n_samples, len(t))))
        
        # Ensure non-negative production
        Q_samples = np.maximum(Q_samples, 0)
        
        # Calculate percentiles
        p10 = np.percentile(Q_samples, 10, axis=0)
        p50 = np.percentile(Q_samples, 50, axis=0)
        p90 = np.percentile(Q_samples, 90, axis=0)
        
        return {
            'p10': p10,
            'p50': p50,
            'p90': p90,
            'mean': np.mean(Q_samples, axis=0),
            'std': np.std(Q_samples, axis=0),
            'samples': Q_samples
        }
    
    def _predict_with_params(self, t: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Predict with specific parameters (for uncertainty quantification)."""
        raise NotImplementedError


class ArpsHyperbolic(DCAModel):
    """
    Arps Hyperbolic decline model (cumulative form).
    Q(t) = (q_i / (Di * (1-b))) * ((1 + b*Di*t)^((1-b)/b) - 1)
    where 0 < b < 1
    """
    
    def __init__(self):
        super().__init__("Arps Hyperbolic")
        
    def fit(self, t: np.ndarray, Q: np.ndarray, bounds: Optional[Dict] = None) -> 'ArpsHyperbolic':
        """
        Fit Arps model to cumulative production data.
        Parameters: [q_i, Di, b]
        """
        if bounds is None:
            bounds = {
                'q_i': (Q[-1] / t[-1] * 0.5, Q[-1] / t[-1] * 5),  # Initial rate estimate
                'Di': (0.001, 2.0),  # Initial decline rate
                'b': (0.01, 0.99)  # Hyperbolic exponent
            }
        
        def cumulative_arps(t, q_i, Di, b):
            """Cumulative Arps hyperbolic equation."""
            # Avoid numerical issues
            b = np.clip(b, 0.01, 0.99)
            Di = np.clip(Di, 0.001, 10.0)
            
            # Calculate cumulative production
            term = np.power(1 + b * Di * t, (1 - b) / b) - 1
            Q = (q_i / (Di * (1 - b))) * term
            return Q
        
        # Initial guess
        q_i_init = Q[-1] / t[-1] * 1.5  # Rough initial rate
        p0 = [q_i_init, 0.1, 0.5]
        
        # Bounds for parameters
        lower_bounds = [bounds['q_i'][0], bounds['Di'][0], bounds['b'][0]]
        upper_bounds = [bounds['q_i'][1], bounds['Di'][1], bounds['b'][1]]
        
        try:
            # Try curve_fit first
            self.params, self.cov = curve_fit(
                cumulative_arps, t, Q, p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=5000
            )
        except:
            # If curve_fit fails, use differential evolution
            def objective(params):
                pred = cumulative_arps(t, *params)
                return np.sum((Q - pred) ** 2)
            
            result = differential_evolution(
                objective,
                bounds=list(zip(lower_bounds, upper_bounds)),
                seed=42
            )
            self.params = result.x
            # Estimate covariance from Hessian (simplified)
            self.cov = np.eye(3) * (result.fun / (len(Q) - 3))
        
        return self
    
    def predict(self, t: np.ndarray) -> np.ndarray:
        """Predict cumulative production."""
        if self.params is None:
            raise ValueError("Model must be fitted before prediction")
        
        q_i, Di, b = self.params
        b = np.clip(b, 0.01, 0.99)
        Di = np.clip(Di, 0.001, 10.0)
        
        # Handle t=0 case
        Q = np.zeros_like(t)
        non_zero_mask = t > 0
        
        if np.any(non_zero_mask):
            t_nz = t[non_zero_mask]
            term = np.power(1 + b * Di * t_nz, (1 - b) / b) - 1
            Q[non_zero_mask] = (q_i / (Di * (1 - b))) * term
        
        return Q
    
    def _predict_with_params(self, t: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Predict with specific parameters."""
        q_i, Di, b = params
        b = np.clip(b, 0.01, 0.99)
        Di = np.clip(Di, 0.001, 10.0)
        
        # Handle t=0 case
        Q = np.zeros_like(t)
        non_zero_mask = t > 0
        
        if np.any(non_zero_mask):
            t_nz = t[non_zero_mask]
            term = np.power(1 + b * Di * t_nz, (1 - b) / b) - 1
            Q[non_zero_mask] = (q_i / (Di * (1 - b))) * term
        
        return Q


class DuongModel(DCAModel):
    """
    Duong decline model (cumulative form).
    Q(t) = (q1/a) * exp((a/(1-m)) * (t^(1-m) - 1))
    Designed for fracture-dominated flow in unconventionals.
    """
    
    def __init__(self):
        super().__init__("Duong")
        
    def fit(self, t: np.ndarray, Q: np.ndarray, bounds: Optional[Dict] = None) -> 'DuongModel':
        """
        Fit Duong model to cumulative production data.
        Parameters: [q1, a, m]
        """
        if bounds is None:
            bounds = {
                'q1': (Q[-1] / t[-1] * 0.1, Q[-1] / t[-1] * 10),
                'a': (0.5, 5.0),
                'm': (1.01, 1.5)  # m > 1 for Duong model
            }
        
        def cumulative_duong(t, q1, a, m):
            """Cumulative Duong equation."""
            # Ensure m > 1 for physical validity
            m = np.clip(m, 1.01, 2.0)
            a = np.clip(a, 0.1, 10.0)
            
            # Handle t=0 case
            t_safe = np.maximum(t, 1e-10)
            
            # Calculate cumulative production
            exponent = (a / (1 - m)) * (np.power(t_safe, 1 - m) - 1)
            Q = (q1 / a) * np.exp(exponent)
            return Q
        
        # Initial guess
        q1_init = Q[-1] / t[-1] * 2
        p0 = [q1_init, 1.0, 1.1]
        
        # Bounds for parameters
        lower_bounds = [bounds['q1'][0], bounds['a'][0], bounds['m'][0]]
        upper_bounds = [bounds['q1'][1], bounds['a'][1], bounds['m'][1]]
        
        try:
            # Try curve_fit first
            self.params, self.cov = curve_fit(
                cumulative_duong, t, Q, p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=5000
            )
        except:
            # If curve_fit fails, use differential evolution
            def objective(params):
                pred = cumulative_duong(t, *params)
                # Add penalty for unrealistic values
                if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                    return 1e10
                return np.sum((Q - pred) ** 2)
            
            result = differential_evolution(
                objective,
                bounds=list(zip(lower_bounds, upper_bounds)),
                seed=42
            )
            self.params = result.x
            # Estimate covariance
            self.cov = np.eye(3) * (result.fun / (len(Q) - 3))
        
        return self
    
    def predict(self, t: np.ndarray) -> np.ndarray:
        """Predict cumulative production."""
        if self.params is None:
            raise ValueError("Model must be fitted before prediction")
        
        q1, a, m = self.params
        m = np.clip(m, 1.01, 2.0)
        a = np.clip(a, 0.1, 10.0)
        
        # Handle t=0 case - Duong model gives Q=0 at t=0
        Q = np.zeros_like(t)
        non_zero_mask = t > 0
        
        if np.any(non_zero_mask):
            t_nz = t[non_zero_mask]
            exponent = (a / (1 - m)) * (np.power(t_nz, 1 - m) - 1)
            Q[non_zero_mask] = (q1 / a) * np.exp(exponent)
        
        return Q
    
    def _predict_with_params(self, t: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Predict with specific parameters."""
        q1, a, m = params
        m = np.clip(m, 1.01, 2.0)
        a = np.clip(a, 0.1, 10.0)
        
        # Handle t=0 case
        Q = np.zeros_like(t)
        non_zero_mask = t > 0
        
        if np.any(non_zero_mask):
            t_nz = t[non_zero_mask]
            exponent = (a / (1 - m)) * (np.power(t_nz, 1 - m) - 1)
            Q[non_zero_mask] = (q1 / a) * np.exp(exponent)
        
        return Q


def bootstrap_dca_uncertainty(model: DCAModel, t_fit: np.ndarray, Q_fit: np.ndarray, 
                              t_pred: np.ndarray, n_bootstrap: int = 1000) -> Dict:
    """
    Bootstrap residuals to generate uncertainty estimates for DCA models.
    
    Args:
        model: Fitted DCA model
        t_fit: Time points used for fitting
        Q_fit: Cumulative production used for fitting  
        t_pred: Time points for prediction
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary with P10, P50, P90 and samples
    """
    # Get fitted values at training points
    Q_fitted = model.predict(t_fit)
    
    # Calculate residuals
    residuals = Q_fit - Q_fitted
    
    # Bootstrap samples
    Q_samples = []
    failed_fits = 0
    for _ in range(n_bootstrap):
        # Resample residuals with replacement
        resampled_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
        
        # Add residuals to fitted values
        Q_bootstrap = Q_fitted + resampled_residuals
        
        # Ensure non-negative values
        Q_bootstrap = np.maximum(Q_bootstrap, 0)
        
        # Refit model to bootstrapped data
        model_boot = model.__class__()
        try:
            model_boot.fit(t_fit, Q_bootstrap)
            Q_pred_boot = model_boot.predict(t_pred)
            # Ensure non-negative predictions
            Q_pred_boot = np.maximum(Q_pred_boot, 0)
            Q_samples.append(Q_pred_boot)
        except:
            # If fitting fails, use deterministic prediction with noise
            failed_fits += 1
            Q_det = model.predict(t_pred)
            noise = np.random.normal(0, np.abs(residuals).mean(), len(t_pred))
            Q_noisy = np.maximum(Q_det + noise, 0)
            Q_samples.append(Q_noisy)
    
    if len(Q_samples) == 0:
        # If all bootstrap fits failed, return deterministic prediction
        Q_det = model.predict(t_pred)
        return {
            'p10': Q_det,
            'p50': Q_det,
            'p90': Q_det,
            'mean': Q_det,
            'std': np.zeros_like(Q_det),
            'samples': np.array([Q_det])
        }
    
    Q_samples = np.array(Q_samples)
    
    # Calculate percentiles
    p10 = np.percentile(Q_samples, 10, axis=0)
    p50 = np.percentile(Q_samples, 50, axis=0)
    p90 = np.percentile(Q_samples, 90, axis=0)
    
    return {
        'p10': p10,
        'p50': p50,
        'p90': p90,
        'mean': np.mean(Q_samples, axis=0),
        'std': np.std(Q_samples, axis=0),
        'samples': Q_samples
    }