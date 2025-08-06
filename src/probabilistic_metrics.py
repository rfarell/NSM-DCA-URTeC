# src/probabilistic_metrics.py
"""
Probabilistic metrics for evaluating forecast quality and calibration.
Includes CRPS, sharpness, and calibration metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def crps_empirical(y_true: np.ndarray, y_samples: np.ndarray) -> float:
    """
    Calculate the Continuous Ranked Probability Score (CRPS) using empirical samples.
    
    CRPS measures both calibration and sharpness of probabilistic forecasts.
    Lower values are better.
    
    Args:
        y_true: Observed values [n_observations]
        y_samples: Forecast samples [n_samples, n_observations]
        
    Returns:
        Average CRPS across observations
    """
    n_samples = y_samples.shape[0]
    
    # Sort samples for each observation
    y_samples_sorted = np.sort(y_samples, axis=0)
    
    crps_values = []
    for i in range(y_true.shape[0]):
        y_obs = y_true[i]
        y_fcst = y_samples_sorted[:, i]
        
        # Calculate CRPS using empirical formula
        # CRPS = E|Y_fcst - y_obs| - 0.5 * E|Y_fcst - Y_fcst'|
        term1 = np.mean(np.abs(y_fcst - y_obs))
        term2 = np.mean(np.abs(y_fcst[:, np.newaxis] - y_fcst[np.newaxis, :]))
        crps = term1 - 0.5 * term2
        
        crps_values.append(crps)
    
    return np.mean(crps_values)


def crps_gaussian(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """
    Calculate CRPS assuming Gaussian distribution.
    
    Args:
        y_true: Observed values
        mean: Forecast means
        std: Forecast standard deviations
        
    Returns:
        Average CRPS
    """
    # Standardize observations
    z = (y_true - mean) / std
    
    # CRPS for standard normal
    crps_standard = -z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1/np.sqrt(np.pi)
    
    # Scale back
    crps = std * crps_standard
    
    return np.mean(crps)


def interval_sharpness(p10: np.ndarray, p90: np.ndarray) -> Dict[str, float]:
    """
    Calculate sharpness metrics (width of prediction intervals).
    
    Args:
        p10: 10th percentile predictions
        p90: 90th percentile predictions
        
    Returns:
        Dictionary with sharpness metrics
    """
    width = p90 - p10
    
    return {
        'mean_width': np.mean(width),
        'median_width': np.median(width),
        'std_width': np.std(width),
        'relative_width': np.mean(width / np.maximum(p90, 1e-10))  # Width relative to P90
    }


def calibration_metrics(y_true: np.ndarray, y_samples: np.ndarray, 
                        intervals: list = [50, 80, 90, 95]) -> Dict:
    """
    Calculate calibration metrics for prediction intervals.
    
    Args:
        y_true: Observed values [n_observations]
        y_samples: Forecast samples [n_samples, n_observations]
        intervals: List of prediction interval widths to evaluate (in %)
        
    Returns:
        Dictionary with calibration metrics
    """
    results = {}
    
    for interval in intervals:
        alpha = (100 - interval) / 2
        p_lower = np.percentile(y_samples, alpha, axis=0)
        p_upper = np.percentile(y_samples, 100 - alpha, axis=0)
        
        # Calculate coverage (fraction of observations within interval)
        coverage = np.mean((y_true >= p_lower) & (y_true <= p_upper))
        
        # Calculate interval width
        width = np.mean(p_upper - p_lower)
        
        results[f'coverage_{interval}'] = coverage
        results[f'width_{interval}'] = width
        results[f'calibration_error_{interval}'] = abs(coverage - interval/100)
    
    # Overall calibration error
    results['mean_calibration_error'] = np.mean([results[f'calibration_error_{i}'] for i in intervals])
    
    return results


def pit_values(y_true: np.ndarray, y_samples: np.ndarray) -> np.ndarray:
    """
    Calculate Probability Integral Transform (PIT) values for calibration assessment.
    
    PIT values should be uniformly distributed [0,1] for well-calibrated forecasts.
    
    Args:
        y_true: Observed values [n_observations]
        y_samples: Forecast samples [n_samples, n_observations]
        
    Returns:
        PIT values [n_observations]
    """
    pit = np.zeros(y_true.shape[0])
    
    for i in range(y_true.shape[0]):
        # Calculate empirical CDF value at observed point
        pit[i] = np.mean(y_samples[:, i] <= y_true[i])
    
    return pit


def plot_calibration_diagram(y_true: np.ndarray, y_samples: np.ndarray, 
                             model_name: str = "Model",
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Create calibration reliability diagram.
    
    Args:
        y_true: Observed values
        y_samples: Forecast samples [n_samples, n_observations]
        model_name: Name for the plot title
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure
    """
    # Calculate PIT values
    pit = pit_values(y_true.flatten(), y_samples.reshape(y_samples.shape[0], -1))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left plot: Calibration reliability diagram
    n_bins = min(10, len(pit) // 5)  # Adaptive binning
    expected = np.linspace(0, 1, 100)
    
    # Calculate empirical quantiles
    quantiles = np.linspace(0, 1, n_bins + 1)
    empirical = []
    for q in quantiles[1:]:
        empirical.append(np.mean(pit <= q))
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax1.plot(quantiles[1:], empirical, 'bo-', label=model_name)
    ax1.fill_between(quantiles[1:], quantiles[1:], empirical, alpha=0.3)
    ax1.set_xlabel('Expected cumulative probability')
    ax1.set_ylabel('Observed cumulative probability')
    ax1.set_title('Calibration Reliability Diagram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Right plot: PIT histogram
    ax2.hist(pit, bins=min(20, len(pit) // 3), density=True, alpha=0.7, edgecolor='black')
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Uniform distribution')
    ax2.set_xlabel('PIT value')
    ax2.set_ylabel('Density')
    ax2.set_title('PIT Histogram')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Calibration Assessment: {model_name}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def compare_models_crps(models_results: Dict[str, Dict], y_true: np.ndarray) -> pd.DataFrame:
    """
    Compare multiple models using CRPS and other metrics.
    
    Args:
        models_results: Dictionary mapping model names to their results
                       Each result should have 'samples' or 'mean'/'std'
        y_true: True observed values
        
    Returns:
        DataFrame with comparison metrics
    """
    
    metrics_list = []
    
    for model_name, results in models_results.items():
        metrics = {'Model': model_name}
        
        # Ensure we're comparing the same shape
        y_true_flat = y_true.flatten()
        
        # Calculate CRPS
        if 'samples' in results:
            # Use empirical CRPS
            # Reshape samples to [n_samples, n_observations]
            samples = results['samples']
            if samples.ndim == 3:
                # [n_samples, n_wells, n_time] -> [n_samples, n_wells*n_time]
                samples_flat = samples.reshape(samples.shape[0], -1)
            else:
                samples_flat = samples
            
            # Make sure dimensions match
            if samples_flat.shape[1] == y_true_flat.shape[0]:
                crps = crps_empirical(y_true_flat, samples_flat)
            else:
                crps = np.nan
                
        elif 'mean' in results and 'std' in results:
            # Use Gaussian CRPS
            mean_flat = results['mean'].flatten()
            std_flat = results['std'].flatten()
            
            # Make sure dimensions match
            if mean_flat.shape[0] == y_true_flat.shape[0] and std_flat.shape[0] == y_true_flat.shape[0]:
                crps = crps_gaussian(y_true_flat, mean_flat, std_flat)
            else:
                crps = np.nan
        else:
            crps = np.nan
        
        metrics['CRPS'] = crps
        
        # Calculate sharpness if we have percentiles
        if 'p10' in results and 'p90' in results:
            sharpness = interval_sharpness(results['p10'].flatten(), 
                                          results['p90'].flatten())
            metrics['Sharpness (P90-P10)'] = sharpness['mean_width']
            metrics['Relative Sharpness'] = sharpness['relative_width']
        
        # Calculate calibration if we have samples
        if 'samples' in results:
            cal_metrics = calibration_metrics(
                y_true.flatten(),
                results['samples'].reshape(results['samples'].shape[0], -1),
                intervals=[80, 90]
            )
            metrics['Coverage 80%'] = cal_metrics['coverage_80']
            metrics['Coverage 90%'] = cal_metrics['coverage_90']
            metrics['Calibration Error'] = cal_metrics['mean_calibration_error']
        
        # Calculate RMSE for point forecast
        if 'p50' in results:
            rmse = np.sqrt(np.mean((results['p50'].flatten() - y_true.flatten())**2))
            metrics['RMSE (P50)'] = rmse
        elif 'mean' in results:
            rmse = np.sqrt(np.mean((results['mean'].flatten() - y_true.flatten())**2))
            metrics['RMSE (Mean)'] = rmse
        
        metrics_list.append(metrics)
    
    df = pd.DataFrame(metrics_list)
    
    # Sort by CRPS (lower is better)
    df = df.sort_values('CRPS')
    
    # Calculate relative improvements
    if len(df) > 0:
        baseline_crps = df.iloc[-1]['CRPS']  # Use worst model as baseline
        df['CRPS Improvement (%)'] = ((baseline_crps - df['CRPS']) / baseline_crps * 100)
    
    return df


def plot_interval_comparison(models_results: Dict[str, Dict], y_true: np.ndarray,
                            time_points: np.ndarray, well_idx: int = 0,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot prediction intervals for multiple models for visual comparison.
    
    Args:
        models_results: Dictionary of model results
        y_true: True values [n_wells, n_time]
        time_points: Time points in days
        well_idx: Which well to plot
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure
    """
    n_models = len(models_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4), sharey=True)
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, results) in enumerate(models_results.items()):
        ax = axes[idx]
        
        # Plot true values
        ax.plot(time_points, y_true[well_idx, :], 'ko-', label='Observed', markersize=4)
        
        # Plot prediction intervals
        if 'p10' in results and 'p90' in results:
            p10 = results['p10'][well_idx, :] if results['p10'].ndim > 1 else results['p10']
            p50 = results['p50'][well_idx, :] if results['p50'].ndim > 1 else results['p50']
            p90 = results['p90'][well_idx, :] if results['p90'].ndim > 1 else results['p90']
            
            ax.fill_between(time_points, p10, p90, alpha=0.3, label='P10-P90')
            ax.plot(time_points, p50, 'b-', label='P50', linewidth=2)
        
        ax.set_xlabel('Time (days)')
        if idx == 0:
            ax.set_ylabel('Cumulative Production')
        ax.set_title(model_name)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Prediction Interval Comparison - Well {well_idx}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig