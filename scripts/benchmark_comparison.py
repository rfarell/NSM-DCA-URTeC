#!/usr/bin/env python
# scripts/benchmark_fair_comparison.py
"""
Fair benchmark comparison script for production forecasting models.
All models are conditioned on the same initial production history.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import yaml
import pickle
import time
from typing import Dict, Tuple
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data_processor import DataProcessor
from src.utils import load_config, build_model_from_config, load_model_checkpoint
from src.dca_models import ArpsHyperbolic, DuongModel, bootstrap_dca_uncertainty
from src.ml_benchmarks import LightGBMQuantile
from src.probabilistic_metrics import (
    crps_empirical, interval_sharpness, calibration_metrics,
    plot_calibration_diagram, compare_models_crps, plot_interval_comparison
)

def fit_single_well_dca_conditional(args):
    """
    Fit DCA models for a single well using only historical data.
    
    Args:
        args: Tuple of (well_idx, Q_history, t_history, t_future, phase_name, n_bootstrap)
    
    Returns:
        Tuple of (well_idx, arps_predictions, duong_predictions)
    """
    well_idx, Q_history, t_history, t_future, phase_name, n_bootstrap = args
    
    # Import here to avoid pickling issues in multiprocessing
    from src.dca_models import ArpsHyperbolic, DuongModel, bootstrap_dca_uncertainty
    
    # Fit Arps using only historical data
    arps_predictions = None
    try:
        arps = ArpsHyperbolic()
        arps.fit(t_history, Q_history)
        # Predict future time points
        Q_pred = arps.predict(t_future)
        # Generate uncertainty
        arps_predictions = bootstrap_dca_uncertainty(
            arps, t_history, Q_history, t_future, n_bootstrap=n_bootstrap
        )
    except Exception as e:
        # Use naive forecast as fallback
        last_Q = Q_history[-1]
        arps_predictions = {
            'p10': np.full_like(t_future, last_Q * 0.9),
            'p50': np.full_like(t_future, last_Q),
            'p90': np.full_like(t_future, last_Q * 1.1),
            'mean': np.full_like(t_future, last_Q),
            'std': np.full_like(t_future, last_Q * 0.1)
        }
    
    # Fit Duong using only historical data
    duong_predictions = None
    try:
        duong = DuongModel()
        duong.fit(t_history, Q_history)
        # Predict future time points
        Q_pred = duong.predict(t_future)
        # Generate uncertainty
        duong_predictions = bootstrap_dca_uncertainty(
            duong, t_history, Q_history, t_future, n_bootstrap=n_bootstrap
        )
    except Exception as e:
        # Use naive forecast as fallback
        last_Q = Q_history[-1]
        duong_predictions = {
            'p10': np.full_like(t_future, last_Q * 0.9),
            'p50': np.full_like(t_future, last_Q),
            'p90': np.full_like(t_future, last_Q * 1.1),
            'mean': np.full_like(t_future, last_Q),
            'std': np.full_like(t_future, last_Q * 0.1)
        }
    
    return well_idx, arps_predictions, duong_predictions


def evaluate_nsm_conditional(model, x_test, z_test, t_vals, scale_test, device, 
                            n_history=3, n_mc=100, uncertainty_scale=3.0):
    """
    Evaluate NSM-DCA model conditioned on initial production history.
    
    Args:
        model: NSM-DCA model
        x_test: Test data [n_wells, n_time, n_phases]
        z_test: Static features [n_wells, n_features]
        t_vals: Time points
        scale_test: Scaling factors
        device: Torch device
        n_history: Number of historical time points to condition on
        n_mc: Number of MC samples
        uncertainty_scale: Factor to scale uncertainty (default 3.0 for calibration)
    
    Returns:
        Dictionary with predictions for future time points only
    """
    from torchdiffeq import odeint
    
    # Ensure model and data are on the correct device
    model = model.to(device)
    x_test = x_test.to(device)
    z_test = z_test.to(device)
    t_vals = t_vals.to(device)
    scale_test = scale_test.to(device)
    
    n_wells, n_time, n_phases = x_test.shape
    n_future = n_time - n_history
    
    # Get historical data and future time points
    x_history = x_test[:, :n_history, :]  # [n_wells, n_history, n_phases]
    t_future = t_vals[n_history-1:]  # Start from last historical point for continuity
    
    # Generate MC samples
    all_predictions = []
    
    model.eval()
    with torch.no_grad():
        for mc_idx in range(n_mc):
            # Start from the last historical point
            x0 = x_history[:, -1, :]  # [n_wells, n_phases]
            
            def ode_func(t_scalar, x_state):
                return model(t_scalar, x_state, z_test)
            
            # Solve ODE from last historical point forward
            pred_traj = odeint(
                ode_func, x0, t_future,
                method='midpoint',
                atol=1e-2,
                rtol=1e-2
            )
            
            # Reshape: [T, B, D] -> [B, T, D]
            pred_traj = pred_traj.permute(1, 0, 2)
            
            # Denormalize
            pred_traj_denorm = pred_traj * scale_test.unsqueeze(1)
            
            # Only keep future predictions (exclude the starting point)
            pred_future = pred_traj_denorm[:, 1:, :]  # [n_wells, n_future, n_phases]
            
            all_predictions.append(pred_future.cpu().numpy())
    
    # Stack predictions: [n_mc, n_wells, n_future, n_phases]
    predictions = np.array(all_predictions)
    
    # Calculate statistics
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    # Post-hoc calibration: scale uncertainty to address overconfidence
    std_scaled = std_pred * uncertainty_scale
    
    # Recalculate percentiles using scaled uncertainty
    # Assume normal distribution for simplicity
    from scipy import stats
    p10_scaled = mean_pred - stats.norm.ppf(0.9) * std_scaled
    p90_scaled = mean_pred + stats.norm.ppf(0.9) * std_scaled
    
    # Generate new samples with scaled uncertainty for calibration metrics
    # This ensures coverage metrics are calculated correctly
    n_mc = predictions.shape[0]
    np.random.seed(42)  # For reproducibility
    scaled_samples = []
    for _ in range(n_mc):
        # Generate samples from scaled normal distribution
        scaled_sample = mean_pred + np.random.randn(*mean_pred.shape) * std_scaled
        scaled_samples.append(scaled_sample)
    scaled_samples = np.array(scaled_samples)
    
    results = {
        'samples': scaled_samples,  # Use scaled samples for coverage calculation
        'p10': p10_scaled,
        'p50': mean_pred,  # Median stays the same
        'p90': p90_scaled,
        'mean': mean_pred,
        'std': std_scaled  # Return scaled std
    }
    
    return results


def evaluate_dca_conditional(x_test, t_vals_days, n_history=3, n_bootstrap=500, 
                            n_workers=None, max_wells=None):
    """
    Evaluate DCA models using only historical data for fitting.
    
    Args:
        x_test: Test data [n_wells, n_time, n_phases]
        t_vals_days: Time points in days
        n_history: Number of historical points to use for fitting
        n_bootstrap: Number of bootstrap samples
        n_workers: Number of parallel workers
        max_wells: Maximum number of wells to process
    """
    n_wells, n_time, n_phases = x_test.shape
    
    # Limit wells if requested
    if max_wells is not None and max_wells < n_wells:
        print(f"  Limiting to {max_wells} wells (out of {n_wells}) for faster testing")
        x_test = x_test[:max_wells]
        n_wells = max_wells
    
    # Set number of workers
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    print(f"  Using {n_workers} parallel workers")
    
    # Split time into history and future
    t_history = t_vals_days[1:n_history]  # Skip t=0 for DCA
    t_future = t_vals_days[n_history:]
    
    results = {}
    
    # Process each phase
    for phase_idx, phase_name in enumerate(['Gas', 'Oil', 'Water']):
        print(f"\n  Processing {phase_name} phase...")
        results[phase_name] = {}
        
        # Prepare arguments for parallel processing
        args_list = []
        for well_idx in range(n_wells):
            Q_full = x_test[well_idx, :, phase_idx]
            Q_history = Q_full[1:n_history]  # Historical data (skip t=0)
            args_list.append((well_idx, Q_history, t_history, t_future, phase_name, n_bootstrap))
        
        # Process wells in parallel with progress bar
        arps_predictions = [None] * n_wells
        duong_predictions = [None] * n_wells
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(fit_single_well_dca_conditional, args): idx 
                      for idx, args in enumerate(args_list)}
            
            with tqdm(total=n_wells, desc=f"    Fitting DCA models ({phase_name})", 
                     unit="wells", leave=True) as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        well_idx, arps_result, duong_result = future.result()
                        arps_predictions[well_idx] = arps_result
                        duong_predictions[well_idx] = duong_result
                    except Exception as e:
                        print(f"Warning: Failed to fit well {idx}: {e}")
                        # Use fallback predictions
                        Q_last = x_test[idx, n_history-1, phase_idx]
                        fallback = {
                            'p10': np.full(len(t_future), Q_last * 0.9),
                            'p50': np.full(len(t_future), Q_last),
                            'p90': np.full(len(t_future), Q_last * 1.1),
                            'mean': np.full(len(t_future), Q_last),
                            'std': np.full(len(t_future), Q_last * 0.1)
                        }
                        arps_predictions[idx] = fallback
                        duong_predictions[idx] = fallback
                    pbar.update(1)
        
        # Aggregate results
        arps_samples = []
        for p in arps_predictions:
            if 'samples' in p:
                arps_samples.append(p['samples'])
        
        arps_dict = {
            'p10': np.array([p['p10'] for p in arps_predictions]),
            'p50': np.array([p['p50'] for p in arps_predictions]),
            'p90': np.array([p['p90'] for p in arps_predictions]),
            'mean': np.array([p['mean'] for p in arps_predictions]),
            'std': np.array([p['std'] for p in arps_predictions])
        }
        
        if arps_samples:
            arps_dict['samples'] = np.array(arps_samples).transpose(1, 0, 2)
        
        results[phase_name]['Arps'] = arps_dict
        
        # Aggregate Duong results
        duong_samples = []
        for p in duong_predictions:
            if 'samples' in p:
                duong_samples.append(p['samples'])
        
        duong_dict = {
            'p10': np.array([p['p10'] for p in duong_predictions]),
            'p50': np.array([p['p50'] for p in duong_predictions]),
            'p90': np.array([p['p90'] for p in duong_predictions]),
            'mean': np.array([p['mean'] for p in duong_predictions]),
            'std': np.array([p['std'] for p in duong_predictions])
        }
        
        if duong_samples:
            duong_dict['samples'] = np.array(duong_samples).transpose(1, 0, 2)
        
        results[phase_name]['Duong'] = duong_dict
    
    return results


def evaluate_lightgbm_conditional(x_train, x_test, z_train, z_test, t_vals_days, n_history=3):
    """
    Evaluate LightGBM using historical production as features.
    """
    n_wells_test, n_time, n_phases = x_test.shape
    n_future = n_time - n_history
    
    results = {}
    
    for phase_idx, phase_name in enumerate(['Gas', 'Oil', 'Water']):
        # Prepare training data with historical features
        Y_train_full = x_train[:, :, phase_idx]
        Y_test_full = x_test[:, :, phase_idx]
        
        # Create features including historical production
        # For training: use first n_history points as features
        X_train_history = Y_train_full[:, :n_history].reshape(x_train.shape[0], -1)
        X_train_enhanced = np.concatenate([z_train.cpu().numpy(), X_train_history], axis=1)
        
        # For testing: use first n_history points as features
        X_test_history = Y_test_full[:, :n_history].reshape(n_wells_test, -1)
        X_test_enhanced = np.concatenate([z_test.cpu().numpy(), X_test_history], axis=1)
        
        # Target is future production
        Y_train_future = Y_train_full[:, n_history:]
        Y_test_future = Y_test_full[:, n_history:]
        
        # Train LightGBM model
        lgb_model = LightGBMQuantile(quantiles=[0.1, 0.5, 0.9])
        
        try:
            # Fit model on future time points
            t_future = t_vals_days[n_history:]
            lgb_model.fit(X_train_enhanced, Y_train_future, t_future)
            
            # Predict
            predictions = lgb_model.predict(X_test_enhanced)
            
            # Generate uncertainty samples
            pred_with_samples = lgb_model.predict_with_uncertainty(
                X_test_enhanced, n_samples=500
            )
            
            results[phase_name] = {
                'p10': predictions['p10'],
                'p50': predictions['p50'],
                'p90': predictions['p90'],
                'mean': pred_with_samples['mean'],
                'std': pred_with_samples['std'],
                'samples': pred_with_samples['samples']
            }
            
        except Exception as e:
            print(f"LightGBM failed for phase {phase_name}: {e}")
            # Fallback to persistence forecast
            last_values = Y_test_full[:, n_history-1:n_history]
            results[phase_name] = {
                'p10': np.tile(last_values * 0.9, (1, n_future)),
                'p50': np.tile(last_values, (1, n_future)),
                'p90': np.tile(last_values * 1.1, (1, n_future)),
                'mean': np.tile(last_values, (1, n_future)),
                'std': np.tile(last_values * 0.1, (1, n_future))
            }
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Fair benchmark comparison')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--experiment', type=str, default='exp1',
                      help='Experiment name for NSM model')
    parser.add_argument('--output-dir', type=str, default='benchmarks_fair',
                      help='Output directory for results')
    parser.add_argument('--n-history', type=int, default=6,
                      help='Number of historical time points to condition on (default: 6 = up to 360 days)')
    parser.add_argument('--n-bootstrap', type=int, default=100,
                      help='Number of bootstrap samples for DCA models')
    parser.add_argument('--n-mc', type=int, default=100,
                      help='Number of MC samples for NSM model')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Test set size')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--n-workers', type=int, default=None,
                      help='Number of parallel workers')
    parser.add_argument('--max-wells', type=int, default=None,
                      help='Maximum number of wells to process')
    parser.add_argument('--force-dca', action='store_true',
                      help='Force re-evaluation of DCA models even if cached results exist')
    parser.add_argument('--skip-dca', action='store_true',
                      help='Skip DCA evaluation entirely (use only if cached results exist)')
    parser.add_argument('--skip-lightgbm', action='store_true',
                      help='Skip LightGBM evaluation')
    parser.add_argument('--uncertainty-scale', type=float, default=3.0,
                      help='Factor to scale NSM-DCA uncertainty for calibration (default: 3.0)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    print("Loading data...")
    data_processor = DataProcessor(config)
    x_train, z_train, x_test, z_test, t_vals = data_processor.prepare_data()
    
    # Get time points in days
    t_vals_days = np.array(data_processor.days, dtype=np.float64)
    
    # Get scaling factors
    scale_train = data_processor.scale_train
    scale_test = data_processor.scale_test
    
    print(f"Data shapes: Train={x_train.shape}, Test={x_test.shape}")
    print(f"Time points (days): {t_vals_days.tolist()}")
    print(f"Using {args.n_history} historical points for conditioning (up to {t_vals_days[args.n_history-1] if args.n_history <= len(t_vals_days) else t_vals_days[-1]:.0f} days)")
    print(f"Predicting {len(t_vals_days) - args.n_history} future points ({', '.join([f'{d:.0f}' for d in t_vals_days[args.n_history:]])} days)")
    
    # Limit data if requested
    if args.max_wells is not None and args.max_wells < x_test.shape[0]:
        x_test_limited = x_test[:args.max_wells]
        z_test_limited = z_test[:args.max_wells]
        scale_test_limited = scale_test[:args.max_wells]
    else:
        x_test_limited = x_test
        z_test_limited = z_test
        scale_test_limited = scale_test
    
    # Convert to numpy for DCA models
    x_test_denorm = (x_test_limited * scale_test_limited.unsqueeze(1)).cpu().numpy()
    x_train_denorm = (x_train * scale_train.unsqueeze(1)).cpu().numpy()
    
    # Load NSM-DCA model
    print("\nEvaluating NSM-DCA model (conditional)...")
    if args.uncertainty_scale != 1.0:
        print(f"  Using uncertainty scaling factor: {args.uncertainty_scale}x for calibration")
    experiment_path = os.path.join('experiments', args.experiment)
    model_path = os.path.join(experiment_path, 'model.pth')
    
    if os.path.exists(model_path):
        model = build_model_from_config(config, device)
        model = load_model_checkpoint(model, model_path, device)
        model.eval()
        
        nsm_results = evaluate_nsm_conditional(
            model, x_test_limited, z_test_limited, t_vals, scale_test_limited,
            device, n_history=args.n_history, n_mc=args.n_mc, 
            uncertainty_scale=args.uncertainty_scale
        )
    else:
        print(f"NSM model not found at {model_path}")
        nsm_results = None
    
    # Handle DCA model evaluation with caching
    dca_cache_file = os.path.join(args.output_dir, f'dca_cache_h{args.n_history}_b{args.n_bootstrap}_w{args.max_wells}.pkl')
    dca_results = None
    
    # Check if we should use cached DCA results
    if not args.skip_dca:
        if os.path.exists(dca_cache_file) and not args.force_dca:
            print(f"\n" + "="*60)
            print("CACHED DCA RESULTS FOUND")
            print("="*60)
            print(f"Cache file: {dca_cache_file}")
            print(f"Created: {time.ctime(os.path.getmtime(dca_cache_file))}")
            
            user_input = input("\nDo you want to use cached DCA results? This saves ~5-10 minutes. [Y/n]: ").strip().lower()
            
            if user_input != 'n':
                print("Loading cached DCA results...")
                with open(dca_cache_file, 'rb') as f:
                    dca_results = pickle.load(f)
                print("✓ Cached DCA results loaded successfully")
            else:
                print("Re-evaluating DCA models...")
        
        # Evaluate DCA models if not loaded from cache
        if dca_results is None:
            print(f"\nEvaluating DCA models (conditional on {args.n_history} points)...")
            print(f"  Bootstrap samples: {args.n_bootstrap}")
            
            dca_results = evaluate_dca_conditional(
                x_test_denorm, t_vals_days,
                n_history=args.n_history,
                n_bootstrap=args.n_bootstrap,
                n_workers=args.n_workers,
                max_wells=args.max_wells
            )
            
            # Save DCA results to cache
            print(f"Saving DCA results to cache: {dca_cache_file}")
            with open(dca_cache_file, 'wb') as f:
                pickle.dump(dca_results, f)
            print("✓ DCA results cached for future use")
    else:
        # Try to load from cache when skipping
        if os.path.exists(dca_cache_file):
            print(f"\nLoading cached DCA results (--skip-dca flag)...")
            with open(dca_cache_file, 'rb') as f:
                dca_results = pickle.load(f)
            print("✓ Cached DCA results loaded")
        else:
            print(f"\nWARNING: No cached DCA results found at {dca_cache_file}")
            print("DCA models will be excluded from comparison")
    
    # Handle LightGBM evaluation with caching
    lgb_cache_file = os.path.join(args.output_dir, f'lgb_cache_h{args.n_history}_w{args.max_wells}.pkl')
    lgb_results = None
    
    if not args.skip_lightgbm:
        # Check for cached LightGBM results
        if os.path.exists(lgb_cache_file) and not args.force_dca:
            print(f"\n" + "="*60)
            print("CACHED LIGHTGBM RESULTS FOUND")
            print("="*60)
            print(f"Cache file: {lgb_cache_file}")
            print(f"Created: {time.ctime(os.path.getmtime(lgb_cache_file))}")
            
            user_input = input("\nDo you want to use cached LightGBM results? This saves ~1-2 minutes. [Y/n]: ").strip().lower()
            
            if user_input != 'n':
                print("Loading cached LightGBM results...")
                with open(lgb_cache_file, 'rb') as f:
                    lgb_results = pickle.load(f)
                print("✓ Cached LightGBM results loaded successfully")
            else:
                print("Re-evaluating LightGBM...")
        
        # Evaluate LightGBM if not loaded from cache
        if lgb_results is None:
            print(f"\nEvaluating LightGBM (conditional on {args.n_history} points)...")
            lgb_results = evaluate_lightgbm_conditional(
                x_train_denorm, x_test_denorm, z_train, z_test_limited, t_vals_days,
                n_history=args.n_history
            )
            
            # Save LightGBM results to cache
            print(f"Saving LightGBM results to cache: {lgb_cache_file}")
            with open(lgb_cache_file, 'wb') as f:
                pickle.dump(lgb_results, f)
            print("✓ LightGBM results cached for future use")
    else:
        print("\nSkipping LightGBM evaluation (--skip-lightgbm flag)")
    
    # Compare results on future predictions only
    print("\nComputing comparison metrics on future predictions...")
    
    all_metrics = []
    
    # Process each phase
    for phase_idx, phase_name in enumerate(['Gas', 'Oil', 'Water']):
        print(f"\n{phase_name} Phase:")
        
        # True future values only
        if args.max_wells is not None and args.max_wells < x_test_denorm.shape[0]:
            y_true = x_test_denorm[:args.max_wells, args.n_history:, phase_idx]
        else:
            y_true = x_test_denorm[:, args.n_history:, phase_idx]
        
        # Collect model results (all should have same shape now)
        model_results = {}
        
        # NSM-DCA
        if nsm_results is not None:
            model_results['NSM-DCA'] = {
                'p10': nsm_results['p10'][:, :, phase_idx],
                'p50': nsm_results['p50'][:, :, phase_idx],
                'p90': nsm_results['p90'][:, :, phase_idx],
                'samples': nsm_results['samples'][:, :, :, phase_idx] if 'samples' in nsm_results else None
            }
        
        # DCA models
        if dca_results and phase_name in dca_results:
            for dca_name in ['Arps', 'Duong']:
                if dca_name in dca_results[phase_name]:
                    model_results[dca_name] = dca_results[phase_name][dca_name]
        
        # LightGBM
        if lgb_results and phase_name in lgb_results:
            model_results['LightGBM'] = lgb_results[phase_name]
        
        # Calculate metrics
        metrics_df = compare_models_crps(model_results, y_true)
        metrics_df['Phase'] = phase_name
        all_metrics.append(metrics_df)
        
        print(metrics_df.to_string())
        
        # Generate plots for first well
        fig = plot_interval_comparison(
            model_results, y_true, t_vals_days[args.n_history:], well_idx=0,
            save_path=os.path.join(args.output_dir, f"comparison_{phase_name}_well0.png")
        )
        plt.close(fig)
    
    # Combine all metrics
    full_metrics_df = pd.concat(all_metrics, ignore_index=True)
    
    # Save results
    output_file = os.path.join(args.output_dir, 'benchmark_metrics_fair.csv')
    full_metrics_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Create summary table
    print("\n" + "="*80)
    print(f"FAIR BENCHMARK SUMMARY (Conditioned on {args.n_history} historical points)")
    print("="*80)
    
    # Average across phases
    summary = full_metrics_df.groupby('Model').agg({
        'CRPS': 'mean',
        'Sharpness (P90-P10)': 'mean',
        'Coverage 80%': 'mean',
        'Coverage 90%': 'mean',
        'Calibration Error': 'mean',
        'RMSE (P50)': 'mean'
    }).round(4)
    
    print(summary.to_string())
    
    # Save summary
    summary_file = os.path.join(args.output_dir, 'benchmark_summary_fair.csv')
    summary.to_csv(summary_file)
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()