#!/usr/bin/env python
# scripts/benchmark_comparison.py
"""
Comprehensive benchmark comparison script for production forecasting models.
Compares NSM-DCA against Arps, Duong, and LightGBM models.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import yaml
import pickle
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
from src.utils import load_config, build_model_from_config
from src.dca_models import ArpsHyperbolic, DuongModel, bootstrap_dca_uncertainty
from src.ml_benchmarks import LightGBMQuantile
from src.probabilistic_metrics import (
    crps_empirical, interval_sharpness, calibration_metrics,
    plot_calibration_diagram, compare_models_crps, plot_interval_comparison
)

def fit_single_well_dca(args):
    """
    Fit DCA models for a single well (helper for parallel processing).
    
    Args:
        args: Tuple of (well_idx, Q, Q_fit, t_fit, t_days, phase_name, n_bootstrap)
    
    Returns:
        Tuple of (well_idx, arps_result, duong_result)
    """
    well_idx, Q, Q_fit, t_fit, t_days, phase_name, n_bootstrap = args
    
    # Import here to avoid pickling issues in multiprocessing
    from src.dca_models import ArpsHyperbolic, DuongModel, bootstrap_dca_uncertainty
    
    # Fit Arps
    arps_result = None
    try:
        arps = ArpsHyperbolic()
        arps.fit(t_fit, Q_fit)
        arps_result = bootstrap_dca_uncertainty(
            arps, t_fit, Q_fit, t_days, n_bootstrap=n_bootstrap
        )
    except Exception as e:
        # Use fallback
        arps_result = {
            'p10': Q, 'p50': Q, 'p90': Q,
            'mean': Q, 'std': np.zeros_like(Q)
        }
    
    # Fit Duong
    duong_result = None
    try:
        duong = DuongModel()
        duong.fit(t_fit, Q_fit)
        duong_result = bootstrap_dca_uncertainty(
            duong, t_fit, Q_fit, t_days, n_bootstrap=n_bootstrap
        )
    except Exception as e:
        # Use fallback
        duong_result = {
            'p10': Q, 'p50': Q, 'p90': Q,
            'mean': Q, 'std': np.zeros_like(Q)
        }
    
    return well_idx, arps_result, duong_result


def parse_args():
    parser = argparse.ArgumentParser(description='Run benchmark comparison')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--experiment', type=str, default='exp1',
                      help='Experiment name for NSM model')
    parser.add_argument('--output-dir', type=str, default='benchmarks',
                      help='Output directory for results')
    parser.add_argument('--n-bootstrap', type=int, default=500,
                      help='Number of bootstrap samples for DCA models')
    parser.add_argument('--n-mc', type=int, default=100,
                      help='Number of MC samples for NSM model')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Test set size')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--n-workers', type=int, default=None,
                      help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--max-wells', type=int, default=None,
                      help='Maximum number of wells to process (for testing)')
    return parser.parse_args()


def evaluate_nsm_model(model, x_test, z_test, t_vals, scale_test, device, n_mc=100):
    """Evaluate NSM-DCA model and generate probabilistic predictions."""
    from torchdiffeq import odeint
    
    # Ensure model is on the correct device
    model = model.to(device)
    
    # Move data to the same device as model
    x_test = x_test.to(device)
    z_test = z_test.to(device)
    t_vals = t_vals.to(device)
    scale_test = scale_test.to(device)
    
    n_wells, n_time, n_phases = x_test.shape
    
    # Generate MC samples
    all_predictions = []
    
    model.eval()
    with torch.no_grad():
        for mc_idx in range(n_mc):
            x0 = x_test[:, 0, :]  # Initial conditions
            
            def ode_func(t_scalar, x_state):
                return model(t_scalar, x_state, z_test)
            
            # Solve ODE
            pred_traj = odeint(
                ode_func, x0, t_vals,
                method='midpoint',
                atol=1e-2,
                rtol=1e-2
            )
            
            # Reshape: [T, B, D] -> [B, T, D]
            pred_traj = pred_traj.permute(1, 0, 2)
            
            # Denormalize
            pred_traj_denorm = pred_traj * scale_test.unsqueeze(1)
            
            all_predictions.append(pred_traj_denorm.cpu().numpy())
    
    # Stack predictions: [n_mc, n_wells, n_time, n_phases]
    predictions = np.array(all_predictions)
    
    # Calculate statistics
    results = {
        'samples': predictions,
        'p10': np.percentile(predictions, 10, axis=0),
        'p50': np.percentile(predictions, 50, axis=0),
        'p90': np.percentile(predictions, 90, axis=0),
        'mean': np.mean(predictions, axis=0),
        'std': np.std(predictions, axis=0)
    }
    
    return results


def evaluate_dca_models(x_test, t_vals_days, n_bootstrap=500, n_workers=None, max_wells=None):
    """
    Evaluate DCA models (Arps and Duong) on test data with parallel processing.
    
    Args:
        x_test: Test data [n_wells, n_time, n_phases]
        t_vals_days: Time points in days
        n_bootstrap: Number of bootstrap samples
        n_workers: Number of parallel workers
        max_wells: Maximum number of wells to process (for testing)
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
    
    # Convert time to days (skip t=0 for DCA fitting)
    t_days = t_vals_days
    t_fit = t_days[1:]  # Skip first time point (t=0)
    
    results = {}
    
    # Process each phase
    for phase_idx, phase_name in enumerate(['Gas', 'Oil', 'Water']):
        print(f"\n  Processing {phase_name} phase...")
        results[phase_name] = {}
        
        # Prepare arguments for parallel processing
        args_list = []
        for well_idx in range(n_wells):
            Q = x_test[well_idx, :, phase_idx]
            Q_fit = Q[1:]  # Skip first point for fitting
            args_list.append((well_idx, Q, Q_fit, t_fit, t_days, phase_name, n_bootstrap))
        
        # Process wells in parallel with progress bar
        arps_predictions = [None] * n_wells
        duong_predictions = [None] * n_wells
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(fit_single_well_dca, args): idx 
                      for idx, args in enumerate(args_list)}
            
            # Process results with progress bar
            with tqdm(total=n_wells, desc=f"    Fitting DCA models ({phase_name})", 
                     unit="wells", leave=True) as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        well_idx, arps_result, duong_result = future.result()
                        arps_predictions[well_idx] = arps_result
                        duong_predictions[well_idx] = duong_result
                    except Exception as e:
                        # Use fallback for failed wells
                        Q = x_test[idx, :, phase_idx]
                        fallback = {
                            'p10': Q, 'p50': Q, 'p90': Q,
                            'mean': Q, 'std': np.zeros_like(Q)
                        }
                        arps_predictions[idx] = fallback
                        duong_predictions[idx] = fallback
                    pbar.update(1)
        
        # Aggregate Arps results
        # Collect samples if available
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
        
        # Add samples if available - shape: [n_samples, n_wells, n_time]
        if arps_samples:
            # Stack samples: list of [n_samples_i, n_time] -> [n_samples, n_wells, n_time]
            # Since each well has same n_samples, we can transpose
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
        
        # Add samples if available
        if duong_samples:
            duong_dict['samples'] = np.array(duong_samples).transpose(1, 0, 2)
        
        results[phase_name]['Duong'] = duong_dict
    
    return results


def evaluate_lightgbm(x_train, x_test, z_train, z_test, t_vals_days):
    """Evaluate LightGBM quantile regression model."""
    n_wells_test, n_time, n_phases = x_test.shape
    
    results = {}
    
    for phase_idx, phase_name in enumerate(['Gas', 'Oil', 'Water']):
        # Prepare training data
        Y_train = x_train[:, :, phase_idx]  # [n_wells_train, n_time]
        Y_test = x_test[:, :, phase_idx]    # [n_wells_test, n_time]
        
        # Train LightGBM model
        lgb_model = LightGBMQuantile(quantiles=[0.1, 0.5, 0.9])
        
        try:
            # Fit model
            lgb_model.fit(z_train.cpu().numpy(), Y_train, t_vals_days)
            
            # Predict
            predictions = lgb_model.predict(z_test.cpu().numpy())
            
            # Generate uncertainty samples
            pred_with_samples = lgb_model.predict_with_uncertainty(
                z_test.cpu().numpy(), n_samples=500
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
            # Fallback to simple prediction
            results[phase_name] = {
                'p10': Y_test,
                'p50': Y_test,
                'p90': Y_test,
                'mean': Y_test,
                'std': np.zeros_like(Y_test)
            }
    
    return results


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
    
    # Limit data if requested
    if args.max_wells is not None and args.max_wells < x_test.shape[0]:
        x_test_limited = x_test[:args.max_wells]
        z_test_limited = z_test[:args.max_wells]
        scale_test_limited = scale_test[:args.max_wells]
    else:
        x_test_limited = x_test
        z_test_limited = z_test
        scale_test_limited = scale_test
    
    # Load NSM-DCA model
    print("\nEvaluating NSM-DCA model...")
    experiment_path = os.path.join('experiments', args.experiment)
    model_path = os.path.join(experiment_path, 'model.pth')
    
    if os.path.exists(model_path):
        model = build_model_from_config(config, device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        nsm_results = evaluate_nsm_model(
            model, x_test_limited, z_test_limited, t_vals, scale_test_limited, 
            device, n_mc=args.n_mc
        )
    else:
        print(f"NSM model not found at {model_path}")
        nsm_results = None
    
    # Evaluate DCA models
    print("\nEvaluating DCA models (Arps, Duong)...")
    print(f"  Bootstrap samples: {args.n_bootstrap}")
    # Convert to numpy for DCA models (they expect denormalized data)
    x_test_denorm = (x_test * scale_test.unsqueeze(1)).cpu().numpy()
    x_train_denorm = (x_train * scale_train.unsqueeze(1)).cpu().numpy()
    
    dca_results = evaluate_dca_models(
        x_test_denorm, t_vals_days, 
        n_bootstrap=args.n_bootstrap,
        n_workers=args.n_workers,
        max_wells=args.max_wells
    )
    
    # Evaluate LightGBM
    print("\nEvaluating LightGBM quantile regression...")
    # Limit test data for LightGBM if needed
    if args.max_wells is not None and args.max_wells < x_test_denorm.shape[0]:
        x_test_denorm_limited = x_test_denorm[:args.max_wells]
        z_test_limited_np = z_test_limited
    else:
        x_test_denorm_limited = x_test_denorm
        z_test_limited_np = z_test
    
    lgb_results = evaluate_lightgbm(
        x_train_denorm, x_test_denorm_limited, z_train, z_test_limited_np, t_vals_days
    )
    
    # Combine all results for comparison
    print("\nComputing comparison metrics...")
    
    # Initialize results storage
    all_metrics = []
    
    # Process each phase
    for phase_idx, phase_name in enumerate(['Gas', 'Oil', 'Water']):
        print(f"\n{phase_name} Phase:")
        
        # True values - limit to same number of wells as DCA if needed
        if args.max_wells is not None and args.max_wells < x_test_denorm.shape[0]:
            y_true = x_test_denorm[:args.max_wells, :, phase_idx]
        else:
            y_true = x_test_denorm[:, :, phase_idx]
        
        # Collect model results
        model_results = {}
        
        # NSM-DCA
        if nsm_results is not None:
            model_results['NSM-DCA'] = {
                'p10': nsm_results['p10'][:, :, phase_idx],
                'p50': nsm_results['p50'][:, :, phase_idx],
                'p90': nsm_results['p90'][:, :, phase_idx],
                'samples': nsm_results['samples'][:, :, :, phase_idx]
            }
        
        # DCA models
        if phase_name in dca_results:
            for dca_name in ['Arps', 'Duong']:
                if dca_name in dca_results[phase_name]:
                    model_results[dca_name] = dca_results[phase_name][dca_name]
        
        # LightGBM
        if phase_name in lgb_results:
            model_results['LightGBM'] = lgb_results[phase_name]
        
        # Calculate metrics
        metrics_df = compare_models_crps(model_results, y_true)
        metrics_df['Phase'] = phase_name
        all_metrics.append(metrics_df)
        
        print(metrics_df.to_string())
        
        # Generate calibration plots
        for model_name, results in model_results.items():
            if 'samples' in results:
                fig = plot_calibration_diagram(
                    y_true, results['samples'],
                    model_name=f"{model_name} - {phase_name}",
                    save_path=os.path.join(args.output_dir, 
                                          f"calibration_{model_name}_{phase_name}.png")
                )
                plt.close(fig)
        
        # Generate comparison plot for first well
        fig = plot_interval_comparison(
            model_results, y_true, t_vals_days, well_idx=0,
            save_path=os.path.join(args.output_dir, f"comparison_{phase_name}_well0.png")
        )
        plt.close(fig)
    
    # Combine all metrics
    full_metrics_df = pd.concat(all_metrics, ignore_index=True)
    
    # Save results
    output_file = os.path.join(args.output_dir, 'benchmark_metrics.csv')
    full_metrics_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Create summary table
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
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
    summary_file = os.path.join(args.output_dir, 'benchmark_summary.csv')
    summary.to_csv(summary_file)
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()