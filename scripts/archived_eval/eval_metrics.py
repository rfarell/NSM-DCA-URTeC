#!/usr/bin/env python
# scripts/eval_metrics.py
"""
Calculate and report model evaluation metrics.
"""

import os
import sys
import torch
import argparse
import random
import numpy as np
import pandas as pd
from tabulate import tabulate

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.evaluation import load_experiment, predict_from_t
from src.utils import evaluate_model, print_metrics

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Calculate and report model evaluation metrics')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Name of the experiment to evaluate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--num-mc', type=int, default=10,
                        help='Number of Monte Carlo samples for prediction')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save metrics CSV file (optional)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed metrics')
    return parser.parse_args()

def calculate_time_specific_metrics(model, z_data, t_vals, x_data, days, device,
                                    *, scale_tensor=None, num_mc=10):
    """
    Calculate metrics for predictions at specific time points.
    
    Args:
        model: Trained model
        z_data: Well static features
        t_vals: Time values tensor
        x_data: Actual production data
        days: Time points in days
        device: Computation device
        num_mc: Number of Monte Carlo samples
        
    Returns:
        pandas.DataFrame: DataFrame with metrics
    """
    phase_names = ['Gas', 'Oil', 'Water']
    time_indices = list(range(len(days)))
    
    # Initialize results storage
    metrics_data = []
    
    # For each starting time point
    for start_idx in range(len(days)-1):  # Skip the last day as starting point
        start_day = days[start_idx]
        
        # For each ending time point (must be after start)
        for end_idx in range(start_idx+1, len(days)):
            end_day = days[end_idx]
            duration = end_day - start_day
            
            # Generate prediction
            predictions = predict_from_t(
                model, z_data, t_vals, x_data, 
                start_idx=start_idx, 
                num_mc=num_mc, 
                device=device
            )
            
            # Calculate the relative ending position
            rel_end_idx = end_idx - start_idx
            
            # For each phase
            for phase_idx, phase_name in enumerate(phase_names):
                # Extract predictions and actual values for this phase
                pred_value = predictions[:, rel_end_idx-1, phase_idx].flatten()
                actual_value = x_data[:, end_idx, phase_idx].cpu().numpy().flatten()

                # De-normalise if scale tensor provided
                if scale_tensor is not None:
                    pred_value   = pred_value   * scale_tensor[:, phase_idx].cpu().numpy().flatten()
                    actual_value = actual_value * scale_tensor[:, phase_idx].cpu().numpy().flatten()
                
                # Calculate metrics
                r2 = 1 - np.sum((actual_value - pred_value) ** 2) / max(1e-10, np.sum((actual_value - np.mean(actual_value)) ** 2))
                rmse = np.sqrt(np.mean((actual_value - pred_value) ** 2))
                mae = np.mean(np.abs(actual_value - pred_value))
                mape = np.mean(np.abs((actual_value - pred_value) / np.maximum(1e-10, actual_value))) * 100
                
                # Store results
                metrics_data.append({
                    'Start Day': start_day,
                    'End Day': end_day,
                    'Duration': duration,
                    'Phase': phase_name,
                    'R2': r2,
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE': mape
                })
    
    return pd.DataFrame(metrics_data)

def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load experiment
    print(f"Loading experiment: {args.experiment}")
    exp_data = load_experiment(args.experiment, device=args.device)
    
    model = exp_data['model']
    device = exp_data['device']
    eval_dir = exp_data['eval_dir']
    x_train = exp_data['x_train']
    z_train = exp_data['z_train']
    x_test  = exp_data['x_test']
    z_test  = exp_data['z_test']
    dp      = exp_data['data_processor']

    # Per-well scaling tensors for de-normalisation
    scale_train = getattr(dp, 'scale_train', None)
    scale_test  = getattr(dp, 'scale_test',  None)
    t_vals = exp_data['t_vals']
    
    # Time points from the data processor
    days = [0, 30, 60, 90, 180, 360, 720, 1080]
    
    # Calculate overall metrics
    print("Calculating overall metrics...")
    train_metrics, _ = evaluate_model(model, x_train, z_train, t_vals,
                                      scale_tensor=scale_train,
                                      num_mc=args.num_mc, device=device)

    test_metrics, _ = evaluate_model(model, x_test, z_test, t_vals,
                                     scale_tensor=scale_test,
                                     num_mc=args.num_mc, device=device)
    
    # Print metrics if verbose
    if args.verbose:
        print("\n=== Training Set Metrics ===")
        print_metrics(train_metrics, prefix="Training")
        
        print("\n=== Test Set Metrics ===")
        print_metrics(test_metrics, prefix="Test")
    
    # Calculate time-specific metrics
    print("Calculating time-specific metrics...")
    metrics_df = calculate_time_specific_metrics(
        model, z_test, t_vals, x_test, days, device,
        scale_tensor=scale_test, num_mc=args.num_mc
    )
    
    # Group metrics by phase and calculate averages (including only numeric columns)
    # Get all numeric columns except the grouping columns
    numeric_cols = metrics_df.select_dtypes(include=['number']).columns.tolist()
    if 'Phase' in numeric_cols: numeric_cols.remove('Phase')
    if 'Duration' in numeric_cols: numeric_cols.remove('Duration')
    
    # Calculate mean for each group
    phase_avg = metrics_df.groupby('Phase')[numeric_cols].mean().reset_index()
    duration_avg = metrics_df.groupby('Duration')[numeric_cols].mean().reset_index()
    
    # Print summary tables
    print("\n=== Metrics by Production Phase ===")
    print(tabulate(phase_avg, headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    print("\n=== Metrics by Prediction Duration ===")
    print(tabulate(duration_avg, headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    # Save metrics to file if output path provided
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(eval_dir, 'metrics.csv')
    
    metrics_df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")
    
    # Also save summary metrics
    phase_avg.to_csv(os.path.join(eval_dir, 'metrics_by_phase.csv'), index=False)
    duration_avg.to_csv(os.path.join(eval_dir, 'metrics_by_duration.csv'), index=False)
    
    # Create a summary text file with overall metrics
    summary_path = os.path.join(eval_dir, 'metrics_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"=== Overall Metrics for Experiment: {args.experiment} ===\n\n")
        
        f.write("Training Set Metrics:\n")
        f.write(f"Overall R²: {train_metrics['Overall_R2']:.4f}\n")
        f.write(f"Overall RMSE: {train_metrics['Overall_RMSE']:.4f}\n\n")
        
        f.write("Test Set Metrics:\n")
        f.write(f"Overall R²: {test_metrics['Overall_R2']:.4f}\n")
        f.write(f"Overall RMSE: {test_metrics['Overall_RMSE']:.4f}\n\n")
        
        for phase in ['Gas', 'Oil', 'Water']:
            f.write(f"{phase} Test R² (final): {test_metrics[f'{phase}_R2_final']:.4f}\n")
            f.write(f"{phase} Test RMSE (final): {test_metrics[f'{phase}_RMSE_final']:.4f}\n")
        
    print(f"Summary metrics saved to {summary_path}")

if __name__ == "__main__":
    main()