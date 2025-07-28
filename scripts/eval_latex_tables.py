#!/usr/bin/env python
# scripts/eval_latex_tables.py
"""
Generate LaTeX tables of model performance metrics for different time periods.
This script creates tables for RMSE and R² values between predicted and actual
production for gas, oil, and water across various prediction horizons.
"""

import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.evaluation import load_experiment, predict_from_t

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate LaTeX tables of model metrics')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Name of the experiment to evaluate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--num-mc', type=int, default=10,
                        help='Number of Monte Carlo samples for prediction')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save LaTeX tables (defaults to experiment evaluation dir)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def calculate_metrics_table(model, z_data, t_vals, x_data, days, device, num_mc=10):
    """
    Calculate R² and RMSE metrics for predictions from different start times to different end times.
    
    Args:
        model: Trained model
        z_data: Static features tensor
        t_vals: Time values tensor
        x_data: Actual production data tensor
        days: List of day values
        device: Computation device
        num_mc: Number of Monte Carlo samples
        
    Returns:
        dict: Dictionary with metric tables for each phase and metric type
    """
    # Initialize result dictionaries
    phases = ['Gas', 'Oil', 'Water']
    phase_indices = {phase: idx for idx, phase in enumerate(phases)}
    
    # Create empty DataFrames for each metric and phase
    rmse_tables = {phase: pd.DataFrame(index=days[:-1], columns=days[1:]) for phase in phases}
    r2_tables = {phase: pd.DataFrame(index=days[:-1], columns=days[1:]) for phase in phases}
    
    # Iterate through all possible start times (except the last one)
    for start_idx, start_day in enumerate(days[:-1]):
        print(f"Calculating metrics for predictions starting at t={start_day} days...")
        
        # Generate predictions starting from this time point
        predictions = predict_from_t(
            model, z_data, t_vals, x_data, 
            start_idx=start_idx, 
            num_mc=num_mc, 
            device=device
        )
        
        # For each possible end time after the start time
        for rel_end_idx, end_day in enumerate(days[start_idx+1:], 1):
            end_idx = start_idx + rel_end_idx
            
            # Calculate metrics for each phase
            for phase, phase_idx in phase_indices.items():
                # Extract predictions and actual values for this phase
                pred_value = predictions[:, rel_end_idx, phase_idx].flatten()  # Removed -1 to match scatter plots
                actual_value = x_data[:, end_idx, phase_idx].cpu().numpy().flatten()
                
                # Calculate R²
                r2 = 1 - np.sum((actual_value - pred_value) ** 2) / max(1e-10, np.sum((actual_value - np.mean(actual_value)) ** 2))
                
                # Calculate RMSE
                rmse = np.sqrt(np.mean((actual_value - pred_value) ** 2))
                
                # Store metrics in tables
                rmse_tables[phase].loc[start_day, end_day] = rmse
                r2_tables[phase].loc[start_day, end_day] = r2
    
    return {
        'rmse': rmse_tables,
        'r2': r2_tables
    }

def format_latex_table(df, metric_name, phase_name, days):
    """
    Format a DataFrame as a LaTeX table.
    
    Args:
        df: DataFrame with metrics
        metric_name: Name of the metric (RMSE or R²)
        phase_name: Name of the phase (Gas, Oil, Water)
        days: List of day values
        
    Returns:
        str: LaTeX table
    """
    # Format the metric name for LaTeX
    if metric_name.lower() == 'rmse':
        metric_label = 'RMSE'
    else:
        metric_label = 'R$^2$'
    
    # Determine a better format for numbers
    if metric_name.lower() == 'rmse':
        float_format = lambda x: f"{x:.4f}" if pd.notnull(x) else ""
    else:  # R²
        float_format = lambda x: f"{x:.3f}" if pd.notnull(x) else ""
    
    # Create LaTeX table
    latex = []
    latex.append("\\begin{table}")
    latex.append("    \\centering")
    latex.append(f"    \\caption{{{metric_label} of Predicted vs Actual {phase_name} Production at Different Time Periods}}")
    latex.append(f"    \\label{{tab:{metric_name.lower()}_{phase_name.lower()}}}")
    
    # Start tabular environment
    num_cols = len(days) - 1
    latex.append(f"    \\begin{{tabular}}{{c|{('c' * num_cols)}}}")
    latex.append("        \\toprule")
    
    # Header row with multirow for "Initial Condition (Days)"
    latex.append("        \\multirow{2}{*}{Initial Condition (Days)} & \\multicolumn{" + 
                f"{num_cols}" + "}{c}{Prediction Time (Days)} \\\\")
    latex.append("        \\cmidrule{2-" + f"{num_cols+1}" + "}")
    
    # Column headers (days)
    header_row = "        & " + " & ".join([f"{day}" for day in days[1:]]) + " \\\\"
    latex.append(header_row)
    latex.append("        \\midrule")
    
    # Data rows
    for start_day in days[:-1]:
        row_values = []
        for end_day in days[1:]:
            if end_day > start_day:
                value = df.loc[start_day, end_day]
                row_values.append(float_format(value))
            else:
                row_values.append("")
        
        row = f"        {start_day}   & " + " & ".join(row_values) + " \\\\"
        latex.append(row)
    
    # End of table
    latex.append("        \\bottomrule")
    latex.append("    \\end{tabular}")
    latex.append("    \\vspace{0.2cm}")
    
    # Determine appropriate caption based on metric
    if metric_name.lower() == 'rmse':
        caption = f"    \\caption*{{The table presents RMSE values for predicted versus actual {phase_name.lower()} production across varying time periods. Lower RMSE values indicate better predictive accuracy.}}"
    else:  # R²
        caption = f"    \\caption*{{The table presents R$^2$ values for predicted versus actual {phase_name.lower()} production across varying time periods. Higher values (closer to 1.0) indicate better predictive accuracy.}}"
    
    latex.append(caption)
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def main():
    """Main function to generate LaTeX tables."""
    # Parse arguments
    args = parse_args()
    
    # Load experiment
    print(f"Loading experiment: {args.experiment}")
    exp_data = load_experiment(args.experiment, device=args.device)
    
    model = exp_data['model']
    device = exp_data['device']
    eval_dir = exp_data['eval_dir']
    x_test = exp_data['x_test']
    z_test = exp_data['z_test']
    t_vals = exp_data['t_vals']
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Create tables directory in evaluation dir
        output_dir = os.path.join(eval_dir, 'tables')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"LaTeX tables will be saved to: {output_dir}")
    
    # Time points
    days = [0, 30, 60, 90, 180, 360, 720, 1080]
    
    # Calculate metrics tables
    print("Calculating metrics tables...")
    metrics_tables = calculate_metrics_table(
        model, z_test, t_vals, x_test, days, device, num_mc=args.num_mc
    )
    
    # Generate and save LaTeX tables
    phases = ['Gas', 'Oil', 'Water']
    metrics = ['rmse', 'r2']
    
    for metric in metrics:
        for phase in phases:
            # Format table
            latex_table = format_latex_table(
                metrics_tables[metric][phase], 
                metric.upper(), 
                phase, 
                days
            )
            
            # Save to file
            filename = f"{metric.lower()}_{phase.lower()}.tex"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(latex_table)
            
            print(f"Saved {metric.upper()} table for {phase} to {filepath}")
    
    print(f"\nAll LaTeX tables have been saved to {output_dir}")

if __name__ == "__main__":
    main()