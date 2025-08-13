#!/usr/bin/env python
"""
Comprehensive evaluation script with organized output structure for publication-ready figures.
Output structure:
    experiments/exp_name/evaluation/
        ├── figures/           # Publication-ready figures (300 DPI PNG)
        │   ├── scatter_plot.png
        │   ├── trajectories.png
        │   └── gradients.png
        ├── metrics/           # CSV and text metrics
        │   ├── summary.csv
        │   ├── by_phase.csv
        │   └── by_duration.csv
        └── tables/            # LaTeX tables
            ├── r2_metrics.tex
            └── rmse_metrics.tex
"""

import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.evaluation import load_experiment, predict_from_t
from src.utils import evaluate_model, print_metrics

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
})

def parse_args():
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment name')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cpu/cuda)')
    parser.add_argument('--num-mc', type=int, default=10,
                       help='Monte Carlo samples')
    parser.add_argument('--components', nargs='+', 
                       default=['metrics', 'scatter', 'trajectories', 'tables'],
                       choices=['metrics', 'scatter', 'trajectories', 'gradients', 'tables'],
                       help='Evaluation components to run')
    parser.add_argument('--clean', action='store_true',
                       help='Clean existing evaluation directory first')
    return parser.parse_args()

def setup_directories(exp_dir, clean=False):
    """Create organized output directory structure."""
    eval_dir = Path(exp_dir) / 'evaluation'
    
    if clean and eval_dir.exists():
        import shutil
        shutil.rmtree(eval_dir)
    
    # Create organized subdirectories
    dirs = {
        'figures': eval_dir / 'figures',
        'metrics': eval_dir / 'metrics',
        'tables': eval_dir / 'tables',
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def evaluate_metrics(exp, output_dirs):
    """Calculate and save comprehensive metrics."""
    print("\n📊 Calculating Metrics...")
    
    model = exp['model']
    x_test = exp['x_test']
    z_test = exp['z_test']
    t_vals = exp['t_vals']
    device = exp['device']
    scale_test = exp.get('scale_test', None)
    
    # Calculate metrics
    metrics, predictions = evaluate_model(
        model, x_test, z_test, t_vals,
        scale_tensor=scale_test,
        num_mc=10,
        device=device
    )
    
    # Save summary metrics
    summary_df = pd.DataFrame([metrics])
    summary_path = output_dirs['metrics'] / 'summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✓ Saved summary metrics to {summary_path}")
    
    # Calculate phase-specific metrics
    phase_metrics = []
    for phase in ['Gas', 'Oil', 'Water']:
        phase_data = {
            'Phase': phase,
            'R2_final': metrics[f'{phase}_R2_final'],
            'RMSE_final': metrics[f'{phase}_RMSE_final'],
            'R2_all': metrics[f'{phase}_R2_all'],
            'RMSE_all': metrics[f'{phase}_RMSE_all']
        }
        phase_metrics.append(phase_data)
    
    phase_df = pd.DataFrame(phase_metrics)
    phase_path = output_dirs['metrics'] / 'by_phase.csv'
    phase_df.to_csv(phase_path, index=False)
    print(f"  ✓ Saved phase metrics to {phase_path}")
    
    return metrics, predictions

def create_scatter_plot(exp, output_dirs):
    """Create publication-ready scatter plot."""
    print("\n🎨 Creating Scatter Plot...")
    
    model = exp['model']
    x_test = exp['x_test']
    z_test = exp['z_test']
    t_vals = exp['t_vals']
    device = exp['device']
    scale_test = exp.get('scale_test', None)
    
    # Get predictions
    _, predictions = evaluate_model(
        model, x_test, z_test, t_vals,
        scale_tensor=scale_test,
        num_mc=10,
        device=device
    )
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    phase_names = ['Gas', 'Oil', 'Water']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Professional colors
    
    for i, (phase, color) in enumerate(zip(phase_names, colors)):
        ax = axes[i]
        
        # Get final time point data
        if scale_test is not None:
            actual = (x_test[:, -1, i] * scale_test[:, i]).detach().cpu().numpy()
        else:
            actual = x_test[:, -1, i].detach().cpu().numpy()
        pred = predictions[:, -1, i].detach().cpu().numpy()
        
        # Calculate R² and RMSE
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        
        # Create scatter plot
        ax.scatter(actual, pred, alpha=0.6, s=20, c=color, edgecolors='none')
        
        # Add diagonal line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, linewidth=1)
        
        # Labels and title
        ax.set_xlabel(f'Actual {phase} Production', fontsize=10)
        ax.set_ylabel(f'Predicted {phase} Production', fontsize=10)
        ax.set_title(f'{phase} (R²={r2:.3f}, RMSE={rmse:.1f})', fontsize=11)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.suptitle('Production Predictions at Final Time (t=1080 days)', fontsize=12, y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = output_dirs['figures'] / 'scatter_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved scatter plot to {output_path}")

def create_trajectory_plot(exp, output_dirs, num_wells=5):
    """Create publication-ready trajectory plots."""
    print("\n📈 Creating Trajectory Plots...")
    
    model = exp['model']
    x_test = exp['x_test']
    z_test = exp['z_test']
    t_vals = exp['t_vals']
    device = exp['device']
    scale_test = exp.get('scale_test', None)
    days = exp['data_processor'].days
    
    # Select random wells
    n_wells = x_test.shape[0]
    well_indices = np.random.choice(n_wells, min(num_wells, n_wells), replace=False)
    
    # Create figure
    fig, axes = plt.subplots(num_wells, 3, figsize=(12, 2.5 * num_wells))
    if num_wells == 1:
        axes = axes.reshape(1, -1)
    
    phase_names = ['Gas', 'Oil', 'Water']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for well_idx, well_id in enumerate(well_indices):
        # Get predictions for this well
        x0 = x_test[well_id:well_id+1, 0, :]
        z = z_test[well_id:well_id+1]
        
        # Run ODE solver for predictions
        from torchdiffeq import odeint
        def ode_func(t, x):
            return model(t, x, z)
        
        # Generate multiple trajectories
        trajectories = []
        for _ in range(10):
            pred = odeint(ode_func, x0, t_vals, method='midpoint')
            pred = pred.squeeze(1)  # Remove batch dimension
            if scale_test is not None:
                pred = pred * scale_test[well_id]
            trajectories.append(pred.detach().cpu().numpy())
        
        trajectories = np.array(trajectories)
        mean_traj = trajectories.mean(axis=0)
        std_traj = trajectories.std(axis=0)
        
        # Get actual data
        if scale_test is not None:
            actual = (x_test[well_id] * scale_test[well_id]).detach().cpu().numpy()
        else:
            actual = x_test[well_id].detach().cpu().numpy()
        
        # Plot each phase
        for phase_idx, (phase, color) in enumerate(zip(phase_names, colors)):
            ax = axes[well_idx, phase_idx]
            
            # Plot actual
            ax.plot(days, actual[:, phase_idx], 'o', color=color, 
                   markersize=4, label='Actual', alpha=0.8)
            
            # Plot prediction with uncertainty
            ax.plot(days, mean_traj[:, phase_idx], '-', color=color, 
                   linewidth=2, label='Predicted')
            ax.fill_between(days, 
                           mean_traj[:, phase_idx] - 2*std_traj[:, phase_idx],
                           mean_traj[:, phase_idx] + 2*std_traj[:, phase_idx],
                           color=color, alpha=0.2)
            
            # Formatting
            ax.set_xlabel('Time (days)' if well_idx == num_wells-1 else '')
            ax.set_ylabel(f'{phase} Production')
            ax.set_title(f'Well {well_id} - {phase}')
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_xlim(0, 1100)
            if well_idx == 0 and phase_idx == 0:
                ax.legend(loc='upper left', frameon=True, fancybox=False)
    
    plt.suptitle('Production Trajectories with Uncertainty (±2σ)', fontsize=12, y=1.01)
    plt.tight_layout()
    
    # Save figure
    output_path = output_dirs['figures'] / 'trajectories.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved trajectories to {output_path}")

def create_latex_tables(metrics, output_dirs):
    """Create LaTeX tables for publication."""
    print("\n📝 Creating LaTeX Tables...")
    
    # R² table
    r2_data = []
    for phase in ['Gas', 'Oil', 'Water']:
        r2_data.append({
            'Phase': phase,
            'Final Time': f"{metrics[f'{phase}_R2_final']:.4f}",
            'All Times': f"{metrics[f'{phase}_R2_all']:.4f}"
        })
    
    r2_df = pd.DataFrame(r2_data)
    r2_latex = r2_df.to_latex(index=False, escape=False, 
                              column_format='lcc',
                              caption='R² values for production predictions',
                              label='tab:r2_metrics')
    
    r2_path = output_dirs['tables'] / 'r2_metrics.tex'
    with open(r2_path, 'w') as f:
        f.write(r2_latex)
    print(f"  ✓ Saved R² table to {r2_path}")
    
    # RMSE table
    rmse_data = []
    for phase in ['Gas', 'Oil', 'Water']:
        rmse_data.append({
            'Phase': phase,
            'Final Time': f"{metrics[f'{phase}_RMSE_final']:.2f}",
            'All Times': f"{metrics[f'{phase}_RMSE_all']:.2f}"
        })
    
    rmse_df = pd.DataFrame(rmse_data)
    rmse_latex = rmse_df.to_latex(index=False, escape=False,
                                  column_format='lcc',
                                  caption='RMSE values for production predictions',
                                  label='tab:rmse_metrics')
    
    rmse_path = output_dirs['tables'] / 'rmse_metrics.tex'
    with open(rmse_path, 'w') as f:
        f.write(rmse_latex)
    print(f"  ✓ Saved RMSE table to {rmse_path}")

def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    print(f"\n{'='*60}")
    print(f"Comprehensive Evaluation: {args.experiment}")
    print(f"{'='*60}")
    
    # Load experiment
    exp = load_experiment(args.experiment, device=args.device)
    exp_dir = os.path.join('experiments', args.experiment)
    
    # Setup directories
    output_dirs = setup_directories(exp_dir, clean=args.clean)
    print(f"✓ Output directories created in {exp_dir}/evaluation/")
    
    # Run selected components
    metrics = None
    
    if 'metrics' in args.components:
        metrics, _ = evaluate_metrics(exp, output_dirs)
    
    if 'scatter' in args.components:
        create_scatter_plot(exp, output_dirs)
    
    if 'trajectories' in args.components:
        create_trajectory_plot(exp, output_dirs, num_wells=5)
    
    if 'tables' in args.components and metrics is not None:
        create_latex_tables(metrics, output_dirs)
    
    print(f"\n{'='*60}")
    print("✓ Evaluation Complete!")
    print(f"  Results saved to: {exp_dir}/evaluation/")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()