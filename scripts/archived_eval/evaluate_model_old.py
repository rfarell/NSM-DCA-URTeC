#!/usr/bin/env python
# scripts/evaluate_model.py
"""
Main evaluation script that runs all evaluation components.
"""

import os
import sys
import torch
import argparse
import random
import numpy as np
import subprocess

# Enable TensorFloat32 (TF32) for better performance on compatible GPUs
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Detailed evaluation of a trained model')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Name of the experiment to evaluate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--num-mc', type=int, default=10,
                        help='Number of Monte Carlo samples for prediction for scatter plots')
    parser.add_argument('--num-traj', type=int, default=100,
                        help='Number of trajectory samples for trajectory plots')
    parser.add_argument('--num-wells', type=int, default=10,
                        help='Number of random wells to sample for trajectory plots')
    parser.add_argument('--eval-type', type=str, default='all', 
                        choices=['scatter', 'trajectories', 'metrics', 'gradients', 'tables', 'all'],
                        help='Type of evaluation to perform')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--specific-wells', type=str, default=None,
                        help='Comma-separated list of specific well indices to plot (for trajectory plots)')
    return parser.parse_args()

def main():
    """Main function to run evaluation components."""
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Choose which scripts to run based on eval-type
    scripts_to_run = []
    
    if args.eval_type in ['scatter', 'all']:
        cmd = [
            sys.executable, 'scripts/eval_scatter_plots.py',
            '--experiment', args.experiment,
            '--num-mc', str(args.num_mc),
            '--seed', str(args.seed)
        ]
        if args.device:
            cmd.extend(['--device', args.device])
        scripts_to_run.append(cmd)
    
    if args.eval_type in ['trajectories', 'all']:
        cmd = [
            sys.executable, 'scripts/eval_trajectories.py',
            '--experiment', args.experiment,
            '--num-traj', str(args.num_traj),
            '--num-wells', str(args.num_wells),
            '--seed', str(args.seed)
        ]
        if args.device:
            cmd.extend(['--device', args.device])
        if args.specific_wells:
            cmd.extend(['--specific-wells', args.specific_wells])
        scripts_to_run.append(cmd)
    
    if args.eval_type in ['metrics', 'all']:
        cmd = [
            sys.executable, 'scripts/eval_metrics.py',
            '--experiment', args.experiment,
            '--num-mc', str(args.num_mc),
            '--seed', str(args.seed),
            '--verbose'
        ]
        if args.device:
            cmd.extend(['--device', args.device])
        scripts_to_run.append(cmd)
        
    if args.eval_type in ['gradients', 'all']:
        cmd = [
            sys.executable, 'scripts/eval_production_gradients.py',
            '--experiment', args.experiment,
            '--seed', str(args.seed),
            '--well-idx', '0',  # Use well #0 by default
            '--vis-type', 'both'
        ]
        if args.device:
            cmd.extend(['--device', args.device])
        scripts_to_run.append(cmd)
        
    if args.eval_type in ['tables', 'all']:
        cmd = [
            sys.executable, 'scripts/eval_latex_tables.py',
            '--experiment', args.experiment,
            '--num-mc', str(args.num_mc),
            '--seed', str(args.seed)
        ]
        if args.device:
            cmd.extend(['--device', args.device])
        scripts_to_run.append(cmd)
    
    # Run each script
    for i, cmd in enumerate(scripts_to_run):
        print(f"\n[{i+1}/{len(scripts_to_run)}] Running: {' '.join(cmd)}\n")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {cmd}: {e}")
    
    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()