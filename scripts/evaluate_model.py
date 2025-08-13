#!/usr/bin/env python
"""
Master evaluation script with clean, organized output for publication.
Replaces the old evaluate_model.py with better organization.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Run comprehensive model evaluation')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment name to evaluate')
    parser.add_argument('--components', nargs='+',
                       default=['all'],
                       choices=['metrics', 'figures', 'tables', 'gradients', 'all'],
                       help='Components to evaluate')
    parser.add_argument('--clean', action='store_true',
                       help='Clean existing evaluation directory first')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu/cuda)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Determine which components to run
    if 'all' in args.components:
        components = ['metrics', 'scatter', 'trajectories', 'tables']
        run_gradients = True
    else:
        component_map = {
            'metrics': ['metrics'],
            'figures': ['scatter', 'trajectories'],
            'tables': ['tables'],
        }
        components = []
        for comp in args.components:
            if comp in component_map:
                components.extend(component_map[comp])
        run_gradients = 'gradients' in args.components
    
    print(f"\n{'='*60}")
    print(f"Model Evaluation Pipeline: {args.experiment}")
    print(f"{'='*60}\n")
    
    # Run comprehensive evaluation
    if components:
        cmd = [
            sys.executable,
            'scripts/evaluate_comprehensive.py',
            '--experiment', args.experiment,
            '--components'] + components
        
        if args.clean:
            cmd.append('--clean')
        if args.device:
            cmd.extend(['--device', args.device])
        
        print("📊 Running comprehensive evaluation...")
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print("❌ Evaluation failed")
            return 1
    
    # Run gradient analysis if requested
    if run_gradients:
        print("\n🎨 Running gradient analysis...")
        cmd = [
            sys.executable,
            'scripts/eval_production_gradients.py',
            '--experiment', args.experiment,
            '--well-idx', '0'  # Just first well for publication
        ]
        if args.device:
            cmd.extend(['--device', args.device])
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print("❌ Gradient analysis failed")
            return 1
    
    # Print summary of outputs
    exp_dir = Path('experiments') / args.experiment / 'evaluation'
    if exp_dir.exists():
        print(f"\n{'='*60}")
        print("✅ Evaluation Complete!")
        print(f"{'='*60}\n")
        print("📁 Output Structure:")
        print(f"  {exp_dir}/")
        print(f"    ├── figures/       # Publication-ready figures (300 DPI)")
        print(f"    ├── metrics/       # CSV metrics files")
        print(f"    ├── tables/        # LaTeX tables")
        if run_gradients:
            print(f"    └── gradients/     # Gradient visualizations")
        print("\n📊 Key Files:")
        
        # List key files if they exist
        key_files = [
            ('figures/scatter_plot.png', 'Prediction vs actual scatter plots'),
            ('figures/trajectories.png', 'Production trajectories with uncertainty'),
            ('metrics/summary.csv', 'Overall model metrics'),
            ('metrics/by_phase.csv', 'Phase-specific metrics'),
            ('tables/r2_metrics.tex', 'R² LaTeX table'),
            ('tables/rmse_metrics.tex', 'RMSE LaTeX table'),
        ]
        
        for rel_path, description in key_files:
            full_path = exp_dir / rel_path
            if full_path.exists():
                print(f"    ✓ {rel_path:<30} {description}")
    
    return 0

if __name__ == "__main__":
    exit(main())