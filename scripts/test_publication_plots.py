#!/usr/bin/env python
# scripts/test_publication_plots.py
"""
Test script to generate sample training statistics and create publication plots.
"""

import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from publication_plots import create_all_publication_plots

def generate_sample_stats():
    """Generate sample training statistics for testing plots."""
    
    # Simulate training over 5000 steps
    steps = list(range(0, 5001, 100))
    n_steps = len(steps)
    
    # Generate loss curves (decreasing with noise)
    losses = []
    nll_values = []
    kl_values = []
    
    for i, step in enumerate(steps):
        # Exponentially decreasing losses with some noise
        progress = i / (n_steps - 1)
        
        # NLL starts high and decreases
        nll = 1000 * np.exp(-3 * progress) + 50 + np.random.normal(0, 5)
        nll_values.append(max(10, nll))
        
        # KL starts low and increases slightly then stabilizes
        kl = 0.1 * np.exp(2 * min(progress, 0.2)) + np.random.normal(0, 0.05)
        kl_values.append(max(0.01, kl))
        
        # Total loss
        losses.append(nll_values[-1] + kl_values[-1])
    
    # Generate R² curves (increasing)
    train_r2 = []
    test_r2 = []
    train_rmse = []
    test_rmse = []
    
    for i in range(n_steps):
        progress = i / (n_steps - 1)
        
        # R² increases from 0 to ~0.95
        train_r2_val = 0.95 * (1 - np.exp(-4 * progress)) + np.random.normal(0, 0.01)
        train_r2.append(min(0.99, max(0, train_r2_val)))
        
        # Test R² slightly lower
        test_r2_val = train_r2[-1] - 0.02 - np.random.normal(0, 0.005)
        test_r2.append(min(0.98, max(0, test_r2_val)))
        
        # RMSE decreases (inverse of R²)
        train_rmse_val = 100 * np.exp(-3 * progress) + 5 + np.random.normal(0, 1)
        train_rmse.append(max(1, train_rmse_val))
        
        # Test RMSE slightly higher
        test_rmse_val = train_rmse[-1] * 1.1 + np.random.normal(0, 0.5)
        test_rmse.append(max(1, test_rmse_val))
    
    # Generate per-phase R² and RMSE (optional)
    phases = ['gas', 'oil', 'water']
    phase_stats = {}
    
    for phase in phases:
        phase_train_r2 = []
        phase_test_r2 = []
        phase_train_rmse = []
        phase_test_rmse = []
        
        # Add some variation between phases
        phase_offset = {'gas': 0, 'oil': -0.03, 'water': -0.05}[phase]
        rmse_offset = {'gas': 1.0, 'oil': 1.2, 'water': 1.5}[phase]
        
        for i in range(n_steps):
            progress = i / (n_steps - 1)
            
            # Phase-specific R²
            phase_train = 0.95 * (1 - np.exp(-4 * progress)) + phase_offset + np.random.normal(0, 0.01)
            phase_train_r2.append(min(0.99, max(0, phase_train)))
            
            phase_test = phase_train_r2[-1] - 0.02 - np.random.normal(0, 0.005)
            phase_test_r2.append(min(0.98, max(0, phase_test)))
            
            # Phase-specific RMSE (inversely related to R²)
            phase_train_rmse_val = (100 * np.exp(-3 * progress) + 5) * rmse_offset + np.random.normal(0, 1)
            phase_train_rmse.append(max(1, phase_train_rmse_val))
            
            phase_test_rmse_val = phase_train_rmse[-1] * 1.1 + np.random.normal(0, 0.5)
            phase_test_rmse.append(max(1, phase_test_rmse_val))
        
        phase_stats[f'train_r2_{phase}'] = phase_train_r2
        phase_stats[f'test_r2_{phase}'] = phase_test_r2
        phase_stats[f'train_rmse_{phase}'] = phase_train_rmse
        phase_stats[f'test_rmse_{phase}'] = phase_test_rmse
    
    # Create times (milliseconds per step)
    times = [np.random.uniform(50, 150) / 1000 for _ in steps]
    
    # Combine all statistics
    stats = {
        'steps': steps,
        'losses': losses,
        'nll_values': nll_values,
        'kl_values': kl_values,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'times': times,
        **phase_stats
    }
    
    return stats

def main():
    """Generate sample stats and create publication plots."""
    
    print("Generating sample training statistics...")
    stats = generate_sample_stats()
    
    print(f"Generated {len(stats['steps'])} training steps")
    print(f"Final training R²: {stats['train_r2'][-1]:.4f}")
    print(f"Final test R²: {stats['test_r2'][-1]:.4f}")
    print(f"Final training RMSE: {stats['train_rmse'][-1]:.2f}")
    print(f"Final test RMSE: {stats['test_rmse'][-1]:.2f}")
    
    # Create publication plots
    output_dir = 'plots/publication_test'
    print(f"\nGenerating publication plots in {output_dir}/...")
    create_all_publication_plots(stats, output_dir)
    
    print("\nDone! Check the plots in plots/publication_test/")

if __name__ == "__main__":
    main()