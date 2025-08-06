#!/usr/bin/env python
# scripts/publication_plots.py
"""
Publication-quality plot generation for journal submission.
Designed for 2-column journal format with half-page width figures.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set publication-quality defaults
def set_publication_style():
    """
    Set matplotlib parameters for publication-quality figures.
    Optimized for 2-column journal format (half-page width).
    """
    # Use serif fonts like Computer Modern (LaTeX default) or Times
    plt.rcParams.update({
        # Figure size for half-page width (typically 3.5 inches wide for 2-column)
        'figure.figsize': (3.5, 2.8),  # Width, Height in inches
        
        # High DPI for crisp text and lines
        'figure.dpi': 300,
        'savefig.dpi': 300,
        
        # Font settings for readability at small sizes
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 9,  # Base font size
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 11,
        
        # Line widths
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'axes.linewidth': 0.8,
        
        # Grid settings
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        
        # Tick settings
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.fancybox': False,
        'legend.edgecolor': 'black',
        'legend.borderpad': 0.3,
        'legend.columnspacing': 1.0,
        'legend.handlelength': 1.5,
        
        # Better text rendering
        'text.usetex': False,  # Set to True if LaTeX is available
        'mathtext.fontset': 'stix',
        
        # Tight layout
        'figure.autolayout': False,  # We'll use constrained_layout instead
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.02,
        'figure.constrained_layout.w_pad': 0.02,
    })

def plot_training_curves_dual_axis(stats, output_dir='plots', filename='training_curves_journal.pdf'):
    """
    Create publication-quality training curves with dual y-axes.
    Combines R² (left axis) and RMSE (right axis) in a single plot.
    
    Args:
        stats: Dictionary containing training statistics
        output_dir: Directory to save the plot
        filename: Output filename
    """
    set_publication_style()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    if not stats['steps'] or len(stats['steps']) == 0:
        print("No training stats to plot")
        return
    
    # Determine evaluation steps
    eval_steps = []
    if 'train_r2' in stats and len(stats['train_r2']) > 0:
        if len(stats['train_r2']) < len(stats['steps']):
            step_size = max(1, len(stats['steps']) // len(stats['train_r2']))
            for i in range(len(stats['train_r2'])):
                step_idx = min(i * step_size, len(stats['steps']) - 1)
                eval_steps.append(stats['steps'][step_idx])
        else:
            eval_steps = stats['steps'][:len(stats['train_r2'])]
    
    # Create figure with specific size for journal
    fig, ax1 = plt.subplots(figsize=(3.5, 2.8))
    
    # Define colors for consistency
    color_r2 = '#2E86AB'  # Blue
    color_rmse = '#A23B72'  # Red/Purple
    
    # Plot R² on left y-axis
    ax1.set_xlabel('Training Step', fontweight='bold')
    ax1.set_ylabel('R²', fontweight='bold', color=color_r2)
    ax1.tick_params(axis='y', labelcolor=color_r2)
    
    # Adjust y-limits based on data range
    r2_values = []
    if 'train_r2' in stats:
        r2_values.extend(stats['train_r2'])
    if 'test_r2' in stats:
        r2_values.extend(stats['test_r2'])
    
    if r2_values:
        r2_min = min(r2_values)
        r2_max = max(r2_values)
        # If we have negative R² values (model not converged), adjust limits
        if r2_min < 0:
            ax1.set_ylim(min(-1, r2_min * 1.1), min(1.0, r2_max * 1.1 + 0.1))
            ax1.axhline(y=0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        else:
            ax1.set_ylim(0, 1.0)
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
            ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    else:
        ax1.set_ylim(0, 1.0)
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    
    # Add subtle grid
    ax1.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
    
    # Plot training R² (solid line)
    line1 = ax1.plot(eval_steps, stats['train_r2'], 
                     color=color_r2, linestyle='-', linewidth=1.5,
                     marker='o', markersize=2, markevery=max(1, len(eval_steps)//10),
                     label='Train R²', zorder=5)
    
    # Plot test R² if available (dashed line)
    line2 = []
    if 'test_r2' in stats and len(stats['test_r2']) > 0:
        test_eval_steps = eval_steps[:len(stats['test_r2'])]
        line2 = ax1.plot(test_eval_steps, stats['test_r2'][:len(test_eval_steps)], 
                        color=color_r2, linestyle='--', linewidth=1.5,
                        marker='s', markersize=2, markevery=max(1, len(test_eval_steps)//10),
                        label='Test R²', zorder=5)
    
    # Create second y-axis for RMSE
    ax2 = ax1.twinx()
    ax2.set_ylabel('RMSE', fontweight='bold', color=color_rmse)
    ax2.tick_params(axis='y', labelcolor=color_rmse)
    
    # Set RMSE scale based on data range
    if 'train_rmse' in stats and len(stats['train_rmse']) > 0:
        rmse_min = min(min(stats['train_rmse']), 
                      min(stats.get('test_rmse', stats['train_rmse'])))
        rmse_max = max(max(stats['train_rmse']), 
                      max(stats.get('test_rmse', stats['train_rmse'])))
        
        # Use log scale if range is large or values are very large
        if rmse_max / rmse_min > 10 or rmse_max > 1000:
            ax2.set_yscale('log')
            # For very large values, use scientific notation and limit ticks
            if rmse_max > 10000:
                ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.0e}'))
                ax2.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))
            else:
                ax2.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
                ax2.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=[2, 5], numticks=5))
        else:
            # Linear scale with nice ticks
            ax2.yaxis.set_major_locator(ticker.MaxNLocator(5))
    
    # Plot training RMSE (solid line)
    line3 = []
    if 'train_rmse' in stats and len(stats['train_rmse']) > 0:
        line3 = ax2.plot(eval_steps, stats['train_rmse'], 
                        color=color_rmse, linestyle='-', linewidth=1.5,
                        marker='^', markersize=2, markevery=max(1, len(eval_steps)//10),
                        label='Train RMSE', zorder=4)
    
    # Plot test RMSE if available (dashed line)
    line4 = []
    if 'test_rmse' in stats and len(stats['test_rmse']) > 0:
        test_eval_steps = eval_steps[:len(stats['test_rmse'])]
        line4 = ax2.plot(test_eval_steps, stats['test_rmse'][:len(test_eval_steps)], 
                        color=color_rmse, linestyle='--', linewidth=1.5,
                        marker='v', markersize=2, markevery=max(1, len(test_eval_steps)//10),
                        label='Test RMSE', zorder=4)
    
    # Format x-axis
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
    ax1.set_xlim(left=0)
    
    # Combine all lines for legend
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    
    # Add legend in best location
    if len(lines) > 0:
        legend = ax1.legend(lines, labels, 
                           loc='best',
                           ncol=2 if len(lines) > 2 else 1,
                           frameon=True, 
                           fancybox=False,
                           edgecolor='black',
                           framealpha=0.95,
                           borderpad=0.3,
                           columnspacing=0.8,
                           handlelength=1.2)
        legend.get_frame().set_linewidth(0.5)
    
    # Add subtle box around plot
    for spine in ax1.spines.values():
        spine.set_linewidth(0.8)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.8)
    
    # Save the figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', pad_inches=0.02)
    plt.close()
    
    print(f"Publication-quality training curves saved to {output_path}")
    return output_path

def plot_loss_components(stats, output_dir='plots', filename='loss_components_journal.pdf'):
    """
    Create publication-quality plot of NLL and KL divergence with dual y-axes.
    
    Args:
        stats: Dictionary containing training statistics
        output_dir: Directory to save the plot
        filename: Output filename
    """
    set_publication_style()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if not stats['steps'] or len(stats['steps']) == 0:
        print("No training stats to plot")
        return
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(3.5, 2.8))
    
    # Define colors
    color_nll = '#E74C3C'    # Red
    color_kl = '#3498DB'     # Blue
    
    # Plot NLL on left y-axis (primary, usually larger scale)
    ax1.set_xlabel('Training Step', fontweight='bold')
    ax1.set_ylabel('NLL', fontweight='bold', color=color_nll)
    ax1.tick_params(axis='y', labelcolor=color_nll)
    
    # Plot NLL
    line1 = ax1.plot(stats['steps'], stats['nll_values'], 
                     color=color_nll, linestyle='-', linewidth=1.5,
                     marker='o', markersize=2, markevery=max(1, len(stats['steps'])//10),
                     label='NLL', zorder=3)
    
    # Set scale for NLL axis
    nll_min = min(stats['nll_values'])
    nll_max = max(stats['nll_values'])
    if nll_max / nll_min > 100:  # Use log scale for large ranges
        ax1.set_yscale('log')
        if nll_max > 10000:
            ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.0e}'))
    
    # Add grid for primary axis
    ax1.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
    
    # Create second y-axis for KL divergence
    ax2 = ax1.twinx()
    ax2.set_ylabel('KL Divergence', fontweight='bold', color=color_kl)
    ax2.tick_params(axis='y', labelcolor=color_kl)
    
    # Plot KL divergence
    line2 = ax2.plot(stats['steps'], stats['kl_values'], 
                     color=color_kl, linestyle='--', linewidth=1.5,
                     marker='s', markersize=2, markevery=max(1, len(stats['steps'])//10),
                     label='KL Divergence', zorder=2)
    
    # Set scale for KL axis
    kl_min = min(stats['kl_values'])
    kl_max = max(stats['kl_values'])
    if kl_max / kl_min > 100:  # Use log scale for large ranges
        ax2.set_yscale('log')
        if kl_max > 100:
            ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.1e}'))
    
    # Format x-axis
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
    ax1.set_xlim(left=0)
    
    # Combine lines for legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    
    # Add legend
    legend = ax1.legend(lines, labels,
                       loc='upper right', 
                       frameon=True,
                       fancybox=False,
                       edgecolor='black',
                       framealpha=0.95,
                       borderpad=0.3)
    legend.get_frame().set_linewidth(0.5)
    
    # Add subtle box around plot
    for spine in ax1.spines.values():
        spine.set_linewidth(0.8)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.8)
    
    # Save the figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', pad_inches=0.02)
    plt.close()
    
    print(f"Publication-quality loss components plot saved to {output_path}")
    return output_path


def plot_phase_panel_dual_axis(stats, output_dir='plots', filename='phase_panel_journal.pdf'):
    """
    Create a 1x3 panel plot with dual y-axes for each phase (gas, oil, water).
    Designed to span 2 columns in journal format.
    
    Args:
        stats: Dictionary containing training statistics
        output_dir: Directory to save the plot
        filename: Output filename
    """
    set_publication_style()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define phases and their colors
    phases = ['Gas', 'Oil', 'Water']
    phase_keys = ['gas', 'oil', 'water']
    
    # Define consistent colors for R² and RMSE
    color_r2 = '#2E86AB'  # Blue
    color_rmse = '#A23B72'  # Red/Purple
    
    # Create figure with 1x3 subplots
    # Full page width (7 inches) for 2-column span
    # Use constrained_layout=False to allow manual spacing adjustment
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey='row', 
                            constrained_layout=False)
    
    # Determine evaluation steps
    eval_steps = []
    if 'train_r2' in stats and len(stats['train_r2']) > 0:
        if len(stats['train_r2']) < len(stats['steps']):
            step_size = max(1, len(stats['steps']) // len(stats['train_r2']))
            for i in range(len(stats['train_r2'])):
                step_idx = min(i * step_size, len(stats['steps']) - 1)
                eval_steps.append(stats['steps'][step_idx])
        else:
            eval_steps = stats['steps'][:len(stats['train_r2'])]
    
    # If we don't have phase-specific data, use overall data for all phases
    has_phase_data = all(f'train_r2_{phase}' in stats for phase in phase_keys)
    
    # Track min/max values for consistent scaling
    all_rmse_values = []
    all_ax2_axes = []  # Store the twin axes for later access
    
    # Plot each phase
    for idx, (phase_name, phase_key) in enumerate(zip(phases, phase_keys)):
        ax1 = axes[idx]
        
        # Get data for this phase
        if has_phase_data:
            # Use phase-specific data
            train_r2_key = f'train_r2_{phase_key}'
            test_r2_key = f'test_r2_{phase_key}'
            train_rmse_key = f'train_rmse_{phase_key}'
            test_rmse_key = f'test_rmse_{phase_key}'
        else:
            # Use overall data for all phases
            train_r2_key = 'train_r2'
            test_r2_key = 'test_r2'
            train_rmse_key = 'train_rmse'
            test_rmse_key = 'test_rmse'
        
        # Plot R² on left y-axis
        ax1.set_xlabel('Training Step', fontweight='bold', fontsize=8)
        if idx == 0:
            ax1.set_ylabel('R²', fontweight='bold', color=color_r2, fontsize=8)
        ax1.tick_params(axis='y', labelcolor=color_r2, labelsize=7)
        
        # Check if we need to adjust for negative R² values
        r2_values = []
        if train_r2_key in stats:
            r2_values.extend(stats[train_r2_key])
        if test_r2_key in stats:
            r2_values.extend(stats[test_r2_key])
        
        if r2_values and min(r2_values) < 0:
            # Adjust for negative R² (unconverged model)
            ax1.set_ylim(min(-1, min(r2_values) * 1.1), min(1.0, max(r2_values) * 1.1 + 0.1))
            ax1.axhline(y=0, color='gray', linestyle=':', linewidth=0.4, alpha=0.5)
        else:
            ax1.set_ylim(0, 1.0)
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
            ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        
        # Add grid
        ax1.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=0.4)
        ax1.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
        
        # Plot training R²
        if train_r2_key in stats and len(stats[train_r2_key]) > 0:
            line1 = ax1.plot(eval_steps, stats[train_r2_key], 
                           color=color_r2, linestyle='-', linewidth=1.2,
                           marker='o', markersize=1.5, markevery=max(1, len(eval_steps)//8),
                           label='Train R²', zorder=5)
        else:
            line1 = []
        
        # Plot test R²
        line2 = []
        if test_r2_key in stats and len(stats[test_r2_key]) > 0:
            test_eval_steps = eval_steps[:len(stats[test_r2_key])]
            line2 = ax1.plot(test_eval_steps, stats[test_r2_key][:len(test_eval_steps)], 
                           color=color_r2, linestyle='--', linewidth=1.2,
                           marker='s', markersize=1.5, markevery=max(1, len(test_eval_steps)//8),
                           label='Test R²', zorder=5)
        
        # Create second y-axis for RMSE
        ax2 = ax1.twinx()
        all_ax2_axes.append(ax2)  # Store for later access
        if idx == 2:  # Only show label on rightmost plot
            ax2.set_ylabel('RMSE', fontweight='bold', color=color_rmse, fontsize=8)
        ax2.tick_params(axis='y', labelcolor=color_rmse, labelsize=7)
        
        # Plot RMSE
        line3 = []
        line4 = []
        if train_rmse_key in stats and len(stats[train_rmse_key]) > 0:
            # Collect RMSE values for scaling
            all_rmse_values.extend(stats[train_rmse_key])
            if test_rmse_key in stats:
                all_rmse_values.extend(stats[test_rmse_key])
            
            # Plot training RMSE
            line3 = ax2.plot(eval_steps, stats[train_rmse_key], 
                           color=color_rmse, linestyle='-', linewidth=1.2,
                           marker='^', markersize=1.5, markevery=max(1, len(eval_steps)//8),
                           label='Train RMSE', zorder=4)
            
            # Plot test RMSE
            if test_rmse_key in stats and len(stats[test_rmse_key]) > 0:
                test_eval_steps = eval_steps[:len(stats[test_rmse_key])]
                line4 = ax2.plot(test_eval_steps, stats[test_rmse_key][:len(test_eval_steps)], 
                               color=color_rmse, linestyle='--', linewidth=1.2,
                               marker='v', markersize=1.5, markevery=max(1, len(test_eval_steps)//8),
                               label='Test RMSE', zorder=4)
        
        # Format x-axis
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(4, integer=True))
        ax1.set_xlim(left=0)
        ax1.tick_params(axis='x', labelsize=7)
        
        # Add phase title
        ax1.set_title(phase_name, fontsize=9, fontweight='bold', pad=3)
        
        # Store lines for combined legend (only from first subplot)
        if idx == 0:
            all_lines = line1 + line2 + line3 + line4
            all_labels = [l.get_label() for l in all_lines]
        
        # Add subtle box around plot
        for spine in ax1.spines.values():
            spine.set_linewidth(0.6)
        for spine in ax2.spines.values():
            spine.set_linewidth(0.6)
    
    # Set consistent RMSE scale across all subplots
    if all_rmse_values and all_ax2_axes:
        rmse_min = min(all_rmse_values)
        rmse_max = max(all_rmse_values)
        
        # Use log scale if range is large or values are very large
        use_log = (rmse_max / rmse_min > 10 or rmse_max > 1000) if rmse_min > 0 else False
        
        # Apply consistent scaling to all RMSE axes
        for ax2 in all_ax2_axes:
            if use_log:
                ax2.set_yscale('log')
                # For very large values, use scientific notation and limit ticks
                if rmse_max > 10000:
                    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.0e}'))
                    ax2.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))
                else:
                    ax2.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=4))
                # Don't set minor locator for very large ranges to avoid tick overflow
                if rmse_max / rmse_min < 1000:
                    ax2.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=[2, 5], numticks=4))
            else:
                # For linear scale, limit the number of ticks
                ax2.set_ylim(rmse_min * 0.9, rmse_max * 1.1)
                ax2.yaxis.set_major_locator(ticker.MaxNLocator(4, min_n_ticks=2))
    
    # Adjust spacing between subplots and make room for legend
    plt.subplots_adjust(wspace=0.4, bottom=0.25)
    
    # Add horizontal legend below the entire figure
    if 'all_lines' in locals() and all_lines:
        fig.legend(all_lines, all_labels,
                  loc='lower center',
                  ncol=4,  # Horizontal layout with 4 columns
                  frameon=True,
                  fancybox=False,
                  edgecolor='black',
                  framealpha=0.95,
                  borderpad=0.3,
                  columnspacing=1.2,
                  handlelength=1.5,
                  fontsize=7,
                  bbox_to_anchor=(0.5, -0.08))  # Position further below to avoid covering x-label
    
    # Save the figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', pad_inches=0.02)
    plt.close()
    
    print(f"Publication-quality phase panel plot saved to {output_path}")
    return output_path

def create_all_publication_plots(stats, output_dir='plots'):
    """
    Generate all publication-quality plots for journal submission.
    
    Args:
        stats: Dictionary containing training statistics
        output_dir: Directory to save all plots
    """
    print("\nGenerating publication-quality plots...")
    
    # Create main training curves with dual axes
    plot_training_curves_dual_axis(stats, output_dir)
    
    # Create 1x3 panel plot for phases
    plot_phase_panel_dual_axis(stats, output_dir)
    
    # Create loss components plot
    plot_loss_components(stats, output_dir)
    
    print(f"\nAll publication plots saved to {output_dir}/")
    print("Files generated:")
    print("  - training_curves_journal.pdf/png (main dual-axis plot)")
    print("  - phase_panel_journal.pdf/png (1x3 panel with dual axes for each phase, 2-column span)")
    print("  - loss_components_journal.pdf/png (NLL and KL with dual axes, 1-column)")

if __name__ == "__main__":
    # Example usage: Load existing training stats and generate plots
    import torch
    import sys
    import os
    
    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)
    
    # Check if training stats exist
    stats_path = 'experiments/exp1/training_stats.pt'
    if not os.path.exists(stats_path):
        # Try default location
        stats_path = 'training_stats.pt'
    
    if os.path.exists(stats_path):
        print(f"Loading training statistics from {stats_path}...")
        try:
            stats = torch.load(stats_path, weights_only=False, map_location='cpu')
        except TypeError:
            stats = torch.load(stats_path, map_location='cpu')
        
        # Generate publication plots
        output_dir = 'plots/publication'
        create_all_publication_plots(stats, output_dir)
    else:
        print("No training statistics found. Please train a model first.")
        print(f"Looked for stats at: {stats_path}")