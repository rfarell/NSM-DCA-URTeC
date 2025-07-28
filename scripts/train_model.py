#!/usr/bin/env python
# scripts/train_model.py

import os
import sys
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.utils import load_config, build_model_from_config
from src.data_processor import DataProcessor
from src.trainer import Trainer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MixtureGP model for DCA')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--eval-interval', type=int, default=100,
                        help='Interval for evaluation and plotting')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to a saved model to resume training from')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Experiment ID/path to save and organize results. If it exists, training will resume.')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start a new training run even if experiment exists')
    return parser.parse_args()

def plot_training_stats(stats, output_dir='plots'):
    """Plot training statistics and save to files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Make sure we have data to plot
    if not stats['steps'] or len(stats['steps']) == 0:
        print("No training stats to plot")
        return
    
    # Set larger font sizes for publication-quality plots
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.titlesize': 22,
        'lines.linewidth': 3
    })
    
    # Plot loss
    plt.figure(figsize=(12, 8))
    plt.plot(stats['steps'], stats['losses'], label='Total Loss')
    plt.plot(stats['steps'], stats['nll_values'], label='NLL')
    plt.plot(stats['steps'], stats['kl_values'], label='KL Divergence')
    plt.xlabel('Training Step')
    plt.ylabel('Loss Value')
    plt.yscale('log')
    plt.title('Training Loss Components')
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {loss_path}")
    
    # Plot timing
    plt.figure(figsize=(12, 8))
    plt.plot(stats['steps'], [t*1000 for t in stats['times']])
    plt.xlabel('Training Step')
    plt.ylabel('Time (ms)')
    plt.title('Training Time per Step')
    plt.grid(True)
    time_path = os.path.join(output_dir, 'training_time.png')
    plt.savefig(time_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Timing plot saved to {time_path}")
    
    # Determine the eval steps
    if 'train_r2' in stats and len(stats['train_r2']) > 0:
        if len(stats['train_r2']) < len(stats['steps']):
            # Determine the eval interval from step intervals
            step_size = max(1, len(stats['steps']) // len(stats['train_r2']))
            eval_steps = []
            
            # Select the actual steps when evaluations happened
            for i in range(len(stats['train_r2'])):
                step_idx = min(i * step_size, len(stats['steps']) - 1)
                eval_steps.append(stats['steps'][step_idx])
        else:
            # If we have the same number of eval points as steps, use all steps
            eval_steps = stats['steps']
            
        # Combined plot for both R² and RMSE on a single plot with two y-axes
        if 'train_rmse' in stats and len(stats['train_rmse']) > 0:
            # Create a figure with a single plot and two y-axes
            fig, ax1 = plt.subplots(figsize=(14, 10))
            
            # Plot R² on left y-axis (blue)
            ax1.set_xlabel('Training Step', fontsize=20)
            ax1.set_ylabel('R²', fontsize=20, color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.0)  # R² is usually between 0 and 1
            
            # Plot training R² (solid blue line)
            line1 = ax1.plot(eval_steps, stats['train_r2'], 'b-', linewidth=3, label='Training R²')
            
            # Plot test R² if available (dashed blue line)
            if 'test_r2' in stats and len(stats['test_r2']) > 0:
                if len(stats['test_r2']) == len(stats['train_r2']):
                    line2 = ax1.plot(eval_steps, stats['test_r2'], 'b--', linewidth=3, label='Test R²')
                else:
                    test_steps = stats['steps'][:len(stats['test_r2'])]
                    line2 = ax1.plot(test_steps, stats['test_r2'], 'b--', linewidth=3, label='Test R²')
            
            # Create second y-axis (right) for RMSE (red)
            ax2 = ax1.twinx()
            ax2.set_ylabel('RMSE', fontsize=20, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_yscale('log')  # RMSE often looks better on log scale
            
            # Plot training RMSE (solid red line)
            line3 = ax2.plot(eval_steps, stats['train_rmse'], 'r-', linewidth=3, label='Training RMSE')
            
            # Plot test RMSE if available (dashed red line)
            if 'test_rmse' in stats and len(stats['test_rmse']) > 0:
                if len(stats['test_rmse']) == len(stats['train_rmse']):
                    line4 = ax2.plot(eval_steps, stats['test_rmse'], 'r--', linewidth=3, label='Test RMSE')
                else:
                    test_steps = stats['steps'][:len(stats['test_rmse'])]
                    line4 = ax2.plot(test_steps, stats['test_rmse'], 'r--', linewidth=3, label='Test RMSE')
            
            # Add title
            plt.title('Model Performance Metrics', fontsize=24)
            
            # Combine all lines for the legend
            lines = line1
            labels = ['Training R²']
            
            if 'test_r2' in stats and len(stats['test_r2']) > 0:
                lines += line2
                labels.append('Test R²')
                
            lines += line3
            labels.append('Training RMSE')
            
            if 'test_rmse' in stats and len(stats['test_rmse']) > 0:
                lines += line4
                labels.append('Test RMSE')
            
            # Add legend with all lines
            fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.97),
                      ncol=4, fontsize=16, frameon=True, facecolor='white', edgecolor='black')
            
            # Adjust layout to make room for the legend
            plt.subplots_adjust(top=0.85)
            
            # Save the combined plot
            combined_path = os.path.join(output_dir, 'performance_metrics.png')
            plt.savefig(combined_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Combined performance metrics plot saved to {combined_path}")
            
            # Also save individual plots for backward compatibility
            
            # R² plot
            plt.figure(figsize=(12, 8))
            plt.plot(eval_steps, stats['train_r2'], 'b-', linewidth=3, label='Training')
            if 'test_r2' in stats and len(stats['test_r2']) > 0:
                if len(stats['test_r2']) == len(stats['train_r2']):
                    plt.plot(eval_steps, stats['test_r2'], 'r-', linewidth=3, label='Test')
                else:
                    test_steps = stats['steps'][:len(stats['test_r2'])]
                    plt.plot(test_steps, stats['test_r2'], 'r-', linewidth=3, label='Test')
            
            plt.xlabel('Training Step')
            plt.ylabel('R²')
            plt.title('Model Performance (R²)')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.ylim(0, 1.0)
            r2_path = os.path.join(output_dir, 'r_squared.png')
            plt.savefig(r2_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"R² plot saved to {r2_path}")
            
            # RMSE plot
            plt.figure(figsize=(12, 8))
            plt.plot(eval_steps, stats['train_rmse'], 'b-', linewidth=3, label='Training')
            if 'test_rmse' in stats and len(stats['test_rmse']) > 0:
                if len(stats['test_rmse']) == len(stats['train_rmse']):
                    plt.plot(eval_steps, stats['test_rmse'], 'r-', linewidth=3, label='Test')
                else:
                    test_steps = stats['steps'][:len(stats['test_rmse'])]
                    plt.plot(test_steps, stats['test_rmse'], 'r-', linewidth=3, label='Test')
            
            plt.xlabel('Training Step')
            plt.ylabel('RMSE')
            plt.title('Model Performance (RMSE)')
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.yscale('log')
            rmse_path = os.path.join(output_dir, 'rmse.png')
            plt.savefig(rmse_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"RMSE plot saved to {rmse_path}")

def main():
    """Main training script."""
    args = parse_args()
    
    # Initialize training variables
    experiment_path = None
    resume_training = False
    existing_stats = None
    
    # Process experiment path if provided
    if args.experiment:
        # Make sure experiments directory exists
        os.makedirs('experiments', exist_ok=True)
        
        experiment_path = os.path.join('experiments', args.experiment)
        os.makedirs(experiment_path, exist_ok=True)
        
        # Setup experiment-specific paths
        experiment_config_path = os.path.join(experiment_path, 'config.yaml')
        experiment_model_path = os.path.join(experiment_path, 'model.pth')
        experiment_stats_path = os.path.join(experiment_path, 'training_stats.pt')
        experiment_scaler_path = os.path.join(experiment_path, 'scaler.pkl')
        experiment_plots_dir = os.path.join(experiment_path, 'plots')
        
        # Check if this experiment already exists and we should resume
        
        if os.path.exists(experiment_config_path) and os.path.exists(experiment_model_path) and not args.no_resume:
            print(f"Found existing experiment at {experiment_path}")
            resume_training = True
            
            # Load previous training stats if they exist
            if os.path.exists(experiment_stats_path):
                print("Loading existing training statistics...")
                # PyTorch >=2.6 sets weights_only=True per default which breaks
                # loading arbitrary python objects (our stats dict contains
                # numpy scalars).  Explicitly set weights_only=False when the
                # argument is accepted.
                try:
                    existing_stats = torch.load(experiment_stats_path, weights_only=False, map_location='cpu')
                except TypeError:
                    # Older torch versions (<2.6) do not have weights_only arg
                    existing_stats = torch.load(experiment_stats_path, map_location='cpu')
                
            # Use the experiment's config
            args.config = experiment_config_path
            args.model_path = experiment_model_path
    
    # Load configuration first
    config = load_config(args.config)
    
    # Use seed from config if not specified via command line
    if args.seed == 42:  # Default value, so use config
        seed = config["training"]["random_seed"]
    else:
        seed = args.seed
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set device-specific seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Update paths in config if using experiment
    if experiment_path:
        if 'paths' not in config:
            config['paths'] = {}
            
        config['paths']['model_checkpoint'] = experiment_model_path
        config['paths']['scaler'] = experiment_scaler_path
        config['paths']['plots_dir'] = experiment_plots_dir
        
        # Save the config to experiment directory if new experiment
        if not resume_training or args.no_resume:
            print(f"Saving experiment configuration to {experiment_config_path}")
            with open(experiment_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        from src.utils import get_device
        device = get_device()
    print(f"Using device: {device}")
    
    # Initialize data processor and prepare data
    print("Loading and preparing data...")
    data_processor = DataProcessor(config)
    x_train, z_train, x_test, z_test, t_vals = data_processor.prepare_data()

    # Retrieve per-well scaling factors for (train/test) sets
    scale_train = data_processor.scale_train
    scale_test  = data_processor.scale_test
    print(f"Data shapes: x_train={x_train.shape}, z_train={z_train.shape}, "
          f"x_test={x_test.shape}, z_test={z_test.shape}, t_vals={t_vals.shape}")
    
    # Build model or load from checkpoint
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}...")
        model = build_model_from_config(config, device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print("Building new model...")
        model = build_model_from_config(config, device)
    
    # Initialize trainer
    trainer = Trainer(model, config, device)
    
    # Train model
    print("Starting training...")
    plots_dir = config["paths"].get("plots_dir", "plots")
    
    # Get eval_interval from config or command line (default to 10 if not specified)
    eval_interval = args.eval_interval
    if not eval_interval and "eval_interval" in config["training"]:
        eval_interval = config["training"]["eval_interval"]
    if not eval_interval:
        eval_interval = 10
    
    # Make sure evaluation interval is small enough to see progress
    if config["training"]["total_steps"] < eval_interval:
        eval_interval = max(1, config["training"]["total_steps"] // 2)
        print(f"Adjusting evaluation interval to {eval_interval} based on total steps")
    
    # Initialize or resume stats
    initial_stats = None
    if resume_training and existing_stats:
        print("Resuming training from previous run...")
        initial_stats = existing_stats
    
    stats = trainer.train(
        x_train, z_train, x_test, z_test, t_vals,
        scale_train=scale_train, scale_test=scale_test,
        eval_interval=eval_interval,
        checkpoint_path=config["paths"]["model_checkpoint"],
        plots_dir=plots_dir,
        initial_stats=initial_stats
    )
    
    # Save scaler
    print("Saving scaler...")
    data_processor.save_scaler(config["paths"]["scaler"])
    
    # Plot training statistics
    print("Plotting training statistics...")
    plot_training_stats(stats, plots_dir)
    
    # Save training statistics if in experiment mode
    if experiment_path:
        print(f"Saving training statistics to {experiment_stats_path}")
        torch.save(stats, experiment_stats_path)
    
    print(f"Training complete. Model saved to {config['paths']['model_checkpoint']}")

if __name__ == "__main__":
    main()