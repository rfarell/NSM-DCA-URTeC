#!/usr/bin/env python
# scripts/train_model.py

import os
import sys
import torch
import argparse
import numpy as np
import yaml

# Enable TensorFloat32 (TF32) for better performance on compatible GPUs
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.utils import load_config, build_model_from_config, load_model_checkpoint
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
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile for faster startup (recommended for GPU issues)')
    return parser.parse_args()

def plot_training_stats(stats, output_dir='plots'):
    """Generate publication-quality plots for training statistics."""
    # Import publication plotting functions
    try:
        from publication_plots import create_all_publication_plots
        # Generate publication-quality plots
        pub_dir = os.path.join(output_dir, 'publication')
        create_all_publication_plots(stats, pub_dir)
    except ImportError:
        print("Warning: Could not import publication_plots module")
    except Exception as e:
        print(f"Error generating publication plots: {e}")

def main():
    """Main training script."""
    args = parse_args()
    
    # Handle --no-compile flag
    if args.no_compile:
        os.environ['TORCH_COMPILE_DISABLE'] = '1'
        print("Torch compilation disabled via --no-compile flag")
    
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
        model = load_model_checkpoint(model, args.model_path, device)
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