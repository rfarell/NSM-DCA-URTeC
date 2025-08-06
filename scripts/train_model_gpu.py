#!/usr/bin/env python
# scripts/train_model_gpu.py
"""
Optimized training script for GPU execution.
Disables torch.compile and uses better GPU settings.
"""

import os
import sys
import argparse
import torch
import yaml
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data_processor import DataProcessor
from src.trainer import Trainer
from src.utils import load_config, plot_training_stats, build_model_from_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train NSM-DCA model (GPU optimized)')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--experiment', type=str, default=None,
                      help='Experiment name (creates subdirectory under experiments/)')
    parser.add_argument('--model-path', type=str, default=None,
                      help='Path to existing model to continue training')
    parser.add_argument('--eval-interval', type=int, default=None,
                      help='Evaluation interval in steps')
    parser.add_argument('--no-compile', action='store_true',
                      help='Disable torch.compile (recommended for debugging)')
    parser.add_argument('--resume', action='store_true',
                      help='Resume training from last checkpoint in experiment')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Force disable compilation for GPU
    os.environ['TORCH_COMPILE_DISABLE'] = '1'
    
    # Set CUDA settings for better performance
    if args.device == 'cuda' and torch.cuda.is_available():
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable cudnn autotuner
        torch.backends.cudnn.benchmark = True
        # Reduce memory fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Load configuration
    config = load_config(args.config)
    
    # Update device in config if specified
    if args.device:
        config['device'] = args.device
    
    # Handle experiment mode
    experiment_path = None
    resume_training = False
    existing_stats = None
    
    if args.experiment:
        experiment_path = os.path.join('experiments', args.experiment)
        os.makedirs(experiment_path, exist_ok=True)
        
        # Save a copy of the config to the experiment directory
        experiment_config_path = os.path.join(experiment_path, 'config.yaml')
        print(f"Saving experiment configuration to {experiment_config_path}")
        with open(experiment_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        # Update model checkpoint path for experiment
        config['paths']['model_checkpoint'] = os.path.join(experiment_path, 'model.pth')
        
        # Check for existing training stats if resuming
        experiment_stats_path = os.path.join(experiment_path, 'training_stats.pth')
        if args.resume and os.path.exists(experiment_stats_path):
            print(f"Loading existing training stats from {experiment_stats_path}")
            existing_stats = torch.load(experiment_stats_path)
            resume_training = True
            
            # Also check for existing model
            model_checkpoint = config['paths']['model_checkpoint']
            if os.path.exists(model_checkpoint):
                args.model_path = model_checkpoint
                print(f"Will resume from model checkpoint: {model_checkpoint}")
    
    # Set device
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Print GPU info if using CUDA
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load and prepare data
    print("Loading and preparing data...")
    data_processor = DataProcessor(config)
    x_train, z_train, x_test, z_test, t_vals = data_processor.prepare_data()
    
    # Get scaling factors
    scale_train = data_processor.scale_train
    scale_test = data_processor.scale_test
    
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
    
    # Create custom Trainer without compilation
    class TrainerNoCompile(Trainer):
        def __init__(self, model, config, device):
            self.model = model
            self.config = config
            self.device = device
            
            # Skip torch.compile completely
            print("Running without torch.compile for better GPU performance...")
            
            # Training hyperparameters
            self.learning_rate = config["training"]["learning_rate"]
            self.batch_size = config["training"]["batch_size"]
            self.total_steps = config["training"]["total_steps"]
            self.num_mc = config["model"]["num_mc"]

            # KL annealing schedule
            self.kl_anneal_steps = config["training"].get("kl_anneal_steps", 1000)
            
            # Cache constants
            self.time_scale = 100.0
            self.kl_weights = config.get('training', {}).get('kl_weights', None)
            
            # Loss hyperparameters
            self.lambda_ent = 1e-2
            self.lambda_1 = 1.0
            self.lambda_2 = 1.0
            
            # Apply weight decay only to the gating network parameters
            gating_params = []
            other_params = []
            for name, param in self.model.named_parameters():
                if 'gating_net' in name:
                    gating_params.append(param)
                else:
                    other_params.append(param)
            
            # Create parameter groups with different weight decay
            param_groups = [
                {'params': gating_params, 'weight_decay': 0.01},
                {'params': other_params, 'weight_decay': 0.0}
            ]
            
            # Optimizer with parameter groups
            self.optimizer = torch.optim.AdamW(param_groups, lr=self.learning_rate)
            
            # Initialize gradient scaler for mixed precision training
            from torch.amp import GradScaler
            self.scaler = GradScaler('cuda', enabled=(self.device.type == 'cuda'))
    
    # Initialize trainer
    trainer = TrainerNoCompile(model, config, device)
    
    # Train model
    print("Starting training...")
    plots_dir = config["paths"].get("plots_dir", "plots")
    
    # Get eval_interval from config or command line
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
    
    # Run a warmup iteration if on GPU
    if device.type == 'cuda':
        print("Running GPU warmup...")
        with torch.no_grad():
            # Create dummy batch
            dummy_x = x_train[:config["training"]["batch_size"]]
            dummy_z = z_train[:config["training"]["batch_size"]]
            dummy_t = t_vals
            
            # Run forward pass to warm up GPU
            x0 = dummy_x[:, 0, :]
            from torchdiffeq import odeint
            
            def ode_func(t_scalar, x_state):
                return model(t_scalar, x_state, dummy_z)
            
            _ = odeint(ode_func, x0, dummy_t, method='midpoint')
        
        # Clear cache after warmup
        torch.cuda.empty_cache()
        print("GPU warmup complete")
    
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
        experiment_stats_path = os.path.join(experiment_path, 'training_stats.pth')
        print(f"Saving training statistics to {experiment_stats_path}")
        torch.save(stats, experiment_stats_path)
    
    print(f"Training complete. Model saved to {config['paths']['model_checkpoint']}")

if __name__ == "__main__":
    main()