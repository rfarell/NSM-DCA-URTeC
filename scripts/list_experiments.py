#!/usr/bin/env python
# scripts/list_experiments.py

import os
import sys
import argparse
import yaml
import glob
import torch
from datetime import datetime
from tabulate import tabulate

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='List available experiments')
    parser.add_argument('--detail', action='store_true',
                        help='Show detailed information for each experiment')
    return parser.parse_args()

def get_experiments():
    """Get all available experiments."""
    experiments_dir = os.path.join(project_root, 'experiments')
    
    if not os.path.exists(experiments_dir):
        return []
    
    # List all subdirectories in the experiments directory
    return [d for d in os.listdir(experiments_dir) 
            if os.path.isdir(os.path.join(experiments_dir, d))]

def get_experiment_info(exp_name):
    """Get information about a specific experiment."""
    exp_path = os.path.join(project_root, 'experiments', exp_name)
    
    # Initialize info dictionary
    info = {
        'name': exp_name,
        'config_file': None,
        'model_file': None,
        'stats_file': None,
        'last_modified': None,
        'training_steps': 0,
        'config': None,
        'last_loss': None,
        'last_r2': None
    }
    
    # Check for config file
    config_path = os.path.join(exp_path, 'config.yaml')
    if os.path.exists(config_path):
        info['config_file'] = config_path
        try:
            with open(config_path, 'r') as f:
                info['config'] = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config for {exp_name}: {e}")
    
    # Check for model file
    model_path = os.path.join(exp_path, 'model.pth')
    if os.path.exists(model_path):
        info['model_file'] = model_path
        info['last_modified'] = datetime.fromtimestamp(os.path.getmtime(model_path))
    
    # Check for training stats
    stats_path = os.path.join(exp_path, 'training_stats.pt')
    if os.path.exists(stats_path):
        info['stats_file'] = stats_path
        try:
            stats = torch.load(stats_path)
            if 'steps' in stats and len(stats['steps']) > 0:
                info['training_steps'] = stats['steps'][-1]
            if 'losses' in stats and len(stats['losses']) > 0:
                info['last_loss'] = stats['losses'][-1]
            if 'train_r2' in stats and len(stats['train_r2']) > 0:
                info['last_r2'] = stats['train_r2'][-1]
        except Exception as e:
            print(f"Error loading stats for {exp_name}: {e}")
    
    return info

def main():
    """Main function to list experiments."""
    args = parse_args()
    
    # Get all experiments
    experiments = get_experiments()
    
    if not experiments:
        print("No experiments found. Run training with --experiment NAME to create one.")
        return
    
    # Collect info for each experiment
    experiment_info = [get_experiment_info(exp) for exp in experiments]
    
    # Basic table
    if not args.detail:
        table_data = []
        for info in experiment_info:
            status = "✅ Complete" if info['model_file'] else "❌ Incomplete"
            steps = info['training_steps'] if info['training_steps'] > 0 else "N/A"
            last_run = info['last_modified'].strftime("%Y-%m-%d %H:%M") if info['last_modified'] else "Never"
            
            table_data.append([
                info['name'],
                status,
                steps,
                last_run
            ])
        
        print(tabulate(table_data, 
                      headers=["Experiment", "Status", "Steps", "Last Run"],
                      tablefmt="grid"))
    else:
        # Detailed table
        for info in experiment_info:
            print(f"\n{'='*60}")
            print(f"Experiment: {info['name']}")
            print(f"{'='*60}")
            
            # Status
            if info['model_file'] and info['config_file'] and info['stats_file']:
                print("Status: Complete")
            else:
                missing = []
                if not info['model_file']: missing.append("model")
                if not info['config_file']: missing.append("config")
                if not info['stats_file']: missing.append("stats")
                print(f"Status: Incomplete (missing: {', '.join(missing)})")
            
            # Training stats
            print(f"Training steps: {info['training_steps']}")
            if info['last_loss'] is not None:
                print(f"Last loss: {info['last_loss']:.6f}")
            if info['last_r2'] is not None:
                print(f"Last R²: {info['last_r2']:.4f}")
            
            # Last modified
            if info['last_modified']:
                print(f"Last run: {info['last_modified'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Config details
            if info['config']:
                print("\nConfiguration:")
                if 'model' in info['config']:
                    model_config = info['config']['model']
                    print(f"  Model: {model_config.get('num_experts', 'N/A')} experts, "
                          f"{model_config.get('num_basis', 'N/A')} basis functions")
                
                if 'training' in info['config']:
                    train_config = info['config']['training']
                    print(f"  Training: {train_config.get('total_steps', 'N/A')} steps, "
                          f"batch size {train_config.get('batch_size', 'N/A')}, "
                          f"lr={train_config.get('learning_rate', 'N/A')}")
            
            # Files
            print("\nFiles:")
            if info['config_file']:
                print(f"  Config: {os.path.relpath(info['config_file'], project_root)}")
            if info['model_file']:
                print(f"  Model: {os.path.relpath(info['model_file'], project_root)}")
            if info['stats_file']:
                print(f"  Stats: {os.path.relpath(info['stats_file'], project_root)}")
            
            # Command to resume
            print("\nTo resume this experiment:")
            print(f"  python scripts/train_model.py --experiment {info['name']}")
            print(f"To start over with same configuration:")
            print(f"  python scripts/train_model.py --experiment {info['name']} --no-resume")

if __name__ == "__main__":
    main()