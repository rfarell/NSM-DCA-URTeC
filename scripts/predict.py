#!/usr/bin/env python
# scripts/predict.py

import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.utils import load_config, build_model_from_config
from src.data_processor import DataProcessor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions with trained MixtureGP model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (defaults to config value)')
    parser.add_argument('--scaler', type=str, default=None,
                        help='Path to scaler pickle file (defaults to config value)')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output path for predictions CSV')
    parser.add_argument('--well-id', type=int, default=None,
                        help='Optional: ID of specific well to predict and plot')
    parser.add_argument('--num-mc', type=int, default=10,
                        help='Number of Monte Carlo samples for prediction')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu or cuda)')
    return parser.parse_args()

def predict_trajectories(model, z_data, t_vals, x0=None, num_mc=10, device=torch.device('cpu')):
    """
    Predicts production trajectories for wells based on their static features.
    
    Args:
        model: Trained MixtureGP model
        z_data: Static features tensor [N, z_dim]
        t_vals: Time values tensor [T]
        x0: Initial production state [N, 3], if None uses zeros
        num_mc: Number of Monte Carlo samples for prediction
        device: Device for computation
        
    Returns:
        np.ndarray: Average predicted trajectories [N, T, 3]
    """
    model.eval()
    z_data = z_data.to(device)
    t_vals = t_vals.to(device)
    
    # If no initial state provided, use zeros
    N = z_data.shape[0]
    if x0 is None:
        x0 = torch.zeros(N, 3, device=device)
    else:
        x0 = x0.to(device)
    
    # Make predictions with Monte Carlo sampling
    all_trajectories = []
    with torch.no_grad():
        for _ in range(num_mc):
            def ode_func(t_scalar, x_state):
                return model(t_scalar, x_state, z_data)
            
            # Get ODE solver options from config
            config = load_config()
            
            # Set up the ODE solver
            ode_options = {
                'method': config["ode"]["method"],
                'atol': float(config["ode"]["atol"]),
                'rtol': float(config["ode"]["rtol"])
            }
            
            # Adjust for specific devices if needed
            if device.type == "mps":
                # Apple MPS may need more conservative tolerances
                ode_options['atol'] = max(ode_options['atol'], 1e-4)
                ode_options['rtol'] = max(ode_options['rtol'], 1e-4)
            
            # Run the ODE solver
            pred_traj = odeint(ode_func, x0, t_vals, **ode_options)
            # pred_traj has shape [T, N, 3]
            pred_traj = pred_traj.permute(1, 0, 2)  # => [N, T, 3]
            all_trajectories.append(pred_traj)
    
    # Average over Monte Carlo samples
    avg_traj = torch.stack(all_trajectories, dim=0).mean(dim=0)
    return avg_traj.cpu().numpy()

def plot_well_predictions(predictions, actual=None, well_idx=0, days=None, output_dir='plots'):
    """
    Plot predictions for a specific well.
    
    Args:
        predictions: Predicted trajectories array [N, T, 3]
        actual: Actual production data [N, T, 3] (optional)
        well_idx: Index of well to plot
        days: Days array corresponding to time points
        output_dir: Directory to save output plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create time axis
    if days is None:
        days = np.arange(predictions.shape[1])
    
    # Get predictions for the well
    pred_well = predictions[well_idx]  # [T, 3]
    
    # Production types
    prod_types = ['Gas', 'Oil', 'Water']
    colors = ['blue', 'green', 'red']
    
    plt.figure(figsize=(12, 8))
    for i, (prod_type, color) in enumerate(zip(prod_types, colors)):
        plt.subplot(3, 1, i+1)
        plt.plot(days, pred_well[:, i], color=color, label=f'Predicted {prod_type}')
        
        if actual is not None:
            act_well = actual[well_idx]  # [T, 3]
            plt.scatter(days, act_well[:, i], color=color, alpha=0.6, marker='o', 
                       label=f'Actual {prod_type}')
        
        plt.xlabel('Days')
        plt.ylabel(f'Normalized {prod_type} Production')
        plt.title(f'{prod_type} Production for Well #{well_idx}')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'well_{well_idx}_predictions.png'))
    plt.close()

def main():
    """Main prediction script."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        from src.utils import get_device
        device = get_device()
    print(f"Using device: {device}")
    
    # Get model and scaler paths
    model_path = args.model if args.model else config["paths"]["model_checkpoint"]
    scaler_path = args.scaler if args.scaler else config["paths"]["scaler"]
    
    # Initialize data processor
    data_processor = DataProcessor(config)
    
    # Load data
    x_train, z_train, x_test, z_test, t_vals = data_processor.prepare_data()
    
    # Load scaler
    data_processor.load_scaler(scaler_path)
    
    # Build and load model
    model = build_model_from_config(config, device)
    
    # Load model state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Check if we need to fix the state dict (compiled model issue)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        # Remove the '_orig_mod.' prefix from keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[10:]] = v  # Remove the '_orig_mod.' prefix
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    print(f"Model loaded from {model_path}")
    
    # Make predictions
    print("Generating predictions...")
    predictions = predict_trajectories(
        model, z_test, t_vals, 
        x0=x_test[:, 0], num_mc=args.num_mc, device=device
    )
    
    # Save predictions to CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Convert predictions to dataframe
    df_predictions = pd.DataFrame()
    days = [0, 30, 60, 90, 180, 360, 720, 1080]
    
    for well_idx in range(predictions.shape[0]):
        for t_idx, day in enumerate(days):
            df_predictions = pd.concat([df_predictions, pd.DataFrame({
                'well_id': well_idx,
                'day': day,
                'gas_pred': predictions[well_idx, t_idx, 0],
                'oil_pred': predictions[well_idx, t_idx, 1],
                'water_pred': predictions[well_idx, t_idx, 2],
                'gas_actual': x_test[well_idx, t_idx, 0].item(),
                'oil_actual': x_test[well_idx, t_idx, 1].item(),
                'water_actual': x_test[well_idx, t_idx, 2].item()
            }, index=[0])], ignore_index=True)
    
    # Save to CSV
    df_predictions.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    
    # Plot specific well if requested
    if args.well_id is not None:
        well_idx = args.well_id
        if well_idx >= len(x_test):
            print(f"Error: Well ID {well_idx} out of range (max: {len(x_test)-1})")
        else:
            print(f"Plotting predictions for well #{well_idx}...")
            plot_well_predictions(
                predictions, 
                actual=x_test.numpy(), 
                well_idx=well_idx,
                days=days
            )
            print(f"Plot saved to plots/well_{well_idx}_predictions.png")

if __name__ == "__main__":
    main()