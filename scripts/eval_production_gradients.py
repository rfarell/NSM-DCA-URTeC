#!/usr/bin/env python
# scripts/eval_production_gradients.py
"""
Visualize production gradients to analyze model dynamics.

This script creates two types of visualizations:
1. Gradient vector fields between two phases at a specific time point
2. Gradient vector fields of each phase over time
"""

import os
import sys
import torch
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from matplotlib.patches import FancyBboxPatch

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.evaluation import load_experiment

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize production gradients')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Name of the experiment to evaluate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--well-idx', type=int, default=0,
                        help='Index of well to visualize (use -1 for random selection)')
    parser.add_argument('--time-step', type=int, default=0,
                        help='Time step index to visualize (0-7)')
    parser.add_argument('--resolution', type=int, default=50,
                        help='Grid resolution for phase-phase visualization')
    parser.add_argument('--vis-type', type=str, default='both',
                        choices=['phase-phase', 'time-phase', 'both'],
                        help='Type of visualization to generate')
    parser.add_argument('--grid-size-q', type=int, default=40,
                        help='Number of grid points for production rates (time-phase vis)')
    parser.add_argument('--grid-size-t', type=int, default=40,
                        help='Number of grid points for time (time-phase vis)')
    parser.add_argument('--max-time', type=float, default=2160,
                        help='Maximum time in days for visualization (time-phase vis)')
    parser.add_argument('--scale', type=float, default=None,
                        help='Scale factor for arrows (lower = larger arrows)')
    parser.add_argument('--dt', type=float, default=30.0,
                        help='Time interval in days for gradient calculation (time-phase vis)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

#####################################################################
# 1. Phase-Phase Gradient Visualization Functions
#####################################################################

def compute_production_gradients(model, well_z, time_point, phase1_idx=0, phase2_idx=1, 
                                 res=50, min_val=0.01, max_val=3.0, device=torch.device('cpu')):
    """
    Compute production gradients over a grid of production values for two phases.
    
    Args:
        model: Trained model
        well_z: Static features for a well
        time_point: Time point to evaluate at
        phase1_idx: Index of first phase (x-axis)
        phase2_idx: Index of second phase (y-axis)
        res: Resolution of the grid
        min_val, max_val: Min and max production values
        device: Computation device
        
    Returns:
        tuple: (x_grid, y_grid, dx_dt, dy_dt) gradient field data
    """
    model.eval()
    
    # Prepare grid
    phase1_vals = np.linspace(min_val, max_val, res)
    phase2_vals = np.linspace(min_val, max_val, res)
    x_grid, y_grid = np.meshgrid(phase1_vals, phase2_vals)
    
    # Initialize gradient field
    dx_dt = np.zeros((res, res))
    dy_dt = np.zeros((res, res))
    
    # Calculate gradients for each grid point
    with torch.no_grad():
        well_z_tensor = well_z.unsqueeze(0)  # [1, z_dim]
        
        # The model expects t_scalar as a tensor that it can unsqueeze
        # Based on the model.py implementation
        
        for i in range(res):
            for j in range(res):
                # Create state vector
                x_val = x_grid[i, j]
                y_val = y_grid[i, j]
                
                # Set the third phase to a constant small value
                third_phase_idx = 3 - phase1_idx - phase2_idx
                state = torch.zeros(1, 3, device=device)
                state[0, phase1_idx] = float(x_val)
                state[0, phase2_idx] = float(y_val)
                state[0, third_phase_idx] = 0.1  # Small constant value
                
                # Compute derivative - time_point should be a scalar tensor
                deriv = model(torch.tensor(time_point, device=device), state, well_z_tensor)
                
                # Store gradients
                dx_dt[i, j] = deriv[0, phase1_idx].item()
                dy_dt[i, j] = deriv[0, phase2_idx].item()
    
    return x_grid, y_grid, dx_dt, dy_dt

def plot_production_gradients(x_grid, y_grid, dx_dt, dy_dt, phase1_name, phase2_name, time_value, save_path):
    """
    Plot production gradients as a vector field.
    
    Args:
        x_grid, y_grid: Grid coordinates
        dx_dt, dy_dt: Gradient components
        phase1_name, phase2_name: Names of the phases
        time_value: Time value in days
        save_path: Path to save the plot
    """
    # Calculate vector magnitudes for coloring
    magnitudes = np.sqrt(dx_dt**2 + dy_dt**2)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create colormap for vector magnitudes
    norm = Normalize()
    norm.autoscale(magnitudes)
    
    # Plot vector field
    plt.quiver(x_grid, y_grid, dx_dt, dy_dt, magnitudes, 
               cmap='viridis', norm=norm, scale=30, width=0.002)
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Gradient Magnitude', fontsize=12)
    
    # Add labels and title
    plt.xlabel(f'{phase1_name} Production', fontsize=12)
    plt.ylabel(f'{phase2_name} Production', fontsize=12)
    plt.title(f'Production Gradient Vector Field at t={time_value} days', fontsize=14)
    
    # Adjust tick label sizes
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    cbar.ax.tick_params(labelsize=10)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_phase_combinations(model, well_z, time_point, time_value, days, resolution, eval_dir, well_idx, device):
    """
    Plot gradient vector fields for all phase combinations.
    
    Args:
        model: Trained model
        well_z: Well static features
        time_point: Time point tensor value
        time_value: Time value in days
        days: Time points in days
        resolution: Grid resolution
        eval_dir: Directory to save plots
        well_idx: Well index
        device: Computation device
    """
    phase_names = ['Gas', 'Oil', 'Water']
    phase_combinations = [
        (0, 1, 'Gas', 'Oil'),
        (0, 2, 'Gas', 'Water'),
        (1, 2, 'Oil', 'Water')
    ]
    
    for phase1_idx, phase2_idx, phase1_name, phase2_name in phase_combinations:
        print(f"  Computing gradients for {phase1_name} vs {phase2_name}...")
        x_grid, y_grid, dx_dt, dy_dt = compute_production_gradients(
            model, well_z, time_point,
            phase1_idx=phase1_idx, phase2_idx=phase2_idx,
            res=resolution, device=device
        )
        
        save_path = os.path.join(
            eval_dir, 
            f'gradients_well{well_idx}_t{time_value}_{phase1_name.lower()}_{phase2_name.lower()}.png'
        )
        
        plot_production_gradients(
            x_grid, y_grid, dx_dt, dy_dt,
            phase1_name, phase2_name, time_value,
            save_path
        )
        print(f"  Saved to {save_path}")

#####################################################################
# 2. Time-Phase Gradient Visualization Functions
#####################################################################

def compute_gradients(model, z_data, q_points, t_points, device, dt=30.0):
    """
    Compute oil production gradients across a grid of production rates and time points.
    
    Args:
        model: Trained MixtureGP model
        z_data: Static features tensor [1, z_dim]
        q_points: Dictionary containing oil production rate points
        t_points: Array of time points
        device: Device for computation
        dt: Time interval in days to use for gradient scaling
        
    Returns:
        dict: Contains grid coordinates and gradient vectors
    """
    model.eval()
    z_data = z_data.to(device)
    
    # Scale factor for time as used in data_processor
    time_scale = 0.01
    
    # Create meshgrid for oil plotting
    T_oil, Q_oil = np.meshgrid(t_points, q_points['oil'])
    
    # Initialize array to store gradients
    grad_oil = np.zeros_like(T_oil)
    
    # Scale times for model input
    t_tensor_points = torch.tensor(t_points, dtype=torch.float32) * time_scale
    t_tensor_points = t_tensor_points.to(device)
    
    with torch.no_grad():
        print("Computing oil gradients...")
        for i, q_oil in enumerate(q_points['oil']):
            for j, t_idx in enumerate(range(len(t_points))):
                # Convert to scalar tensor for model
                t_scalar = t_tensor_points[t_idx].item()
                
                # Create state tensor with just oil values
                x_state = torch.zeros(1, 3, device=device)
                x_state[0, 1] = float(q_oil)  # Oil value as Python float to avoid dtype mismatch
                
                # Compute gradient - using scalar tensor time
                derivative = model(torch.tensor(t_scalar, device=device), x_state, z_data)
                
                # Store gradient component, scaled by dt
                grad_oil[i, j] = derivative[0, 1].item() * dt  # dOil over dt days
    
    return {
        't_points': t_points,
        'q_points': q_points,
        'T_oil': T_oil,
        'Q_oil': Q_oil,
        'grad_oil': grad_oil
    }

def plot_gradient_vector_fields(results, well_idx, z_data, plot_save_path, z_data_save_path, scale=None, dt=30.0):
    """
    Plot a vector field showing oil production rate gradients vs time.
    
    Args:
        results: Dictionary with grid coordinates and gradients
        well_idx: Well index for the title
        z_data: Static features tensor
        plot_save_path: Path to save the figure
        z_data_save_path: Path to save the static features text file
        scale: Scale factor for arrows (None for auto-scaling)
        dt: Time interval in days used for gradient calculation
    """
    # Create figure with just one plot for oil
    fig, ax = plt.subplots(figsize=(10, 6))
    # Reduce padding
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Focus only on oil phase
    phase = 'oil'
    phase_name = 'Oil'
    
    # Get corresponding data
    T = results[f'T_{phase}']
    Q = results[f'Q_{phase}']
    gradients = results[f'grad_{phase}']
    
    # We need two components for quiver plot:
    # U is 1 (constant for time movement)
    # V is the production gradient
    U = np.ones_like(T)  # Constant time flow
    V = gradients
    
    # Normalize for better visualization
    norm = np.sqrt(U**2 + V**2)
    U = U / norm
    V = V / norm
    
    # Plot vector field with cleaner appearance
    quiver = ax.quiver(T, Q, U, V, scale=scale, width=0.002, 
                      color='blue', alpha=0.7, headwidth=4, headlength=4, headaxislength=3.5)
    
    # Add colorful contour plot of gradient values in the background
    smoothed_for_contour = gaussian_filter(gradients, sigma=0.8)
    contour = ax.contourf(T, Q, smoothed_for_contour, 
                         levels=20, cmap='coolwarm', alpha=0.3, antialiased=True)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(f'd{phase_name}/dt (Production Gradient)', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    
    # Add a key/scale for reference with dt information
    # Move legend to left side to avoid overlap with colorbar
    quiverkey = ax.quiverkey(quiver, 0.2, 0.95, 1, 
                 f'Normalized Gradient\n(over {dt} days)', 
                 labelpos='E', fontproperties={'size': 14})
    
    # Add pure white background to the quiverkey
    if quiverkey.text and hasattr(quiverkey.text, 'set_bbox'):
        quiverkey.text.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='lightgray'))
    
    # Add zero-gradient contour line
    try:
        # Apply a small smoothing to the gradients to avoid artifacts
        smoothed_gradients = gaussian_filter(gradients, sigma=1.0)
        
        # Draw zero contour with clean settings
        zero_contour = ax.contour(T, Q, smoothed_gradients, 
                                 levels=[0], colors='red', linewidths=1.5,
                                 alpha=0.7, antialiased=True)
        
        # Only label if contours exist and aren't too complex
        if len(zero_contour.collections) > 0 and len(zero_contour.collections[0].get_paths()) < 10:
            ax.clabel(zero_contour, inline=True, fontsize=14, fmt='%.2f')
    except Exception as e:
        print(f"Note: Could not draw zero-gradient contour for {phase_name}: {e}")
    
    # Set labels and title with clearer phase indication
    ax.set_xlabel('Time (days)', fontsize=16)
    ax.set_ylabel(f'{phase_name} Production Rate', fontsize=16)
    ax.set_title(f'{phase_name} Production Gradient Vector Field (Well {well_idx})', fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Adjust y-axis limit
    original_max_q = np.max(results['q_points'][phase])
    ax.set_ylim(bottom=0, top=original_max_q * 0.6)
    
    # Use tight layout with reduced padding to minimize whitespace
    plt.tight_layout(pad=0.5)
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save static features to a text file
    if z_data_save_path:
        # Format z-value information for saving
        z_feature_names = [
            'LateralLength_FT', 'FracStages', 'AverageStageSpacing_FT',
            'ProppantIntensity_LBSPerFT', 'FluidIntensity_BBLPerFT', 'Isopach_FT',
            'EffectivePorosity_PCT', 'TopOfZone_FT', 'GammaRay_API',
            'Resistivity_OHMSM', 'ClayVolume_PCT', 'WaterSaturation_PCT', 'PhiH_FT',
            'HCPV_PCT', 'TotalOrganicCarbon_WTPCT', 'BottomOfZone_FT',
            'TrueVerticalDepth_FT', 'Latitude', 'Longitude'
        ]
        
        # Format the feature values into a organized table format
        z_numpy = z_data.cpu().numpy().flatten()
        
        # Create a structured layout with categories
        feature_categories = {
            'Completion Parameters': ['LateralLength_FT', 'FracStages', 'AverageStageSpacing_FT', 
                                     'ProppantIntensity_LBSPerFT', 'FluidIntensity_BBLPerFT'],
            'Geological Properties': ['Isopach_FT', 'EffectivePorosity_PCT', 'TopOfZone_FT', 
                                     'GammaRay_API', 'Resistivity_OHMSM', 'ClayVolume_PCT'],
            'Reservoir Characteristics': ['WaterSaturation_PCT', 'PhiH_FT', 'HCPV_PCT', 
                                         'TotalOrganicCarbon_WTPCT', 'BottomOfZone_FT', 'TrueVerticalDepth_FT'],
            'Location': ['Latitude', 'Longitude']
        }
        
        # Create a formatted text with categories
        formatted_text = ""
        for category, features in feature_categories.items():
            if formatted_text:
                formatted_text += "\n\n"
            formatted_text += f"{category}:\n"
            
            # Format features in this category
            feature_list = []
            for feature in features:
                if feature in z_feature_names:
                    idx = z_feature_names.index(feature)
                    # Format numbers nicely
                    val = z_numpy[idx]
                    if val.is_integer() or abs(val - round(val)) < 1e-5:
                        val_str = f"{int(val)}"
                    else:
                        val_str = f"{val:.2f}"
                    feature_list.append(f"{feature.replace('_', ' ')}: {val_str}")
            
            # Join features with commas
            formatted_text += ", ".join(feature_list)

        try:
            with open(z_data_save_path, 'w') as f:
                f.write(f"Static Features for Well {well_idx}\n")
                f.write("="*len(f"Static Features for Well {well_idx}") + "\n\n")
                f.write(formatted_text)
            print(f"  Static features saved to {z_data_save_path}")
        except IOError as e:
            print(f"Warning: Could not save static features to {z_data_save_path}: {e}")

#####################################################################
# Main Function
#####################################################################

def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load experiment
    print(f"Loading experiment: {args.experiment}")
    exp_data = load_experiment(args.experiment, device=args.device)
    
    model = exp_data['model']
    device = exp_data['device']
    eval_dir = exp_data['eval_dir']
    x_test = exp_data['x_test']
    z_test = exp_data['z_test']
    t_vals = exp_data['t_vals']
    
    # Create gradient visualizations directory
    gradients_dir = os.path.join(eval_dir, 'gradients')
    os.makedirs(gradients_dir, exist_ok=True)
    
    # Get days
    days = [0, 30, 60, 90, 180, 360, 720, 1080]
    
    # Validate time step for phase-phase visualization
    if args.time_step < 0 or args.time_step >= len(days):
        print(f"Error: Time step must be between 0 and {len(days)-1}")
        sys.exit(1)
    
    # Select well index
    num_wells = z_test.shape[0]
    if args.well_idx < 0:
        # Random selection
        well_idx = random.randint(0, num_wells-1)
        print(f"Randomly selected well index: {well_idx}")
    else:
        # Use specified well index
        if args.well_idx >= num_wells:
            print(f"Error: Well index must be less than {num_wells}")
            sys.exit(1)
        well_idx = args.well_idx
    
    # Get well features
    well_z = z_test[well_idx].to(device)
    
    #------------------------------------------------------------
    # 1. Phase-Phase Gradient Visualization
    #------------------------------------------------------------
    if args.vis_type in ['phase-phase', 'both']:
        print("\n=== Generating Phase-Phase Gradient Visualizations ===")
        time_point = t_vals[args.time_step].to(device)
        time_value = days[args.time_step]
        
        print(f"Computing phase-phase gradients for well {well_idx} at t={time_value} days")
        plot_all_phase_combinations(
            model, well_z, time_point, time_value, days,
            args.resolution, gradients_dir, well_idx, device
        )
    
    #------------------------------------------------------------
    # 2. Time-Phase Gradient Visualization for Oil Only
    #------------------------------------------------------------
    if args.vis_type in ['time-phase', 'both']:
        print("\n=== Generating Time-Phase Gradient Visualization for Oil ===")
        
        # Create grid points for time
        t_points = np.linspace(0, args.max_time, args.grid_size_t)
        
        # Get the maximum production rates for this well's data
        well_data = x_test[well_idx].cpu().numpy()
        q_max = np.max(well_data, axis=0)  # Max across time for each phase
        
        # Create production rate grid points - only for oil
        q_points = {
            'oil': np.linspace(0, q_max[1] * 1.1, args.grid_size_q)
        }
        
        # Compute gradients across the grid with time interval dt
        print(f"Computing time-phase gradient for oil production for well {well_idx}")
        results = compute_gradients(
            model, well_z.unsqueeze(0), q_points, t_points, device, dt=args.dt
        )
        
        # Create save path for the plot
        plot_save_path = os.path.join(gradients_dir, f'oil_time_rate_gradient_well_{well_idx}_dt{int(args.dt)}.png')
        # Create save path for the static features text file
        z_data_save_path = os.path.join(gradients_dir, f'static_features_well_{well_idx}.txt')
        
        # Plot and save
        print(f"Plotting oil time-phase vector field to {plot_save_path}...")
        plot_gradient_vector_fields(
            results, well_idx, well_z, plot_save_path, z_data_save_path, scale=args.scale, dt=args.dt
        )
    
    print(f"\nAll gradient visualizations saved to {gradients_dir}")

if __name__ == "__main__":
    main()