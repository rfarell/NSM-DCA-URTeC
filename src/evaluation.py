"""
Core evaluation utilities for the NSM-DCA model.
This module contains common functions used across different evaluation scripts.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from matplotlib.gridspec import GridSpec

def load_experiment(experiment_name, device=None, config_loader=None, model_builder=None):
    """
    Load an experiment's model, config, and data.
    
    Args:
        experiment_name: Name of the experiment to load
        device: Device to use (defaults to best available)
        config_loader: Function to load config (defaults to utils.load_config)
        model_builder: Function to build model (defaults to utils.build_model_from_config)
        
    Returns:
        dict: Dictionary containing experiment data:
            - model: Loaded model
            - config: Configuration
            - data_processor: DataProcessor instance with loaded data
            - experiment_path: Path to experiment directory
            - eval_dir: Path to evaluation directory
    """
    import os
    import sys
    
    # Import required modules
    if config_loader is None:
        from src.utils import load_config
        config_loader = load_config
    
    if model_builder is None:
        from src.utils import build_model_from_config
        model_builder = build_model_from_config
    
    if device is None:
        from src.utils import get_device
        device = get_device()
        
    from src.data_processor import DataProcessor
    
    # Get project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Validate experiment name
    experiment_path = os.path.join(project_root, 'experiments', experiment_name)
    if not os.path.exists(experiment_path):
        raise FileNotFoundError(f"Experiment '{experiment_name}' not found in experiments directory")
    
    # Setup paths
    config_path = os.path.join(experiment_path, 'config.yaml')
    model_path = os.path.join(experiment_path, 'model.pth')
    scaler_path = os.path.join(experiment_path, 'scaler.pkl')
    
    # Create evaluation directory
    eval_dir = os.path.join(experiment_path, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Check if required files exist (config and model are mandatory)
    for path, name in [(config_path, "Config"), (model_path, "Model")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found at {path}")
    
    # Load configuration
    config = config_loader(config_path)
    
    # Initialize data processor
    data_processor = DataProcessor(config)
    
    # Load data
    x_train, z_train, x_test, z_test, t_vals = data_processor.prepare_data()
    
    # Load scaler if it exists, otherwise the scaler from prepare_data is used
    if os.path.exists(scaler_path):
        data_processor.load_scaler(scaler_path)
    else:
        print(f"Warning: Scaler file not found at {scaler_path}")
        print("Using scaler fitted during data preparation. This may lead to different results if the original training used a different data split.")
        # Save the scaler for future use
        data_processor.save_scaler(scaler_path)
        print(f"Saved new scaler to {scaler_path}")
    
    # Build and load model
    model = model_builder(config, device)
    
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
    
    # Try loading the state dict
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Handle missing keys for cached tril indices
        if "_tril_idx" in str(e):
            print("Note: Model was saved before tril indices optimization. Loading with strict=False.")
            model.load_state_dict(state_dict, strict=False)
            # The model __init__ will have already created the cached indices
        else:
            raise e
    
    model.eval()
    
    # Move data tensors to device
    x_train = x_train.to(device)
    z_train = z_train.to(device)
    x_test = x_test.to(device)
    z_test = z_test.to(device)
    t_vals = t_vals.to(device)
    
    return {
        'model': model,
        'config': config,
        'data_processor': data_processor,
        'experiment_path': experiment_path,
        'eval_dir': eval_dir,
        'device': device,
        'x_train': x_train,
        'z_train': z_train,
        'x_test': x_test,
        'z_test': z_test,
        't_vals': t_vals
    }

def predict_from_t(model, z_data, t_vals, x_data, start_idx, num_mc=10, device=torch.device('cpu')):
    """
    Predicts production trajectories starting from a specific time point.
    
    Args:
        model: Trained MixtureGP model
        z_data: Static features tensor [N, z_dim]
        t_vals: Time values tensor [T]
        x_data: Production data tensor [N, T, 3]
        start_idx: Index of starting time point
        num_mc: Number of Monte Carlo samples for prediction
        device: Device for computation
        
    Returns:
        np.ndarray: Average predicted trajectories [N, T-start_idx, 3]
    """
    model.eval()
    z_data = z_data.to(device)
    
    # Get time values from start_idx onwards - using absolute time values
    t_sub = t_vals[start_idx:]  # Use absolute time values without resetting to zero
    t_sub = t_sub.to(device)
    
    # Get initial state at start_idx
    x0 = x_data[:, start_idx, :].to(device)  # [N, 3]
    
    # Make predictions with Monte Carlo sampling
    all_trajectories = []
    with torch.no_grad():
        for _ in range(num_mc):
            def ode_func(t_scalar, x_state):
                return model(t_scalar, x_state, z_data)
            
            # ODE solver options
            ode_options = {
                'method': "rk4",
                'atol': 1.0e-4,
                'rtol': 1.0e-4
            }
            
            # Run the ODE solver
            pred_traj = odeint(ode_func, x0, t_sub, **ode_options)
            # pred_traj has shape [T-start_idx, N, 3]
            pred_traj = pred_traj.permute(1, 0, 2)  # => [N, T-start_idx, 3]
            all_trajectories.append(pred_traj)
    
    # Average over Monte Carlo samples
    avg_traj = torch.stack(all_trajectories, dim=0).mean(dim=0)
    return avg_traj.cpu().numpy()

def predict_trajectories(model, z_data, t_vals, x0, num_samples=100, device=torch.device('cpu')):
    """
    Generate multiple trajectory predictions to show uncertainty.
    
    Args:
        model: Trained MixtureGP model
        z_data: Static features tensor [N, z_dim]
        t_vals: Time values tensor [T]
        x0: Initial state tensor [N, 3]
        num_samples: Number of trajectory samples to generate
        device: Device for computation
        
    Returns:
        np.ndarray: Sampled trajectories [num_samples, N, T, 3]
    """
    model.eval()
    z_data = z_data.to(device)
    x0 = x0.to(device)
    t_vals = t_vals.to(device)
    
    # Generate multiple trajectories to show uncertainty
    all_trajectories = []
    with torch.no_grad():
        for _ in range(num_samples):
            def ode_func(t_scalar, x_state):
                return model(t_scalar, x_state, z_data)
            
            # ODE solver options
            ode_options = {
                'method': "rk4",
                'atol': 1.0e-4,
                'rtol': 1.0e-4
            }
            
            # Run the ODE solver
            pred_traj = odeint(ode_func, x0, t_vals, **ode_options)
            # pred_traj has shape [T, N, 3]
            pred_traj = pred_traj.permute(1, 0, 2)  # => [N, T, 3]
            all_trajectories.append(pred_traj)
    
    # Stack along a new dimension for samples
    trajectories = torch.stack(all_trajectories, dim=0)  # [num_samples, N, T, 3]
    return trajectories.cpu().numpy()

# -------------------------------------------
# src/evaluation/plot_utils.py
# -------------------------------------------
"""
Publication-ready scatter plots for zero-history forecasts.

Layout (per call)
-----------------
* 1 row  × 3 columns  :  horizons 0→{360, 720, 1080} d
* Called once per phase (oil, gas, water).
* Colour-blind-safe palette, serif fonts, light grid.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl

# ────────── global style ──────────
sns.set_theme(style="whitegrid", context="paper")
PALETTE = sns.color_palette("colorblind")

FS_LBL, FS_TICK, FS_TITLE, FS_SUP = 11, 9, 10, 14

mpl.rcParams.update({
    "font.family":      "serif",
    "axes.labelsize":   FS_LBL,
    "axes.titlesize":   FS_TITLE,
    "xtick.labelsize":  FS_TICK,
    "ytick.labelsize":  FS_TICK,
    "legend.fontsize":  FS_TICK,
})

# ────────── helper ──────────
def create_selected_scatter(
        pred, actual,
        phase_idx, phase_name,
        time_points, pairs,
        save_path,
        marker="o", ms=16, alpha=0.45,
        color=None):
    """
    pred        : list[np.ndarray] – pred[i] shape (n_wells, n_end, n_phases)
    actual      : torch.Tensor    – (n_wells, n_times, n_phases)
    phase_idx   : int             – 0=gas, 1=oil, 2=water
    pairs       : list[(int,int)] – calendar-day (start,end) pairs
                                   (here only start=0 is used)
    """

    # Map calendar → index
    idx  = {d: i for i, d in enumerate(time_points)}
    pidx = [(idx[s], idx[e]) for s, e in pairs]       # e.g. (0,360) …

    starts = sorted({p[0] for p in pidx})             # == [0]
    ends   = sorted({p[1] for p in pidx})             # 360,720,1080
    n_r, n_c = len(starts), len(ends)                 # 1 × 3

    fig, axes = plt.subplots(
        n_r, n_c,
        figsize=(3.25 * n_c, 3.25),   # single-row
        squeeze=False
    )

    default_col = color or PALETTE[0]

    # --- plot each pair ---
    for si, sj in pidx:
        r, c = starts.index(si), ends.index(sj)
        ax = axes[r, c]

        pred_v = pred[si][:, sj - si, phase_idx].flatten()
        true_v = actual[:, sj, phase_idx].cpu().numpy().flatten()

        ax.scatter(true_v, pred_v,
                   s=ms, marker=marker, alpha=alpha,
                   edgecolors="k", linewidths=0.25,
                   color=default_col)

        # 1:1 line with 10 % padding
        lo, hi = true_v.min(), true_v.max()
        pad = 0.1 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color="0.3")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")

        if c == 0:
            ax.set_ylabel("Predicted")
        ax.set_xlabel("Actual")

        r2 = 1 - np.sum((true_v - pred_v) ** 2) / max(
                 1e-10, np.sum((true_v - true_v.mean()) ** 2))
        ax.set_title(f'0→{time_points[sj]} d\n$R^2={r2:.3f}$', fontsize=FS_TITLE)

        ax.tick_params(labelsize=FS_TICK, direction="in")
        ax.locator_params(nbins=4, tight=True)
        ax.grid(True, which="both", ls=":", lw=.5, alpha=.6)

    fig.suptitle(f'Prediction vs Actual – {phase_name}', fontsize=FS_SUP, y=0.92)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# append to the end of plot_utils.py
# -----------------------------------
def create_phase_grid_scatter(pred_start0,
                              actual,
                              time_points,
                              phase_tuples,
                              end_days,
                              save_path,
                              marker="o", ms=16, alpha=0.45):
    """
    Parameters
    ----------
    pred_start0 : np.ndarray
        Predictions for start_idx = 0; shape (n_wells, n_end, n_phases).
    actual      : torch.Tensor
        Ground-truth cumulative tensor (n_wells, n_times, n_phases).
    phase_tuples: list[(idx, name)]
        Row order, e.g. [(1,"Oil"), (0,"Gas"), (2,"Water")].
    end_days    : list[int]
        Forecast horizons in calendar days, e.g. [360, 720, 1080].
    save_path   : str
        Output filename.
    """

    idx_map = {d: i for i, d in enumerate(time_points)}
    start_i = idx_map[0]                     # always 0-day history
    end_idx = [idx_map[d] for d in end_days] # column indices

    n_r, n_c = len(phase_tuples), len(end_days)
    fig, axes = plt.subplots(
        n_r, n_c,
        figsize=(3.25 * n_c, 3.25 * n_r),
        squeeze=False
    )

    default_col = PALETTE[0]

    # iterate over rows (phases) and columns (horizons)
    for r, (p_idx, p_name) in enumerate(phase_tuples):
        for c, ej in enumerate(end_idx):
            ax = axes[r, c]

            pred_v = pred_start0[:, ej - start_i, p_idx].flatten()
            true_v = actual[:, ej, p_idx].cpu().numpy().flatten()

            ax.scatter(true_v, pred_v,
                       s=ms, marker=marker, alpha=alpha,
                       edgecolors="k", linewidths=0.25,
                       color=default_col)

            lo, hi = true_v.min(), true_v.max()
            pad = 0.1 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color="0.3")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")

            # labels only on outer edges
            if r == n_r - 1: ax.set_xlabel("Actual")
            if c == 0:       ax.set_ylabel("Predicted")

            # title: horizon + R^2
            ss_res = np.sum((true_v - pred_v)**2)
            ss_tot = np.sum((true_v - true_v.mean())**2)
            r2 = 1 - ss_res / max(1e-10, ss_tot)
            ax.set_title(f'0→{end_days[c]} d\n$R^2={r2:.3f}$',
                         fontsize=FS_TITLE, pad=6)

            ax.tick_params(labelsize=FS_TICK, direction="in")
            ax.locator_params(nbins=4, tight=True)
            ax.grid(True, which="both", ls=":", lw=.5, alpha=.6)
            
            # Add row label to identify the phase (only for the first column)
            if c == 0:
                # Add text label at the left margin
                ax.text(-0.25, 0.5, p_name, 
                       transform=ax.transAxes,
                       fontsize=FS_LBL+1, fontweight='bold',
                       ha='right', va='center',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    fig.suptitle("Prediction vs Actual – Zero-history forecasts",
                 fontsize=FS_SUP, y=0.92)
    # Adjust layout to make room for the row labels
    fig.tight_layout(rect=[0.02, 0, 1, 0.88])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==============================================================
#  Updated plotting helpers — 2×-wider prediction intervals
# ==============================================================

def plot_well_trajectories(trajectories, actual, well_indices,
                           days, phase_names, save_path):
    """
    Plot trajectories for selected wells with uncertainty bounds.
    Prediction band widened by a factor of two relative to the
    empirical 90 % CI (5th–95th percentiles).
    """
    n_wells, n_phases = len(well_indices), len(phase_names)
    fig, axes = plt.subplots(n_wells, n_phases,
                             figsize=(15, 2.8 * n_wells),
                             constrained_layout=True)

    for i, w in enumerate(well_indices):
        for j, ph in enumerate(phase_names):
            ax = axes[i, j] if n_wells > 1 else axes[j]

            # --- raw data ------------------------------------------------
            y_true = actual[w, :, j].cpu().numpy()
            ax.plot(days, y_true, "-o", color="firebrick",
                    lw=1.2, label="Actual")

            # --- prediction samples -------------------------------------
            y_samp = trajectories[:, w, :, j]         # [S, T]
            for k in range(y_samp.shape[0]):
                ax.plot(days, y_samp[k], color="royalblue",
                        alpha=0.03, lw=0.5)

            # mean + widened CI
            mean = np.mean(y_samp, axis=0)
            p5  = np.percentile(y_samp, 5, axis=0)
            half = mean - p5
            lower = mean - 2.0 * half
            upper = mean + 2.0 * half

            ax.plot(days, mean, color="royalblue", lw=2, label="p50")
            ax.fill_between(days, lower, upper,
                            color="royalblue", alpha=0.20,
                            label="p10-p90")

            # titles / labels
            if i == 0:
                ax.set_title(ph, fontsize=14)
            if j == 0:
                ax.set_ylabel(f"Well {w}", fontsize=12)
            if i == n_wells - 1:
                ax.set_xlabel("Days", fontsize=12)

            ax.grid(ls="--", alpha=0.7)
            ax.tick_params(labelsize=10)
            if i == 0 and j == 0:
                ax.legend(fontsize=10)

    fig.suptitle("Production Forecasts with 2× Uncertainty Band",
                 fontsize=18)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_well_trajectories_by_start(traj_by_start, actual, well_indices,
                                    days, phase_names, save_path,
                                    start_indices):
    """
    Per-well plots for multiple history lengths, using 2×-wider CI.
    """
    n_phases, n_rows = len(phase_names), len(start_indices)

    for w in well_indices:
        fig, axes = plt.subplots(n_rows, n_phases,
                                 figsize=(4.0 * n_phases, 2.8 * n_rows),
                                 constrained_layout=True)

        for r, s_idx in enumerate(start_indices):
            s_day = days[s_idx]
            y_true_all = actual[w].cpu().numpy()      # [T, 3]

            for c, ph in enumerate(phase_names):
                ax = axes[r, c] if n_rows > 1 else axes[c]

                # actual
                ax.plot(days, y_true_all[:, c], "-o",
                        color="firebrick", lw=1.2, label="Actual")

                # predictions for this start index
                y_samp = traj_by_start[r][:, w, :, c]  # [S, T-s_idx]
                # draw sample paths (thin)
                for k in range(y_samp.shape[0]):
                    gap = np.full(len(days), np.nan)
                    gap[s_idx:] = y_samp[k, :len(days)-s_idx]
                    ax.plot(days, gap, color="royalblue",
                            alpha=0.025, lw=0.6)

                # mean + 2× CI
                mean = np.mean(y_samp, axis=0)
                p5   = np.percentile(y_samp, 5, axis=0)
                half = mean - p5
                low  = mean - 2.0 * half
                up   = mean + 2.0 * half

                f = lambda v: np.concatenate([np.full(s_idx, np.nan), v])
                ax.plot(days, f(mean), color="royalblue", lw=2, label="p50")
                ax.fill_between(days, f(low), f(up),
                                color="royalblue", alpha=0.20,
                                label="p10-p90")

                if s_idx > 0:
                    ax.axvline(s_day, color="green", ls="--", lw=1)

                # labels / titles
                if r == 0:           ax.set_title(ph, fontsize=12)
                if c == 0:           ax.set_ylabel(f"Start {s_day} d", fontsize=11)
                if r == n_rows - 1:  ax.set_xlabel("Days", fontsize=11)

                ax.grid(ls="--", alpha=0.6)
                ax.tick_params(labelsize=10)
                if r == 0 and c == 0:
                    ax.legend(loc="upper left", frameon=False)

        fig.suptitle(f"Well {w}: forecasts after 0 d, 360 d, 720 d history",
                     fontsize=14, y=0.995)
        fig.savefig(save_path.replace(".png", f"_well_{w}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
