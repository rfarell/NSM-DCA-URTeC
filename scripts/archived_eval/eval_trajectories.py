#!/usr/bin/env python
# scripts/eval_trajectories.py
"""
Generate trajectory plots showing production forecasts with uncertainty.
"""

import os, sys, argparse, random
import numpy as np
import torch
import matplotlib as mpl

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.evaluation import (load_experiment, predict_trajectories,
                            plot_well_trajectories,
                            plot_well_trajectories_by_start)

# ────────── CLI ──────────
def parse_args():
    p = argparse.ArgumentParser(description="Generate trajectory plots with uncertainty bounds")
    p.add_argument("--experiment", required=True, type=str)
    p.add_argument("--device",      default=None, type=str)
    p.add_argument("--num-traj",    default=100,  type=int, help="# MC samples per trajectory")
    p.add_argument("--num-wells",   default=10,   type=int)
    p.add_argument("--seed",        default=42,   type=int)
    p.add_argument("--specific-wells", default=None,
                   help="Comma-separated well indices; overrides random choice")
    return p.parse_args()

mpl.rcParams.update({
    "axes.labelsize":   12,
    "axes.titlesize":   12,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "lines.markersize": 4,
    "lines.linewidth":  1.5,
})


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # ---------- load experiment ----------
    exp = load_experiment(args.experiment, device=args.device)
    model, device, eval_dir = exp["model"], exp["device"], exp["eval_dir"]
    x_test, z_test, t_vals = exp["x_test"], exp["z_test"], exp["t_vals"]

    days = [0, 30, 60, 90, 180, 360, 720, 1080]
    phase_names = ["Gas", "Oil", "Water"]

    # wells to plot -------------------------------------------------
    n_test = x_test.shape[0]
    if args.specific_wells:
        well_idx_list = [int(i) for i in args.specific_wells.split(",")]
    else:
        well_idx_list = random.sample(range(n_test), min(args.num_wells, n_test))

    print("Wells chosen:", well_idx_list)

    # --------- trajectories from selected start times --------------
    # only 0 d, 360 d, 720 d  → indices 0,5,6 in `days`
    sel_start_idx = [0, 5, 6]
    sel_days      = [days[i] for i in sel_start_idx]

    print("Generating trajectories at start days:", sel_days)
    traj_by_start = []
    for s_i in sel_start_idx:
        t_sub  = t_vals[s_i:]                       # absolute times
        x0_sub = x_test[:, s_i, :]                  # initial state
        traj   = predict_trajectories(model, z_test, t_sub, x0_sub,
                                      num_samples=args.num_traj, device=device)
        traj_by_start.append(traj)

    # --------- per-well figure (3 rows × 3 phases) -----------------
    for w in well_idx_list:
        print(f"Plotting per-start trajectories for well {w}")
        plot_well_trajectories_by_start(traj_by_start, x_test,
                                        [w], days, phase_names,
                                        os.path.join(eval_dir,
                                                     f"traj_by_start_well{w}.png"),
                                        sel_start_idx)

    # --------- combined zero-history figure ------------------------
    print("Plotting combined zero-history trajectories …")
    x0_0 = x_test[:, 0, :]
    traj0 = predict_trajectories(model, z_test, t_vals, x0_0,
                                 num_samples=args.num_traj, device=device)

    plot_well_trajectories(traj0, x_test, well_idx_list, days, phase_names,
                           os.path.join(eval_dir, "traj_all_wells.png"))

    print("Trajectory plots saved in", eval_dir)


if __name__ == "__main__":
    main()
