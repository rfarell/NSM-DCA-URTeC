#!/usr/bin/env python
"""
Create a single 3×3 scatter-grid PNG (oil–gas–water × 3 horizons).
"""

import os, sys, argparse, random
import numpy as np
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.evaluation import load_experiment, predict_from_t, create_phase_grid_scatter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--num-mc", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    exp = load_experiment(args.experiment, device=args.device)
    model   = exp["model"]
    device  = exp["device"]
    eval_dir= exp["eval_dir"]
    x_test  = exp["x_test"]
    z_test  = exp["z_test"]
    t_vals  = exp["t_vals"]
    dp      = exp["data_processor"]
    scale_test = getattr(dp, 'scale_test', None)

    # calendar grid
    days = [0, 30, 60, 90, 180, 360, 720, 1080]
    time_points = days
    end_days = [360, 720, 1080]

    # predictions for start_idx = 0 only
    preds0 = predict_from_t(model, z_test, t_vals, x_test,
                            start_idx=0, num_mc=args.num_mc, device=device)

    # De-normalise if scale provided
    if scale_test is not None:
        # scale_test might be a tensor, convert to numpy for preds0
        if isinstance(scale_test, torch.Tensor):
            st_numpy = scale_test.cpu().numpy()[:, np.newaxis, :]  # [N,1,3] for numpy
            scale_test_device = scale_test.to(x_test.device)
        else:
            st_numpy = scale_test[:, np.newaxis, :]  # [N,1,3] for numpy
            scale_test_device = torch.tensor(scale_test, device=x_test.device, dtype=x_test.dtype)
        
        # Apply scaling
        preds0 = preds0 * st_numpy  # both are numpy arrays now
        x_test_denorm = x_test * scale_test_device.unsqueeze(1)
        # Move back to CPU for plotting
        x_test_denorm = x_test_denorm.cpu()
    else:
        x_test_denorm = x_test.cpu()

    # phase order: Oil (1), Gas (0), Water (2)
    phase_tuples = [(1, "Oil"), (0, "Gas"), (2, "Water")]

    os.makedirs(eval_dir, exist_ok=True)
    out_png = os.path.join(eval_dir, "pred_vs_actual_allphases.png")
    create_phase_grid_scatter(preds0, x_test_denorm, time_points,
                              phase_tuples, end_days, out_png)
    print(f"Saved combined scatter grid to {out_png}")


if __name__ == "__main__":
    main()
