# -*- coding: utf-8 -*-
"""
RBF vs EBF — the headline comparison example.

Fits 50 scattered samples of a 2-D test surface with both a conventional
Scipy RBF (multiquadric, one isotropic length scale) and an EBF
(multiquadric, per-node learned ellipsoids), then plots
Ground Truth / RBF / EBF side by side with a shared color scale.

The surface hides a narrow ridge running diagonally across the domain
(see ``example_utils.test_func``).  The isotropic RBF has to ring around
it; the EBF stretches its node ellipsoids along the ridge and follows it.

The figure is saved to docs/assets/rbf_vs_ebf.png for the documentation.
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import Rbf

import ebf
from example_utils import (test_func, sample_test_func, make_grid,
                           hull_mask, contour_panel)

if __name__ == "__main__":

    # --- Scattered training samples ---
    n_points = 50
    n_nodes = 16
    X_pts, z_pts = sample_test_func(n_points, seed=42)

    # --- Train EBF ---
    model = ebf.EBF(n_nodes=n_nodes, basis='multiquadric')
    model.fit(
        X_pts, z_pts,
        steps=60000,
        loss_threshold=0.04,     # stop early once the loss is good enough
        var_weight=0.01,         # light node-spread regularization
        ellipsoid_weight=0.001,  # light smoothness penalty (ADR-011)
        loss_type='huber',       # robust to any rough spots (ADR-009)
        huber_delta='auto',      # threshold tracks the residual noise floor (ADR-013)
        seed=42,
    )
    nodes = model.get_nodes()

    # --- Evaluate both fits on a grid ---
    n_grid = 100
    xx, yy, grid_pts = make_grid(n_grid)

    z_true = test_func(xx, yy)
    z_ebf = model.predict(grid_pts).reshape(n_grid, n_grid)

    # Scipy RBF baseline: exact interpolation, one node per sample
    rbfi = Rbf(X_pts[:, 0], X_pts[:, 1], z_pts, function='multiquadric')
    z_rbf = rbfi(xx, yy)

    # --- Compare against ground truth inside the convex hull ---
    # Both methods extrapolate freely outside the data; score them only
    # where the fit is supported.
    outside = hull_mask(X_pts, xx, yy)
    z_ebf_masked = np.where(outside, np.nan, z_ebf)
    z_rbf_masked = np.where(outside, np.nan, z_rbf)

    inside = ~outside
    rmse_rbf = np.sqrt(np.mean((z_rbf[inside] - z_true[inside]) ** 2))
    rmse_ebf = np.sqrt(np.mean((z_ebf[inside] - z_true[inside]) ** 2))
    print(f"RMSE vs ground truth (inside hull) — "
          f"RBF: {rmse_rbf:.2f}   EBF: {rmse_ebf:.2f}")

    # --- Plot: Ground Truth / RBF / EBF ---
    # Shared color levels from the ground truth; the unmasked surface is
    # drawn underneath at low alpha to show extrapolation behaviour.
    levels = np.linspace(float(z_true.min()), float(z_true.max()), 25)
    title_size = 12.5

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    contour_panel(axes[0], xx, yy, z_true, levels,
                  samples=X_pts, title_size=title_size,
                  title='Ground Truth\n(narrow diagonal ridge)')
    contour_panel(axes[1], xx, yy, z_rbf_masked, levels, z_full=z_rbf,
                  samples=X_pts, title_size=title_size,
                  title=f'Scipy RBF — 50 centers (one per sample)\n'
                        f'RMSE = {rmse_rbf:.2f}')
    cf = contour_panel(axes[2], xx, yy, z_ebf_masked, levels, z_full=z_ebf,
                       samples=X_pts, nodes=nodes, title_size=title_size,
                       title=f'EBF — {n_nodes} learned nodes\n'
                             f'RMSE = {rmse_ebf:.2f}')

    fig.colorbar(cf, ax=axes[-1], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.set_xlabel('X')
    axes[0].set_ylabel('Y')
    fig.suptitle(
        f'EBF cuts error {rmse_rbf / rmse_ebf:.1f}x using {n_nodes} nodes '
        f'against the RBF\'s {n_points} centers  —  '
        f'{n_points} scattered samples of a ridged test surface',
        fontsize=14)
    plt.tight_layout()

    # --- Save for the documentation ---
    out_path = Path(__file__).parent.parent / "docs" / "assets" / "rbf_vs_ebf.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {out_path}")

    # Skip the interactive window when generating docs figures.
    if "--save-only" not in sys.argv:
        plt.show()
