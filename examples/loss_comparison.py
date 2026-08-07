# -*- coding: utf-8 -*-
"""
Loss type comparison — rmse vs huber vs tukey on imperfect data.

Two scenarios built from the same 2-D test surface as RBF_vs_EBF.py:

* noisy    — Gaussian noise on every sample
* outliers — the same noisy samples plus a few grossly wrong points

Each scenario is fit three times, once per ``loss_type``, and every fit
is scored against the CLEAN ground truth so the numbers show how well
each loss recovers the underlying surface:

* rmse  — squared error; every residual pulls with its full weight, so
          outliers drag the surface toward themselves
* huber — quadratic core, linear tails; outliers keep a constant,
          bounded pull (ADR-009/013)
* tukey — redescending; points beyond the rejection point exert zero
          pull and are effectively discarded (ADR-014)

What the current numbers show: with noise only, all three losses land
within about 10% of each other (rmse 7.81, huber 7.71, tukey 8.44) —
tukey trails slightly because its aggressive rejection also discards
genuine sharp features on the diagonal ridge.  Adding just two gross
outliers collapses squared error (23.33, with a visible gash across the
surface) while both robust losses hold near 7.2 (huber 7.21, tukey 7.57).

So huber is the sensible default: it was at least as good as tukey in
both scenarios here, and it degrades gracefully.  Tukey earns its keep
when outliers are numerous or extreme enough that even huber's bounded
linear pull still drags the surface — raise ``n_outliers`` or
``outlier_size`` below to see it take the lead.

The figure is saved to docs/assets/loss_comparison.png for the docs.
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import ebf
from example_utils import (test_func, sample_test_func, make_grid,
                           hull_mask, contour_panel, ACCENT)

if __name__ == "__main__":

    # --- Scattered training samples ---
    n_points = 100
    n_nodes = 16
    n_outliers = 2
    noise_std = 10.0        # ~3% of the surface range
    outlier_size = 150.0   # gross errors, ~half the surface range

    X_pts, z_clean = sample_test_func(n_points, seed=42)

    rng = np.random.default_rng(7)
    z_noisy = z_clean + rng.normal(0.0, noise_std, n_points)

    # Corrupt a few random points with large errors of random sign
    outlier_idx = rng.choice(n_points, size=n_outliers, replace=False)
    z_outliers = z_noisy.copy()
    z_outliers[outlier_idx] += rng.choice([-1.0, 1.0], n_outliers) * outlier_size

    scenarios = [
        (f'noisy (std = {noise_std:g})', z_noisy),
        (f'noisy + {n_outliers} outliers', z_outliers),
    ]
    loss_types = ['rmse', 'huber', 'tukey']

    # --- Evaluation grid & clean reference ---
    n_grid = 100
    xx, yy, grid_pts = make_grid(n_grid)
    z_true = test_func(xx, yy)
    outside = hull_mask(X_pts, xx, yy)
    inside = ~outside

    # --- Fit every (scenario, loss_type) pair ---
    # results[scenario_name][loss_type] = (masked grid surface, rmse vs truth)
    results = {}
    for scen_name, z_data in scenarios:
        results[scen_name] = {}
        for loss_type in loss_types:
            model = ebf.EBF(n_nodes=n_nodes, basis='multiquadric')
            model.fit(
                X_pts, z_data,
                steps=40000,
                var_weight=0.01,
                ellipsoid_weight=0.001,
                loss_type=loss_type,
                huber_delta='auto',    # only used by 'huber' (ADR-013)
                tukey_c='auto',        # only used by 'tukey' (ADR-014)
                seed=42,
                verbose=False,
            )
            z_fit = model.predict(grid_pts).reshape(n_grid, n_grid)
            rmse = np.sqrt(np.mean((z_fit[inside] - z_true[inside]) ** 2))
            results[scen_name][loss_type] = (np.where(outside, np.nan, z_fit),
                                             rmse)
            print(f"{scen_name:24s} loss_type={loss_type:6s} "
                  f"RMSE vs clean truth = {rmse:.2f}")

    # --- Plot: one row per scenario, ground truth + one panel per loss ---
    levels = np.linspace(float(z_true.min()), float(z_true.max()), 25)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9),
                             sharex=True, sharey=True)

    for row, (scen_name, z_data) in enumerate(scenarios):
        contour_panel(axes[row, 0], xx, yy, z_true, levels,
                      samples=X_pts,
                      title=f'Ground Truth\nsamples: {scen_name}')
        if row == 1:  # mark the corrupted points on the outlier row
            axes[row, 0].scatter(X_pts[outlier_idx, 0], X_pts[outlier_idx, 1],
                                 c=ACCENT, s=90, marker='x', linewidths=2.5,
                                 zorder=6, label='outliers')
            axes[row, 0].legend(loc='upper right', fontsize=8)

        for col, loss_type in enumerate(loss_types, start=1):
            z_fit, rmse = results[scen_name][loss_type]
            cf = contour_panel(axes[row, col], xx, yy, z_fit, levels,
                               samples=X_pts,
                               title=f"loss_type='{loss_type}'\n"
                                     f"RMSE vs truth = {rmse:.2f}")

    fig.colorbar(cf, ax=axes[:, -1], fraction=0.046, pad=0.04)
    for ax in axes[-1]:
        ax.set_xlabel('X')
    for ax in axes[:, 0]:
        ax.set_ylabel('Y')
    fig.suptitle(f'EBF loss types on imperfect data — {n_points} samples, '
                 f'{n_nodes} nodes', fontsize=13)

    # --- Save for the documentation ---
    out_path = Path(__file__).parent.parent / "docs" / "assets" / "loss_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {out_path}")

    # Skip the interactive window when generating docs figures.
    if "--save-only" not in sys.argv:
        plt.show()
