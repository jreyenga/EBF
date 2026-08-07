# -*- coding: utf-8 -*-
"""
Compressor map — the premium end-to-end EBF example (class-based API).

2-D input (corrected mass flow, pressure ratio) -> efficiency.

Data: ``data/GenericMap.xlsx``, sheet ``data`` — 56 operating points
with three columns:

===== ============================== ========
mdot  corrected mass flow            kg/s
PR    total-to-total pressure ratio  -
eta   total-to-total efficiency      -
===== ============================== ========

Demonstrates every user-configurable input and every tool in the repo:

* ``ebf.BASIS_FUNCTIONS``   — the basis function registry
* ``ebf.EBF``               — constructor, fit (all parameters), predict,
                              get_nodes, save, load
* ``EBF.history_``          — per-step training history
* ``ebf.convergence_plot``  — training loss curve
* ``ebf.correlation_plot``  — data vs prediction with R²
* ``ebf.residual_plot``     — residuals vs prediction with RMSE
* ``ebf.contour_plot_2d``   — fitted surface with data and node overlays
* ``ebf.summary_plot_3d``   — contour + correlation + convergence in one figure
* ``ebf.eval_grid``         — N-dimensional evaluation grid
* ``ebf.export_grid``       — lookup-table export (CSV / NPZ)

For the functional API (``ebf.run`` / ``ebf.run_points``), see the API
reference in the docs.
"""
import sys

import numpy as np
import ebf
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

if __name__ == "__main__":

    # --- Available basis functions ---
    print("Available basis functions:", list(ebf.BASIS_FUNCTIONS))

    # --- Load data ---
    data_path = Path(__file__).parent.parent / "data" / "GenericMap.xlsx"
    df = pd.read_excel(data_path, sheet_name="data")
    mdot = df['mdot'].values     # corrected mass flow  (input 1)
    PR = df['PR'].values         # pressure ratio       (input 2)
    eta = df['eta'].values       # efficiency           (output)

    X = np.column_stack([mdot, PR])
    print(f"Loaded {len(eta)} operating points from {data_path.name}")

    # --- Create model ---
    # All constructor parameters shown explicitly:
    model = ebf.EBF(
        n_nodes=9,               # number of interpolation nodes
        basis='multiquadric',    # basis function — see ebf.BASIS_FUNCTIONS for all options
        eps=1e-8,                # numerical stability offset (only affects some basis functions)
    )

    # --- Train ---
    # All fit parameters shown explicitly:
    model.fit(
        X, eta,                  # inputs (n_points, n_dims) and output (n_points,)
        steps=80000,             # optimizer iterations
        lr=0.01,                 # initial learning rate (Adam with exponential decay)
        var_weight=0.01,         # node spread regularization strength — also the primary smoothing knob
        ellipsoid_weight=0.001,  # ellipsoid shape penalty — explicit smoothness knob, 0 disables (ADR-011)
        loss_type='huber',       # 'rmse' (default), 'huber' (downweights outliers), or 'tukey' (rejects outliers)
        huber_delta='auto',      # 'auto' tracks the residual noise floor (ADR-013); a float fixes the threshold
        tukey_c='auto',          # Tukey rejection point (ADR-014) — keep 'auto'; only used with loss_type='tukey'
        val_fraction=0.0,        # held-out fraction for early stopping — 0 disables; needs ~50+ points (ADR-012)
        patience=10,             # stop after this many val evaluations (1 per 100 steps) without improvement
        verbose=True,            # print training progress every 100 steps
        loss_threshold=0.05,     # early stopping when loss <= this value (None to disable)
        seed=42,                 # reproducible weight initialization (None for non-deterministic)
    )

    # --- Training history ---
    # fit() stores a (step, loss) array — useful for judging convergence
    # and tuning var_weight / loss_threshold / steps.
    print(f"Trained for {len(model.history_)} steps, "
          f"final loss {model.history_[-1, 1]:.4f}")

    # --- Predict & inspect ---
    Out = model.predict(X)       # predictions at training points
    Nodes = model.get_nodes()    # node positions in original (unscaled) space
    print("Node positions:\n", Nodes)

    # --- Save and reload ---
    save_dir = Path(__file__).parent.parent / "checkpoints"
    save_dir.mkdir(exist_ok=True)
    ckpt_path = model.save(str(save_dir), filename='compressor-map')
    print("Saved to:", ckpt_path)

    loaded_model = ebf.EBF.load(ckpt_path)
    Out_loaded = loaded_model.predict(X)
    print("Predictions match after reload:",
          np.allclose(Out, Out_loaded, atol=1e-6))

    # --- Convergence plot ---
    # Pass the model directly (its history_ is used) or the array itself;
    # loss_threshold draws the early-stopping target as a reference line.
    ebf.convergence_plot(model, loss_threshold=0.05)

    # --- Correlation plot ---
    ebf.correlation_plot(eta, Out)

    # --- Residual plot ---
    # Residuals vs prediction — bias trends, error scaling, and
    # outliers that the correlation plot hides along its 1:1 line.
    ebf.residual_plot(eta, Out)

    # --- Contour plot ---
    ebf.contour_plot_2d(
        model, X, eta,
        xlabel='Corrected Mass Flow, mdot',
        ylabel='Pressure Ratio, PR',
        zlabel='Efficiency, eta',
        show_data=True,          # overlay training points
        show_nodes=True,         # overlay EBF node positions (from get_nodes)
    )

    # --- Summary figure ---
    # Fitted surface, correlation, residuals and convergence in one
    # figure; extra keyword arguments are forwarded to contour_plot_2d.
    fig_summary, _ = ebf.summary_plot_3d(
        model, X, eta,
        xlabel='Corrected Mass Flow, mdot',
        ylabel='Pressure Ratio, PR',
        zlabel='Efficiency, eta',
        loss_threshold=0.05,
        show_nodes=True,
    )

    # Saved for the documentation — this is the figure the README uses to
    # show what a real fit looks like end to end.
    summary_path = (Path(__file__).parent.parent / "docs" / "assets"
                    / "compressor_map_summary.png")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fig_summary.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"Summary figure saved to: {summary_path}")

    # --- Evaluation grid & lookup-table export ---
    bounds = list(zip(X.min(axis=0), X.max(axis=0)))
    grid = ebf.eval_grid(model, bounds, n_points=100)

    export_path = save_dir / "compressor_map_lookup.csv"
    ebf.export_grid(
        export_path, grid,
        dim_names=['mdot', 'PR'],
    )
    print(f"Lookup table exported to: {export_path}")

    # Skip the interactive window when generating docs figures.
    if "--save-only" not in sys.argv:
        plt.show()
