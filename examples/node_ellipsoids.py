# -*- coding: utf-8 -*-
"""
What the nodes actually learn — the minimal EBF illustration.

Three nodes, one small surface, and the learned ellipsoids drawn on top.
This is the "why does this library exist" picture: a conventional RBF
measures distance from every center with one shared spherical metric,
so a center can only ever cover a round patch.  An EBF node carries its
own matrix ``A_i``, and the optimizer is free to stretch and rotate it.

The test surface is two narrow Gaussian ridges at different angles
(+30 deg and -40 deg).  After fitting, two of the three nodes have
elongated onto a ridge each and turned to match its angle, while the
third stays broad and carries the background.  Nothing tells the model
to do this — it falls out of minimizing the fit error.

Uses ``inv_quadratic``: decaying bases keep ellipsoids compact enough to
draw.  Growing bases like the default ``multiquadric`` drive them to
aspect ratios in the thousands, where the r=1 contour degenerates into
a pair of lines across the whole domain (informative, but unplottable).
See ``examples/RBF_vs_EBF.py`` for the headline accuracy comparison.

The figure is saved to docs/assets/node_ellipsoids.png for the docs.
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import ebf
from example_utils import (DEFAULT_CMAP, SAMPLE_FACE, SAMPLE_EDGE,
                           NODE_FACE, NODE_EDGE, NODE_SIZE, NODE_MARKER,
                           draw_ellipses, ellipsoid_axes, ellipsoid_extent)

LIM = 3.0          # square domain, [-LIM, LIM] in both axes
N_NODES = 3
N_POINTS = 150

# (center_x, center_y, angle_deg, length, width, amplitude)
RIDGES = [(-1.1,  0.9,  30.0, 1.5, 0.35, 1.00),
          ( 1.1, -0.9, -40.0, 1.3, 0.30, 0.85)]


def two_ridges(x, y):
    """Two narrow Gaussian ridges at different angles."""
    z = np.zeros_like(x, dtype=float)
    for cx, cy, deg, sx, sy, amp in RIDGES:
        t = np.radians(deg)
        dx, dy = x - cx, y - cy
        u = dx * np.cos(t) + dy * np.sin(t)      # along the ridge
        v = -dx * np.sin(t) + dy * np.cos(t)     # across it
        z = z + amp * np.exp(-(u / sx) ** 2 - (v / sy) ** 2)
    return z


if __name__ == "__main__":

    # --- Scattered samples of the surface ---
    rng = np.random.default_rng(3)
    X_pts = rng.random((N_POINTS, 2)) * 2 * LIM - LIM
    z_pts = two_ridges(X_pts[:, 0], X_pts[:, 1])

    # --- Fit ---
    model = ebf.EBF(n_nodes=N_NODES, basis='inv_quadratic')
    model.fit(
        X_pts, z_pts,
        steps=8000,
        var_weight=0.05,     # keep the three nodes from collapsing together
        seed=0,
        verbose=False,
    )
    nodes = model.get_nodes()
    A = model.get_ellipsoids()

    # --- Report what each node learned ---
    direction, aspect = ellipsoid_axes(A)
    angles = np.degrees(np.arctan2(direction[:, 1], direction[:, 0])) % 180
    print(f"Fit RMSE: {np.sqrt(np.mean((model.predict(X_pts) - z_pts)**2)):.4f}")
    print("\n node        center          long-axis angle   elongation")
    for i in range(N_NODES):
        print(f"   {i}   ({nodes[i, 0]:6.2f}, {nodes[i, 1]:6.2f})"
              f"        {angles[i]:6.1f} deg        {aspect[i]:5.1f}x")
    print("\nRidge angles in the data: "
          + ", ".join(f"{deg % 180:.0f} deg" for _, _, deg, _, _, _ in RIDGES))

    # --- Evaluate both surfaces on a grid ---
    g = np.linspace(-LIM, LIM, 220)
    xx, yy = np.meshgrid(g, g)
    z_true = two_ridges(xx, yy)
    z_fit = model.predict(np.column_stack([xx.ravel(), yy.ravel()])
                          ).reshape(xx.shape)
    levels = np.linspace(float(z_true.min()), float(z_true.max()), 20)

    # --- Plot: samples vs fit + learned ellipsoids ---
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6), constrained_layout=True)

    axes[0].contourf(xx, yy, z_true, levels=levels, cmap=DEFAULT_CMAP,
                     extend='both')
    axes[0].scatter(X_pts[:, 0], X_pts[:, 1], c=SAMPLE_FACE, s=16,
                    edgecolors=SAMPLE_EDGE, linewidths=0.6, zorder=5,
                    label='samples')
    axes[0].set_title(f'The data — {N_POINTS} scattered samples\n'
                      'of two narrow ridges at different angles',
                      fontsize=12)
    axes[0].legend(loc='upper right', fontsize=9, framealpha=0.9)

    cf = axes[1].contourf(xx, yy, z_fit, levels=levels, cmap=DEFAULT_CMAP,
                          extend='both')

    # Separate the compact feature nodes from any broad background node,
    # so one domain-spanning ellipse doesn't read as a plotting error.
    extent = ellipsoid_extent(A)
    is_broad = extent > 2.0 * np.median(extent)
    draw_ellipses(axes[1], nodes[~is_broad], A[~is_broad],
                  label='learned ellipsoid (r = 1)')
    if is_broad.any():
        draw_ellipses(axes[1], nodes[is_broad], A[is_broad],
                      linestyle='--', lw=1.5, alpha=0.7,
                      label='broad background node')

    axes[1].scatter(nodes[:, 0], nodes[:, 1], c=NODE_FACE, s=NODE_SIZE,
                    marker=NODE_MARKER, edgecolors=NODE_EDGE,
                    linewidths=0.7, zorder=7, label='nodes')
    axes[1].set_title(f'The EBF fit — only {N_NODES} nodes\n'
                      'each stretched and rotated onto its feature',
                      fontsize=12)
    axes[1].legend(loc='upper right', fontsize=9, framealpha=0.9)

    for ax in axes:
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.set_aspect('equal')       # required: ellipses shear otherwise
        ax.set_xlabel('X')
    axes[0].set_ylabel('Y')

    fig.colorbar(cf, ax=axes[-1], fraction=0.046, pad=0.04)
    fig.suptitle('How EBF nodes adapt to the shape of the data', fontsize=14)

    # --- Save for the documentation ---
    out_path = Path(__file__).parent.parent / "docs" / "assets" / "node_ellipsoids.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {out_path}")

    # Skip the interactive window when generating docs figures.
    if "--save-only" not in sys.argv:
        plt.show()
