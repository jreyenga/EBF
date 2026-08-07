# -*- coding: utf-8 -*-
"""
Basis function gallery — plots all EBF basis functions as a function of radius.

The registry is split into the two families that matter when choosing one:

* Increasing (global)  — grow without bound with distance; every node
  influences the whole domain, like a conventional RBF surface term.
* Decreasing (local)   — decay toward zero with distance; each node only
  shapes the surface near its own ellipsoid.

Functions are pulled from ``ebf.BASIS_FUNCTIONS``, so a newly registered
function shows up automatically (in an "Ungrouped" section until it is
added to a family below).  Output is saved to
docs/assets/basis_functions.png for the documentation.
"""
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

from ebf.basis_functions import BASIS_FUNCTIONS

# Family assignment for every registered basis function
INCREASING = ['linear', 'quadratic', 'cubic', 'thin_plate',
              'multiquadric', 'cosh']
DECREASING = ['gaussian', 'inv_multiquadric', 'inv_quadratic',
              'matern32', 'matern52', 'inv_cosh']

# Human-readable expressions for subplot titles
EXPRESSIONS = {
    'linear':           r'$a_1 \cdot r$',
    'quadratic':        r'$a_1 \cdot r^2$',
    'thin_plate':       r'$a_1 \cdot r^2 \ln(r^2)$',
    'multiquadric':     r'$a_1 (\sqrt{r^2+1} - 1)$',
    'inv_multiquadric': r'$a_1 / \sqrt{r^2+1}$',
    'inv_quadratic':    r'$a_1 / (1+r^2)$',
    'gaussian':         r'$a_1 \cdot e^{-r^2}$',
    'matern32':         r'$a_1 (1+\sqrt{3}r) e^{-\sqrt{3}r}$',
    'matern52':         r'$a_1 (1+\sqrt{5}r+\frac{5}{3}r^2) e^{-\sqrt{5}r}$',
    'cosh':             r'$a_1 \cdot \cosh(\sqrt{r^2+\varepsilon})$',
    'inv_cosh':         r'$a_1 / \cosh(\sqrt{r^2+\varepsilon})$',
    'cubic':            r'$a_1 \cdot r^3$',
}

GROUP_COLORS = {
    'Increasing (global)': '#1f77b4',
    'Decreasing (local)': '#2ca02c',
    'Ungrouped': '#d62728',
}


def evaluate_basis(fn, n_params, r_values, eps=1e-8):
    """Evaluate a basis function over an array of radius values.

    The registry functions expect r2 shaped (n_points, n_nodes) and return
    (n_points,).  We use n_nodes=1 so the reduce_sum over axis=1 is a no-op.
    """
    r2 = tf.constant(r_values ** 2, dtype=tf.float32)
    r2 = tf.reshape(r2, (-1, 1))  # (n_points, 1)
    a1 = tf.constant([1.0])       # (1,) — one node
    a2 = tf.constant([1.0])
    a3 = tf.constant([1.0])

    if n_params == 1:
        y = fn(r2, a1, eps)
    elif n_params == 2:
        y = fn(r2, a1, a2, eps)
    else:
        y = fn(r2, a1, a2, a3, eps)

    return y.numpy()


if __name__ == "__main__":
    r = np.linspace(0, 5, 500).astype(np.float32)

    # Anything registered but not yet assigned to a family still gets plotted
    ungrouped = [n for n in BASIS_FUNCTIONS
                 if n not in INCREASING and n not in DECREASING]
    groups = [('Increasing (global)', INCREASING),
              ('Decreasing (local)', DECREASING)]
    if ungrouped:
        print("Ungrouped basis functions (add them to a family):", ungrouped)
        groups.append(('Ungrouped', ungrouped))

    ncols = 3
    row_counts = [(len(names) + ncols - 1) // ncols for _, names in groups]

    fig = plt.figure(figsize=(13, 3.2 * sum(row_counts)),
                     constrained_layout=True)
    subfigs = np.atleast_1d(fig.subfigures(len(groups), 1,
                                           height_ratios=row_counts))

    for subfig, (group_name, names), nrows in zip(subfigs, groups, row_counts):
        subfig.suptitle(group_name, fontsize=13, fontweight='bold')
        axes = np.atleast_1d(subfig.subplots(nrows, ncols)).flatten()

        for ax, name in zip(axes, names):
            fn, n_params = BASIS_FUNCTIONS[name]
            y = evaluate_basis(fn, n_params, r)

            ax.plot(r, y, linewidth=2, color=GROUP_COLORS[group_name])
            ax.set_title(f"{name}\n{EXPRESSIONS.get(name, '')}", fontsize=10)
            ax.set_xlabel('r')
            ax.set_ylabel(r'$\varphi(r)$')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 5)

        # Hide unused subplots
        for ax in axes[len(names):]:
            ax.set_visible(False)

    fig.suptitle('EBF Basis Function Gallery  (a = 1)', fontsize=14)

    out_path = Path(__file__).parent.parent / "docs" / "assets" / "basis_functions.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {out_path}")

    # Skip the interactive window when generating docs figures.
    if "--save-only" not in sys.argv:
        plt.show()
