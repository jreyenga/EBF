# -*- coding: utf-8 -*-
"""
Loss function gallery — compares the three EBF loss types (rmse, huber, tukey).

Two panels as a function of the residual r (in units of the robust residual
scale sigma, since the adaptive thresholds are calibrated as multiples of it):

* rho(r)  — the per-point loss each residual contributes
* psi(r)  — the influence (d rho / d r): how hard a point pulls on the surface

Squared error grows without bound, Huber caps the pull at a constant beyond
delta (ADR-013), and Tukey redescends to exactly zero beyond c — points out
there are effectively discarded (ADR-014).

Thresholds are drawn at the classical tuning constants used by the 'auto'
calibration in ebf.train (delta = 1.345*sigma, c = 4.685*sigma).
Output is saved to docs/assets/loss_functions.png.
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ebf.train import HUBER_K, TUKEY_K

COLORS = {'squared': '#888880', 'huber': '#2a78d6', 'tukey': '#1baf7a'}
STYLES = {'squared': '-', 'huber': '--', 'tukey': '-.'}


def rho_squared(r):
    """Per-point squared-error loss (the 'rmse' data term is sqrt(mean(r^2)))."""
    return 0.5 * r ** 2


def rho_huber(r, delta=HUBER_K):
    """Huber: quadratic core, linear beyond delta (ADR-009/013)."""
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r ** 2, delta * (a - 0.5 * delta))


def rho_tukey(r, c=TUKEY_K):
    """Tukey biweight: quadratic core, constant (c^2/6) beyond c (ADR-014)."""
    u = np.maximum(0.0, 1.0 - (r / c) ** 2)
    return (c ** 2 / 6.0) * (1.0 - u ** 3)


def psi_squared(r):
    return r


def psi_huber(r, delta=HUBER_K):
    return np.clip(r, -delta, delta)


def psi_tukey(r, c=TUKEY_K):
    u = np.maximum(0.0, 1.0 - (r / c) ** 2)
    return r * u ** 2


if __name__ == "__main__":
    r = np.linspace(-6, 6, 1000)

    losses = {
        'squared (rmse)': (rho_squared, psi_squared, 'squared'),
        f'huber ($\\delta = {HUBER_K}\\sigma$)': (rho_huber, psi_huber, 'huber'),
        f'tukey ($c = {TUKEY_K}\\sigma$)': (rho_tukey, psi_tukey, 'tukey'),
    }

    fig, (ax_rho, ax_psi) = plt.subplots(1, 2, figsize=(12, 4.5),
                                         constrained_layout=True)

    for label, (rho, psi, key) in losses.items():
        ax_rho.plot(r, rho(r), STYLES[key], color=COLORS[key],
                    linewidth=2, label=label)
        ax_psi.plot(r, psi(r), STYLES[key], color=COLORS[key],
                    linewidth=2, label=label)

    for ax in (ax_rho, ax_psi):
        for x, key in ((HUBER_K, 'huber'), (TUKEY_K, 'tukey')):
            ax.axvline(x, color=COLORS[key], linewidth=0.8, alpha=0.4)
            ax.axvline(-x, color=COLORS[key], linewidth=0.8, alpha=0.4)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(r'residual $r$  (units of robust scale $\sigma$)')
        ax.legend(loc='upper center', fontsize=9)

    ax_rho.set_title(r'Loss $\rho(r)$ — what the optimizer sums')
    ax_rho.set_ylabel(r'$\rho(r)$')
    ax_rho.set_ylim(0, 8)

    ax_psi.set_title(r'Influence $\psi(r) = d\rho/dr$ — pull on the surface')
    ax_psi.set_ylabel(r'$\psi(r)$')
    ax_psi.set_ylim(-2.5, 2.5)

    fig.suptitle('EBF loss types: squared error vs. Huber vs. Tukey biweight',
                 fontsize=13)

    out_path = Path(__file__).parent.parent / "docs" / "assets" / "loss_functions.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {out_path}")

    # Skip the interactive window when generating docs figures.
    if "--save-only" not in sys.argv:
        plt.show()
