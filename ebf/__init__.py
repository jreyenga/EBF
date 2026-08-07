# -*- coding: utf-8 -*-
"""
EBF — Elliptical Basis Function interpolation library.

Public API
----------
EBF(n_nodes, ...)              Class-based API: fit / predict / get_nodes
run(data, n_nodes, ...)        Train an EBF model; returns (Scale, Offset, file)
run_points(points, ...)        Evaluate a trained model at new points
BASIS_FUNCTIONS                Dict of available basis functions
"""
from ebf.ebf import EBF
from ebf.train import run
from ebf.predict import run_points
from ebf.basis_functions import BASIS_FUNCTIONS, DEFAULT_BASIS
from ebf.viz import (correlation_plot, residual_plot, convergence_plot,
                     contour_plot_2d, summary_plot_3d, eval_grid,
                     export_grid)

__all__ = ["EBF", "run", "run_points", "BASIS_FUNCTIONS", "DEFAULT_BASIS",
           "correlation_plot", "residual_plot", "convergence_plot",
           "contour_plot_2d", "summary_plot_3d", "eval_grid", "export_grid"]
