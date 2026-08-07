# API Reference

## Class API (recommended)

::: ebf.EBF
    options:
      members:
        - __init__
        - fit
        - predict
        - get_nodes
        - get_ellipsoids
        - save
        - load

## Functional API

The functional API is the original interface, kept for backwards compatibility.
For new code, prefer the class API above.

::: ebf.train.run

::: ebf.predict.run_points

## Visualization Utilities

::: ebf.viz.convergence_plot

::: ebf.viz.correlation_plot

::: ebf.viz.residual_plot

::: ebf.viz.contour_plot_2d

::: ebf.viz.summary_plot_3d

::: ebf.viz.eval_grid

::: ebf.viz.export_grid

## Basis Function Registry

::: ebf.basis_functions
    options:
      members:
        - BASIS_FUNCTIONS
        - DEFAULT_BASIS
      show_if_no_docstring: true
