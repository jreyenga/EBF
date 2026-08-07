# -*- coding: utf-8 -*-
"""
EBF visualization utilities.

Reusable plotting and grid-evaluation helpers for fitted EBF models.
All plot functions return ``(fig, ax)`` and accept an optional *ax*
argument so they can be composed into multi-panel figures.
"""
import numpy as np
from scipy.stats import linregress
from scipy.interpolate import griddata


# ------------------------------------------------------------------
# Shared plot style
# ------------------------------------------------------------------
# One source of truth, so figures from these helpers and the example
# scripts in examples/ read as the same family.  The default colormap is
# a sequential blue ramp, which constrains the overlays:
#
#   * Samples sit on top of the map, so they must not be blue — white
#     with a dark edge stays legible at both ends of the ramp.
#   * Nodes need to be distinguishable from samples at a glance, hence a
#     different color *and* a different marker.  Muted orange is the
#     colorblind-safe complement to blue and reads calmer than a
#     saturated red.
#   * Diagnostic plots (correlation, residual) draw on white rather than
#     over a colormap, so white markers would vanish; they use a mid
#     blue from the same family instead.

DEFAULT_CMAP = 'Blues_r'

SAMPLE_FACE = 'white'      # data points overlaid on a contour map
SAMPLE_EDGE = '#222222'
SAMPLE_SIZE = 26

NODE_FACE = '#e08a45'      # EBF node positions — muted orange
NODE_EDGE = '#4a2f12'
NODE_SIZE = 80
NODE_MARKER = '^'

POINT_FACE = '#4c8dc0'     # scatter on white-background diagnostic plots
POINT_EDGE = '#1f3b57'

ACCENT = '#d1603d'         # reference lines that mark a limit or target
GRID_LINE = 'black'        # contour lines drawn over the filled map
GRID_ALPHA = 0.3

# Error-shaded variant: points are colored by |error| instead of a flat
# fill, so the same red ramp carries the same meaning on every panel of a
# summary figure.  Sequential red reads as "worse = darker" without a
# legend, and stays distinct from the blue surface ramp underneath.  The
# markers are drawn larger than the flat-fill ones because a small marker
# gives the eye too little area to judge its shade.
ERROR_CMAP = 'Reds'
ERROR_SIZE = 46


# ------------------------------------------------------------------
# Correlation plot (any dimensionality)
# ------------------------------------------------------------------

def correlation_plot(y_true, y_pred, ax=None, *,
                     c=None, cmap=None, norm=None):
    """Scatter plot of data vs prediction with 1:1 line and R².

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Observed values.
    y_pred : array-like, shape (n,)
        Predicted values.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.
    c : array-like, shape (n,), optional
        Per-point values to color the markers by (e.g. absolute error).
        Uses the flat style colour when *None*.
    cmap, norm : optional
        Colormap and normalization applied to *c*.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
    r2 = r_value ** 2

    if c is None:
        ax.scatter(y_true, y_pred, c=POINT_FACE, edgecolors=POINT_EDGE,
                   linewidths=0.6, s=36, alpha=0.85, zorder=3)
    else:
        # No alpha here — a translucent marker mixes with the white
        # background and reads as a lower error than it is.
        ax.scatter(y_true, y_pred, c=c, cmap=cmap, norm=norm,
                   edgecolors=POINT_EDGE, linewidths=0.6, s=ERROR_SIZE,
                   zorder=3)
    lo, hi = y_true.min(), y_true.max()
    ax.plot([lo, hi], [lo, hi], 'k--', zorder=2)
    ax.set_xlabel('Data')
    ax.set_ylabel('Prediction')
    ax.set_title(f'R\u00b2: {r2:.4f}')

    return fig, ax


# ------------------------------------------------------------------
# Residual-vs-predicted plot (any dimensionality)
# ------------------------------------------------------------------

def residual_plot(y_true, y_pred, ax=None, *,
                  c=None, cmap=None, norm=None):
    """Scatter plot of residuals against predictions with a zero line.

    Complements :func:`correlation_plot`: structure in this plot that
    the correlation chart compresses along its 1:1 line becomes
    visible here — a curve means systematic bias (too few nodes or
    over-smoothing), a funnel means the error scales with the output
    level, and outliers stand apart from the cloud.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Observed values.
    y_pred : array-like, shape (n,)
        Predicted values.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.
    c : array-like, shape (n,), optional
        Per-point values to color the markers by (e.g. absolute error).
        Uses the flat style colour when *None*.
    cmap, norm : optional
        Colormap and normalization applied to *c*.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    rmse = np.sqrt(np.mean(residuals ** 2))

    if c is None:
        ax.scatter(y_pred, residuals, c=POINT_FACE, edgecolors=POINT_EDGE,
                   linewidths=0.6, s=36, alpha=0.85, zorder=3)
    else:
        ax.scatter(y_pred, residuals, c=c, cmap=cmap, norm=norm,
                   edgecolors=POINT_EDGE, linewidths=0.6, s=ERROR_SIZE,
                   zorder=3)
    ax.axhline(0.0, color='k', linestyle='--', zorder=2)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Residual (data - prediction)')
    ax.set_title(f'RMSE: {rmse:.4f}')

    return fig, ax


# ------------------------------------------------------------------
# Training convergence plot
# ------------------------------------------------------------------

def convergence_plot(history, ax=None, *, log_scale=True,
                     loss_threshold=None):
    """Training (and validation) loss curve from a training history.

    Parameters
    ----------
    history : array-like, shape (n_steps, 2) or (n_steps, 3), or ebf.EBF
        Training history with columns ``(step, loss)`` — either
        ``EBF.history_``, the fourth return value of
        ``run(..., return_history=True)``, or a fitted ``EBF``
        instance (its ``history_`` attribute is used).  Histories from
        a run with ``val_fraction > 0`` have a third ``val_loss``
        column (NaN except at evaluation steps), plotted as a second
        curve.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.
    log_scale : bool, optional
        Plot the loss on a logarithmic axis.  Default ``True`` — the
        loss typically spans orders of magnitude over a run.
    loss_threshold : float, optional
        Draw the early-stopping threshold as a horizontal reference
        line (pass the same value given to ``fit()`` / ``run()``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    history = getattr(history, 'history_', history)
    if history is None:
        raise ValueError(
            "No training history — fit the model before plotting.")
    history = np.asarray(history)
    if history.ndim != 2 or history.shape[1] not in (2, 3):
        raise ValueError(
            "history must have shape (n_steps, 2) or (n_steps, 3) with "
            "columns (step, loss[, val_loss]).")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    steps = history[:, 0]
    loss = history[:, 1]

    ax.plot(steps, loss, label='Training loss')

    has_val = history.shape[1] == 3 and np.any(np.isfinite(history[:, 2]))
    if has_val:
        val_mask = np.isfinite(history[:, 2])
        ax.plot(steps[val_mask], history[val_mask, 2],
                label='Validation loss')

    if log_scale:
        ax.set_yscale('log')

    if loss_threshold is not None:
        ax.axhline(loss_threshold, color=ACCENT, linestyle='--',
                   label=f'loss_threshold = {loss_threshold:g}')
    if loss_threshold is not None or has_val:
        ax.legend(loc='best', fontsize=8)

    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'Final loss: {loss[-1]:.4f} after {int(steps[-1])} steps')
    ax.minorticks_on()

    return fig, ax


# ------------------------------------------------------------------
# 2-D contour plot
# ------------------------------------------------------------------

def contour_plot_2d(model, X_data, y_data=None, ax=None, *,
                    n_grid=400, mask=True,
                    n_contours=7, n_contourf=31,
                    cmap=DEFAULT_CMAP, alpha=0.9,
                    xlabel=None, ylabel=None, zlabel=None,
                    show_data=True, show_nodes=False,
                    data_color=None, data_cmap=None, data_norm=None,
                    clabel_fmt='$Z=%.2f$'):
    """Filled contour map for a 2-D input EBF model.

    Parameters
    ----------
    model : ebf.EBF
        A fitted EBF model with 2-D input.
    X_data : array-like, shape (n_points, 2)
        Training input points (used for grid bounds and optional mask).
    y_data : array-like, shape (n_points,), optional
        Training output values — only needed when *mask* is ``True``
        (for convex-hull masking via scipy griddata).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.
    n_grid : int, optional
        Grid resolution per axis.  Default ``400``.
    mask : bool, optional
        Mask predictions outside the convex hull of training data.
        Requires *y_data*.  Default ``True``.
    n_contours : int, optional
        Number of labelled contour lines.  Default ``7``.
    n_contourf : int, optional
        Number of filled contour levels.  Default ``31``.
    cmap : str, optional
        Matplotlib colormap.  Default ``'Blues_r'``.  Colormaps from
        cmasher (``'cmr.*'``) are registered and may also be passed.
    alpha : float, optional
        Fill opacity.  Default ``0.9``.
    xlabel, ylabel, zlabel : str, optional
        Axis / colorbar labels.
    show_data : bool, optional
        Overlay training data points.  Default ``True``.
    show_nodes : bool, optional
        Overlay EBF node positions.  Default ``False``.
    data_color : array-like, shape (n_points,), optional
        Per-point values to color the data overlay by (e.g. absolute
        error).  Uses the flat white marker face when *None*.
    data_cmap, data_norm : optional
        Colormap and normalization applied to *data_color*.  No colorbar
        is drawn for it here — the caller owns that (see
        :func:`summary_plot_3d`), since the axes already carry the
        surface colorbar.
    clabel_fmt : str or None, optional
        Format string for contour labels.  ``None`` disables labels.
        Default ``'$Z=%.2f$'``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    import cmasher  # noqa: F401 — registers the 'cmr.*' colormaps

    X_data = np.asarray(X_data)
    if X_data.shape[1] != 2:
        raise ValueError("contour_plot_2d requires 2-D input data.")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # --- Build evaluation grid ---
    mins = X_data.min(axis=0)
    maxes = X_data.max(axis=0)
    x1 = np.linspace(mins[0], maxes[0], n_grid)
    x2 = np.linspace(mins[1], maxes[1], n_grid)
    xx, yy = np.meshgrid(x1, x2)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])

    z_pred = model.predict(grid_pts).reshape(n_grid, n_grid)

    # --- Convex-hull mask ---
    if mask:
        if y_data is None:
            raise ValueError("y_data is required when mask=True.")
        y_data = np.asarray(y_data)
        zmask = griddata(
            (X_data[:, 0], X_data[:, 1]), y_data,
            (xx, yy), method='linear')
        z_pred[~np.isfinite(zmask)] = np.nan

    # --- Filled contours ---
    cs_fill = ax.contourf(xx, yy, z_pred, levels=n_contourf,
                          zorder=1, cmap=cmap, alpha=alpha)
    cs_fine = ax.contour(xx, yy, z_pred, levels=n_contourf,
                         colors=GRID_LINE, linewidths=0.3,
                         zorder=2, alpha=GRID_ALPHA)
    # cs_coarse = ax.contour(xx, yy, z_pred, levels=n_contours,
    #                       colors='black', zorder=2, alpha=1.0)

    # if clabel_fmt is not None:
    #    ax.clabel(cs_coarse, inline=1, fontsize=10, fmt=clabel_fmt)

    # --- Colorbar ---
    cbar = fig.colorbar(cs_fill, ax=ax)
    cbar.add_lines(cs_fine)
    # cbar.add_lines(cs_coarse)
    if zlabel is not None:
        cbar.ax.set_ylabel(zlabel)

    # --- Overlays ---
    # Both must contrast against the sequential blue fill; see the shared
    # style block at the top of this module.
    if show_data:
        if data_color is None:
            ax.scatter(X_data[:, 0], X_data[:, 1], c=SAMPLE_FACE,
                       s=SAMPLE_SIZE, edgecolors=SAMPLE_EDGE,
                       linewidths=0.8, zorder=3, label='data')
        else:
            ax.scatter(X_data[:, 0], X_data[:, 1], c=data_color,
                       cmap=data_cmap, norm=data_norm,
                       s=ERROR_SIZE, edgecolors=SAMPLE_EDGE,
                       linewidths=0.8, zorder=3, label='data')

    if show_nodes:
        nodes = model.get_nodes()
        ax.scatter(nodes[:, 0], nodes[:, 1], c=NODE_FACE,
                   marker=NODE_MARKER, s=NODE_SIZE, edgecolors=NODE_EDGE,
                   linewidths=0.7, zorder=4, label='EBF nodes')

    if show_data or show_nodes:
        ax.legend(loc='best', fontsize=8, framealpha=0.9)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Keep the view on the data region — stray nodes far outside the data
    # would otherwise stretch the axes and shrink the map.
    ax.set_xlim(mins[0], maxes[0])
    ax.set_ylim(mins[1], maxes[1])

    ax.minorticks_on()

    return fig, ax


# ------------------------------------------------------------------
# 3-D data summary figure (contour + correlation + convergence)
# ------------------------------------------------------------------

def summary_plot_3d(model, X_data, y_data, *, figsize=(12, 8),
                    loss_threshold=None,
                    xlabel=None, ylabel=None, zlabel=None,
                    error_color=True, error_cmap=ERROR_CMAP,
                    **contour_kwargs):
    """One-figure fit summary for 3-D data (two inputs, one output).

    The fitted surface (:func:`contour_plot_2d`) is the dominant
    element, filling the full height of the figure on the left; the
    data-vs-prediction plot (:func:`correlation_plot`), the
    residual-vs-predicted plot (:func:`residual_plot`), and the
    training loss curve (:func:`convergence_plot`) are stacked in a
    narrower column on the right.

    By default every data point is shaded by its absolute error on one
    shared red scale, so the same shade means the same error on all
    three data panels and a bad point can be traced from the residual
    plot back to where it sits on the map (``error_color=False``
    restores flat markers).

    Parameters
    ----------
    model : ebf.EBF
        A fitted EBF model with 2-D input and a training history.
    X_data : array-like, shape (n_points, 2)
        Training input points.
    y_data : array-like, shape (n_points,)
        Training output values.
    figsize : tuple, optional
        Figure size in inches.  Default ``(12, 8)``.
    loss_threshold : float, optional
        Early-stopping threshold reference line for the convergence
        panel (pass the same value given to ``fit()``).
    xlabel, ylabel, zlabel : str, optional
        Axis / colorbar labels for the contour panel.
    error_color : bool, optional
        Shade every data point by its absolute error — same values, same
        colormap and same scale on all three data panels, with a single
        shared colorbar down the right-hand edge.  Default ``True``;
        pass ``False`` for flat-filled markers.
    error_cmap : str, optional
        Colormap for the *error_color* shading.  Default ``'Reds'``.
    **contour_kwargs
        Extra keyword arguments forwarded to :func:`contour_plot_2d`
        (e.g. ``n_grid``, ``mask``, ``cmap``, ``show_nodes``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib.axes.Axes, shape (4,)
        ``(contour, correlation, residual, convergence)`` axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    X_data = np.asarray(X_data)
    y_data = np.asarray(y_data)
    if X_data.ndim != 2 or X_data.shape[1] != 2:
        raise ValueError(
            "summary_plot_3d requires 3-D data: X_data with two input "
            "columns plus a 1-D output y_data.")

    y_pred = model.predict(X_data)

    if error_color:
        abs_err = np.abs(y_data - y_pred)
        # A shared norm is what makes the three panels comparable: the
        # same shade means the same error wherever it appears.  Anchor
        # the low end at zero so shade reads as magnitude, not rank.
        err_max = float(abs_err.max())
        err_norm = Normalize(vmin=0.0, vmax=err_max if err_max > 0 else 1.0)
        point_kwargs = dict(c=abs_err, cmap=error_cmap, norm=err_norm)
        data_kwargs = dict(data_color=abs_err, data_cmap=error_cmap,
                           data_norm=err_norm)
    else:
        point_kwargs = {}
        data_kwargs = {}

    fig = plt.figure(figsize=figsize)
    if error_color:
        # A dedicated slim column keeps the shared colorbar clear of the
        # panels; a figure-level colorbar would instead steal width from
        # whichever axes it attached to.
        gs = fig.add_gridspec(3, 3, width_ratios=(2.5, 1, 0.07))
        ax_cbar = fig.add_subplot(gs[:, 2])
    else:
        gs = fig.add_gridspec(3, 2, width_ratios=(2.5, 1))
        ax_cbar = None
    ax_contour = fig.add_subplot(gs[:, 0])
    ax_corr = fig.add_subplot(gs[0, 1])
    ax_resid = fig.add_subplot(gs[1, 1])
    ax_conv = fig.add_subplot(gs[2, 1])

    contour_plot_2d(model, X_data, y_data, ax=ax_contour,
                    xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                    **data_kwargs, **contour_kwargs)

    correlation_plot(y_data, y_pred, ax=ax_corr, **point_kwargs)
    residual_plot(y_data, y_pred, ax=ax_resid, **point_kwargs)

    convergence_plot(model, ax=ax_conv, loss_threshold=loss_threshold)

    if error_color:
        cbar = fig.colorbar(
            ScalarMappable(norm=err_norm, cmap=error_cmap), cax=ax_cbar)
        cbar.ax.set_ylabel('|error| (data - prediction)', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    for ax in (ax_corr, ax_resid, ax_conv):
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
        ax.title.set_size(10)

    fig.tight_layout()

    return fig, np.array([ax_contour, ax_corr, ax_resid, ax_conv])


# ------------------------------------------------------------------
# N-dimensional evaluation grid
# ------------------------------------------------------------------

def eval_grid(model, bounds, n_points=50):
    """Create an n-dimensional rectilinear grid and predict on it.

    Parameters
    ----------
    model : ebf.EBF
        A fitted EBF model.
    bounds : list of (min, max)
        Per-dimension ``(min, max)`` bounds.
    n_points : int or list of int, optional
        Grid resolution.  A single int uses the same resolution for
        every dimension; a list specifies resolution per dimension.
        Default ``50``.

    Returns
    -------
    result : dict
        ``"coords"``      — ``ndarray (n_total, n_dims)`` flat grid points.
        ``"predictions"``  — ``ndarray (n_total,)`` model output.
        ``"grid_shape"``   — ``tuple`` of ints, shape for reshaping.
        ``"axes"``         — ``list`` of 1-D arrays (tick values per dim).
    """
    n_dims = len(bounds)

    if isinstance(n_points, int):
        n_points = [n_points] * n_dims

    axes = [np.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, n_points)]
    grids = np.meshgrid(*axes, indexing='ij')
    grid_shape = grids[0].shape
    coords = np.column_stack([g.ravel() for g in grids])

    predictions = model.predict(coords)

    return {
        "coords": coords,
        "predictions": predictions,
        "grid_shape": grid_shape,
        "axes": axes,
    }


# ------------------------------------------------------------------
# Export grid to file
# ------------------------------------------------------------------

def export_grid(filepath, grid_result, dim_names=None):
    """Save evaluation-grid results to CSV or NPZ.

    The format is chosen by file extension:

    * ``.csv`` — flat table with one column per input dimension plus a
      ``prediction`` column.  Universal format readable by Excel,
      MATLAB, C++, etc.
    * ``.npz`` — NumPy archive containing ``coords``, ``predictions``,
      ``grid_shape``, and per-dimension axis arrays ``axis_0``, …

    Parameters
    ----------
    filepath : str or pathlib.Path
        Output file path.  Extension determines format.
    grid_result : dict
        Output of :func:`eval_grid`.
    dim_names : list of str, optional
        Column names for each input dimension.  Defaults to
        ``["dim_0", "dim_1", …]``.
    """
    from pathlib import Path
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    coords = grid_result["coords"]
    predictions = grid_result["predictions"]
    n_dims = coords.shape[1]

    if dim_names is None:
        dim_names = [f"dim_{i}" for i in range(n_dims)]

    if ext == '.csv':
        header = ','.join(dim_names + ['prediction'])
        table = np.column_stack([coords, predictions])
        np.savetxt(filepath, table, delimiter=',', header=header, comments='')

    elif ext == '.npz':
        save_kwargs = {
            "coords": coords,
            "predictions": predictions,
            "grid_shape": np.array(grid_result["grid_shape"]),
        }
        for i, ax in enumerate(grid_result["axes"]):
            save_kwargs[f"axis_{i}"] = ax
        np.savez(filepath, **save_kwargs)

    else:
        raise ValueError(
            f"Unsupported extension '{ext}'. Use '.csv' or '.npz'.")
