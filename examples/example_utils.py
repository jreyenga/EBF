# -*- coding: utf-8 -*-
"""
Shared helpers for the synthetic-surface examples.

``test_func`` is a 2-D test surface deliberately built to stress a
conventional (isotropic) RBF: a smooth polynomial swell plus a narrow
ridge running diagonally across the domain.  An isotropic kernel has a
single length scale, so it either averages the ridge away or rings
around it — an EBF node can stretch its ellipsoid along the ridge and
capture it with far fewer centers.

Used by ``RBF_vs_EBF.py`` and ``loss_comparison.py`` so those examples
are directly comparable.  The plotting helpers at the bottom
(``contour_panel``, ``draw_ellipses``) are domain-agnostic and are also
used by ``node_ellipsoids.py``, which brings its own surface.
"""
import numpy as np
from matplotlib.patches import Ellipse
from scipy.interpolate import griddata
import cmasher  # noqa: F401 — registers the 'cmr.*' colormaps

# Domain shared by every example built on this surface.
# Note the 10:1 aspect ratio — the data standardization inside EBF
# handles it, scipy's Rbf sees the raw coordinates.
X_MIN, X_MAX = -20.0, 20.0
Y_MIN, Y_MAX = -2.0,   2.0

# Plot style is defined once in ebf.viz and reused here, so the figures
# in the docs match what a user gets from ebf.contour_plot_2d and friends.
from ebf.viz import (DEFAULT_CMAP, SAMPLE_FACE, SAMPLE_EDGE, SAMPLE_SIZE,
                     NODE_FACE, NODE_EDGE, NODE_SIZE, NODE_MARKER,
                     ACCENT, GRID_LINE, GRID_ALPHA)  # noqa: F401


def test_func(x, y):
    """2-D test surface: polynomial swell + narrow diagonal ridge."""
    xn = x / 10.0
    base  = (xn**5 + 3*xn*y + 1.5*y**2 + 5) * (xn + y) + 200 + np.sin(xn * y) * 50
    ridge = 80 * np.exp(-10 * (xn + y)**2)
    return base + ridge


def sample_test_func(n_points, seed=42):
    """Draw scattered samples of ``test_func`` over the domain.

    Returns ``(X, z)`` with X shaped (n_points, 2) and z (n_points,).
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n_points, 2)) * [X_MAX - X_MIN, Y_MAX - Y_MIN] \
        + [X_MIN, Y_MIN]
    z = test_func(X[:, 0], X[:, 1])
    return X, z


def make_grid(n_grid=100):
    """Rectilinear evaluation grid over the standard domain.

    Returns ``(xx, yy, grid_pts)`` — meshgrid arrays shaped
    (n_grid, n_grid) and the flattened (n_grid**2, 2) point list.
    """
    gx = np.linspace(X_MIN, X_MAX, n_grid)
    gy = np.linspace(Y_MIN, Y_MAX, n_grid)
    xx, yy = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, grid_pts


def hull_mask(X_pts, xx, yy):
    """Boolean mask of grid cells OUTSIDE the convex hull of the samples.

    Both RBF and EBF extrapolate freely; masking to the hull keeps the
    comparison to where the fit is actually supported by data.
    """
    ref = griddata((X_pts[:, 0], X_pts[:, 1]), np.zeros(len(X_pts)),
                   (xx, yy), method='linear')
    return ~np.isfinite(ref)


def contour_panel(ax, xx, yy, z, levels, *, z_full=None, cmap=DEFAULT_CMAP,
                  extrap_alpha=0.18, samples=None, nodes=None, title=None,
                  title_size=11, legend=True):
    """One filled-contour comparison panel with shared color levels.

    Parameters
    ----------
    ax           : axes to draw on
    z            : surface masked to the convex hull (NaN outside)
    levels       : shared contour levels — pass the same array to every
                   panel so colors are comparable across the figure
    z_full       : optional unmasked surface, drawn underneath at low
                   alpha to show how the fit extrapolates
    extrap_alpha : opacity of that extrapolation underlay.  Kept low so
                   it reads as "unsupported by data" rather than
                   competing with the masked surface on top
    samples      : optional (n, 2) training points to overlay
    nodes        : optional (n_nodes, 2) EBF node positions to overlay
    title        : optional axes title
    legend       : draw the sample/node legend

    Returns the filled-contour set (for a shared colorbar).
    """
    if z_full is not None:
        ax.contourf(xx, yy, z_full, levels=levels, cmap=cmap,
                    extend='both', alpha=extrap_alpha)
    cf = ax.contourf(xx, yy, z, levels=levels, cmap=cmap, extend='both')
    ax.contour(xx, yy, z, levels=levels, colors=GRID_LINE,
               linewidths=0.3, alpha=GRID_ALPHA)
    if samples is not None:
        # Dark edge and full opacity: the light end of the sequential
        # colormap washes out faint white markers.
        ax.scatter(samples[:, 0], samples[:, 1],
                   c=SAMPLE_FACE, s=SAMPLE_SIZE, edgecolors=SAMPLE_EDGE,
                   linewidths=0.8, zorder=5, label='samples')
    if nodes is not None:
        ax.scatter(nodes[:, 0], nodes[:, 1],
                   c=NODE_FACE, s=NODE_SIZE, marker=NODE_MARKER,
                   edgecolors=NODE_EDGE, linewidths=0.7, zorder=7,
                   label='EBF nodes')
    if legend and (samples is not None or nodes is not None):
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    if title is not None:
        ax.set_title(title, fontsize=title_size)
    return cf


def ellipsoid_axes(A):
    """Slow-decay direction and elongation of each learned ellipsoid.

    For ``A = Q diag(lambda) Q^T`` the iso-distance curve
    ``(x - v)^T A (x - v) = r^2`` has semi-axis ``r / sqrt(lambda_k)``
    along eigenvector ``q_k`` — so the *smallest* eigenvalue gives the
    *longest* axis, i.e. the direction in which the node's influence
    decays most slowly.

    Returns
    -------
    direction : (n_nodes, 2) unit vectors along the long axis
    aspect    : (n_nodes,) ratio of long to short axis.  1 is circular;
                large values mean the node has collapsed to a slab and
                is effectively measuring distance in one direction only.
    """
    eigvals, eigvecs = np.linalg.eigh(A)          # ascending eigenvalues
    eigvals = np.maximum(eigvals, 1e-12)          # guard degeneracy
    direction = eigvecs[..., 0]                   # long axis
    aspect = np.sqrt(eigvals[..., 1] / eigvals[..., 0])
    return direction, aspect


def draw_ellipses(ax, nodes, A, *, r=1.0, color=NODE_FACE, lw=2.0,
                  alpha=0.95, linestyle='-', label=None):
    """Overlay each node's learned ellipsoid as its ``r`` iso-distance curve.

    Draws the set ``(x - v)^T A (x - v) = r^2`` for every node — the
    contour on which that node's basis function has fallen to
    ``phi(r)``.  This is the thing EBF adds over a conventional RBF:
    each node owns a matrix the optimizer may stretch and rotate, rather
    than sharing one spherical length scale.

    Only worth drawing when the ellipsoids stay reasonably round.  With a
    *growing* basis (``multiquadric``, ``matern52``) the learned matrices
    routinely reach aspect ratios in the thousands and each contour
    degenerates into a pair of near-parallel lines spanning the domain —
    unreadable once you have more than a couple of nodes.  Decaying bases
    (``gaussian``, ``inv_quadratic``) keep them legible.  Check with
    :func:`ellipsoid_axes` before plotting.

    Parameters
    ----------
    ax    : axes to draw on — use an equal aspect ratio, or the ellipses
            render sheared
    nodes : (n_nodes, 2) node centers, from ``EBF.get_nodes()``
    A     : (n_nodes, 2, 2) ellipsoid matrices, from
            ``EBF.get_ellipsoids()`` — same units as *nodes*
    r     : iso-distance level to draw
    """
    for k, (center, A_k) in enumerate(zip(nodes, A)):
        eigvals, eigvecs = np.linalg.eigh(A_k)
        eigvals = np.maximum(eigvals, 1e-12)
        semi = r / np.sqrt(eigvals)               # long axis first
        major = eigvecs[:, 0]
        angle = np.degrees(np.arctan2(major[1], major[0]))

        ax.add_patch(Ellipse(
            xy=center, width=2 * semi[0], height=2 * semi[1], angle=angle,
            facecolor='none', edgecolor=color, linewidth=lw, alpha=alpha,
            linestyle=linestyle, zorder=6, clip_on=True,
            label=label if k == 0 else None,
        ))


def ellipsoid_extent(A):
    """Geometric-mean radius of each node's ``r = 1`` ellipsoid.

    ``det(A)^(-1/(2*n_dims))`` — one number per node summarizing how far
    its influence reaches, independent of orientation.  Useful for
    separating compact "feature" nodes from broad background ones.
    """
    n_dims = A.shape[-1]
    return np.linalg.det(A) ** (-1.0 / (2 * n_dims))
