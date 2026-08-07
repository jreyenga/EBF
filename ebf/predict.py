# -*- coding: utf-8 -*-
"""
EBF inference on new points using a saved checkpoint (TF2 — eager).
"""
import warnings

import numpy as np

from ebf.scaling import unscale_output, unscale_nodes
from ebf.io import restore


def _resolve_sidecar(value, name, config):
    """Resolve Scale/Offset from the user argument and the JSON sidecar.

    ``None`` → use the sidecar copy (error if the checkpoint predates
    sidecar storage).  User-supplied values win, with a warning when they
    disagree with the sidecar — a mismatch produces silently wrong
    predictions (ADR-003).
    """
    stored = np.asarray(config[name]) if name in config else None
    if value is None:
        if stored is None:
            raise ValueError(
                f"'{name}' is not stored in the checkpoint sidecar "
                f"(older checkpoint?) — pass {name} explicitly.")
        return stored
    value = np.asarray(value)
    if stored is not None and not np.allclose(value, stored):
        warnings.warn(
            f"User-supplied {name} differs from the value stored in the "
            f"checkpoint sidecar; using the user-supplied value.")
    return value


def run_points(points, Scale=None, Offset=None, file=None):
    """Evaluate the trained EBF model at new input points.

    Parameters
    ----------
    points : (n_points, n_dims) — input points in original (unscaled) space
    Scale  : (n_dims+1,) array or None — 1/std per column.  ``None``
             (default) reads the value stored in the checkpoint's JSON
             sidecar by train.run()
    Offset : (n_dims+1,) array or None — mean per column.  ``None``
             (default) reads the sidecar value
    file   : str — checkpoint file stem returned by train.run() (required)

    Returns
    -------
    Y     : (n_points,) — predicted output in original space
    Nodes : (n_nodes, n_dims) — node positions in original space
    """
    if file is None:
        raise TypeError("run_points() missing required argument: 'file'")

    model, config = restore(file)

    Scale = _resolve_sidecar(Scale, 'Scale', config)
    Offset = _resolve_sidecar(Offset, 'Offset', config)

    points_scaled = (points - Offset[:-1]) * Scale[:-1]

    Y_pred, _, _ = model(points_scaled.astype(np.float32))
    Nodes_scaled = model.Nodes.numpy()

    Y = unscale_output(Y_pred.numpy(), Scale, Offset)
    Nodes = unscale_nodes(Nodes_scaled, Scale, Offset)
    return Y, Nodes
