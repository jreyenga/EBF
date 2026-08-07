# -*- coding: utf-8 -*-
"""
Data standardization utilities.

Convention: data array is (n_points, n_dims + 1) — last column is output.
Scale and Offset arrays match data columns (length n_dims + 1).
"""
import numpy as np


def compute_scale_offset(data):
    """Compute standardization parameters for zero-mean unit-variance scaling.

    Parameters
    ----------
    data : numpy.ndarray, shape (n_points, n_dims+1)
        Raw data array (last column is the output variable).

    Returns
    -------
    Scale : numpy.ndarray, shape (n_dims+1,)
        ``1 / std`` per column.
    Offset : numpy.ndarray, shape (n_dims+1,)
        Column means.
    """
    Scale = 1.0 / np.std(data, axis=0)
    Offset = np.mean(data, axis=0)
    return Scale, Offset


def scale_data(data, Scale, Offset):
    """Standardize data: ``(data - Offset) * Scale``.

    Parameters
    ----------
    data : numpy.ndarray, shape (n_points, n_dims+1)
        Raw data array.
    Scale : numpy.ndarray, shape (n_dims+1,)
        Scale factors from :func:`compute_scale_offset`.
    Offset : numpy.ndarray, shape (n_dims+1,)
        Offsets from :func:`compute_scale_offset`.

    Returns
    -------
    data_scaled : numpy.ndarray, shape (n_points, n_dims+1)
        Standardized data with zero mean and unit variance per column.
    """
    return (data - Offset) * Scale


def unscale_output(y_scaled, Scale, Offset):
    """Reverse standardization for the output column (last index).

    Parameters
    ----------
    y_scaled : numpy.ndarray, shape (n_points,)
        Predicted output in scaled space.
    Scale : numpy.ndarray, shape (n_dims+1,)
        Scale factors from :func:`compute_scale_offset`.
    Offset : numpy.ndarray, shape (n_dims+1,)
        Offsets from :func:`compute_scale_offset`.

    Returns
    -------
    y : numpy.ndarray, shape (n_points,)
        Predicted output in original space.
    """
    return y_scaled / Scale[-1] + Offset[-1]


def unscale_nodes(nodes_scaled, Scale, Offset):
    """Reverse standardization for node positions (input columns only).

    Parameters
    ----------
    nodes_scaled : numpy.ndarray, shape (n_nodes, n_dims)
        Node positions in scaled space.
    Scale : numpy.ndarray, shape (n_dims+1,)
        Scale factors from :func:`compute_scale_offset`.
    Offset : numpy.ndarray, shape (n_dims+1,)
        Offsets from :func:`compute_scale_offset`.

    Returns
    -------
    nodes : numpy.ndarray, shape (n_nodes, n_dims)
        Node positions in original data space.
    """
    return nodes_scaled / Scale[:-1] + Offset[:-1]
