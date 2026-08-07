# -*- coding: utf-8 -*-
"""
Basis function registry for EBF.

``BASIS_FUNCTIONS`` is a dict mapping basis function names to
``(callable, n_params)`` tuples.

callable signature
    ``(r2, a1, [a2, [a3,]] eps) -> tf.Tensor`` of shape ``(n_points,)``

Parameters consumed by every callable:

r2  : squared non-Euclidean distance, shape ``(n_points, n_nodes)``
a1  : per-node weight tensor, shape ``(n_nodes,)``
a2  : per-node weight tensor, shape ``(n_nodes,)`` — only for ``n_params >= 2``
a3  : per-node weight tensor, shape ``(n_nodes,)`` — only for ``n_params == 3``
eps : small float for numerical stability (user-configurable, default ``1e-8``)

All current functions use ``n_params=1``.  The registry tuple format and the
``n_params`` branching in ``EBFModel`` are retained to support future
multi-parameter basis functions without structural changes.

``DEFAULT_BASIS`` is ``'multiquadric'`` (see ADR-007).
"""
import tensorflow as tf


BASIS_FUNCTIONS = {
    # name               (function,                                                                                             n_params)
    'linear':            (lambda r2, a1, eps: tf.reduce_sum(a1 * r2**0.5, axis=1),                                             1),
    'quadratic':         (lambda r2, a1, eps: tf.reduce_sum(a1 * r2, axis=1),                                                  1),
    'thin_plate':        (lambda r2, a1, eps: tf.reduce_sum(tf.math.xlogy(r2, r2) * a1, axis=1),                               1),
    'multiquadric':      (lambda r2, a1, eps: tf.reduce_sum(a1 * ((r2 + 1)**0.5 - 1), axis=1),                                 1),
    'inv_multiquadric':  (lambda r2, a1, eps: tf.reduce_sum(a1 / (r2 + 1)**0.5, axis=1),                                       1),
    'inv_quadratic':     (lambda r2, a1, eps: tf.reduce_sum(a1 / (1 + r2), axis=1),                                            1),
    'gaussian':          (lambda r2, a1, eps: tf.reduce_sum(a1 * tf.math.exp(-r2), axis=1),                                    1),
    'matern32':          (lambda r2, a1, eps: tf.reduce_sum(a1 * (1 + 3**0.5 * r2**0.5) * tf.math.exp(-3**0.5 * r2**0.5), axis=1), 1),
    'matern52':          (lambda r2, a1, eps: tf.reduce_sum(a1 * (1 + 5**0.5 * r2**0.5 + 5/3 * r2) * tf.math.exp(-5**0.5 * r2**0.5), axis=1), 1),
    'cosh':              (lambda r2, a1, eps: tf.reduce_sum(a1 * tf.math.cosh((r2 + eps)**0.5), axis=1),                       1),
    'cubic':             (lambda r2, a1, eps: tf.reduce_sum(a1 * r2**1.5, axis=1),                                             1),
    'inv_cosh':          (lambda r2, a1, eps: tf.reduce_sum(a1 / tf.math.cosh((r2 + eps)**0.5), axis=1),                       1),
}

DEFAULT_BASIS = 'multiquadric'
