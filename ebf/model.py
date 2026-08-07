# -*- coding: utf-8 -*-
"""
Core EBF model.

All public functions operate on scaled (standardized) data.

Tensor shape convention (see CONVENTIONS.md):
  (n_points, n_nodes, n_dims)
"""
import tensorflow as tf
from ebf.basis_functions import BASIS_FUNCTIONS, DEFAULT_BASIS


def DeltaAll(x1, x2):
    """Pairwise delta vectors between two sets of points.

    Parameters
    ----------
    x1 : tf.Tensor, shape (n_points, n_dims)
        First set of points.
    x2 : tf.Tensor, shape (n_nodes, n_dims)
        Second set of points (typically node positions).

    Returns
    -------
    deltas : tf.Tensor, shape (n_points, n_nodes, n_dims)
        ``x1[i] - x2[j]`` for every ``(i, j)`` pair.
    """
    x1 = tf.transpose(x1)
    x2 = tf.transpose(x2)
    xIn1 = tf.expand_dims(x1, -2)
    xIn2 = tf.expand_dims(x2, -1)
    r = tf.transpose(xIn1 - xIn2)
    return r


def NonEuclidDistance(deltas, W, D):
    """Generalized (non-Euclidean) squared distance via per-node ellipsoid matrix.

    Constructs a positive-definite matrix ``A = L L^T + eps*I`` from the
    upper-triangular part of *W* (see ADR-001) and computes
    ``delta^T A delta`` for each point-node pair.

    Parameters
    ----------
    deltas : tf.Tensor, shape (n_points, n_nodes, n_dims)
        Point-to-node difference vectors.
    W : tf.Tensor, shape (n_nodes, n_dims, n_dims)
        Raw ellipsoid weight matrix (upper-triangular part is used).
    D : int
        Number of input dimensions.

    Returns
    -------
    r2 : tf.Tensor, shape (n_points, n_nodes)
        Non-negative squared distances.
    """
    eps = 1e-6  # stabilization for positive-definite matrix — see ADR-001
    L = tf.linalg.band_part(W, 0, -1)          # upper triangular, shape (n_nodes, D, D)
    A = tf.linalg.matmul(L, L, transpose_b=True) + eps * tf.eye(D)  # PD matrix

    deltas_exp = tf.expand_dims(deltas, axis=-1)  # (n_points, n_nodes, D, 1)
    Rsquare = tf.squeeze(
        tf.linalg.matmul(
            tf.linalg.matmul(deltas_exp, A, transpose_a=True),
            deltas_exp
        ),
        axis=[-1, -2]
    )  # (n_points, n_nodes)
    return tf.math.abs(Rsquare)


def LinearBias(x, b1, b2):
    """Linear + constant bias term (see ADR-004).

    Parameters
    ----------
    x : tf.Tensor, shape (n_points, n_dims)
        Input points.
    b1 : tf.Variable, shape (n_dims,)
        Linear weights.
    b2 : tf.Variable, shape (1,)
        Constant bias.

    Returns
    -------
    out : tf.Tensor, shape (n_points,)
        ``sum(x * b1, axis=1) + b2``.
    """
    return tf.reduce_sum(tf.multiply(x, b1), axis=1) + b2


class EBFModel(tf.Module):
    """EBF model as a tf.Module for native TF2 eager execution.

    All trainable variables are instance attributes, enabling
    ``tf.train.Checkpoint`` to discover them automatically.

    This is the low-level model that operates on **scaled** data.  For a
    high-level interface that handles scaling automatically, see
    :class:`ebf.EBF`.

    Parameters
    ----------
    n_dims : int
        Number of input dimensions.
    n_nodes : int
        Number of EBF nodes.
    basis : str, optional
        Basis function name (see ``ebf.BASIS_FUNCTIONS``).
    eps : float, optional
        Numerical stability offset for basis functions.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from ebf.model import EBFModel
    >>> model = EBFModel(n_dims=2, n_nodes=5)
    >>> X = tf.constant([[0.0, 0.0], [1.0, 1.0]])
    >>> Y, dist_nodes, dist = model(X)
    >>> Y.shape
    TensorShape([2])
    """

    def __init__(self, n_dims, n_nodes, basis=DEFAULT_BASIS, eps=1e-8, seed=None, name=None):
        super().__init__(name=name)
        if basis not in BASIS_FUNCTIONS:
            raise ValueError(f"Unknown basis '{basis}'. Choose from: {list(BASIS_FUNCTIONS)}")

        self.n_dims = n_dims
        self.n_nodes = n_nodes
        self.basis = basis
        self.eps = eps
        self._fn, self._n_params = BASIS_FUNCTIONS[basis]

        if seed is not None:
            tf.random.set_seed(seed)

        # --- Trainable variables ---
        self.Nodes = tf.Variable(
            tf.random.normal([n_nodes, n_dims], stddev=0.5, mean=0.0),
            name='Nodes')
        self.EllipsoidWeights = tf.Variable(
            tf.random.normal([n_nodes, n_dims, n_dims], stddev=0.1, mean=0.0),
            name='EWeights')
        self.b1 = tf.Variable(
            tf.random.normal([n_dims], stddev=0.001, mean=0.0),
            name='B1')
        self.b2 = tf.Variable(
            tf.random.normal([1], stddev=0.001, mean=0.0),
            name='B2')
        self.a1 = tf.Variable(
            tf.random.normal([n_nodes], stddev=0.1, mean=0.0),
            name='A1')
        if self._n_params >= 2:
            self.a2 = tf.Variable(
                tf.random.normal([n_nodes], stddev=0.01, mean=0.0),
                name='A2')
        if self._n_params >= 3:
            self.a3 = tf.Variable(
                tf.random.normal([n_nodes], stddev=0.001, mean=0.0),
                name='A3')

    def ellipsoid_factors(self):
        """Upper-triangular factors L of the ellipsoid matrices (ADR-001).

        Returns
        -------
        L : tf.Tensor, shape (n_nodes, n_dims, n_dims)
            The factors from which each node's ellipsoid matrix
            ``A = L L^T + eps*I`` is built.  Used by the training loop
            for the ellipsoid shape penalty (ADR-011).
        """
        return tf.linalg.band_part(self.EllipsoidWeights, 0, -1)

    def __call__(self, X):
        """Forward pass.

        X : tf.Tensor or ndarray, shape (n_points, n_dims)

        Returns
        -------
        Y          : (n_points,) — predicted output (scaled)
        dist_nodes : (n_nodes, n_nodes) — pairwise node distances (for regularization)
        dist       : (n_points, n_nodes) — point-to-node distances
        """
        X = tf.cast(X, tf.float32)

        deltas = DeltaAll(X, self.Nodes)                                        # (n_points, n_nodes, n_dims)
        dist = NonEuclidDistance(deltas, self.EllipsoidWeights, self.n_dims)     # (n_points, n_nodes)

        if self._n_params == 1:
            Y1 = self._fn(dist, self.a1, self.eps)
        elif self._n_params == 2:
            Y1 = self._fn(dist, self.a1, self.a2, self.eps)
        else:
            Y1 = self._fn(dist, self.a1, self.a2, self.a3, self.eps)

        Y2 = LinearBias(X, self.b1, self.b2)
        Y = Y1 + Y2

        # Node-to-node distances (used for spread regularization — see ADR-002)
        delta_nodes = DeltaAll(self.Nodes, self.Nodes)
        dist_nodes = NonEuclidDistance(delta_nodes, self.EllipsoidWeights, self.n_dims)

        return Y, dist_nodes, dist
