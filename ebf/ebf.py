# -*- coding: utf-8 -*-
"""
High-level EBF wrapper — fit / predict / get_nodes API.

This module provides the ``EBF`` class, a user-friendly interface that
manages data scaling, model construction, training, and checkpoint I/O
internally.  The lower-level functional API (``ebf.run`` / ``ebf.run_points``)
remains available for backwards compatibility.
"""
import numpy as np

from ebf.model import EBFModel
from ebf.basis_functions import BASIS_FUNCTIONS, DEFAULT_BASIS
from ebf.scaling import (compute_scale_offset, scale_data,
                         unscale_output, unscale_nodes)
from ebf.io import save, restore
from ebf.train import _train, _validate_fit_params


class EBF:
    """Elliptical Basis Function interpolation model.

    Parameters
    ----------
    n_nodes : int
        Number of EBF nodes.
    basis : str, optional
        Basis function name.  See ``ebf.BASIS_FUNCTIONS`` for available
        options.  Default is ``'multiquadric'``.
    eps : float, optional
        Numerical stability offset for basis functions that need it.
        Default is ``1e-8``.

    Examples
    --------
    >>> import numpy as np
    >>> import ebf
    >>> model = ebf.EBF(n_nodes=8)
    >>> X = np.linspace(0, 2 * np.pi, 30).reshape(-1, 1)
    >>> y = np.sin(X).ravel()
    >>> model.fit(X, y, steps=5000)
    >>> y_pred = model.predict(X)
    >>> nodes = model.get_nodes()
    """

    def __init__(self, n_nodes, basis=DEFAULT_BASIS, eps=1e-8):
        if basis not in BASIS_FUNCTIONS:
            raise ValueError(
                f"Unknown basis '{basis}'. "
                f"Choose from: {list(BASIS_FUNCTIONS)}")
        self.n_nodes = n_nodes
        self.basis = basis
        self.eps = eps

        # Set after fit()
        self._model = None
        self._optimizer = None
        self._scale = None
        self._offset = None
        self._is_fitted = False
        self.history_ = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y=None, *, steps=60000, lr=0.01, var_weight=0.2,
            ellipsoid_weight=0.0, loss_type='rmse', huber_delta='auto',
            tukey_c='auto',
            val_fraction=0.0, patience=10,
            verbose=True, loss_threshold=None, seed=None):
        """Train the model on input data.

        Accepts either separate arrays ``X`` and ``y``, or a single
        combined array where the last column is the output variable.

        Parameters
        ----------
        X : array-like, shape (n_points, n_dims) or (n_points, n_dims+1)
            Input features.  If *y* is ``None``, the last column of *X*
            is treated as the output variable.
        y : array-like, shape (n_points,), optional
            Output variable.  Required when *X* contains only input
            features.
        steps : int, optional
            Number of optimizer iterations.  Default is ``60000``.
        lr : float, optional
            Initial learning rate for Adam.  Default is ``0.01``.
        var_weight : float, optional
            Regularization strength for node spread (see ADR-002).
            Default is ``0.2``.
        ellipsoid_weight : float, optional
            Ellipsoid shape penalty strength (see ADR-011).  Penalizes
            the mean squared Frobenius norm of the per-node ellipsoid
            factors L, keeping node influence zones small and round for
            a smoother surface.  Default is ``0.0`` (penalty disabled).
        loss_type : str, optional
            ``'rmse'`` (default), ``'huber'``, or ``'tukey'``.  Huber gives
            outliers linear (reduced) weight; Tukey biweight is
            redescending — residuals beyond the rejection point exert zero
            pull on the surface, so gross outliers are effectively
            discarded.  See ADR-009/013/014.
        huber_delta : ``'auto'`` or float, optional
            Huber loss threshold in scaled data space.  Default is
            ``'auto'`` (ADR-013): the threshold is recalibrated every 100
            steps from the current residual spread (a robust MAD estimate),
            so roughly the largest ~18% of residuals get linear,
            outlier-resistant treatment as the fit tightens.  Pass a float
            to fix the threshold instead.  Only used when
            ``loss_type='huber'``.
        tukey_c : ``'auto'`` or float, optional
            Tukey biweight rejection point in scaled data space.  Default
            is ``'auto'`` (ADR-014, recommended): ``4.685 * sigma`` with
            the same MAD recalibration, which anneals from an effectively
            quadratic start — important because the Tukey loss is
            non-convex and a fixed small ``c`` can reject most points at
            initialization and stall training.  Only used when
            ``loss_type='tukey'``.
        val_fraction : float, optional
            Fraction of points held out as a validation set for early
            stopping (see ADR-012).  Default is ``0.0`` — no split,
            identical to previous behavior.  When > 0, the validation
            loss is evaluated every 100 steps, training stops once it
            has not improved for *patience* consecutive evaluations, and
            the weights from the best-validation step are restored.
            Replaces guessing *steps* on noisy data, where training loss
            keeps falling while the model memorizes noise.  Only
            reliable with ~50+ points — below that the held-out loss is
            too noisy to give a stable stopping signal (a ``UserWarning``
            is issued); prefer regularization (``var_weight``,
            ``ellipsoid_weight``, ``loss_type='huber'``) on small
            datasets.
        patience : int, optional
            Number of consecutive validation evaluations without
            improvement before stopping.  Default is ``10`` (i.e. 1000
            steps).  Only used when ``val_fraction > 0``.
        verbose : bool, optional
            Print training progress every 100 steps.  Default is ``True``.
        loss_threshold : float or None, optional
            Stop early when the training loss drops to or below this
            value.  ``None`` disables.  Default is ``None``.
        seed : int or None, optional
            Random seed for reproducible weight initialization and
            validation split.  ``None`` (default) is non-deterministic.

        Returns
        -------
        self
            The fitted model (allows method chaining).

        Notes
        -----
        After fitting, the per-step training history is available as
        ``self.history_`` — an ``(n_steps_run, 2)`` array with columns
        ``(step, loss)``, useful for convergence plots and tuning
        ``var_weight`` / ``loss_threshold``.  When ``val_fraction > 0``
        a third ``val_loss`` column is added (NaN except at evaluation
        steps).
        """
        _validate_fit_params(loss_type, val_fraction, patience, huber_delta,
                             tukey_c)

        X = np.asarray(X)
        if y is None:
            if X.ndim != 2 or X.shape[1] < 2:
                raise ValueError(
                    "When y is None, X must be a 2-D array with at least "
                    "2 columns (inputs + output).")
            data = X
        else:
            y = np.asarray(y)
            if X.ndim != 2:
                raise ValueError("X must be 2-D, shape (n_points, n_dims).")
            if y.ndim != 1 or y.shape[0] != X.shape[0]:
                raise ValueError(
                    "y must be 1-D with the same number of rows as X.")
            data = np.column_stack([X, y])

        n_dims = data.shape[1] - 1

        # --- Scaling (ADR-003) ---
        Scale, Offset = compute_scale_offset(data)
        if verbose:
            print("Scale:", Scale)
            print("Offset:", Offset)
        data_scaled = scale_data(data, Scale, Offset)

        In = data_scaled[:, :-1].astype(np.float32)   # (n_points, n_dims)
        Out = data_scaled[:, -1].astype(np.float32)    # (n_points,)

        # --- Model ---
        model = EBFModel(n_dims, self.n_nodes, basis=self.basis, eps=self.eps, seed=seed)

        # --- Shared training loop (see ebf/train.py) ---
        optimizer, history = _train(model, In, Out,
                                    steps=steps, lr=lr,
                                    var_weight=var_weight,
                                    ellipsoid_weight=ellipsoid_weight,
                                    loss_type=loss_type,
                                    huber_delta=huber_delta,
                                    tukey_c=tukey_c,
                                    loss_threshold=loss_threshold,
                                    val_fraction=val_fraction,
                                    patience=patience,
                                    seed=seed,
                                    verbose=verbose)

        # --- Store state ---
        self._model = model
        self._optimizer = optimizer
        self._scale = Scale
        self._offset = Offset
        self._is_fitted = True
        self.history_ = history
        return self

    def predict(self, X):
        """Predict output values at new input points.

        Parameters
        ----------
        X : array-like, shape (n_points, n_dims)
            Input points in original (unscaled) space.

        Returns
        -------
        Y : numpy.ndarray, shape (n_points,)
            Predicted output in original space.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float32)
        X_scaled = (X - self._offset[:-1]) * self._scale[:-1]

        Y_pred, _, _ = self._model(X_scaled)
        return unscale_output(Y_pred.numpy(), self._scale, self._offset)

    def get_nodes(self):
        """Return node positions in original (unscaled) space.

        Returns
        -------
        nodes : numpy.ndarray, shape (n_nodes, n_dims)
            Node center positions in the original data space.
        """
        self._check_fitted()
        nodes_scaled = self._model.Nodes.numpy()
        return unscale_nodes(nodes_scaled, self._scale, self._offset)

    def get_ellipsoids(self):
        """Return per-node ellipsoid matrices in original (unscaled) space.

        Each node's influence region is the quadratic form
        ``r_i^2 = (x - v_i)^T A_i (x - v_i)``, where ``v_i`` is the node
        position from :meth:`get_nodes`.  This method returns the ``A_i``
        expressed in the original data units, so the two can be used
        together directly (e.g. to draw iso-distance ellipses on a plot).

        The model trains on standardized data, where
        ``delta_scaled = S * delta`` with ``S = diag(Scale[:-1])``.  Since
        ``r^2`` is invariant, the original-space matrix is ``S A S``.

        Returns
        -------
        A : numpy.ndarray, shape (n_nodes, n_dims, n_dims)
            Symmetric positive-definite ellipsoid matrices.  Small
            eigenvalues correspond to long ellipsoid axes (slow decay in
            that direction), large eigenvalues to short axes.

        Examples
        --------
        Semi-axis lengths and orientation of the ``r = 1`` contour for
        node ``i``, via the eigendecomposition ``A = Q L Q^T``::

            A = model.get_ellipsoids()[i]
            eigvals, eigvecs = np.linalg.eigh(A)
            semi_axes = 1.0 / np.sqrt(eigvals)   # along columns of eigvecs

        See Also
        --------
        get_nodes : the matching node center positions.
        """
        self._check_fitted()
        L = self._model.ellipsoid_factors().numpy()   # (n_nodes, n_dims, n_dims)
        n_dims = L.shape[-1]
        # Matches NonEuclidDistance: A = L L^T + eps*I (ADR-001)
        A_scaled = L @ np.transpose(L, (0, 2, 1)) + 1e-6 * np.eye(n_dims)

        S = np.diag(self._scale[:-1])
        return S @ A_scaled @ S

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path, filename='ebf-model'):
        """Save model to a checkpoint directory.

        Parameters
        ----------
        path : str
            Directory for checkpoint files.
        filename : str, optional
            Checkpoint filename stem.  Default is ``'ebf-model'``.

        Returns
        -------
        file : str
            Checkpoint file stem (pass to ``EBF.load()``).
        """
        self._check_fitted()
        return save(self._model, self._optimizer, path, filename,
                    scale=self._scale, offset=self._offset)

    @classmethod
    def load(cls, file):
        """Restore an EBF model from a checkpoint.

        Parameters
        ----------
        file : str
            Checkpoint file stem as returned by ``save()``.

        Returns
        -------
        model : EBF
            A fitted ``EBF`` instance ready for ``predict()`` and
            ``get_nodes()``.
        """
        model_tf, config = restore(file)
        obj = cls(
            n_nodes=config['n_nodes'],
            basis=config['basis'],
            eps=config['eps'])
        obj._model = model_tf
        obj._scale = np.array(config['Scale'])
        obj._offset = np.array(config['Offset'])
        obj._is_fitted = True
        return obj

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "Model has not been fitted yet. Call .fit() first.")
