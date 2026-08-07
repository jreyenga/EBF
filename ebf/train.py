# -*- coding: utf-8 -*-
"""
EBF training loop (TF2 — GradientTape).

_train() is the single shared training loop; both the functional API
(run(), below) and the class API (EBF.fit()) call it, so the loss
definition and optimizer setup exist in exactly one place.

run() trains the model on input data and saves a checkpoint.
"""
import warnings

import tensorflow as tf
import numpy as np

from ebf.model import EBFModel
from ebf.scaling import compute_scale_offset, scale_data
from ebf.io import save
from ebf.basis_functions import DEFAULT_BASIS

# Steps between validation-loss evaluations when val_fraction > 0 (ADR-012).
# One patience unit = VAL_EVERY optimizer steps.
VAL_EVERY = 100

# Adaptive robust-loss threshold (ADR-013/ADR-014): steps between
# recalibrations from the current residuals when the threshold is 'auto'.
DELTA_EVERY = 100

# threshold = K * sigma_hat, with sigma_hat = MAD_TO_SIGMA * MAD(residuals).
# The MAD (median absolute deviation) is a residual-scale estimate that
# outliers cannot corrupt; 1.4826 converts it to a Gaussian-equivalent
# sigma.  HUBER_K = 1.345 is the classical tuning constant that retains 95%
# efficiency on Gaussian noise (~18% of points in the linear zone);
# TUKEY_K = 4.685 is its biweight counterpart (also 95% efficiency, with
# influence redescending to zero at the threshold).  The floor keeps the
# threshold from collapsing to 0 on noise-free data, where Huber would
# degenerate to pure L1 and Tukey would reject everything.
HUBER_K = 1.345
TUKEY_K = 4.685
MAD_TO_SIGMA = 1.4826
THRESHOLD_FLOOR = 1e-3


def _validate_threshold(name, value):
    if isinstance(value, str):
        if value != 'auto':
            raise ValueError(
                f"{name} must be 'auto' or a positive number, got '{value}'")
    elif not value > 0:
        raise ValueError(
            f"{name} must be 'auto' or a positive number, got {value}")


def _validate_fit_params(loss_type, val_fraction=0.0, patience=10,
                         huber_delta='auto', tukey_c='auto'):
    """Shared hyperparameter validation for run() and EBF.fit()."""
    if loss_type not in ('huber', 'rmse', 'tukey'):
        raise ValueError(
            f"loss_type must be 'rmse', 'huber', or 'tukey', got '{loss_type}'")
    _validate_threshold('huber_delta', huber_delta)
    _validate_threshold('tukey_c', tukey_c)
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")


def _train(model, In, Out, *, steps, lr, var_weight,
           ellipsoid_weight=0.0,
           loss_type='rmse', huber_delta='auto', tukey_c='auto',
           loss_threshold=None,
           val_fraction=0.0, patience=10, seed=None,
           verbose=True):
    """Shared GradientTape training loop (full-batch, per ADR-005).

    Operates on **scaled** data — callers are responsible for
    standardization (ADR-003) and for saving Scale/Offset.

    Parameters
    ----------
    model          : EBFModel — freshly constructed model to train in place
    In             : (n_points, n_dims) float32 array — scaled inputs
    Out            : (n_points,) float32 array — scaled outputs
    steps          : int — number of optimizer steps
    lr             : float — initial learning rate
    var_weight     : float — node spread regularization strength (ADR-002)
    ellipsoid_weight : float — ellipsoid shape penalty strength (ADR-011);
                     ``0.0`` disables the penalty
    loss_type      : str — ``'rmse'``, ``'huber'`` (ADR-009/013), or
                     ``'tukey'`` (ADR-014, redescending — rejects outliers)
    huber_delta    : ``'auto'`` or float — Huber threshold in scaled data
                     space.  ``'auto'`` (default, ADR-013) recalibrates the
                     threshold every ``DELTA_EVERY`` steps from the current
                     residuals (``1.345 * 1.4826 * MAD``), so it tracks the
                     noise floor as the fit tightens; a float fixes it
    tukey_c        : ``'auto'`` or float — Tukey biweight rejection point in
                     scaled data space (ADR-014).  ``'auto'`` (default) uses
                     ``4.685 * 1.4826 * MAD`` with the same recalibration
                     cadence; a float fixes it (not recommended — 'auto'
                     anneals from an effectively quadratic start, a fixed
                     small c can reject most points at initialization and
                     stall training)
    loss_threshold : float or None — stop early when the training loss
                     drops to or below this value; ``None`` disables
    val_fraction   : float — fraction of points held out for validation
                     (ADR-012).  ``0.0`` (default) disables the split and
                     reproduces the previous behavior exactly
    patience       : int — stop when the validation loss has not improved
                     for this many consecutive evaluations (one evaluation
                     every ``VAL_EVERY`` steps); only used when
                     *val_fraction* > 0
    seed           : int or None — seeds the validation split permutation;
                     only used when *val_fraction* > 0
    verbose        : bool — print progress every 100 steps

    Returns
    -------
    optimizer : tf.keras.optimizers.Adam
        The optimizer, for checkpointing alongside the model.
    history : numpy.ndarray, shape (n_steps_run, 2) or (n_steps_run, 3)
        Per-step training history — columns are ``(step, loss)``, plus a
        third ``val_loss`` column when *val_fraction* > 0 (NaN except at
        evaluation steps).  ``n_steps_run`` may be less than *steps* when
        *loss_threshold* or validation patience triggers early stopping.

    Notes
    -----
    The validation loss is the **data-fit term only** (RMSE or Huber on
    the held-out points) — the ADR-002/ADR-011 regularizers do not measure
    generalization, so including them would confound the stopping signal.
    When the split is active, the weights from the best-validation step are
    restored at the end of training (whether or not patience triggered).
    """
    # --- Optional validation split (SMOOTHNESS S2, ADR-012) ---
    use_val = val_fraction > 0.0
    if use_val:
        n_points = In.shape[0]
        if n_points < 50:
            warnings.warn(
                f"val_fraction > 0 with only {n_points} points: validation "
                "splitting is unreliable below ~50 points — the held-out "
                "loss is too noisy to give a stable stopping signal. "
                "Prefer regularization (var_weight, ellipsoid_weight, "
                "loss_type='huber') on small datasets.")
        n_val = max(1, int(round(val_fraction * n_points)))
        if n_points - n_val < 2:
            raise ValueError(
                f"val_fraction={val_fraction} leaves {n_points - n_val} of "
                f"{n_points} points for training — need at least 2.")
        perm = np.random.default_rng(seed).permutation(n_points)
        X_val_tensor = tf.constant(In[perm[:n_val]])
        Y_val_tensor = tf.constant(Out[perm[:n_val]])
        In, Out = In[perm[n_val:]], Out[perm[n_val:]]

    # --- Optimizer with exponential LR decay (ADR-005) ---
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # --- Convert to tensors once (full-batch, per ADR-005) ---
    X_tensor = tf.constant(In)
    Y_tensor = tf.constant(Out)
    var_weight_t = tf.constant(var_weight, dtype=tf.float32)
    ellipsoid_weight_t = tf.constant(ellipsoid_weight, dtype=tf.float32)

    # --- Robust-loss threshold (ADR-009 / ADR-013 / ADR-014) ---
    # 'auto' uses a tf.Variable so the threshold can be reassigned between
    # steps without retracing the compiled train_step graph.  The threshold
    # is calibrated against the *residual* scale, not the data scale — a
    # fixed data-scale threshold sits far above converged residuals, and
    # the robust loss silently degenerates to a pure quadratic with no
    # outlier resistance.
    robust = loss_type in ('huber', 'tukey')
    if robust:
        thresh_param = huber_delta if loss_type == 'huber' else tukey_c
        thresh_k = HUBER_K if loss_type == 'huber' else TUKEY_K
        adaptive_thresh = isinstance(thresh_param, str)
        if adaptive_thresh:
            thresh_t = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        else:
            thresh_t = tf.constant(thresh_param, dtype=tf.float32)
    else:
        adaptive_thresh = False

    def refresh_thresh():
        # Recalibrate the threshold from the current training residuals
        # (ADR-013).  MAD is used instead of std so outliers cannot inflate
        # their own threshold; training rows only, so the validation loss
        # stays a clean held-out measurement.
        Y_pred, _dist_nodes, _dist = model(X_tensor)
        r = Out - Y_pred.numpy()
        mad = np.median(np.abs(r - np.median(r)))
        thresh = max(thresh_k * MAD_TO_SIGMA * mad, THRESHOLD_FLOOR)
        thresh_t.assign(thresh)
        return thresh

    def data_loss(y_true, y_pred):
        # Data-fit term (ADR-009/013/014) — shared by train_step and
        # val_step.  Both robust losses use sqrt(2 * mean(rho)): a
        # "pseudo-RMSE" that equals sqrt(MSE) exactly (Huber) or to first
        # order (Tukey) when residuals sit inside the threshold, so
        # loss_threshold and the regularizer weights keep their meaning
        # across loss types.
        if loss_type == 'huber':
            residuals = y_true - y_pred
            abs_r = tf.abs(residuals)
            huber = tf.where(
                abs_r <= thresh_t,
                0.5 * tf.square(residuals),
                thresh_t * (abs_r - 0.5 * thresh_t))
            return tf.sqrt(2.0 * tf.reduce_mean(huber))
        if loss_type == 'tukey':
            # Tukey biweight (ADR-014): quadratic core, influence
            # redescends to exactly zero at |r| = c — points beyond c
            # exert no pull on the surface.  The branchless max(0, .)
            # form keeps the gradient exact (zero beyond c).
            residuals = y_true - y_pred
            u = tf.maximum(0.0, 1.0 - tf.square(residuals / thresh_t))
            rho = (tf.square(thresh_t) / 6.0) * (1.0 - u * u * u)
            return tf.sqrt(2.0 * tf.reduce_mean(rho))
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        return tf.sqrt(mse)

    # --- Compiled training step ---
    # @tf.function traces the forward/backward pass into a single graph so
    # TensorFlow can parallelise ops across CPU cores — without it, eager
    # mode dispatches each op individually with Python overhead between them,
    # losing the multi-core throughput that TF1 sess.run() had automatically.
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            Y_pred, dist_nodes, _dist = model(X_tensor)

            # Loss: data term + node-spread regularization (ADR-002)
            _mean_dist, var_dist = tf.nn.moments(dist_nodes, axes=[0, 1])

            loss = data_loss(Y_tensor, Y_pred) + var_weight_t / var_dist

            # Ellipsoid shape penalty (ADR-011) — mean squared Frobenius
            # norm of the per-node factors L; the branch is resolved at
            # trace time, so ellipsoid_weight=0 adds nothing to the graph
            if ellipsoid_weight > 0.0:
                L = model.ellipsoid_factors()
                frob = tf.reduce_mean(tf.reduce_sum(tf.square(L), axis=[1, 2]))
                loss = loss + ellipsoid_weight_t * frob

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    if use_val:
        @tf.function
        def val_step():
            Y_pred, _dist_nodes, _dist = model(X_val_tensor)
            return data_loss(Y_val_tensor, Y_pred)

    display_step = 100
    history = []

    best_val = np.inf
    best_step = 0
    best_weights = None
    evals_no_improve = 0

    # Initial threshold from the untrained model's residuals — these are
    # O(1) in scaled space, so training starts effectively quadratic (fast
    # convergence, no points rejected) and the threshold shrinks with the
    # residuals as the fit tightens.  For Tukey this annealing is what
    # keeps the non-convex loss out of bad basins (ADR-014).
    thresh_val = refresh_thresh() if adaptive_thresh else None
    thresh_label = 'delta' if loss_type == 'huber' else 'c'

    for step in range(1, steps + 1):
        loss = train_step()
        loss_val = loss.numpy()

        if adaptive_thresh and step % DELTA_EVERY == 0:
            thresh_val = refresh_thresh()

        val_loss = None
        if use_val and (step % VAL_EVERY == 0 or step == steps):
            val_loss = val_step().numpy()

        if use_val:
            history.append((step, loss_val,
                            np.nan if val_loss is None else val_loss))
        else:
            history.append((step, loss_val))

        if verbose and (step % display_step == 0 or step == 1):
            msg = f"Step {step}, Loss= {loss_val:.4f}"
            if thresh_val is not None:
                msg += f", {thresh_label}= {thresh_val:.4f}"
            if val_loss is not None:
                msg += f", Val= {val_loss:.4f}"
            print(msg)

        # Patience-based early stopping on validation loss (ADR-012)
        if val_loss is not None:
            if val_loss < best_val:
                best_val, best_step = val_loss, step
                best_weights = [v.numpy() for v in model.trainable_variables]
                evals_no_improve = 0
            else:
                evals_no_improve += 1
                if evals_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at step {step}: no validation "
                              f"improvement in {patience} evaluations "
                              f"(best {best_val:.4f} at step {best_step})")
                    break

        if loss_threshold is not None and loss_val <= loss_threshold:
            if verbose:
                print(f"Converged at step {step}: Loss {loss_val:.4f} <= threshold {loss_threshold}")
            break

    # Roll back to the best-validation weights (ADR-012) — the steps past
    # best_step were, by the validation signal, fitting noise
    if use_val and best_weights is not None:
        for var, weights in zip(model.trainable_variables, best_weights):
            var.assign(weights)
        if verbose and best_step != step:
            print(f"Restored best weights from step {best_step} "
                  f"(val loss {best_val:.4f})")

    if verbose:
        print("Optimization Finished!")

    return optimizer, np.asarray(history, dtype=np.float32)


def run(data, n_nodes,
        basis=DEFAULT_BASIS,
        eps=1e-8,
        var_weight=0.2,
        ellipsoid_weight=0.0,
        loss_type='rmse',
        huber_delta='auto',
        tukey_c='auto',
        path="./",
        filename="my-model",
        train_steps=60000,
        start=0.01,
        loss_threshold=None,
        seed=None,
        verbose=True,
        return_history=False,
        val_fraction=0.0,
        patience=10):
    """Train an EBF model and save a checkpoint.

    Parameters
    ----------
    data            : (n_points, n_dims+1) array — last column is the output variable
    n_nodes         : int — number of EBF nodes
    basis           : str — basis function name (default: 'multiquadric')
    eps             : float — numerical stability offset for basis functions (default: 1e-8)
    var_weight      : float — regularization strength for node spread (default: 0.2)
    ellipsoid_weight : float — ellipsoid shape penalty strength; penalizes the mean
                      squared Frobenius norm of the per-node ellipsoid factors L,
                      keeping node influence zones small and round for a smoother
                      surface.  ``0.0`` (default) disables the penalty; see ADR-011
    loss_type       : str — ``'rmse'`` (default), ``'huber'`` (robust — outliers
                      get linear treatment; ADR-009/013), or ``'tukey'``
                      (redescending — outliers beyond the rejection point exert
                      zero pull; ADR-014)
    huber_delta     : ``'auto'`` or float — Huber threshold in scaled data space.
                      ``'auto'`` (default) recalibrates the threshold every 100
                      steps from the current residual spread so roughly the
                      largest ~18% of residuals get linear (outlier-resistant)
                      treatment; a float fixes the threshold instead.  See ADR-013
    tukey_c         : ``'auto'`` or float — Tukey biweight rejection point in
                      scaled data space; residuals beyond it are ignored entirely.
                      ``'auto'`` (default, recommended) tracks the residual noise
                      floor at ``4.685 * sigma``, annealing from an effectively
                      quadratic start.  See ADR-014
    path            : str — directory for checkpoint files (default: './')
    filename        : str — checkpoint filename stem (default: 'my-model')
    train_steps     : int — number of optimizer steps (default: 60000)
    start           : float — initial learning rate (default: 0.01)
    loss_threshold  : float or None — stop early when the training loss drops to
                      or below this value.  ``None`` disables (default: None)
    seed            : int or None — random seed for reproducible weight
                      initialization and validation split.  ``None`` (default)
                      is non-deterministic
    verbose         : bool — print scaling info and training progress
                      (default: True)
    return_history  : bool — when True, also return the per-step training
                      history as a fourth value (default: False)
    val_fraction    : float — fraction of points held out as a validation set
                      for early stopping (default: 0.0 = disabled, identical to
                      previous behavior).  When > 0, the validation loss is
                      evaluated every 100 steps, training stops once it has not
                      improved for *patience* consecutive evaluations, and the
                      weights from the best-validation step are restored.  Only
                      reliable with ~50+ points — below that the held-out loss
                      is too noisy to give a stable stopping signal (a
                      ``UserWarning`` is issued); prefer regularization on
                      small datasets.  See ADR-012
    patience        : int — number of consecutive validation evaluations without
                      improvement before stopping (default: 10, i.e. 1000 steps).
                      Only used when ``val_fraction > 0``

    Returns
    -------
    Scale   : (n_dims+1,) — 1/std per column
    Offset  : (n_dims+1,) — mean per column
    file    : str — checkpoint file stem for use with predict.run_points()
    history : (n_steps_run, 2) ndarray — ``(step, loss)`` per step; only
              returned when ``return_history=True``.  When ``val_fraction > 0``
              a third ``val_loss`` column is added (NaN except at evaluation
              steps)
    """
    _validate_fit_params(loss_type, val_fraction, patience, huber_delta,
                         tukey_c)

    Scale, Offset = compute_scale_offset(data)
    if verbose:
        print("Scale:", Scale)
        print("Offset:", Offset)
    data_scaled = scale_data(data, Scale, Offset)

    In = data_scaled[:, :-1].astype(np.float32)   # (n_points, n_dims)
    Out = data_scaled[:, -1].astype(np.float32)    # (n_points,)
    n_dims = data.shape[-1] - 1

    model = EBFModel(n_dims, n_nodes, basis=basis, eps=eps, seed=seed)

    optimizer, history = _train(model, In, Out,
                                steps=train_steps, lr=start,
                                var_weight=var_weight,
                                ellipsoid_weight=ellipsoid_weight,
                                loss_type=loss_type,
                                huber_delta=huber_delta,
                                tukey_c=tukey_c,
                                loss_threshold=loss_threshold,
                                val_fraction=val_fraction, patience=patience,
                                seed=seed, verbose=verbose)

    file = save(model, optimizer, path, filename, scale=Scale, offset=Offset)

    if return_history:
        return Scale, Offset, file, history
    return Scale, Offset, file
