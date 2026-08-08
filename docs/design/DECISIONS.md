# Architecture Decision Records (ADRs)

Consult this file before changing any architectural choice. If a decision needs to be
revised, update the existing entry rather than adding a conflicting one, and log the change
in `SESSION_LOG.md`.

---

## ADR-001: Positive-Definite Matrix via LLᵀ Factorization

**Decision:** The per-node ellipsoid matrix Aᵢ is parameterized as `Lᵢ Lᵢᵀ + ε·I`, where
Lᵢ is the upper-triangular part of a freely-trained `(n_dims, n_dims)` matrix.

**Rationale:** A general matrix is not guaranteed to be positive-definite during gradient
descent. Parameterizing via a triangular factor guarantees Aᵢ is PD for any values of Lᵢ,
eliminating projection steps or PD-penalty terms. The `ε·I` term prevents degeneracy when
Lᵢ → 0 (which would make Aᵢ singular).

**Implementation:** `tf.linalg.band_part(W, 0, -1)` zeros the lower triangle. Do not remove
this call — without it, the full matrix would be used and PD is not guaranteed.

**Trade-off:** Lᵢ stores D² values but only D(D+1)/2 are meaningful. Slightly wasteful but
simplifies indexing. Can be optimized in a future version.

---

## ADR-002: Node Spread Regularization via 1/var(dist_nodes)

**Decision:** The regularization term in the loss is `var_weight × (1 / var(dist_nodes))`,
where `dist_nodes` is the matrix of pairwise non-Euclidean distances between node centers.

**Rationale:** Without regularization, all nodes can collapse to the same location (a
degenerate local minimum where the model behaves as a single-node RBF). Penalizing low
variance of pairwise distances prevents this. The inverse form sends the loss to infinity
as nodes collapse, creating a hard barrier. Found empirically to outperform simpler
alternatives (mean distance penalty, spread bonus).

**Implications:** `var_weight` is a sensitive hyperparameter. Too small → node collapse;
too large → nodes spread outside the data hull and RMSE degrades. Default 0.2 is calibrated
for standardized (unit-variance) data (raised from 0.1 when the L2 amplitude penalty was
disabled — see ADR-010). Must be preserved in the TF2 migration.

---

## ADR-003: Standardization (Zero-Mean, Unit-Variance) over Min-Max Scaling

**Decision:** All input and output dimensions are standardized by subtracting the mean
and dividing by the standard deviation before training.

**Rationale:** Standardization is more robust to outliers than min-max scaling, and keeps
the optimizer in a well-conditioned region regardless of the absolute scale of each
dimension. Physical data (flow rates, pressure ratios, efficiencies) can have very different
scales; unit-variance normalization ensures each dimension contributes equally to the
non-Euclidean distance before training begins.

**Implications:** `Scale` and `Offset` must always be saved alongside model weights — they
are required to un-scale predictions and node positions. A model saved without them is
not recoverable. This is a hard constraint on all save/load implementations.

---

## ADR-004: Global Linear + Constant Trend Term

**Decision:** A linear bias `b1·x + b2` is always added to the RBF sum. It is always
trained; it is not optional.

**Rationale:** Pure RBF interpolation struggles when data has a strong global trend (e.g.
efficiency increasing monotonically with flow rate). The linear term captures the trend
cheaply, leaving RBF nodes to model the residual. This is standard practice in polyharmonic
spline literature and improves convergence speed and extrapolation behavior.

**Implications:** `b1` and `b2` are always part of the variable set. Do not make them
optional or removable.

---

## ADR-005: Adam Optimizer, Full-Batch, with Exponential LR Decay

**Decision:** Adam optimizer with exponential learning rate decay. Training is always
full-batch (all data points per step).

**Rationale:** Adam was chosen for robustness to hyperparameter choice and handling of
the varied gradient magnitudes that arise from optimizing node positions, ellipsoid
matrices, and amplitude weights simultaneously. Full-batch is appropriate because datasets
are small (typically < 1000 points) and the loss landscape benefits from the exact
gradient. Exponential LR decay allows aggressive early exploration and fine convergence
later.

**Implications:** Mini-batching is not currently implemented and would require revisiting
the LR decay schedule. Do not add mini-batching without considering this interaction.

---

## ADR-006: TensorFlow as the Compute Backend

**Decision:** TensorFlow (migrating from TF1 compat mode to native TF2) is used rather
than NumPy, PyTorch, or JAX.

**Rationale:** Original implementation used TF1. Automatic differentiation is required —
the gradient of the non-Euclidean distance through the LLᵀ construction is non-trivial to
derive analytically. TF2 is the migration target because it adds eager execution (easier
debugging) while retaining the same math ops and avoiding a full rewrite.

**Trade-off:** PyTorch would be equally valid technically and has a larger research
community. Switching backends is a larger scope change than TF1→TF2 and is deferred
indefinitely unless a concrete reason arises (e.g. a required library only available
in PyTorch).

---

## ADR-007: Multiquadric as the Default Basis Function

**Decision:** `multiquadric` (`a1·(√(r²+1)−1)`) is the default basis function.

**Rationale:** Multiquadric is smooth, strictly positive-definite in any dimension, and
well-conditioned for gradient-based optimization. Found empirically to give good results on
compressor map data without tuning. The shifted form `(√(r²+1)−1)` rather than `√(r²+c²)`
avoids a separate shape hyperparameter by using c=1 in the normalized data space.

**Implication:** The effective shape of the basis function is tied to the data
standardization. If the scaling approach is changed (ADR-003), the basis function behavior
will change.

---

## ADR-008: n_points Argument is Vestigial

**Decision:** The `n_points` parameter passed to `EBF_Graph` is not used in any computation
and should be removed in Phase 2.

**Rationale:** It was added in an earlier design where the placeholder shape was fixed. The
current implementation uses `[None, n_dims]` for the placeholder, making `n_points`
unnecessary. Removing it simplifies the API.

**Action:** Remove from `EBF_Graph` signature and all call sites during Phase 2 restructure.

---

## ADR-009: Huber Loss as Default Loss Function

**Decision:** Huber loss is available as an alternative to RMSE via `loss_type='huber'`.
The default remains RMSE (`loss_type='rmse'`).

**Rationale:** The target datasets are noisy sensor and test-rig measurements. RMSE squares
every residual, so a single noisy outlier exerts disproportionate influence (a point 10×
noisier than the rest becomes 100× more important). Huber loss behaves quadratically for
small residuals (preserving gradient efficiency near the solution) but switches to linear
beyond a threshold `huber_delta`, capping outlier influence. This changes what the optimizer
prioritizes — fitting the underlying trend rather than chasing measurement noise.

**Hyperparameter:** `huber_delta` (originally default `1.3`). Since data is standardized
to unit variance (ADR-003), `delta = 1.3` meant residuals larger than 1.3 standard
deviations got linear treatment — intended to mirror the classical robust-statistics
choice of `1.345·sigma`. **Superseded by ADR-013:** the classical rule calibrates against
the *residual* scale, not the data scale, so the fixed 1.3 provided almost no robustness
in practice; the default is now `'auto'` (adaptive, residual-calibrated).

**Implications:** RMSE remains the default. Users working with noisy data should set
`loss_type='huber'` explicitly. ~~The `loss_threshold` parameter is calibrated against
the active loss function, so thresholds tuned for RMSE will need adjustment for Huber.~~
(Resolved by ADR-013's RMSE-comparable scaling.)

---

## ADR-010: L2 Amplitude Penalty (Smooth Weight) — Rejected

**Decision:** The L2 penalty on the amplitude weights (`smooth_weight * mean(a1²)`,
SMOOTHNESS.md Method 1 / Phase S1a) is **rejected**. The `smooth_weight` parameter has
been removed from both training APIs (2026-07-08); passing it now raises `TypeError`.
Do not re-add it under this or any other name — the mechanism itself is unsound for EBF,
not merely mistuned.

**Rationale (original):** Large amplitude weights cause tall, narrow bumps in the fitted
surface. Penalizing `||a||²` is the kernel-machine equivalent of Tikhonov regularization —
it shrinks the RKHS norm of the interpolant, which upper-bounds surface curvature for basis
functions like multiquadric, gaussian, and inv_multiquadric.

**Why it was rejected:** The RKHS argument assumes a *fixed* kernel. In EBF the kernel is
learned — node positions and per-node ellipsoid matrices are trainable — and that breaks
the penalty in two ways:

1. **It penalizes height, not sharpness.** Near a node the multiquadric contribution is
   `a1·δᵀAδ/2`, so local curvature scales with the product `a1·λ(A)`. Spikes come from
   large ellipsoid eigenvalues (narrow influence), not large amplitudes. A moderate-`a1`,
   huge-`A` needle is barely penalized, while a broad legitimate feature with large `a1`
   is punished hard.
2. **The optimizer dodges the penalty in a direction that sharpens the surface.** For the
   default multiquadric, `φ ≈ r` at large distance, so the rescaling `L → c·L, a1 → a1/c`
   leaves the far field unchanged while cutting the penalty by `c²` — and increasing
   near-node curvature by `c`. Worse, the ADR-002 term compounds this: inflating `L`
   inflates `var(dist_nodes)`, shrinking `var_weight/var(dist_nodes)` too. The two
   regularizers jointly reward the "small amplitudes, big ellipsoids" reparameterization —
   a spikier fit. This matches the empirical finding that `smooth_weight` was ineffective
   compared to `var_weight` for producing smooth-but-conformal fits.

**History:** The default was tried at `0.05`, then `0.01`, then the penalty was disabled
(commit `8b9ff35`; `var_weight` raised 0.1 → 0.2 to compensate — see ADR-002). The dead
parameter was removed entirely in the Session C API-polish batch (2026-07-08).

**Successors:** Smoothness is controlled by `var_weight`, node count, and basis choice.
For an explicit knob, the EBF-appropriate mechanisms act on the ellipsoids directly:
SMOOTHNESS.md Phase S3 (`ellipsoid_weight`, Frobenius penalty on L — also breaks the
rescaling degeneracy above) and Phase S4 (`grad_weight`, immune to reparameterization).
S3 is scheduled as Session D in `ROADMAP.md`. Note the S1c L1 penalty
(`sparse_weight`) shares part of the same degeneracy for growing bases — evaluate it
separately under Phase S7 (node pruning), not as a smoothness control.

---

## ADR-011: Ellipsoid Shape Penalty via Frobenius Norm on L

**Decision:** An optional smoothness term `ellipsoid_weight × mean_i(‖Lᵢ‖_F²)` is added to
the loss (SMOOTHNESS.md Method 4 Option B / Phase S3), where Lᵢ is the upper-triangular
factor of node i's ellipsoid matrix (ADR-001). Default `ellipsoid_weight = 0.0` — the
penalty is off unless the user enables it, so default behavior is unchanged. Exposed in
both `train.run()` and `EBF.fit()`; computed in the shared `_train()` loop with a
trace-time branch, so the zero-weight path adds nothing to the graph.

**Rationale:** This is the successor to the rejected L2 amplitude penalty (ADR-010). EBF
sharpness lives in the ellipsoid eigenvalues — near-node curvature scales with `a1·λ(A)` —
so the penalty acts on that mechanism directly. Since `Aᵢ = Lᵢ Lᵢᵀ + ε·I`, we have
`tr(Aᵢ) = ‖Lᵢ‖_F² + D·ε ≥ λmax(Aᵢ)`, so bounding the Frobenius norm bounds the largest
eigenvalue and hence how sharp/narrow any node's influence can become. It also breaks the
rescaling degeneracy that sank ADR-010: the reparameterization `L → c·L, a1 → a1/c` that
dodged the amplitude penalty is now punished by `c²`.

**Why Option B (Frobenius):** Option A (log condition number) needs an eigendecomposition
every training step; Option C (off-diagonal only) misses per-axis sharpening, which
produces spikes just as easily as rotation/shear. Frobenius is one matmul-free line and
targets the right quantity (λmax).

**What it does *not* control:** the condition number `λmax/λmin` is scale-invariant, so a
magnitude penalty does not bound it — under strong penalty the ellipsoids get *smaller*
(lower λmax, smoother surface) but not necessarily *rounder*, and empirically the mean
condition number can even rise while λmax falls. The Session D acceptance test therefore
asserts a reduction in λmax and ‖L‖_F, not condition number. If pure shape control
(round-but-large ellipsoids allowed) is ever needed, upgrade to Option A.

**Interaction with ADR-002:** the `1/var(dist_nodes)` term remains active and unmodified.
The two terms oppose each other in equilibrium — shrinking all L shrinks `dist_nodes` and
its variance, inflating the ADR-002 barrier — which is what prevents the penalty from
collapsing every ellipsoid to zero.

**Hyperparameter:** `ellipsoid_weight`. Sensitivity is problem-dependent: ~0.01 is a mild
nudge that preserves fit quality; ~0.5 visibly shrinks ellipsoids and trades accuracy for
smoothness. Tune upward from small values, ideally against held-out data (S2/S6).

---

## ADR-012: Early Stopping via Validation Split

**Decision:** Optional patience-based early stopping (SMOOTHNESS.md Method 3 / Phase S2)
via `val_fraction` (default `0.0` = off) and `patience` (default `10`) in both
`train.run()` and `EBF.fit()`, implemented once in the shared `_train()` loop. When
`val_fraction > 0`, a random `val_fraction` of the points is held out before training,
the validation loss is evaluated every `VAL_EVERY = 100` steps, training stops after
`patience` consecutive evaluations without improvement, and the weights from the
best-validation step are restored. With the default `val_fraction = 0.0`, behavior is
bit-identical to the previous code.

**Rationale:** On noisy data the training loss keeps falling as the model memorizes
individual noisy points while generalization error *rises* — the divergence point is
exactly where the model transitions from learning signal to learning noise. `loss_threshold`
monitors training loss and therefore cannot detect this; a held-out set can. This replaces
`train_steps` guesswork for the primary use case (noisy sensor / test-rig data).

**Design choices:**

1. **Validation loss is the data-fit term only** (RMSE or Huber on the held-out points,
   matching the active `loss_type`). The ADR-002 and ADR-011 regularizers do not measure
   generalization; including them would confound the stopping signal with regularizer
   dynamics.
2. **Best-weight restore, always.** Whenever the split is active, the final weights are
   those of the best-validation step — even when the loop runs to completion or
   `loss_threshold` fires first. The steps past the best step were, by the validation
   signal, fitting noise; keeping them would silently deliver a worse model than was
   observed during training.
3. **Evaluation cadence is a constant (`VAL_EVERY = 100`),** matching the progress-print
   interval, not a user parameter — `patience` already controls the effective stopping
   horizon (`patience × 100` steps) and a second knob would add tuning surface without
   value.
4. **The split lives in `_train()`,** after the callers' scaling. Scale/Offset are thus
   computed on the *full* dataset (including held-out points) — acceptable, standard
   practice for a split whose purpose is a stopping signal rather than an unbiased error
   estimate.
5. **`seed` drives the split** (same parameter that seeds weight initialization), so a
   seeded run is fully reproducible; unseeded runs get a different split each time
   (documented in SMOOTHNESS.md Method 3 cons).
6. **History gains a third column** `(step, loss, val_loss)` when the split is active
   (NaN except at evaluation steps); `convergence_plot` draws it as a second curve. The
   2-column shape is preserved when `val_fraction = 0` for backward compatibility.

**Small-data guard:** below ~50 points the held-out loss (1–10 samples) is too
high-variance to give a stable stopping signal — a `UserWarning` recommends
regularization (`var_weight`, `ellipsoid_weight`, `loss_type='huber'`) instead. This is
a warning, not an error, because the threshold is heuristic.

**Not implemented (deliberately):** retraining on the full dataset for the selected
number of steps after stopping (mentioned as an option in SMOOTHNESS.md). It doubles
training time to reclaim ≤ 20 % of the data, and the selected step count is not
transferable — the loss trajectory on a different training set diverges from the run
that selected it. Revisit only if the held-out fraction proves costly in practice.

**Interactions:** `loss_threshold` still monitors the *training* loss and both stopping
mechanisms can be active simultaneously; whichever fires first ends the loop (best-weight
restore still applies). Reported per-step progress shows both losses when the split is
active.

---

## ADR-013: Adaptive Huber Threshold and RMSE-Comparable Loss Scaling

**Decision:** `huber_delta` defaults to `'auto'`: the threshold is recalibrated every
100 steps (`DELTA_EVERY`) from the current training residuals as
`delta = 1.345 * 1.4826 * MAD(residuals)`, floored at `1e-3` in scaled space. A float
still fixes the threshold. Additionally, the Huber data term is reported and optimized
as `sqrt(2 * mean(huber))` instead of `mean(huber)`.

**Rationale (threshold):** ADR-009 calibrated `delta = 1.3` against the *data* scale
(unit variance after ADR-003 standardization), but the classical `1.345·σ` rule refers
to the *residual* scale. Converged residuals are typically 0.01–0.05 in scaled space —
orders of magnitude below 1.3 — so every residual sat in the quadratic zone and Huber
silently degenerated to `0.5·MSE` with **no outlier robustness at all** (a bad point
10× the noise level was still fully squared). Calibrating against the residuals fixes
this: the MAD (median absolute deviation) estimates the residual spread in a way
outliers cannot corrupt (they cannot inflate their own threshold), `1.4826` converts
MAD to a Gaussian-equivalent sigma, and `1.345·σ` retains 95% statistical efficiency on
Gaussian noise while putting roughly the largest ~18% of residuals in the linear zone.
The threshold is self-calibrating over training: initial residuals are O(1) so training
starts effectively quadratic (fast convergence), and delta shrinks with the residuals
so the linear zone stays meaningful relative to the current noise floor.

**Rationale (scaling):** the `'rmse'` branch returns `sqrt(MSE)` while Huber returned
`mean(huber) ≈ 0.5·MSE` for in-zone residuals — at a residual scale of 0.05 that is
~40× smaller, so switching loss types silently re-weighted the total loss toward the
ADR-002/ADR-011 regularizers and invalidated any tuned `loss_threshold`.
`sqrt(2 * mean(huber))` is a "pseudo-RMSE": it reduces *exactly* to `sqrt(MSE)` when
all residuals are inside delta, so `loss_threshold`, `var_weight`, and
`ellipsoid_weight` keep their meaning across loss types.

**Implementation notes:**
- The adaptive threshold lives in a non-trainable `tf.Variable`, assigned between steps —
  no retracing of the compiled `train_step`, and no gradient path through delta (the
  optimizer cannot game its own threshold).
- Recalibration uses **training rows only**; the validation loss (ADR-012) stays a clean
  held-out measurement. The extra forward pass every 100 steps costs ~1% overhead.
- The floor (`1e-3`) prevents delta chasing residuals to zero on noise-free data, where
  the loss would become pure L1 with jumpy gradients near the solution.
- With `'auto'`, successive validation evaluations use slightly different deltas; the
  drift is negligible because the pseudo-RMSE equals RMSE up to the ~18% linear tail.
- Verbose output prints the current delta alongside the loss, so the residual noise
  floor is visible during training.

**Supersedes:** the `huber_delta = 1.3` default from ADR-009 (fixed floats remain
supported for users who want manual control).

---

## ADR-014: Tukey Biweight Loss (Redescending — Outlier Rejection)

**Decision:** A third loss type `loss_type='tukey'` implements the Tukey biweight
(bisquare) loss with rejection point `tukey_c` (default `'auto'`). It reuses the entire
ADR-013 adaptive-threshold machinery: `c = 4.685 * 1.4826 * MAD(residuals)`, recalibrated
every 100 steps (`DELTA_EVERY`), floored at `THRESHOLD_FLOOR = 1e-3`, and the same
`sqrt(2 * mean(rho))` pseudo-RMSE scaling.

**Rationale:** Huber (ADR-009/013) *downweights* outliers — beyond delta their influence
is capped at a constant, but they still pull on the surface forever. The Tukey biweight is
a *redescending* M-estimator: influence rises through the quadratic core, rolls over, and
reaches exactly zero at `|r| = c`. Points beyond the rejection point exert zero pull —
they are discarded in every practical sense (the loss plateaus at `c²/6`; it cannot
decrease for far points, or the optimizer would be rewarded for pushing the surface away
from them). This is the right tool when large residuals represent *erroneous* data (bad
sensor readings) rather than merely *noisy* data. `TUKEY_K = 4.685` is the classical
tuning constant retaining 95% efficiency on Gaussian noise.

**Rejection is not permanent (by design):** influence depends on the current residual, so
a rejected point regains its vote if the surface later moves back toward it — continuous
reconsideration, unlike manually deleting rows.

**Non-convexity and annealing:** the Tukey loss is non-convex; a bad start can reject
correct data and settle in a wrong basin. `'auto'` provides the standard annealing remedy
for free: initial residuals are O(1) in scaled space, so c starts large (everything in the
quadratic core) and tightens onto the noise floor as the fit converges. A fixed float `c`
is supported but not recommended — a small fixed c can reject most points at
initialization, zeroing their gradients and stalling training (documented in the
docstrings).

**Implementation:** branchless form `rho = (c²/6)(1 - max(0, 1 - (r/c)²)³)` — the
`max(0, ·)` clamps the gradient to exactly zero beyond c while staying differentiable
inside. For small residuals `rho ≈ r²/2`, so the ADR-013 pseudo-RMSE scaling reduces to
`sqrt(MSE)` to first order and `loss_threshold` / regularizer weights keep their meaning.
Shared internals were renamed generic (`thresh_t`, `refresh_thresh`,
`THRESHOLD_FLOOR`); verbose output prints `c=` for Tukey, `delta=` for Huber.

**Verification (sine + 2 gross outliers, 12 nodes, `var_weight=0.01`, 8k steps),
R² vs *clean* truth:** RMSE 0.9895, Huber-auto 0.9994, **Tukey-auto 0.9996** — with
Tukey's final loss (0.046) also closest to the inlier noise floor since rejected points
contribute only a constant.

---

## ADR-015: User-Settable Tuning Constant via `'<k>sigma'` Threshold Specs

**Decision:** `huber_delta` and `tukey_c` accept a third form — a sigma-relative string
such as `'2.5sigma'` — alongside the existing `'auto'` and fixed float. It keeps the full
ADR-013 adaptive machinery (MAD estimate, `DELTA_EVERY` recalibration, `THRESHOLD_FLOOR`)
and only substitutes the caller's K for the built-in `HUBER_K` / `TUKEY_K`. Parsing lives
in `_parse_threshold(name, value, default_k) -> (adaptive, number)` in `ebf/train.py`,
which is also the validator called from `_validate_fit_params`. `'auto'` remains the
default and `'1.345sigma'` is exactly equivalent to it for Huber.

**Rationale:** the two parameters conflated *policy* (adaptive vs fixed) with *value*.
The knob a user actually wants to tune is K — it sets the efficiency/robustness
trade-off, and lowering it is the standard response to heavy contamination — but K was
hardcoded and reachable only by abandoning adaptivity for a fixed absolute threshold.
That is the wrong trade: a fixed threshold in scaled data space is precisely the failure
mode ADR-013 was written to eliminate, and for Tukey it also discards the annealing that
keeps the non-convex loss out of bad basins (ADR-014). Users were being pushed toward the
worst option to reach the knob they wanted.

**Alternatives rejected:**
- *Separate `huber_k` / `tukey_k` parameters* — simpler to implement, but widens an
  already-large `fit()` signature by two and admits silently-ignored combinations
  (`huber_k=2.0` with `huber_delta=0.4` does nothing, with no diagnostic).
- *Tuple form `('auto', 2.5)`* — avoids parsing but reads poorly at the call site and
  still needs its own validation branch.

The string form makes invalid combinations unrepresentable, keeps every public signature
unchanged, and reads the way engineers state the quantity ("three sigma"). Being
stringly-typed is the accepted cost; the parser rejects malformed specs eagerly, at the
same validation point as before.

**Grammar:** `^\s*<float>\s*\*?\s*sigma\s*$` — whitespace- and `*`-tolerant, so
`'3sigma'`, `'3 sigma'`, `'3 * sigma'`, and `'3.sigma'` are equivalent; scientific
notation is accepted. K must be strictly positive. `'auto'` is matched before the regex.

**Guidance (documented, not enforced):** for Tukey, K below ~2 can stall training for the
same reason a small fixed `c` does. Not made a hard error — it is a legitimate setting on
severely contaminated data, and the failure is visible in the loss trace.

**Unchanged:** the fixed-float form, the default constants, `refresh_thresh()` (it already
read K from the enclosing scope), the checkpoint format (thresholds are training-time
hyperparameters and were never serialized), and every public signature.
