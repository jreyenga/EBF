# EBF Smoothness Promotion — Methods & Implementation Roadmap

## Context

The EBF model fits a surface `y = sum_i[ a1_i * phi(r_i^2) ] + b1*x + b2`, where each node
has its own ellipsoid matrix controlling anisotropic distance. The current loss is:

```
L = RMSE(y, y_hat) + var_weight / var(dist_nodes)
```

The only existing regularization (ADR-002) prevents **node collapse** — it says nothing about
the smoothness of the fitted surface. This document surveys methods to explicitly promote
smoothness, ranked by viability for the EBF architecture.

**Primary motivation:** the target datasets are inherently noisy (sensor measurements,
test-rig data). The model must fit the underlying trend without chasing measurement noise.
Rankings below are weighted toward noise-robustness, not just geometric smoothness.

---

## Method 1: Amplitude Weight Penalty (L2 on `a1`)

**Rank: 1 — Best starting point**

> **Status: implemented, then REJECTED (ADR-010, 2026-07-08).** The RKHS argument below
> assumes a fixed kernel; in EBF the ellipsoids are learned, so the penalty targets height
> instead of sharpness and the optimizer can dodge it via the rescaling `L → c·L, a1 → a1/c`
> (which sharpens the near field). Empirically ineffective vs `var_weight`. See ADR-010 for
> the full mechanism. The EBF-appropriate replacements are Method 4 (S3) and Method 5 (S4).

### Idea

Add `lambda_a * sum(a1^2)` to the loss. Large `|a1|` values cause tall, narrow bumps in the
surface; penalizing them forces the optimizer to distribute the fit across many nodes with
moderate amplitudes, which is inherently smoother.

### Why it works for EBF

- In standard RBF theory, the norm of the weight vector `||w||` is directly proportional
  to the RKHS norm of the interpolant (Reproducing Kernel Hilbert Space). Penalizing it
  is the kernel-machine equivalent of Tikhonov regularization.
- For basis functions like `multiquadric`, `gaussian`, `inv_multiquadric`, the RKHS norm
  equals or upper-bounds the surface curvature. Shrinking `a1` shrinks curvature.
- Dead simple to implement: one line added to the loss function.

### Formula

```
L = RMSE + var_weight / var(dist_nodes) + lambda_a * reduce_mean(a1^2)
```

If `a2`, `a3` exist (multi-parameter basis functions), include them too:

```
+ lambda_a * (reduce_mean(a1^2) + reduce_mean(a2^2) [+ reduce_mean(a3^2)])
```

### Hyperparameter

`lambda_a` (float, default ~0.01). Same sensitivity profile as `var_weight` — too high
degrades fit accuracy, too low has no smoothing effect. Should be tuned on a per-problem
basis or via cross-validation.

### Pros / Cons

| Pros | Cons |
|------|------|
| One line of code | Only indirect smoothness control |
| Zero computational overhead | Doesn't constrain ellipsoid shapes |
| Well-understood theory (ridge regression) | May under-smooth if basis func is inherently rough |
| Works for all basis functions | Requires tuning `lambda_a` |

---

## Method 2: Robust Loss Function (Huber Loss)

**Rank: 2 — Critical for noisy data**

### Idea

Replace RMSE with Huber loss. RMSE squares every residual, so a single noisy outlier can
warp the entire surface toward it. Huber loss behaves quadratically for small residuals
(preserving gradient efficiency near the solution) but switches to linear beyond a threshold
`delta`, dramatically reducing the influence of outliers.

### Formula

```
huber(r) = 0.5 * r^2            if |r| <= delta
           delta * (|r| - 0.5 * delta)  otherwise

L = mean(huber(y - y_hat)) + var_weight / var(dist_nodes) + ...
```

Note: this replaces `sqrt(MSE)` with `mean(huber(...))`. The gradient signal near the
solution is similar but outliers no longer dominate.

### Why it matters for noisy data

RMSE treats a point that's 10x noisier than the rest as 100x more important (because of
the square). Huber loss caps that influence at 10x. This changes *what the optimizer is
trying to do* — not just how it does it — making it fundamentally different from adding
penalty terms.

### Hyperparameter

`delta` (float, default ~1.0 in scaled data space). Points with residuals below `delta`
are fit normally; those above are treated as potential outliers. Since data is standardized
to unit variance, `delta = 1.0` means "residuals larger than 1 standard deviation get
linear treatment." Can also be set to `1.345 * sigma` (the classical robust-statistics
choice that retains 95% efficiency for Gaussian data).

### Implementation

TF provides `tf.keras.losses.Huber` or it can be written directly:

```python
residuals = Y_tensor - Y_pred
huber = tf.where(
    tf.abs(residuals) <= delta,
    0.5 * tf.square(residuals),
    delta * (tf.abs(residuals) - 0.5 * delta)
)
loss = tf.reduce_mean(huber) + var_weight_t / var_dist
```

### Pros / Cons

| Pros | Cons |
|------|------|
| Directly addresses the noise problem at the loss level | Changes loss semantics (not a drop-in) |
| ~5 lines of code | New hyperparameter `delta` |
| Well-understood robust statistics theory | Slightly slower convergence near the optimum |
| Works with all basis functions | Need to decide: Huber vs RMSE as default |

---

## Method 3: Train/Validation Split with Early Stopping

**Rank: 3 — Essential for noisy data, no new hyperparameters**

> **Status: implemented (ADR-012, 2026-07-09)** as `val_fraction` / `patience` in both
> training APIs, default `val_fraction=0.0` (off). Implementation notes: the validation
> loss is the data-fit term only (no regularizers), evaluation runs every 100 steps, the
> best-validation weights are restored at the end, and the training history gains a
> third `val_loss` column that `convergence_plot` draws as a second curve.

### Idea

Hold out a fraction of the data (e.g. 15-20%) as a validation set. Monitor validation loss
during training. Stop when validation loss stops improving (starts increasing = the model is
memorizing noise in the training set).

### Why it matters for noisy data

With noisy data, training loss keeps decreasing as the model memorizes individual noisy
points, but generalization error (validation loss) starts *increasing*. The divergence point
is exactly where the model transitions from learning signal to learning noise. Early stopping
catches this automatically without needing to tune regularization strengths.

### Implementation

```python
# Split before training
n_val = int(0.15 * n_points)
indices = np.random.permutation(n_points)
X_train, X_val = X[indices[n_val:]], X[indices[:n_val]]
Y_train, Y_val = Y[indices[n_val:]], Y[indices[:n_val]]

# In training loop: evaluate validation loss every N steps
# Stop if no improvement for `patience` evaluations
```

### Hyperparameters

`val_fraction` (float, default 0.15) — fraction held out for validation.
`patience` (int, default 10) — number of evaluation intervals without improvement before
stopping.

These replace `train_steps` rather than adding to the hyperparameter count.

### Pros / Cons

| Pros | Cons |
|------|------|
| Automatic capacity control | Reduces training data by val_fraction |
| Replaces `train_steps` guesswork | Small datasets may not have enough points to split |
| No smoothness hyperparameters to tune | Non-deterministic (depends on random split) |
| Well-proven technique | Need to decide what to do with held-out points after stopping |

---

## Method 4: Ellipsoid Condition Number / Shape Penalty

**Rank: 4 — High impact, moderate complexity**

> **Status: Option B implemented (ADR-011, 2026-07-09)** as `ellipsoid_weight` in both
> training APIs, default `0.0` (off). Implementation note: the Frobenius penalty bounds
> `λmax(A)` (the sharpness mechanism) but **not the condition number**, which is
> scale-invariant — under strong penalty ellipsoids get smaller, not necessarily rounder.
> Upgrade to Option A if pure shape control is ever needed.

### Idea

Penalize extreme aspect ratios in the per-node ellipsoid matrices. When a node's `A_i`
has eigenvalues differing by orders of magnitude, the node's influence is a razor-thin
ellipsoid — responsive in one direction, nearly flat in another. This creates sharp ridges.

### Formulation options

**Option A — Log condition number:**
```
penalty_i = log(lambda_max(A_i) / lambda_min(A_i))
smoothness_loss = lambda_c * reduce_mean(penalty)
```
Drives eigenvalue ratios toward 1 (spherical nodes).

**Option B — Frobenius norm of L (simpler):**
```
penalty = reduce_mean(||L_i||_F^2)
```
Since `A = L L^T + eps*I`, penalizing `||L||_F` limits the overall magnitude of A, which
simultaneously limits both scale and anisotropy. This is computationally cheaper than
eigendecomposition.

**Option C — Off-diagonal penalty:**
```
mask = 1 - eye(D)
penalty = reduce_mean((L * mask)^2)
```
Only penalizes off-diagonal elements of L, preserving per-axis scaling while discouraging
rotation/shearing. A gentler constraint that still reduces anisotropy.

### Why it works for EBF

The ellipsoid matrices are the unique feature of EBF vs standard RBF. Unconstrained, they
can create arbitrarily sharp directional features. Bounding their condition number forces
each node to be "reasonably round," which directly limits the directional sharpness of the
fitted surface.

### Pros / Cons

| Pros | Cons |
|------|------|
| Directly constrains the thing that makes EBF surfaces rough | Eigenvalue computation adds cost per step |
| Tunable — can allow moderate anisotropy while banning extremes | Frobenius alternative is cheaper but less precise |
| Interpretable: condition number has geometric meaning | New hyperparameter (`lambda_c`) |

---

## Method 5: Gradient Magnitude Penalty (First-Derivative Regularization)

**Rank: 5 — Effective but computationally expensive**

### Idea

Penalize the norm of the surface gradient at each training point (or at a set of collocation
points). A smooth surface has bounded gradients; an oscillating surface has large gradients
in the valleys and peaks.

### Formula

```
grad_y = d(y_hat) / d(X)            # shape: (n_points, n_dims)
penalty = reduce_mean(||grad_y||^2)
L = RMSE + var_weight/var(dist_nodes) + lambda_g * penalty
```

### Implementation

TF2 makes this feasible via double `GradientTape`:

```python
with tf.GradientTape() as outer_tape:
    outer_tape.watch(X_tensor)
    with tf.GradientTape() as inner_tape:
        inner_tape.watch(X_tensor)
        Y_pred, dist_nodes, dist = model(X_tensor)
    grad_y = inner_tape.gradient(Y_pred, X_tensor)     # (n_points, n_dims)
    grad_penalty = tf.reduce_mean(tf.reduce_sum(grad_y**2, axis=1))
    loss = rmse + var_weight_t / var_dist + lambda_g * grad_penalty
gradients = outer_tape.gradient(loss, model.trainable_variables)
```

### Why it works for EBF

The EBF forward pass is fully differentiable through TF. The gradient of the output w.r.t.
the input captures exactly how fast the surface changes — penalizing it directly penalizes
non-smoothness. This is the most theoretically rigorous approach.

### Pros / Cons

| Pros | Cons |
|------|------|
| Directly penalizes non-smoothness | Nested GradientTape doubles memory and compute |
| Works for any basis function | Gradient only at training points — doesn't constrain between them |
| Doesn't restrict ellipsoid shape unnecessarily | Needs careful scaling of `lambda_g` |

---

## Method 6: Laplacian Penalty (Second-Derivative Regularization)

**Rank: 6 — Gold standard for smoothness, hardest to implement**

### Idea

Penalize the Laplacian (sum of second partial derivatives) of the surface at training points.
The Laplacian measures curvature — zero Laplacian is a harmonic function (the smoothest
possible interpolant). This is the continuous analogue of thin-plate spline energy.

### Formula

```
laplacian = sum_d [ d^2(y_hat) / d(x_d)^2 ]    # scalar per point
penalty = reduce_mean(laplacian^2)
```

### Implementation difficulty

Requires computing the diagonal of the Hessian `d^2 y / dx_i^2` for each input dimension.
TF2 can do this via nested `GradientTape`, but:

- For D input dimensions, need D separate second-derivative computations
- Memory cost scales as `O(D * n_points * n_nodes)`
- May need `tf.function` with `experimental_compile` to be practical

### Pros / Cons

| Pros | Cons |
|------|------|
| Most rigorous smoothness measure | Expensive: O(D) Hessian diagonals |
| Thin-plate spline energy is a special case | Complex implementation |
| Directly minimizes curvature | Triple GradientTape nesting can be fragile |

---

## Method 7: Basis Function Selection

**Rank: 7 — Zero-cost smoothness via choice**

### Idea

Some basis functions are inherently smoother than others. Choosing a smoother basis function
is the simplest way to get a smoother fit, with no code changes needed.

### Smoothness ranking of current basis functions

| Basis | Smoothness | Notes |
|-------|-----------|-------|
| `gaussian` | C-infinity | Infinitely differentiable, fastest decay |
| `inv_multiquadric` | C-infinity | Smooth, moderate decay |
| `inv_quadratic` | C-infinity | Smooth, slow decay |
| `inv_cosh` | C-infinity | Smooth, moderate decay |
| `matern52` | C-2 | Twice differentiable, no more |
| `matern32` | C-1 | Once differentiable only |
| `multiquadric` | C-infinity | Smooth but *grows* with distance (extrapolation risk) |
| `cubic` | C-1 | Continuous first derivative, cusp at r=0 |
| `quadratic` | C-0 | Continuous but kink at r=0 |
| `linear` | C-0 | Non-differentiable at r=0 |
| `thin_plate` | C-1 | Standard, but r^2 log(r^2) is only once smooth |
| `thin_plate_2` | C-1 | Higher-order thin plate, similar regularity |

### Recommendation

For maximum smoothness with compact influence: **`gaussian`** or **`inv_cosh`**.
For maximum smoothness with global influence: **`inv_multiquadric`**.
Current default (`multiquadric`) is C-infinity and generally a good all-round choice.

### Pros / Cons

| Pros | Cons |
|------|------|
| No code changes, no hyperparameters | Not adjustable — it's smooth or it isn't |
| No computational cost | May not be smooth *enough* without weight regularization |
| User just picks a different string | Different basis functions have different convergence properties |

---

## Method 8: Reducing Node Count

**Rank: 8 — Simplest lever, but sacrifices accuracy**

### Idea

Fewer nodes = fewer degrees of freedom = less capacity for oscillation. This is the RBF
equivalent of reducing polynomial degree. It's the most blunt smoothness tool.

### Guidance

Rule of thumb: `n_nodes` should be between `sqrt(n_points)` and `n_points / 3`.
Below `sqrt(n_points)`, the model is too constrained to capture real features.

### Pros / Cons

| Pros | Cons |
|------|------|
| Zero implementation effort | Throws away model capacity |
| Faster training | No way to smooth locally while keeping detail elsewhere |
| Fewer parameters = less overfitting | User has to guess the right count |

---

## Method 9: Cross-Validation for Hyperparameter Tuning

**Rank: 9 — Meta-method, works with any of the above**

### Idea

Use LOO-CV error to automatically select `lambda_a`, `lambda_c`, `lambda_g`, or `n_nodes`.
LOO-CV is particularly efficient for linear systems (via the "hat matrix" shortcut) but for
EBF's nonlinear model it requires retraining for each fold.

### Practical alternative: k-fold CV

Split data into k=5 folds, train k models, average validation RMSE. Use grid search or
Bayesian optimization over the smoothness hyperparameters.

### Pros / Cons

| Pros | Cons |
|------|------|
| Principled, automatic tuning | k * train_time per evaluation |
| Avoids overfitting to any single split | Outer loop adds complexity to API |
| Works with any regularization method | Not practical for > 2 hyperparameters simultaneously |

---

## Method 10: Adaptive Node Count (Automatic Model Complexity)

**Rank: 10 — Stretch goal, high value but significant engineering**

The methods below address the question: "how many nodes does this dataset actually need?"
rather than requiring the user to guess. They are ordered from easiest to hardest.

### 10a: L1 Sparsity on `a1` (Soft Pruning) — easiest

Add `sparse_weight * reduce_mean(|a1|)` to the loss alongside the L2 penalty from Method 1.
L1 penalty drives unneeded node weights **exactly to zero** (unlike L2, which shrinks but
never zeros). Start with more nodes than you think you need; the optimizer will "turn off"
the excess ones.

Combined with L2 this is called **Elastic Net** regularization:

```
L = ... + smooth_weight * mean(a1^2) + sparse_weight * mean(|a1|)
```

After training, count nodes with `|a1| > threshold` to see the effective node count.

**Pros:** ~3 lines of code, no graph changes, well-studied theory.
**Cons:** L1 is non-differentiable at zero (use `tf.abs` — TF handles the subgradient).
Doesn't reduce computation during training (all nodes still evaluated).

### 10b: Learnable Per-Node Gate — moderate

Add a scalar gate variable `g_i` per node, passed through sigmoid: `gate_i = sigmoid(g_i)`.
Multiply each node's contribution by its gate:

```python
# In EBFModel.__init__:
self.gates_raw = tf.Variable(tf.zeros([n_nodes]), name='Gates')

# In forward pass:
gates = tf.sigmoid(self.gates_raw)   # (n_nodes,) in [0, 1]
Y1 = reduce_sum(gates * a1 * phi(r2), axis=1)
```

Initialize `g_i = 0` (all gates start at 0.5). During training, gates for unnecessary nodes
drift toward 0 and useful nodes drift toward 1. Optionally add L1 on `gates` to encourage
binary decisions.

**Pros:** More expressive than L1 on `a1` — the gate also suppresses the node's contribution
to distance regularization. Differentiable everywhere. ~20 lines.
**Cons:** New variable set, adds a concept to the model.

### 10c: Information Criteria (AIC/BIC) — simple outer loop

Train separate models at a few node counts (e.g. `n_nodes = [3, 5, 8, 12, 20]`), compute:

```
k = n_nodes * (1 + n_dims + n_dims*(n_dims+1)/2) + n_dims + 1   # total parameters
BIC = n_points * log(MSE) + k * log(n_points)
```

Pick the node count that minimizes BIC. This is the most statistically principled approach
but requires training multiple models.

**Pros:** Completely reliable, easy to implement, well-understood theory.
**Cons:** Trains `len(candidates)` full models. Good as a validation tool even if another
method is the primary selector.

### 10d: Sequential Node Addition with Warm Start — classical RBF approach

Start with `n_nodes=2`, train to convergence. Examine residuals, place a new node at the
location of the largest residual. Create a new model with `n_nodes=3`, copy all existing
weights, initialize only the new node's weights. Train again (converges fast because most
weights are already good). Repeat until validation error stops improving.

**Warm-start transfer for EBF:**
- Copy `Nodes[:old_n]`, `EllipsoidWeights[:old_n]`, `a1[:old_n]`, `b1`, `b2`
- Initialize the new node's position at the point of max residual
- Initialize its ellipsoid weights as small random (≈ spherical)
- Initialize its `a1` as the residual value at that point

The graph recreation cost is real, but warm-starting means each stage converges in a
fraction of the full training steps. The natural stopping criterion is: stop when adding
a node doesn't improve validation error.

**Pros:** Most principled, gives best results, automatic stopping.
**Cons:** Biggest engineering lift — needs warm-start utility, outer training loop, and
validation split. Best suited as a Phase S6+ stretch goal.

---

## Summary Ranking

| Rank | Method | Noise Impact | Effort | New Hyperparams |
|------|--------|-------------|--------|-----------------|
| 1 | L2 amplitude penalty (`a1`) | High | Trivial | `smooth_weight` |
| 2 | Robust loss (Huber) | Very High | Trivial | `delta` |
| 3 | Train/val split + early stopping | Very High | Low | `val_fraction`, `patience` |
| 4 | Ellipsoid shape penalty | Medium | Low-Medium | `ellipsoid_weight` |
| 5 | Gradient magnitude penalty | Medium-High | Medium | `grad_weight` |
| 6 | Laplacian penalty | High | High | `laplacian_weight` |
| 7 | Basis function choice | Low-Medium | None | None |
| 8 | Reduce node count | Medium | None | None |
| 9 | Cross-validation tuning | Meta | Medium-High | None (tunes others) |
| 10a | L1 sparsity on `a1` (soft pruning) | Medium-High | Trivial | `sparse_weight` |
| 10b | Learnable per-node gate | Medium-High | Low | None (or L1 on gates) |
| 10c | AIC/BIC model selection | High | Medium | None |
| 10d | Sequential node addition | High | High | None |

---

## Implementation Roadmap

Not all methods will necessarily be implemented. The phasing below groups features by
dependency and effort so that any subset can be picked up independently.

### Phase S1: Noisy Data Baseline

**Scope:** The three cheapest, highest-impact changes for noisy data, bundled together
because they all modify `train.py` and form a coherent "noise-robust training" feature set.

**S1a — L2 amplitude penalty (Method 1):** ~~Add `smooth_weight` parameter to
`train.run()` and `EBF.fit()`. Add `smooth_weight * reduce_mean(a1^2)` to the loss.~~
**Implemented, then rejected and removed — see ADR-010 and the Method 1 status note.**

**S1b — Huber loss option (Method 2):**
Add `loss_type` parameter (`'rmse'` | `'huber'`) and `huber_delta` to `train.run()` and
`EBF.fit()`. Replace the RMSE computation with a conditional branch. ~15 lines.

**S1c — L1 sparsity on `a1` (Method 10a):**
Add `sparse_weight` parameter. Add `sparse_weight * reduce_mean(|a1|)` to the loss.
~3 lines. **Note (2026-07-08):** shares part of the amplitude/ellipsoid rescaling
degeneracy that sank S1a (ADR-010) for growing bases like multiquadric — evaluate it
under Phase S7 as a node-pruning tool, not as a smoothness control.

**Files changed:**
- `ebf/train.py` — new params, loss term modifications in `train_step()`
- `ebf/api.py` — expose new params in `EBF.fit()`

**Acceptance criteria:**
- All new params default to current behavior (`smooth_weight=0`, `sparse_weight=0`,
  `loss_type='rmse'`)
- Example showing effect of each on noisy synthetic data
- Unit test verifying loss changes when each param is activated

---

### Phase S2: Early Stopping with Validation Split — ✅ Implemented (Session E, 2026-07-09)

> **Done — see ADR-012.** `val_fraction` (default `0.0`) and `patience` (default `10`)
> in `train.run()` and `EBF.fit()`; split and stopping logic in the shared `_train()`
> loop. Deltas from the spec below: default `val_fraction` is `0.0` (off), not 0.15 —
> the design note's "identical to current code" default won; the best-validation
> weights are restored at the end (the "what to do with held-out points / retrain on
> full data" option was deliberately not implemented — see ADR-012); the < 50-points
> warning is implemented; the reported/plotted loss adds validation as a third history
> column rather than replacing the training loss.

**Scope:** Add optional train/validation split with patience-based early stopping (Method 3).

**Files changed:**
- `ebf/train.py` — add `val_fraction`, `patience` params; split data; evaluate val loss
  every N steps; stop when no improvement for `patience` evaluations
- `ebf/api.py` — expose in `EBF.fit()`

**Estimated complexity:** ~40 lines changed.

**Design note:** When `val_fraction=0` (default), behavior is identical to current code.
After early stopping, optionally retrain on full data for the same number of steps that
the validation run selected (a common technique to reclaim the held-out data).

**Acceptance criteria:**
- `val_fraction=0` reproduces current behavior exactly
- Training stops earlier on noisy data than on clean data
- Reported loss is validation loss (not training loss) when split is active

**Documentation note:** The docs and docstring for `val_fraction` must include guidance that
validation splitting is only reliable for datasets with ~50+ points. With fewer than ~50
points, 1-3 held-out samples have too much variance to produce a stable stopping signal —
users should rely on regularization (Phase S1) instead. Consider printing a warning when
`val_fraction > 0` and `n_points < 50`.

---

### Phase S3: Ellipsoid Shape Penalty — ✅ Implemented (Session D, 2026-07-09)

**Scope:** Add Frobenius norm penalty on the upper-triangular L matrices (Option B from
Method 4). This is the simplest form; upgrade to condition-number penalty later if needed.

> **Done — see ADR-011.** `ellipsoid_weight` (default `0.0`) in `train.run()` and
> `EBF.fit()`; penalty computed in the shared `_train()` loop from a new
> `EBFModel.ellipsoid_factors()` helper. One acceptance-criterion revision: tests assert
> lower `λmax(A)` and `‖L‖_F` rather than lower condition number, because the condition
> number is scale-invariant and a magnitude penalty does not bound it (it can even rise
> while the surface gets smoother).

**Files changed:**
- `ebf/train.py` — add `ellipsoid_weight` param, compute `reduce_mean(L_F^2)` in loss
- `ebf/model.py` — expose L matrices (or add a helper to return them) for penalty computation
- `ebf/api.py` — expose `ellipsoid_weight` in `EBF.fit()`

**Estimated complexity:** ~30 lines changed.

**Acceptance criteria:**
- Ellipsoid condition numbers are measurably lower with penalty enabled
- Visual comparison of fitted surface with and without penalty
- Existing regularization (ADR-002) still active and unmodified

---

### Phase S4: Gradient Penalty

**Scope:** Add optional gradient-magnitude regularization via nested `GradientTape`
(Method 5).

**Files changed:**
- `ebf/train.py` — add `grad_weight` param, nested tape logic in `train_step()`
- `ebf/api.py` — expose `grad_weight` in `EBF.fit()`

**Estimated complexity:** ~40 lines changed.

**Design risk:** The `@tf.function`-compiled `train_step` must support double-tape tracing.
A spike/prototype should verify this works under `tf.function` before committing to the
approach.

**Acceptance criteria:**
- Gradient penalty demonstrably reduces surface gradient norms
- Training time increase is < 3x compared to baseline
- Works with all basis functions

---

### Phase S5: Examples & Documentation

**Scope:** Add an example script demonstrating the smoothness methods on a test problem
(e.g., noisy sinusoidal data) with side-by-side plots. Update `docs/` with a smoothness
guide.

**Files:**
- `examples/smoothness_comparison.py`
- `docs/smoothness.md`
- `mkdocs.yml` — add nav entry

---

### Phase S6 (Optional): Cross-Validation Helper

**Scope:** Add a `cv_score()` utility that runs k-fold cross-validation for a given
parameter set, returning mean validation RMSE. Users can wrap this in a grid search to
tune `smooth_weight`, `ellipsoid_weight`, `grad_weight`.

**Files:**
- `ebf/cv.py` — new module
- `ebf/__init__.py` — export `cv_score`

---

### Phase S7 (Stretch): Adaptive Node Count

**Scope:** Implement one or more adaptive node count strategies from Method 10. Recommended
starting point is **10b (learnable per-node gates)** because it requires no graph
recreation and gives clean binary on/off decisions per node.

**If 10b (gates):**
- `ebf/model.py` — add `self.gates_raw` variable, multiply `sigmoid(gates_raw)` into
  basis function sum in `__call__`
- `ebf/train.py` — optionally add L1 penalty on gates to encourage sparsity
- `ebf/api.py` — expose `gate_sparse_weight`, add `get_active_nodes()` method

**If 10d (sequential addition):**
- `ebf/warm_start.py` — new module: create a new `EBFModel(n+1)`, copy weights from
  `EBFModel(n)`, initialize new node at max-residual location
- `ebf/train.py` — add `auto_nodes=True` mode with outer loop
- Requires Phase S2 (early stopping) to know when each stage has converged

**Estimated complexity:** ~50-100 lines depending on approach chosen.

---

### Phase S8 (Stretch): Laplacian Penalty

**Scope:** Add second-derivative (Laplacian) regularization (Method 6). Deferred because it
requires per-dimension Hessian diagonals and may have performance issues. Implement only if
Methods 1-5 prove insufficient for the target use cases.

---

### Dependency Graph

```
S1 (noisy data baseline: L2 + Huber + L1)
  |
  +---> S2 (early stopping)
  |       |
  |       +---> S7 stretch: adaptive nodes (needs S2 for stopping criterion)
  |
  +---> S3 (ellipsoid penalty)  -- independent of S2
  |
  +---> S4 (gradient penalty)   -- independent of S2, S3
  |
  +---> S5 (examples & docs)    -- after S1-S4 are stable
          |
          +---> S6 (CV helper)  -- optional, after S5
          |
          +---> S8 (Laplacian)  -- optional, only if S1-S4 insufficient
```

### Naming Convention

All smoothness hyperparameters use the suffix `_weight` (consistent with existing
`var_weight`):

| Parameter | Controls |
|-----------|----------|
| `var_weight` | Node spread (existing, ADR-002) |
| `smooth_weight` | Amplitude L2 penalty (S1a) — **rejected and removed, ADR-010** |
| `sparse_weight` | Amplitude L1 penalty (S1c) |
| `ellipsoid_weight` | Ellipsoid shape penalty (S3) — **implemented, ADR-011** |
| `grad_weight` | Gradient magnitude penalty (S4) |

Non-`_weight` parameters:

| Parameter | Controls |
|-----------|----------|
| `loss_type` | `'rmse'` or `'huber'` (S1b) |
| `huber_delta` | Huber loss threshold (S1b) |
| `val_fraction` | Validation split ratio (S2) — **implemented, ADR-012** |
| `patience` | Early stopping patience (S2) — **implemented, ADR-012** |
