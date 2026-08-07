# EBF Algorithm Reference

## Model Equation

The EBF model predicts a scalar output ŷ from a D-dimensional input vector x:

```
ŷ = Σᵢ₌₁ᴺ [ a1ᵢ · φ(r²ᵢ, ...) ]  +  b1·x  +  b2
```

where N is the number of nodes, φ is the chosen basis function, and the second term is a
global linear+constant trend.

---

## Non-Euclidean Distance

For node i with center vᵢ, the squared distance from input point x is:

```
r²ᵢ = (x − vᵢ)ᵀ Aᵢ (x − vᵢ)
```

Aᵢ is a D×D **positive-definite matrix unique to each node**, constructed as:

```
Aᵢ = Lᵢ Lᵢᵀ + ε·I
```

where Lᵢ is the upper-triangular matrix stored in `EllipsoidWeights[i]`, and ε (default 1e-8)
prevents degeneracy when Lᵢ → 0. This factorization guarantees Aᵢ is positive-definite for
any real values of Lᵢ — no projection or penalty is needed during training.

The geometric interpretation: each node defines its own ellipsoidal influence region. The
optimizer can freely rotate and stretch these ellipsoids independently per node, which is the
key generalization over standard RBF (where all nodes share a fixed spherical metric).

---

## Trainable Parameters

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `Nodes` (vᵢ) | `(n_nodes, n_dims)` | Node center positions in input space |
| `EllipsoidWeights` (Lᵢ) | `(n_nodes, n_dims, n_dims)` | Upper-triangular factors for Aᵢ |
| `a1` | `(n_nodes,)` | Primary basis function amplitude weights |
| `a2` | `(n_nodes,)` | Secondary weights (multi-parameter basis functions only) |
| `a3` | `(n_nodes,)` | Tertiary weights (multi-parameter basis functions only) |
| `b1` | `(n_dims,)` | Linear trend weights |
| `b2` | `(1,)` | Constant bias |

`a2` and `a3` are only created when the selected basis function declares `n_params >= 2`
or `>= 3` in the registry. All 12 currently registered basis functions use `a1` only, so
these variables are never allocated in practice — the machinery is retained for the
multi-parameter bases planned in Phase 6.

---

## Forward Pass — Step by Step

```
Input
  X                           shape: (n_points, n_dims)

Step 1 — DeltaAll(X, Nodes)
  deltas = x − v              shape: (n_points, n_nodes, n_dims)
  (vector from each node to each input point)

Step 2 — NonEuclidDistance(deltas, EllipsoidWeights)
  L = upper_triangular(EllipsoidWeights)    shape: (n_nodes, n_dims, n_dims)
  A = L @ Lᵀ + ε·I                         shape: (n_nodes, n_dims, n_dims)
  r² = squeeze( deltasᵀ @ A @ deltas )     shape: (n_points, n_nodes)
  r² = abs(r²)   ← guards against FP rounding producing tiny negatives

Step 3 — ActFunc(r², a1, ...)
  Y1 = Σ_nodes [ a1 · φ(r²) ]              shape: (n_points,)

Step 4 — LinearBias(X, b1, b2)
  Y2 = X @ b1 + b2                          shape: (n_points,)

Step 5 — Sum
  Y = Y1 + Y2                               shape: (n_points,)
```

---

## Loss Function

```
L = RMSE(y, ŷ)  +  var_weight · (1 / var(dist_nodes))  [+ ellipsoid_weight · mean(‖L‖_F²)]
```

**RMSE term** — computed on standardized data:
```
RMSE = sqrt( mean( (y_scaled − ŷ_scaled)² ) )
```

**Regularization term** — `dist_nodes` is the `(n_nodes, n_nodes)` matrix of pairwise
non-Euclidean distances between node centers (same distance metric as the forward pass).
`1/var(dist_nodes)` → ∞ as nodes collapse to the same location, preventing degeneracy.
`var_weight` (default 0.2) controls the trade-off. See ADR-002 in `DECISIONS.md`.

**Smoothing effect of `var_weight`** — Beyond preventing collapse, `var_weight` acts as
the primary smoothing control. Forcing nodes apart causes the optimizer to compensate by
reducing the non-Euclidean distance scale (effectively lowering the `r` value seen by the
basis function). This broadens each node's influence zone, producing smoother surfaces.
Higher `var_weight` → more spread → broader influence → smoother fit.

**Ellipsoid shape penalty (optional)** — `ellipsoid_weight · mean(‖Lᵢ‖_F²)`, where `Lᵢ` is
the upper-triangular factor of node i's ellipsoid matrix. Since `tr(Aᵢ) = ‖Lᵢ‖_F² + D·ε`
bounds `λmax(Aᵢ)`, and near-node curvature scales with `a1·λ(A)`, this directly caps how
sharp any node's influence can become. Off by default (`ellipsoid_weight = 0`); enable it
as an explicit smoothness knob for noisy data. See ADR-011 in `DECISIONS.md`.

---

## Data Scaling

All data is standardized before training and un-scaled after inference.

**Scale and Offset computation:**
```python
Scale  = 1 / std(data, axis=0)   # shape (n_dims+1,)
Offset = mean(data, axis=0)       # shape (n_dims+1,)
data_scaled = (data - Offset) * Scale
```

**Training uses `data_scaled` only.**

**Inference un-scaling:**
```python
Y = Y_scaled / Scale[-1] + Offset[-1]          # output
Nodes = Nodes_scaled / Scale[:-1] + Offset[:-1] # node positions
```

`Scale` and `Offset` must be saved alongside model weights. A saved model without them
is not recoverable.

---

## Optimizer

Adam with exponential learning rate decay:

```
lr(step) = lr₀ · decay_rate ^ (step / decay_steps)
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `lr₀` (start) | `0.01` | Initial learning rate |
| `decay_rate` | `0.9` | Multiplicative decay factor |
| `decay_steps` | `10000` | Steps between decay applications |
| `train_steps` | `20000–60000` | Total training steps (problem-dependent) |

Training is always **full-batch** (all points per step). Mini-batching is not implemented
and would require revisiting the LR schedule. See ADR-005 in `DECISIONS.md`.

---

## Basis Functions

See `docs/basis_functions.md` for the full reference with expressions, a visual gallery,
and guidance on choosing between the growing and decaying families; the registry itself is
`ebf/basis_functions.py`. Default: `multiquadric`.

Functions requiring numerical stability epsilon (ε): `cosh`, `inv_cosh`.
Functions where `xlogy` handles the r²=0 case natively: `thin_plate`.
All other functions have no singularity at r²=0.
