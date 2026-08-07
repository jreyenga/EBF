# EBF Roadmap

Current status and planned work. Design rationale for decisions already made lives in
[DECISIONS.md](DECISIONS.md); smoothness research in [SMOOTHNESS.md](SMOOTHNESS.md).

**Categories:** 🐛 Bug/Correctness · 📚 Documentation · 🔧 Maintainability ·
✨ User Improvement · ⚡ Performance · 🏗️ Infrastructure · 🔮 Planned Feature

---

## Status

The library is feature-complete for its original purpose: an installable, documented,
native-TF2 package with a stable public API.

| Milestone | State |
|-----------|-------|
| Package structure, basis registry, examples, tests | ✅ Complete |
| TF1 → TF2 migration (`tf.Module`, `GradientTape`, `tf.train.Checkpoint`) | ✅ Complete |
| Public API (`EBF.fit` / `predict` / `get_nodes` / `get_ellipsoids` / `save` / `load`) | ✅ Complete |
| MkDocs documentation site | ✅ Complete (not yet hosted) |
| Robust losses — Huber (ADR-009/013), Tukey (ADR-014) | ✅ Complete |
| Ellipsoid shape penalty `ellipsoid_weight` (ADR-011) | ✅ Complete |
| Early stopping with validation split (ADR-012) | ✅ Complete |
| Visualization and lookup-table export (`ebf/viz.py`) | ✅ Complete |

---

## Open Work

### P4 — Performance ⚡

**Skip node-to-node distances at inference.** `EBFModel.__call__` always computes
`delta_nodes` + `dist_nodes`, an `O(n_nodes² · n_dims²)` matmul needed only by the training
regularizer. Every `predict()` call pays for it. Fix: a `training=False` flag returning
`None` for `dist_nodes`, or a separate `node_distances()` method called from the training
loop.

**Cache the restored model in `run_points()`.** `ebf/predict.py` re-reads the JSON sidecar
and checkpoint from disk on every call, so grid evaluation via repeated calls pays the
restore cost each time. Either memoize on `file`, or document that repeated evaluation
should use `EBF.load()` + `predict()`, which keeps the model in memory.

> **Not doing:** avoiding the per-step GPU/graph sync. The loop calls `loss.numpy()` every
> step, but the per-step training history that `history_` and `convergence_plot` depend on
> requires it, and per-step `loss_threshold` checking is the documented semantics.
> Sub-sampling to keep both was judged not worth the complexity for CPU-bound full-batch
> training.

### P5 — Infrastructure 🏗️

**Continuous integration.** 121 tests pass locally but nothing runs them automatically. A
minimal GitHub Actions workflow (`poetry install` + `pytest`, Python 3.11) would catch
exactly the class of regression that manual edits have introduced before — divergence
between the two training paths, defaults drifting away from their docstrings.

**Deploy the documentation site.** `mkdocs gh-deploy` to GitHub Pages. The site builds
clean under `--strict` and `site/` is already gitignored.

### P5b — Higher-Dimensional Examples 🏗️

Every worked example fits a 2-D surface, purely because that is what a contour plot can
show. The README states explicitly that the method is n-dimensional — verified 1-D through
5-D, with fit, predict, `get_ellipsoids`, `eval_grid` and `export_grid` all working
unchanged (R² ≈ 0.999 on smooth synthetic targets) — but no *example* demonstrates it.

The open question is presentation, not capability. Candidates:

- **2-D slices** — hold the remaining inputs fixed and contour two at a time; a
  small-multiples grid of slices across a third variable
- **Isosurfaces** — for 3 inputs, marching cubes on a level set of the response
- **Slider-driven slice** — interactive, but a poor fit for static documentation
- **Parallel coordinates / pairwise matrix** — shows behavior without pretending the space
  is spatial

A good candidate dataset would have genuinely heterogeneous units (speed, pressure,
temperature, mass), since that is where the ellipsoid-as-learned-metric argument is
strongest and a Euclidean prior least defensible.

---

## Planned Features 🔮

In recommended order. Specifications for the smoothness items are in
[SMOOTHNESS.md](SMOOTHNESS.md).

1. **S4 — Gradient magnitude penalty** (`grad_weight`, nested `GradientTape`). Prototype the
   double tape under `@tf.function` first, as the design note warns.
2. **S6 — Cross-validation helper** (`ebf/cv.py`, `cv_score()`).
3. **S7 — Adaptive node count.** Start with learnable per-node gates (Method 10b). Note that
   S1c `sparse_weight` was deferred here as a pruning tool rather than a smoothness control,
   because it shares part of the degeneracy that sank ADR-010.
4. **Mixed basis functions** — per-group basis assignment (below).
5. **Harmonic bases** (`sin_rbf`, `damped_sin`, `bessel_j0`) — depends on mixed-basis support.

### Mixed Basis Functions

Per-group basis assignment via a node-group API:

```python
EBF(n_nodes=8, basis='multiquadric')                              # current, unchanged
EBF(n_nodes=8, basis=[('multiquadric', 5), ('inv_multiquadric', 3)])
```

**Forward pass:** `r2` has shape `(n_points, n_nodes)`. Slice by group — group 0 takes
`r2[:, :5]`, group 1 takes `r2[:, 5:]` — evaluate each basis on its slice and sum the
contributions. Each group gets its own weight variables.

**Files affected:** `model.py` (constructor and `__call__` handle a group list; variables
become `A1_g0`, `A1_g1`, …), `io.py` (the JSON sidecar must serialize basis name and node
count per group to reconstruct variables on load), and the `EBF` constructor (`basis`
accepts `str` or `list[tuple[str, int]]`, with the string form a backward-compatible
shorthand).

**Main difficulty:** groups with differing `n_params` — each group should create only the
weight variables its basis actually requires.

### Harmonic Basis Functions

For data with periodic or wave-like structure (blade passing frequencies, tidal data,
vibration modes):

| Name | Expression | Extra params | Notes |
|------|------------|--------------|-------|
| `sin_rbf` | `a1 · sin(a2 · r)` | a2 (frequency) | Pure oscillatory; per-node frequency |
| `damped_sin` | `a1 · sin(a2 · r) · e^(−r²)` | a2 (frequency) | Localized oscillation |
| `bessel_j0` | `a1 · J₀(a2 · r)` | a2 (frequency) | Radial waves; acoustics/optics |

Implementation notes:

- `a2` is the per-node frequency and needs careful initialization (suggest a `freq_init`
  parameter defaulting to `1.0`) or clamping, or the optimizer becomes unstable
- Harmonic bases are prone to local minima in `a2` — may benefit from a warm start
  (pre-train without harmonic nodes, then add) or a separate learning rate for `a2`
- `bessel_j0` needs `tf.math.bessel_i0` or a polynomial approximation; check availability
- Mixed basis is a prerequisite: harmonic nodes are most useful *combined* with smooth
  bases (e.g. 6 `multiquadric` + 2 `sin_rbf`), not as a global replacement

---

## Design Notes

Retained because they still constrain future work.

**Basis function registry.** `BASIS_FUNCTIONS` maps name → `(fn, n_params)`. The `n_params`
value drives how many `a` weight tensors `EBFModel` creates — extra weights waste optimizer
capacity and add noise when the basis doesn't use them. All 12 registered functions
currently use `a1` only; the tuple format and the `n_params` branching exist for the
multi-parameter bases above. `eps` is always the last positional argument so single-weight
functions share a consistent signature.

**Epsilon.** User-configurable via `eps` on the constructor, default `1e-8`. Only `cosh` and
`inv_cosh` need it. `thin_plate` uses `tf.math.xlogy`, which handles `0·log(0) = 0`
natively; the remaining functions have no singularity at `r² = 0`.

**Ellipsoid construction.** `A = LLᵀ` guarantees positive-definiteness but is redundant —
upper-triangular `L` has `D(D+1)/2` free parameters against `D²` stored. Acceptable; could
be optimized later. See ADR-001.

**Node spread regularization.** The loss includes `var_weight / var(dist_nodes)` to
discourage node collapse, where `dist_nodes` is *node-to-node* distance. Using point-to-node
distance instead does not create the collapse barrier. See ADR-002.

**Data scaling.** Inputs and outputs are standardized to zero mean and unit variance before
training. This is load-bearing — model weights are meaningless without it, and the learned
ellipsoids live in scaled space (`get_ellipsoids()` maps them back via `A_orig = S·A·S`).

---

## Completed Work

Condensed; per-session detail is in [SESSION_LOG.md](SESSION_LOG.md) and the git history.

**Phases 1–5** — dead-file cleanup and side-effect fixes; installable `ebf/` package with
basis registry, `examples/`, `tests/`, `data/`; TF1 → TF2 migration; the `EBF` class API;
MkDocs site.

**Post-Phase-5 correctness pass** — the two training APIs were silently training different
models (regularizer mismatch against ADR-002, and defaults diverging from their own
docstrings); `EBF.save()` crashed after `EBF.load()`; `inv_matern32`/`inv_matern52` were
removed as numerically explosive. The root cause of the first two was a duplicated training
loop, now deduplicated into a single shared `_train()`.

**Smoothness work** — Huber loss kept (ADR-009) and given an adaptive MAD-calibrated
threshold (ADR-013); the L2 amplitude penalty `smooth_weight` was implemented, found
mechanically unsound for a learned kernel, and removed (ADR-010, Rejected); ellipsoid shape
penalty added as its successor (ADR-011); early stopping with validation split and
best-weight restore (ADR-012); Tukey biweight loss with adaptive rejection point (ADR-014).

**Documentation and examples** — `ebf/viz.py` plotting and export helpers; shared plot style;
`get_ellipsoids()`; README rebuilt around generated figures; loss-types and RBF-vs-EBF pages;
`make_docs_figures.py` to regenerate every documentation figure in one command.
