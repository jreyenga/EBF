# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Sigma-relative threshold specs.** `huber_delta` and `tukey_c` now accept a
  `'<k>sigma'` string (e.g. `'1.0sigma'`, `'3 * sigma'`) in addition to `'auto'` and a
  fixed float. This keeps the adaptive, residual-calibrated threshold of `'auto'` while
  letting you set the tuning constant K yourself — previously the only way to change K
  was to give up adaptivity entirely and pin an absolute threshold. `'auto'` remains the
  default and is unchanged (`1.345σ` for Huber, `4.685σ` for Tukey). See ADR-015.

## [0.1.0] — 2026-08-07

First public release.

### Added

- **`EBF` class** — the primary API: `fit`, `predict`, `get_nodes`, `get_ellipsoids`,
  `save`, `load`.
- **Per-node learnable ellipsoids.** Each node carries its own positive-definite matrix
  `Aᵢ = LᵢLᵢᵀ + εI`, so the distance metric is inferred from the data rather than assumed
  Euclidean. Works in any number of input dimensions.
- **12 basis functions** via the `BASIS_FUNCTIONS` registry — `multiquadric` (default),
  `gaussian`, `linear`, `quadratic`, `cubic`, `thin_plate`, `inv_multiquadric`,
  `inv_quadratic`, `matern32`, `matern52`, `cosh`, `inv_cosh`.
- **Robust losses.** `loss_type='huber'` downweights outliers and `loss_type='tukey'`
  rejects them; both self-calibrate to the residual noise floor when `huber_delta` /
  `tukey_c` are left at `'auto'`.
- **Smoothness controls.** `var_weight` (node spread regularization) and
  `ellipsoid_weight` (ellipsoid shape penalty).
- **Early stopping.** Optional `val_fraction` validation split with `patience` and
  best-weight restore, plus a `loss_threshold` convergence cutoff.
- **Automatic data standardization.** Inputs and outputs are scaled internally;
  predictions return in original units.
- **Visualization and export** (`ebf.viz`) — `convergence_plot`, `correlation_plot`,
  `residual_plot`, `contour_plot_2d`, `summary_plot_3d`, `eval_grid`, `export_grid`
  (CSV and NPZ).
- **Save/load** via `tf.train.Checkpoint` with a JSON sidecar carrying the basis
  configuration and Scale/Offset values.
- **Functional API** — `run()` and `run_points()` for callers managing their own state.
- **Documentation** — MkDocs site covering the algorithm, basis function gallery, loss
  types, visualization, API reference, and two annotated worked examples.
- **Examples** — `RBF_vs_EBF.py`, `node_ellipsoids.py`, `comp_map_ebf.py`, `1d_fit.py`,
  `loss_comparison.py`, and two gallery scripts. `make_docs_figures.py` regenerates every
  figure used in the README and docs.
- **Test suite** — 121 pytest tests across the model, training loop, and `EBF` class.

### Notes

- Built on native TensorFlow 2 (`tf.Module` + `GradientTape`). No TensorFlow 1
  compatibility shims remain.
- Architecture decisions are recorded in [`docs/design/DECISIONS.md`](docs/design/DECISIONS.md);
  several document changes that look like safe simplifications but are not.

[0.1.0]: https://github.com/jreyenga/EBF/releases/tag/v0.1.0
