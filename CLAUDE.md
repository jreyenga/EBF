# EBF — Session Primer (auto-loaded by Claude Code)

## What This Project Is
Elliptical Basis Function (EBF) interpolation library — a generalization of Radial Basis
Functions where each node has a learnable per-node positive-definite ellipsoid matrix instead
of a scalar Euclidean distance. Used to fit smooth surfaces to scattered engineering data
(compressor maps, geotechnical measurements, etc.). Built on native TF2 (the TF1 → TF2
migration is complete and all legacy TF1 files have been deleted).

## Current State
- **Phase 1 complete** — dead files deleted, side effects fixed, hardcoded paths removed, `__main__` guards added
- **Phase 2 complete** — installable `ebf/` package created; basis function registry; examples/; tests/; data/ moved
- **Phase 3 complete** — TF1 → TF2 migration: `EBFModel(tf.Module)`, `GradientTape` training, `tf.train.Checkpoint` + JSON sidecar
- **Phase 4 complete** — API Polish: `EBF` class with fit/predict/get_nodes/save/load, docstrings
- **Phase 5 complete** — MkDocs documentation site: algorithm guide, basis function gallery, API reference, examples
- **Post-Phase 5** — Smoothness Phase S1 partially adopted: Huber loss kept (ADR-009), L2 amplitude penalty rejected and `smooth_weight` removed (ADR-010); Phase S3 ellipsoid shape penalty added (`ellipsoid_weight`, ADR-011); Phase S2 early stopping added (`val_fraction`/`patience`, ADR-012); adaptive Huber threshold `huber_delta='auto'` + RMSE-comparable Huber scaling (ADR-013); Tukey biweight loss `loss_type='tukey'` with adaptive rejection point `tukey_c='auto'` (ADR-014); `ebf/viz.py` visualization utilities added; legacy TF1 files deleted for distribution
- **Active roadmap** — `docs/design/ROADMAP.md` (open: P4 performance, P5 infrastructure/CI, P5b higher-dimensional examples)

## Key Files
| File | Role |
|------|------|
| `ebf/ebf.py` | High-level `EBF` class: fit / predict / get_nodes / get_ellipsoids / save / load |
| `ebf/model.py` | Core model: `DeltaAll`, `NonEuclidDistance`, `LinearBias`, `EBFModel(tf.Module)` |
| `ebf/train.py` | Shared `_train()` GradientTape loop + functional `run()` — **single source of truth for the loss** |
| `ebf/predict.py` | Inference — functional `run_points()` |
| `ebf/basis_functions.py` | Basis function registry (`BASIS_FUNCTIONS` dict) |
| `ebf/scaling.py` | Data standardization utilities |
| `ebf/io.py` | Checkpoint save/restore (`tf.train.Checkpoint` + JSON sidecar) |
| `ebf/viz.py` | Shared plot style constants + plot/grid helpers: `convergence_plot`, `correlation_plot`, `residual_plot`, `contour_plot_2d`, `summary_plot_3d`, `eval_grid`, `export_grid` |
| `ebf/__init__.py` | Public API: `EBF`, `run`, `run_points`, `BASIS_FUNCTIONS`, viz functions |
| `tests/` | pytest suite: `test_model.py`, `test_train.py`, `test_ebf.py` |
| `examples/comp_map_ebf.py` | **Premium example** — class API end-to-end: all fit inputs, history, every viz/export tool |
| `examples/RBF_vs_EBF.py` | Headline example — scipy RBF vs EBF on the shared synthetic test surface |
| `examples/loss_comparison.py` | rmse vs huber vs tukey on noisy / outlier-corrupted data |
| `examples/node_ellipsoids.py` | Minimal 3-node illustration of how ellipsoids adapt to the data |
| `examples/example_utils.py` | Shared test surface + grid/contour-panel/ellipse helpers |
| `examples/make_docs_figures.py` | Regenerates every figure embedded in the docs (`--list`, `--only NAME`) |
| `ALGORITHM.md` | Math derivation, tensor shapes, loss function |
| `docs/design/ROADMAP.md` | Status, open work, planned features, retained design notes |
| `docs/design/CONVENTIONS.md` | Naming rules, shape notation, data contract |
| `docs/design/DECISIONS.md` | Why architectural choices were made — check before changing anything |
| `docs/design/PRIOR_ART.md` | Where EBF sits in the literature — read before making novelty claims |
| `docs/design/SESSION_LOG.md` | Per-session decisions not captured in git history |
| `docs/design/SMOOTHNESS.md` | Smoothness/noise research and implementation roadmap |
| `mkdocs.yml` | MkDocs documentation site configuration |
| `docs/` | Rendered documentation source (index, algorithm, basis functions, API, examples) |

## Hard Rules
- **Do not** refactor beyond the current phase scope
- **Do not** change core algorithm math without an entry in `docs/design/DECISIONS.md`
- **Do not** remove or bypass data standardization (Scale/Offset) — model weights are meaningless without it
- **Do not** remove the `1/var(dist_nodes)` regularization term — see ADR-002
- **Do not** remove `tf.linalg.band_part` on the ellipsoid weights — see ADR-001
- All scripts must keep `if __name__ == "__main__":` guards
- use `poetry` instead of `pip` to add packages to `.venv` if required

## Quick Reference
- Input data: `(n_points, n_dims+1)` array — **last column is always the output variable**
- Default basis function: `multiquadric`
- Default epsilon: `1e-8` (numerical stability in basis functions)
- Default `var_weight`: `0.2` (node spread regularization strength)
- Default `ellipsoid_weight`: `0.0` (ellipsoid shape penalty off — the explicit smoothness knob, ADR-011)
- Default `huber_delta`: `'auto'` (adaptive, residual-calibrated via MAD; only used with `loss_type='huber'`, ADR-013)
- Default `tukey_c`: `'auto'` (adaptive Tukey rejection point, `4.685·σ̂`; only used with `loss_type='tukey'`, ADR-014)
- Default `val_fraction`: `0.0` (validation split / early stopping off; `patience=10`, ADR-012 — needs ~50+ points)
- Tensor shape comments use `(n_points, n_nodes, n_dims)` notation — see `docs/design/CONVENTIONS.md`
- Status, open work and planned features: `docs/design/ROADMAP.md`
