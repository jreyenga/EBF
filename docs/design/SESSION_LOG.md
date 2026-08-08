# Session Log

One entry per session. Record decisions made in conversation that are **not** captured in
git history or code comments — reasoning, trade-offs rejected, and anything a future session
would otherwise have to re-derive.

> **File renames (2026-08-06).** Entries below refer to files by the names they had at the
> time. `docs/context/` is now `docs/design/`, and `EBF_ROADMAP.md` plus `Fable_Roadmap.md`
> were merged into a single [ROADMAP.md](ROADMAP.md). The historical names are left
> unchanged in the entries so the record stays accurate.

---

## Session 1 — 2026-04-11

**Work done:**
- Full codebase review; file structure and algorithm documented
- Created `EBF_ROADMAP.md` with 4-phase plan, algorithm summary, basis function notes
- Identified canonical active files vs. superseded versions

**Key decisions:**
- `EBF_v4.py` (not v3) is the canonical model — it has named tensors (`tf.identity(Y, name='out')`) required for checkpoint save/restore via `import_meta_graph`
- `EBF_v3.py` is kept because `EBF_tester1.py` and `EBF_tester2_1D.py` still reference it directly
- Phase 3 TF2 migration will use `tf.Module` + `GradientTape` custom training loop rather than `keras.Model` + `model.fit()`, because the `1/var(dist_nodes)` regularization term is non-standard and fits more naturally in a custom loop
- `n_points` argument in `EBF_Graph` is vestigial — flagged for removal in Phase 2 (see ADR-008)

---

## Session 2 — 2026-04-11

**Work done:**
- Executed Phase 1: deleted 7 superseded files, fixed `TestFunc.py` import side effects, removed hardcoded absolute paths, added `if __name__ == "__main__":` guards to all 8 tester scripts, updated `EBF_Runner_v1.py` default path
- Added basis function specification as Section 7 of roadmap: approved function list, epsilon design requirement, parameter count table, `BASIS_FUNCTIONS` registry code example

**Key decisions:**
- `EBF_tester_v2_exp1.py` added to the delete list beyond the original roadmap scope — it imported `EBF_v3_upgraded` (being deleted) and had an undefined `path` variable; confirmed dead code
- `linear` corrected from `a1·r²` to `a1·r`; `quadratic` added as `a1·r²`; `cubic` corrected from `a1·r⁶` to `a1·r³` — user fixed directly in `EBF_v4.py`; roadmap table updated to match
- `eps` (numerical stability epsilon) to be user-configurable with default `1e-8`; always passed as last positional argument to basis functions so signature is consistent regardless of number of `a` weights
- The two "cosh" variants selected are `cosh` (basic, a1 only) and `cosh_alt` (with trainable shape parameter a2) — the other cosh variants in the original code were dropped

---

## Session 3 — 2026-04-11

**Work done:**
- Added 5 new basis functions to roadmap Section 7: `thin_plate_2`, `inv_multiquadric`, `inv_quadratic`, `matern32`, `matern52`; updated parameter count table and registry example
- Created `CLAUDE.md`, `CONVENTIONS.md`, `ALGORITHM.md`, `DECISIONS.md`, `SESSION_LOG.md`
- Added Phase 5 (API Documentation) to roadmap; updated session sequence table

**Key decisions:**
- Matérn 3/2 and 5/2 are the highest-value new additions for engineering data — C¹/C² continuity guarantees are directly relevant for physical fields (pressure, efficiency)
- `inv_multiquadric` and `inv_quadratic` added specifically to address the gap that all existing functions are non-decaying — decaying functions reduce wild extrapolation outside the data hull
- `thin_plate_2` (`r⁴·log(r²)`) restored from the "Mod Thin Plate" commented-out line in `EBF_v4.py` — C³ vs C¹ for `thin_plate`, no additional parameters needed
- Wendland compact-support functions not added — require a support-radius hyperparameter per node; only matter for large datasets (>1000 points); deferred indefinitely
- Oscillatory functions (sinc, damped cosine) not added — not suited for smooth physical surfaces
- Phase 4 (API Polish) covers writing docstrings in code; Phase 5 (API Docs) covers rendering them into navigable documentation — kept separate because they require different tools and effort

---

## Session 4 — 2026-04-11

**Work done:**
- Executed Phase 3: TF1 compat → native TF2 migration across all `ebf/` modules and tests
- `ebf/model.py`: `EBF_Graph` function replaced with `EBFModel(tf.Module)` class; trainable variables are instance attributes; forward pass via `__call__`
- `ebf/train.py`: `tf.Session`/`feed_dict` loop replaced with `tf.GradientTape`; `tf.keras.optimizers.Adam` + `ExponentialDecay` schedule
- `ebf/predict.py`: session restore replaced with eager `EBFModel` instantiation + `tf.train.Checkpoint` restore
- `ebf/io.py`: `tf.train.Saver` replaced with `tf.train.Checkpoint`; JSON sidecar stores model config (n_dims, n_nodes, basis, eps) and Scale/Offset
- `ebf/basis_functions.py`: import changed from `tensorflow.compat.v1` to `tensorflow` (ops identical)
- `tests/test_model.py`: removed all `tf.Session`, `tf.reset_default_graph`, `tf.placeholder`, `tf.disable_v2_behavior`; `TestEBFGraph` → `TestEBFModel`
- `tests/test_train.py`: no changes needed (public API signatures preserved)
- All 63 tests pass (56 unit + 7 integration); zero `compat.v1` references remain

**Key decisions:**
- `EBFModel(tf.Module)` chosen over `keras.Model` as planned in Session 1 — custom `1/var(dist_nodes)` regularization fits naturally in a `GradientTape` loop
- JSON sidecar file (`.json`) added alongside TF2 checkpoints to store model config needed for reconstruction; `tf.train.Checkpoint` only saves variable values, not graph structure
- Old TF1 checkpoints (`.ckpt.meta`, `.ckpt.index`, `.ckpt.data`) are not loadable by the new code — users must retrain; this is expected and accepted
- Public API (`run`, `run_points`, `BASIS_FUNCTIONS`, `DEFAULT_BASIS`) signatures unchanged — no breaking changes to user-facing code
- `ebf/scaling.py` and `ebf/__init__.py` untouched (pure NumPy and re-exports only)
- `examples/compressor_map.py` untouched (calls public API only)

---

## Sessions between 4 and 5 — 2026-04-11 → 2026-04-14 (reconstructed from git, logged retroactively 2026-07-08)

Work in this period was not logged at the time. Summary from git history:

- **Phase 4 (API polish):** `ebf/ebf.py` — `EBF` class with fit/predict/get_nodes/save/load;
  `loss_threshold` convergence option; `seed` parameter on `EBF.fit()`
- **Smoothness Phase S1 (partial):** Huber loss added (`loss_type`, `huber_delta`; ADR-009).
  L2 amplitude penalty (`smooth_weight`) implemented, default tried at 0.05 then 0.01, then
  disabled entirely (commit `8b9ff35`) with `var_weight` raised 0.1→0.2 in compensation —
  see ADR-010 for status
- **Basis registry cleanup:** `thin_plate_2`, `cosh_alt`, and other multi-param variants
  removed (`f39d8d3` "remove a few basis functions"); `inv_matern32`/`inv_matern52` added
  (`45f3af0` "cleaned up basis functions") — removed again in Session 5
- **Visualization:** `ebf/viz.py` — `correlation_plot`, `contour_plot_2d`, `eval_grid`,
  `export_grid`; viz docs page
- **Examples:** `examples/RBF_vs_EBF.py` comparison incl. skewed-data and extrapolation cases
- **Phase 5 (docs):** MkDocs + Material site (`mkdocs.yml`, `docs/`), basis gallery
  script and PNG, install docs, README
- **Distribution cleanup:** legacy TF1 files deleted (`EBF_v4.py`, `EBF_Runner_v1.py`,
  tester scripts); package build in `dist/`

---

## Session 5 — 2026-07-08

**Work done (ROADMAP.md Session A — P1 bug fixes):**
- Fixed `train.run()` regularization: was `1/var(dist)` (point-to-node), now `1/var(dist_nodes)`
  (node-to-node) per ADR-002 — `EBF.fit()` already had the correct term; the two APIs had
  silently diverged
- Fixed `EBF.save()` crash after `EBF.load()`: `io.save()` now builds the `tf.train.Checkpoint`
  without the optimizer when `optimizer is None` (loaded models have no optimizer)
- Canonicalized defaults to `var_weight=0.2`, `huber_delta=1.3` in both APIs; synced docstrings,
  ADR-002, ADR-009, and CLAUDE.md Quick Reference
- Deleted `inv_matern32` / `inv_matern52` from the registry, `docs/basis_functions.md`, and
  the gallery script; regenerated `docs/assets/basis_functions.png` (12 functions)
- Added `seed` parameter to `train.run()` (parity with `EBF.fit()`)
- New regression tests: API-parity (same seed → same predictions from both APIs), shared-defaults
  consistency via `inspect.signature`, and load→save round trip

**Key decisions:**
- Canonical defaults are the class-API values (`var_weight=0.2`, `huber_delta=1.3`): git history
  shows they were deliberate tunings (commits `8b9ff35`, `8dd743a`) — `var_weight` was raised
  0.1→0.2 to compensate when `smooth_weight` was disabled; `huber_delta=1.3` approximates the
  classical `1.345·sigma` robust-statistics choice. The docstrings/ADRs had simply never been
  updated
- `inv_matern32`/`inv_matern52` deleted rather than documented: they grow like `e^(√3·r)`
  (float32 overflow near r≈51 under extrapolation) and had not demonstrated usefulness

---

## Session 6 — 2026-07-08

**Work done (ROADMAP.md Session B — training-loop dedup + doc sync):**
- Extracted shared `_train()` GradientTape loop and `_validate_loss_params()` into
  `ebf/train.py`; both `train.run()` and `EBF.fit()` now call them — the loss definition
  and optimizer setup exist in exactly one place
- `ebf/ebf.py` no longer imports TensorFlow directly; its inline training loop (~60 lines)
  replaced by a `_train()` call. No public API changes; all 88 tests pass unchanged
- CLAUDE.md updated: Key Files table now matches the repo (added `ebf/ebf.py`, `ebf/viz.py`,
  `tests/`, `ROADMAP.md`; removed deleted TF1 files), "Current State" notes post-Phase-5
  work, project description no longer says "currently migrating" to TF2
- ADR-010 rewritten to reflect that the L2 amplitude penalty was implemented then disabled
  (parameter accepted-but-ignored; resolution tracked as ROADMAP item #9)
- SESSION_LOG backfilled with a reconstructed entry covering the unlogged 2026-04-11→14 work

**Key decisions:**
- `_train()` operates on already-scaled arrays and returns the optimizer; callers own
  scaling (ADR-003) and checkpointing — keeps the loop reusable for future S2 early stopping
- `EBF_ROADMAP.md` Section 7 checked against the registry: in sync after Session 5's
  inv_matern removal (12 functions) — no edit needed

---

## Session 7 — 2026-07-08

**Work done (ROADMAP.md Session C — API polish batch, items #7/#9/#10):**
- **#9 `smooth_weight` removed** from `train.run()` and `EBF.fit()` (passing it now raises
  `TypeError`); `_validate_loss_params()` reduced to `loss_type` checking. Removed from
  the example script and the algorithm/example docs
- **#7 sidecar defaults:** `run_points()` `Scale`/`Offset` are now optional — `None` reads
  the copies stored in the checkpoint JSON sidecar; user-supplied values that disagree with
  the sidecar trigger a `UserWarning` (mismatch = silently wrong predictions, ADR-003)
- **#10 training history:** `_train()` records per-step `(step, loss)`; exposed as
  `EBF.history_` and via `train.run(return_history=True)` (default return unchanged).
  `train.run()` gained a `verbose` flag
- ADR-010 rewritten to **Rejected** with the full mechanism; roadmap #9 updated from
  "re-implement recommended" to "removed"; SMOOTHNESS.md Method 1/S1a marked rejected
- New tests: sidecar-vs-explicit prediction parity, mismatch warning, missing-file
  TypeError, history shape/monotonicity for both APIs, smooth_weight-raises. 94 pass
- **`ebf.convergence_plot()`** added to `ebf/viz.py`: loss curve from the new history
  (accepts a fitted `EBF` or a `(step, loss)` array; log scale by default; optional
  `loss_threshold` reference line). Exported from `ebf/__init__.py`, documented in
  `docs/visualization.md` + `docs/api.md`
- `examples/comp_map_ebf.py` promoted to the **premium example**: demonstrates
  `BASIS_FUNCTIONS`, every constructor/fit input, `history_`, save/load, and all four
  plot/export tools. `examples/compressor_map.py` header now declares it the
  functional-API demo — resolves roadmap #16 (examples consolidated by role)

**Key decisions:**
- `smooth_weight` (L2 on `a1`) rejected rather than re-implemented, overturning the
  roadmap's original option-(a) recommendation. Reason: with learnable ellipsoids the
  penalty targets amplitude (height) while EBF sharpness lives in the ellipsoid
  eigenvalues (curvature ∝ `a1·λ(A)`), and the rescaling `L → c·L, a1 → a1/c` preserves
  the multiquadric far field while cutting the penalty by `c²` and sharpening the near
  field by `c` — a degeneracy the ADR-002 term compounds. Matches the empirical finding
  that it was ineffective vs `var_weight`. Full write-up in ADR-010
- The successor smoothness knob is S3 `ellipsoid_weight` (Frobenius on L — also breaks
  the degeneracy), **pulled forward to Session D**; former sessions D/E/F+ shifted to
  E/F/G+. S1c `sparse_weight` deferred to S7 as a pruning tool, not a smoothness control
- `run_points()` keeps its positional signature (`points, Scale, Offset, file`) for
  backward compatibility; sidecar-default usage is `run_points(points, file=file)`

---

## Session 8 — 2026-07-09

**Work done (ROADMAP.md Session D — S3 ellipsoid shape penalty):**
- **`ellipsoid_weight` implemented (ADR-011):** optional loss term
  `ellipsoid_weight × mean(‖L‖_F²)` (SMOOTHNESS Method 4 Option B), default `0.0` (off).
  Added to `_train()` / `train.run()` / `EBF.fit()`; new `EBFModel.ellipsoid_factors()`
  helper returns the upper-triangular L factors. The penalty branch is resolved at
  `@tf.function` trace time, so the default path builds an identical graph
- Documentation: ADR-011, SMOOTHNESS.md status notes (Method 4 / Phase S3 / naming table),
  loss sections of ALGORITHM.md and `docs/algorithm_overview.md`, premium example
  (`ellipsoid_weight` line in the all-parameters `fit()` call)
- New tests: zero-weight bit-parity with the default path, penalty-changes-trajectory,
  finite predictions under strong penalty, Frobenius/λmax reduction (2D), fit quality
  preserved at mild weight (R² > 0.85), and `ellipsoid_weight` added to the API-parity
  defaults check. Full suite passes
- **Roadmap #12 (skip per-step `loss.numpy()`) marked won't-do:** it conflicts with the
  per-step training history from #10 — `history_` / `convergence_plot` need the loss
  materialized every step. Session F scope reduced to #11 + #13

**Key decisions:**
- **Acceptance criterion revised:** SMOOTHNESS S3 said "condition numbers measurably
  lower with penalty enabled" — that criterion was written for Option A. The condition
  number `λmax/λmin` is scale-invariant, so the Frobenius (magnitude) penalty does not
  bound it; in the 2D test it *rose* (173 → 524) while `‖L‖_F` and `λmax` fell sharply.
  Since near-node curvature ∝ `a1·λ(A)`, λmax is the quantity that matters for
  smoothness; tests assert reduced λmax and ‖L‖_F instead. Upgrade path to a
  condition-number penalty (Option A) stays open if scale-free shape control is needed
- Guidance recorded in ADR-011: `ellipsoid_weight ≈ 0.01` is a mild nudge preserving fit
  quality; `≈ 0.5` visibly shrinks ellipsoids and trades accuracy for smoothness

---

## Session 9 — 2026-07-09

**Work done (ROADMAP.md Session E — S2 early stopping with validation split):**
- **`val_fraction` / `patience` implemented (ADR-012):** optional patience-based early
  stopping in both `train.run()` and `EBF.fit()`, logic in the shared `_train()` loop.
  When `val_fraction > 0`: random split (seeded by the existing `seed` param), validation
  loss evaluated every 100 steps (`VAL_EVERY` constant), stop after `patience`
  evaluations without improvement, **best-validation weights restored** at the end.
  Default `val_fraction=0.0` is bit-identical to the previous code (parity test)
- History gains a third `val_loss` column when the split is active (NaN except at
  evaluation steps) — closes the deferred note in roadmap #10; `convergence_plot` now
  accepts 2- or 3-column histories and draws the validation curve with a legend
- `UserWarning` when `val_fraction > 0` and `n_points < 50` (held-out loss too noisy for
  a stable signal); `val_fraction` ∉ [0, 1) and `patience < 1` raise `ValueError` via
  `_validate_fit_params()` (renamed from `_validate_loss_params()`)
- Docs: ADR-012, SMOOTHNESS.md Method 3 / Phase S2 status notes, algorithm overview
  training section, visualization page, premium example (`val_fraction`/`patience` lines),
  CLAUDE.md Quick Reference + roadmap state
- New tests: zero-fraction bit-parity, 3-column history shape/NaN pattern, stops-early on
  pure-noise data, finite predictions after early stop, small-dataset warning, invalid
  param errors (both APIs), `val_fraction`/`patience` added to the API-parity defaults
  check. Full suite passes

**Key decisions (see ADR-012 for full rationale):**
- Validation loss = **data-fit term only** (RMSE/Huber on held-out points) — the
  ADR-002/ADR-011 regularizer terms don't measure generalization and would confound the
  stopping signal
- **Best-weight restore always applies** when the split is active (even on a full run or
  a `loss_threshold` stop) — steps past the best-validation step were fitting noise
- Evaluation cadence is a module constant (100 steps), not a user parameter — `patience`
  already sets the stopping horizon; a second knob adds tuning surface without value
- The spec's optional "retrain on full data for the selected step count" was rejected:
  doubles training time to reclaim ≤ 20 % of the data, and the selected step count isn't
  transferable to a different training set
- `val_fraction` default stays `0.0` (off), not the spec's 0.15 — opt-in feature,
  existing behavior preserved exactly

---

## Session 10 — 2026-07-10

**Work done (adaptive Huber threshold + RMSE-comparable scaling, ADR-013):**
- **`huber_delta='auto'` (new default):** threshold recalibrated every 100 steps
  (`DELTA_EVERY`) from the current training residuals as `1.345 * 1.4826 * MAD`,
  floored at `1e-3` in scaled space. Lives in a non-trainable `tf.Variable` assigned
  between steps — no retracing of the compiled `train_step`, no gradient path through
  delta. Float values still fix the threshold. Verbose output now prints the current
  delta alongside the loss
- **Huber data term rescaled to `sqrt(2 * mean(huber))`** — a "pseudo-RMSE" that equals
  `sqrt(MSE)` exactly when all residuals are inside delta, so `loss_threshold`,
  `var_weight`, and `ellipsoid_weight` keep their meaning across loss types
- **Root cause (why):** ADR-009's fixed `delta=1.3` was calibrated against the *data*
  scale (unit variance), but the classical `1.345·σ` rule refers to the *residual*
  scale (~0.01–0.05 at convergence). Every residual sat in the quadratic zone, Huber
  degenerated to `0.5·MSE` (no robustness), and the `mean(huber)` value was ~40× below
  the RMSE scale, silently re-weighting the loss toward the regularizers
- **Verified end-to-end:** sine data + 2 gross outliers, flexible model
  (15 nodes, `var_weight=0.01`, 20k steps): R² vs *clean* truth — RMSE 0.9783,
  fixed delta=1.3 0.9792 (≈ no benefit, confirming the diagnosis), **auto 0.9992**.
  On clean data, huber-auto matches RMSE convergence step-for-step and final losses
  agree within a few percent (scale comparability confirmed)
- Validation: `huber_delta` must be `'auto'` or a positive number (`ValueError`
  otherwise) via `_validate_fit_params()`; recalibration uses training rows only so
  the ADR-012 validation loss stays a clean held-out measurement
- Docs: ADR-013 (+ ADR-009 superseded notes), algorithm overview, compressor map
  example page, README parameter table, premium example comment, CLAUDE.md
- Tests: huber-auto convergence (both APIs), fixed-float still supported, invalid
  values raise, loss-scale comparability vs RMSE, outlier-robustness regression test
  (huber-auto must beat RMSE vs clean truth). Full suite: 113 passed

**Work done (same session — Tukey biweight loss, ADR-014):**
- **`loss_type='tukey'` added** with `tukey_c='auto'` default: redescending M-estimator —
  influence drops to exactly zero beyond the rejection point, so gross outliers are
  discarded rather than downweighted. Reuses the entire ADR-013 machinery
  (`c = 4.685 * 1.4826 * MAD`, 100-step recalibration, `THRESHOLD_FLOOR`,
  `sqrt(2*mean(rho))` pseudo-RMSE scaling; `rho ≈ r²/2` for small residuals)
- Branchless TF form `rho = (c²/6)(1 - max(0, 1-(r/c)²)³)` — exact zero gradient beyond c
- `'auto'` doubles as annealing for the non-convex loss (starts effectively quadratic,
  tightens onto the noise floor); fixed float c supported but documented as
  not recommended (can reject most points at initialization and stall)
- Internals genericized: `thresh_t` / `refresh_thresh()` / `THRESHOLD_FLOOR` shared by
  Huber and Tukey; verbose prints `delta=` (Huber) or `c=` (Tukey)
- Verified (sine + 2 gross outliers, R² vs clean truth): RMSE 0.9895, Huber 0.9994,
  **Tukey 0.9996**; loss scale stays RMSE-comparable on clean data
- Docs: ADR-014, algorithm overview, compressor map page, README, premium example,
  docs/index.md (also fixed stale "L2 amplitude smoothing on by default" claim — removed
  by ADR-010), CLAUDE.md
- Tests: tukey convergence (both APIs), fixed c, invalid values, scale comparability,
  outlier-robustness regression extended to tukey, `tukey_c` in API-parity check.
  Full suite: 118 passed

**Work done (same session — loss function gallery):**
- `examples/loss_function_gallery.py` added (pattern follows
  `basis_function_gallery.py`): two-panel figure of the per-point loss rho(r) and
  influence psi(r) = d rho/dr for squared error / Huber / Tukey, residual axis in
  units of the robust scale sigma, thresholds drawn at the `HUBER_K` / `TUKEY_K`
  constants imported from `ebf.train`; saves to `docs/assets/loss_functions.png`
- Figure embedded in `docs/algorithm_overview.md` loss section with an explanation
  of the influence-function view (outlier pull: unbounded vs capped vs zero)
- Fixed pre-existing broken link in `docs/index.md` (`algorithm.md` →
  `algorithm_overview.md`) surfaced by `mkdocs build --strict`; strict build now clean

## Session 11 — 2026-07-10

**Work done (examples cleanup):**
- `examples/example_utils.py` added: shared 2-D test surface (`test_func` — polynomial
  swell + narrow diagonal ridge that stresses isotropic RBFs), `sample_test_func`,
  `make_grid`, `hull_mask`, and `contour_panel` helpers. Named without the `test_`
  prefix so pytest never collects it
- `examples/RBF_vs_EBF.py` restructured into comp_map-style sections on the class API;
  now prints and titles RMSE vs ground truth inside the convex hull (scipy RBF 32.10
  vs EBF 5.61 at 50 samples / 16 nodes) and saves `docs/assets/rbf_vs_ebf.png`
- `examples/loss_comparison.py` added: noisy (σ=8) and noisy+5-outlier scenarios each
  fit with rmse/huber/tukey, scored against clean truth. Results: noise-only —
  rmse 10.85 / huber 9.17 / tukey 23.70 (tukey's rejection also discards the genuine
  ridge); outliers — rmse 47.36 / huber 37.08 / tukey 20.02. Saves
  `docs/assets/loss_comparison.png`
- `examples/3d_fit.py` rewritten: was broken (imported nonexistent `TestFunc`, used
  plotly + functional API); now class API, shared test surface, `eval_grid`, and
  matplotlib 3-D surfaces (ground truth vs fit)
- `examples/1d_fit.py` switched to the class API (also demonstrates the combined-array
  `fit(data)` input form); `1d_fit_class_api.py` and `compressor_map.py` deletions
  from the previous session kept — stale references in README.md and CLAUDE.md updated
- `examples/basis_function_gallery.py` regrouped into "Increasing (global)" vs
  "Decreasing (local)" families via subfigures with per-group colors; functions
  registered but not yet assigned to a family land in a visible "Ungrouped" section
- `ebf/viz.py` `contour_plot_2d`: view now clamped to the data bounds — stray nodes far
  outside the data no longer stretch the axes and shrink the map (was making the
  premium example's contour plot unreadable)
- All six example scripts verified end-to-end headlessly (Agg backend); comp_map
  save/load roundtrip and CSV export confirmed. Note: scenario labels must stay ASCII —
  Windows cp1252 console can't print `σ` (crashed the first loss_comparison run)

---

## Session 12 — 2026-08-08

**Work done (ADR-015 — user-settable robust-loss tuning constant):**
- `huber_delta` / `tukey_c` gained a third accepted form: a sigma-relative spec such as
  `'2.5sigma'`. It keeps the full ADR-013 adaptive machinery and only replaces the
  built-in `HUBER_K` / `TUKEY_K`. Motivation: the parameters conflated *policy*
  (adaptive vs fixed) with *value*, so the one knob users actually want to tune — K —
  was reachable only by giving up adaptivity for a fixed absolute threshold, i.e. the
  exact failure mode ADR-013 was written to remove
- New `_parse_threshold(name, value, default_k) -> (adaptive, number)` in `ebf/train.py`
  serves as both parser and validator; `_validate_threshold` removed and
  `_validate_fit_params` now calls the parser. `refresh_thresh()` needed no change — it
  already read K from the enclosing scope
- Grammar `^\s*<float>\s*\*?\s*sigma\s*$`: whitespace- and `*`-tolerant, accepts
  scientific notation and both `'3.sigma'` and `'.5sigma'`. K must be > 0. Note the
  first regex used `\d*\.?\d+`, which rejected the trailing-dot form `'3.sigma'` —
  widened to `(?:\d+\.?\d*|\.\d+)`
- No public signature changed and the checkpoint format is untouched (these thresholds
  are training-time hyperparameters and were never serialized)
- Tests: `TestParseThreshold` (23 fast parser cases, no training), end-to-end sigma-spec
  fits through both `run()` and `EBF.fit()`, and an equivalence test asserting
  `'1.345sigma'` reproduces `'auto'` exactly for Huber
- Docs updated: ADR-015, `docs/loss_types.md` (three-form table + "Choosing K" guidance),
  `algorithm_overview.md`, `docs/examples/*.md`, README parameter table, CHANGELOG,
  CLAUDE.md, and the four examples that pass these parameters
- Examples were **not** re-run and figures were **not** regenerated: the algorithm is
  unchanged for every pre-existing input, and all four example edits are comments
