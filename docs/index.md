# EBF — Elliptical Basis Functions

EBF is an interpolation library that fits smooth surfaces through scattered data points.
It generalizes the well-known Radial Basis Function (RBF) approach by giving each
interpolation node its own stretchable, rotatable ellipsoidal influence zone — so the
model can adapt to data that varies more quickly in some directions than others.

Built on TensorFlow 2, EBF is designed for engineering datasets: compressor maps,
geotechnical measurements, aerodynamic surfaces, and similar problems where you have
a moderate number of irregularly spaced measurements and need a smooth, continuous
prediction surface.

![RBF vs EBF on a ridged test surface](assets/rbf_vs_ebf.png)

*50 scattered samples of a surface with a narrow diagonal ridge. Scipy's RBF places a
center at every sample (50 of them) and still rings around the ridge — RMSE 32.10. EBF
uses 16 learned nodes and follows it — RMSE 5.61.*

## What Makes It Different

A conventional RBF measures distance from every center with one shared, circular metric,
so a center can only ever cover a round patch. An EBF node carries its own
positive-definite matrix, and the optimizer is free to stretch and rotate it during
training.

![How EBF nodes adapt to the data](assets/node_ellipsoids.png)

*Three nodes fit to two narrow ridges at +30° and −40°. Two nodes elongate and turn onto
a ridge each; the third stays broad and carries the background. Nothing instructs the
model to do this — it falls out of minimizing fit error.*

### Why this matters more as dimensions grow

RBF interpolation carries an assumption inherited from the spatial problems it was
invented for: that the input space is **Euclidean**, so distance means something. For
latitude and longitude that holds. But when your axes are speed, power, temperature,
pressure, and price, there is no common unit and therefore no natural notion of distance
— how far is 10 kPa from 3 °C? Standardizing each axis to unit variance (which EBF does
automatically) supplies a default answer, but it is a guess, not a physical fact.

The per-node ellipsoid is that guess made learnable. Because each matrix is full and
symmetric rather than a per-axis scaling, it can rotate as well as stretch — capturing
that two variables act *together*, not merely that one matters more than another.

!!! note "The examples are 2-D; the method is not"
    Two inputs and one output are simply what a contour plot can display. Fitting,
    prediction, evaluation grids, and lookup-table export are all dimension-agnostic —
    only `contour_plot_2d` and `summary_plot_3d` require exactly two inputs. Above that,
    use `correlation_plot` and `residual_plot` to judge fit quality, and sweep two
    variables at a time with `eval_grid` to view slices.

## Installation

```bash
# Clone and install in development mode
git clone https://github.com/jreyenga/EBF.git
cd EBF
poetry install
```

Or with pip:

```bash
pip install -e .
```

## Quickstart

```python
import numpy as np
import ebf

# Create some 1D data
X = np.linspace(0, 2 * np.pi, 30).reshape(-1, 1)
y = np.sin(X).ravel()

# Fit a model
model = ebf.EBF(n_nodes=8)
model.fit(X, y, steps=5000)

# Predict at new points
X_new = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
y_pred = model.predict(X_new)
```

## Key Features

- **Ellipsoidal nodes** — each node learns its own directional influence, unlike
  standard RBF which uses fixed spherical distances
- **12 basis functions** — from multiquadric and Gaussian to Matern kernels;
  see the [Basis Functions](basis_functions.md) reference
- **Automatic data scaling** — inputs and outputs are standardized internally
  so you don't need to normalize your data
- **Noise-robust training** — optional Huber loss reduces outlier influence,
  and Tukey biweight loss rejects gross outliers entirely; both
  self-calibrate to the residual noise floor
- **Simple API** — `fit()`, `predict()`, `get_nodes()`, `save()`, `load()`
- **Save and reload** — checkpoint your trained model and restore it later
- **Visualization utilities** — correlation plots, 2-D contour maps,
  n-dimensional evaluation grids, and CSV/NPZ lookup-table export

## Next Steps

- [How the Algorithm Works](algorithm_overview.md) — a plain-English explanation of what
  EBF does under the hood
- [Basis Functions](basis_functions.md) — gallery and reference for all available
  basis functions
- [Loss Types](loss_types.md) — handling noise and outliers with Huber and Tukey losses
- [Visualization Utilities](visualization.md) — correlation plots, contour maps,
  evaluation grids, and lookup-table export
- [API Reference](api.md) — full parameter documentation
- [RBF vs EBF](examples/rbf_vs_ebf.md) — the headline comparison, and why it works
- [Compressor Map Example](examples/compressor_map.md) — a complete walkthrough
  fitting a 2D engineering dataset
