# Visualization Utilities

EBF includes a set of reusable plotting and data-export helpers in
`ebf.viz`. These cover the most common post-fit tasks: checking model
accuracy, visualizing the fitted surface, building evaluation grids for
arbitrary dimensions, and exporting lookup tables for use in other
programs.

All plot functions accept an optional `ax` argument so they can be
embedded in multi-panel figures, and they return `(fig, ax)` for further
customization.

## Plot Style

Colors are defined once as module-level constants in `ebf.viz`, so figures
from these helpers and from the example scripts read as one family. Import
them when adding your own overlays:

```python
from ebf.viz import DEFAULT_CMAP, SAMPLE_FACE, SAMPLE_EDGE, NODE_FACE, ACCENT

ax.scatter(x, y, c=SAMPLE_FACE, edgecolors=SAMPLE_EDGE)
```

| Constant | Value | Used for |
|----------|-------|----------|
| `DEFAULT_CMAP` | `'Blues_r'` | Filled contour surfaces |
| `SAMPLE_FACE` / `SAMPLE_EDGE` | white / near-black | Training points over a contour map |
| `NODE_FACE` / `NODE_EDGE` | muted orange | EBF node positions (triangle marker) |
| `POINT_FACE` / `POINT_EDGE` | mid blue | Scatter on white-background diagnostics |
| `ACCENT` | muted orange-red | Threshold and limit reference lines |
| `ERROR_CMAP` / `ERROR_SIZE` | `'Reds'` / 46 | Points shaded by absolute error (see the summary figure) |

The constraint driving these choices: the surface colormap is a sequential
blue ramp, so anything drawn *over* it must not be blue. Samples are white
with a dark edge to stay readable at both ends of the ramp, and nodes use
both a different color and a different marker so the two never blur
together. The diagnostic plots draw on white instead of over a colormap,
where white markers would vanish — hence a mid blue there.

## Convergence Plot

`ebf.convergence_plot()` draws the training loss curve from the
`(step, loss)` history recorded during `fit()`. Use it to judge whether
training has converged and to tune `steps`, `var_weight`, and
`loss_threshold` without scraping stdout.

```python
import ebf

model = ebf.EBF(n_nodes=8)
model.fit(X, y, steps=60000, loss_threshold=0.05)

# Pass the fitted model (its history_ attribute is used) ...
fig, ax = ebf.convergence_plot(model, loss_threshold=0.05)

# ... or the history array itself
fig, ax = ebf.convergence_plot(model.history_)
```

The functional API records history too: pass
`run(..., return_history=True)`'s fourth return value.

**Key options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `log_scale` | `True` | Logarithmic loss axis (the loss usually spans orders of magnitude) |
| `loss_threshold` | `None` | Draw the early-stopping threshold as a horizontal reference line |

The title reports the final loss and the number of steps actually run —
fewer steps than requested means `loss_threshold` or validation patience
triggered early stopping.

When the model was fitted with `val_fraction > 0`, the history carries a
third `val_loss` column and the plot shows the validation loss as a
second curve — the gap between the two curves opening up is the visual
signature of overfitting that early stopping guards against.

## Correlation Plot

`ebf.correlation_plot()` compares observed values against predictions.
It works for **any number of input dimensions** — all you need is a
pair of 1-D arrays.

```python
import ebf

model = ebf.EBF(n_nodes=8)
model.fit(X, y, steps=60000)

y_pred = model.predict(X)
fig, ax = ebf.correlation_plot(y, y_pred)
```

The plot shows:

- A scatter of data vs prediction
- A 1:1 reference line (black dashed)
- The R² value in the title

!!! note
    The R² is computed from the slope of a linear regression through the
    scatter, not from the coefficient of determination formula directly.
    For a well-fitted model the two are nearly identical.

## Residual Plot

`ebf.residual_plot()` plots the residuals (data minus prediction)
against the predictions, with a zero reference line and the RMSE in
the title. Like the correlation plot, it works for **any number of
input dimensions**.

```python
y_pred = model.predict(X)
fig, ax = ebf.residual_plot(y, y_pred)
```

Structure that the correlation plot compresses along its 1:1 line
becomes visible here:

- A **curve** in the residuals means systematic bias — too few nodes
  or over-smoothing
- A **funnel** shape means the error scales with the output level
- **Outliers** stand apart from the cloud — useful for judging whether
  a robust loss (`loss_type='huber'` or `'tukey'`) is warranted

Both plots accept `c`, `cmap` and `norm` to shade the markers by a
per-point value instead of the flat style colour — that is how
`summary_plot_3d()` puts every panel on one shared error scale. The
contour plot takes the same thing as `data_color` / `data_cmap` /
`data_norm` for its data overlay.

## 2-D Contour Plot

For models with exactly two input dimensions, `ebf.contour_plot_2d()`
produces a filled contour map with fine and coarse contour line overlays.

```python
fig, ax = ebf.contour_plot_2d(
    model, X, y,
    xlabel='Mass Flow',
    ylabel='Pressure Ratio',
    zlabel='Efficiency',
    show_nodes=True,
    show_data=True,
)
```

**Key options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_grid` | 400 | Grid resolution per axis |
| `mask` | `True` | Hide predictions outside the convex hull of the training data |
| `n_contourf` | 31 | Number of filled contour levels |
| `cmap` | `'Blues_r'` | Matplotlib colormap (cmasher's `cmr.*` maps also work) |
| `alpha` | 0.9 | Fill opacity |
| `show_data` | `True` | Overlay training points |
| `show_nodes` | `False` | Overlay EBF node positions |
| `data_color` | `None` | Per-point values to shade the data overlay by (with `data_cmap` / `data_norm`) |

When `mask=True` (the default), `scipy.interpolate.griddata` with
`method='linear'` is used to identify grid cells outside the convex hull
of the training data, and those cells are set to NaN so they appear
blank.  This prevents the plot from showing potentially misleading
extrapolation far from the data.

## 3-D Data Summary Figure

For 3-D data (two inputs, one output), `ebf.summary_plot_3d()` combines
the diagnostic plots above into a single figure: the fitted contour map
fills the full height on the left as the dominant element, with the
correlation plot, residual plot, and training convergence curve stacked
in a narrower column on the right.

```python
fig, axes = ebf.summary_plot_3d(
    model, X, y,
    xlabel='Mass Flow',
    ylabel='Pressure Ratio',
    zlabel='Efficiency',
    loss_threshold=0.05,
    show_nodes=True,
)
```

By default (`error_color=True`) every data point is shaded by its
absolute error with the `'Reds'` colormap — darker means a worse fit —
using one normalization shared by the contour, correlation and residual
panels, with the colorbar down the right-hand edge. Because the scale is
shared and anchored at zero, the same shade means the same error on every
panel, so a point that stands out in the residual plot can be located on
the map. The scale is set by the largest absolute error, so a single bad
point will pale the rest; pass `error_color=False` for flat white markers
instead, or `error_cmap` for a different ramp.

Extra keyword arguments (`n_grid`, `mask`, `cmap`, `show_nodes`, …) are
forwarded to `contour_plot_2d()`. The returned `axes` array holds the
`(contour, correlation, residual, convergence)` axes for further
customization.

## N-Dimensional Evaluation Grid

`ebf.eval_grid()` creates a rectilinear grid over any number of
dimensions and evaluates the model at every grid point.

```python
# Define bounds as a list of (min, max) per dimension
bounds = [(x1_min, x1_max), (x2_min, x2_max), (x3_min, x3_max)]

# Or derive them from the training data
bounds = list(zip(X.min(axis=0), X.max(axis=0)))

grid = ebf.eval_grid(model, bounds, n_points=50)
```

The `n_points` argument can be a single integer (same resolution for
every dimension) or a list of integers (one per dimension).

The returned dictionary contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `"coords"` | `(n_total, n_dims)` | Flat array of grid-point coordinates |
| `"predictions"` | `(n_total,)` | Model output at each grid point |
| `"grid_shape"` | tuple | Shape for reshaping predictions back to the grid |
| `"axes"` | list of 1-D arrays | Tick values along each dimension |

For 2-D inputs, you can reshape the predictions and use standard
matplotlib contour functions directly:

```python
z = grid["predictions"].reshape(grid["grid_shape"])
xx = grid["axes"][0]
yy = grid["axes"][1]
plt.contourf(xx, yy, z.T)
```

## Exporting Lookup Tables

`ebf.export_grid()` writes the output of `eval_grid()` to a file.  The
format is determined by the file extension:

### CSV (`.csv`) — recommended for interoperability

```python
ebf.export_grid(
    "output/efficiency_table.csv",
    grid,
    dim_names=['Mass Flow', 'Pressure Ratio'],
)
```

Produces a flat table like:

```
Mass Flow,Pressure Ratio,prediction
0.50,1.10,0.821
0.50,1.15,0.834
...
```

This is readable by virtually any program — Excel, MATLAB, C/C++
readers, Python `pandas.read_csv()`, etc.  For a 2-D model with
100 points per axis this produces a 10 000-row file (a few hundred KB).

### NPZ (`.npz`) — compact Python-native format

```python
ebf.export_grid("output/efficiency_table.npz", grid)
```

Stores `coords`, `predictions`, `grid_shape`, and per-dimension axis
arrays.  Reload with:

```python
data = np.load("output/efficiency_table.npz")
predictions = data["predictions"]
x1_axis = data["axis_0"]
x2_axis = data["axis_1"]
shape = tuple(data["grid_shape"])
```

NPZ is much more compact than CSV for large grids and preserves full
floating-point precision.

## Composing Multi-Panel Figures

Because every plot function accepts an `ax` argument, you can compose
them into a single figure:

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ebf.correlation_plot(y, y_pred, ax=ax1)
ebf.contour_plot_2d(model, X, y, ax=ax2, xlabel='X1', ylabel='X2', zlabel='Y')
plt.tight_layout()
plt.show()
```

## Full API

See the [API Reference](api.md#visualization-utilities) for complete
parameter documentation.
