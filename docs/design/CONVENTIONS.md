# EBF Coding Conventions

## Data Contract

- Training array shape: `(n_points, n_dims+1)` — **last column is always the output variable**
- Input slice: `data[:, :-1]` → shape `(n_points, n_dims)`
- Output slice: `data[:, -1]` → shape `(n_points,)`
- Data is always standardized (zero-mean, unit-variance) before the model sees it
- `Scale` and `Offset` must always travel with saved model weights

## Tensor Shape Notation

Shape comments follow the pattern `# shape (dim1, dim2, ...)` using named dimensions:

| Symbol | Meaning |
|--------|---------|
| `n_points` | number of data or query points |
| `n_nodes` | number of EBF nodes (hyperparameter) |
| `n_dims` | number of input dimensions (not counting the output column) |

Key shapes through the forward pass:

```
X (input):               (n_points, n_dims)
Nodes:                   (n_nodes, n_dims)
deltas (x − v):          (n_points, n_nodes, n_dims)
L (upper-triangular):    (n_nodes, n_dims, n_dims)
A = LLᵀ + εI:           (n_nodes, n_dims, n_dims)
r² (distances):          (n_points, n_nodes)
a1, a2, a3 weights:      (n_nodes,)
b1 (linear weights):     (n_dims,)
b2 (constant bias):      (1,)
Y (output):              (n_points,)
```

## Variable Naming

| Name | Type | Description |
|------|------|-------------|
| `n_nodes` | `int` | number of EBF nodes |
| `n_dims` | `int` | number of input dimensions |
| `n_points` | `int` | number of data points |
| `data` | `ndarray (n_points, n_dims+1)` | raw input+output array |
| `Scale` | `ndarray (n_dims+1,)` | `1/std` per dimension for standardization |
| `Offset` | `ndarray (n_dims+1,)` | `mean` per dimension for standardization |
| `var_weight` | `float` | loss weight for node spread regularization |
| `eps` | `float` | numerical stability epsilon, default `1e-8` |
| `r2` | tensor `(n_points, n_nodes)` | squared non-Euclidean distance |
| `deltas` | tensor `(n_points, n_nodes, n_dims)` | point-to-node difference vectors |
| `dist_nodes` | tensor `(n_nodes, n_nodes)` | node-to-node pairwise distances (for regularization) |
| `a1`, `a2`, `a3` | tensor `(n_nodes,)` | basis function amplitude weights |
| `b1` | tensor `(n_dims,)` | linear trend weights |
| `b2` | tensor `(1,)` | constant bias |
| `file` | `str` | checkpoint path returned by `run()`, consumed by `run_points()` |

## Function and Class Naming

**Existing internal math functions** — keep `CamelCase` through Phase 2; rename to `snake_case` in Phase 4 if desired:
- `DeltaAll`, `NonEuclidDistance`, `ActFunc`, `LinearBias`, `EBF_Graph`

**Phase 2+ public API** — `snake_case`:
- `fit`, `predict`, `get_nodes`

**Phase 2+ module-level functions** — `snake_case`:
- `run`, `run_points`, `plot_correlation`

**Basis function registry keys** — `snake_case` strings:
- `'multiquadric'`, `'thin_plate'`, `'matern52'`, etc.

## File and Module Naming

| Context | Convention | Example |
|---------|------------|---------|
| Package modules | `snake_case.py` | `model.py`, `basis_functions.py` |
| Example scripts | `snake_case.py` | `compressor_map.py` |
| Test files | `test_<module>.py` | `test_model.py` |
| Documentation | `UPPERCASE.md` for project-level, `lowercase.md` for topic docs | `CLAUDE.md`, `ROADMAP.md` |

## Code Style

- 4-space indentation, no tabs
- Tensor shape documented in comment on the line that produces it: `# shape (n_points, n_nodes)`
- Commented-out alternative code blocks use `# name` label at the end of the line (existing pattern — preserve it)
- No hardcoded absolute paths anywhere; use relative paths or `pathlib`
- `if __name__ == "__main__":` guard required on all script files
- No bare `except:` clauses
