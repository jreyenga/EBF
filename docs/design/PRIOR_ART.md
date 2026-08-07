# Prior Art and Related Work

**Purpose:** an honest record of where the EBF idea sits in the existing literature, so that
claims made about this library are accurate and so that future work can build on what has
already been done elsewhere.

**Summary in one line:** the core idea — replacing the scalar Euclidean radius of an RBF with
a per-node positive-definite matrix — is *not* novel and has carried the name "elliptical
basis function" since at least 2000. What is distinctive here is the specific combination of
training scheme, regularization, and basis-function generality, plus the absence of any
comparable open-source Python implementation.

**Scope caveat:** this survey is the result of a targeted literature search, not an exhaustive
prior-art review. It is adequate for setting honest expectations in documentation. It is
**not** adequate as a patentability opinion — that would require systematic Google Scholar
sweeps and a patent-database search.

---

## 1. The core concept is established prior art

### 1.1 Elliptical basis function networks (the name is taken)

Mak & Kung introduced full per-node covariance matrices into RBF networks and called the
result *elliptical basis function* networks:

> M. W. Mak and S. Y. Kung, "Estimation of elliptical basis function parameters by the EM
> algorithm with application to speaker verification," *IEEE Transactions on Neural
> Networks*, vol. 11, no. 4, pp. 961–969, 2000.
> <https://ieeexplore.ieee.org/document/857775/>

A companion comparative study exists as well ("Elliptical basis function networks and radial
basis function networks for speaker verification: a comparative study"). This line of work
was pursued through the speaker-verification and speech-processing literature for years.

**Relationship to this library:** the model equation is the same generalization. The
differences are in how the ellipsoids are obtained. Mak & Kung fit them with EM (the network
is effectively a Gaussian mixture model with a discriminative output layer); this library
learns them by end-to-end gradient descent on the regression loss.

### 1.2 Anisotropic RBF interpolation (numerical analysis)

> G. Casciola, D. Lazzaro, L. B. Montefusco, S. Morigi, "Shape preserving surface
> reconstruction using locally anisotropic radial basis function interpolants," *Computers &
> Mathematics with Applications*, vol. 51, no. 8, pp. 1185–1198, 2006.
> <https://www.sciencedirect.com/science/article/pii/S0898122106000721>

Replaces the Euclidean norm with a local metric per center to preserve edges, flat regions,
and corners when reconstructing surfaces from unorganized point sets.

**Relationship to this library:** their metric is *derived analytically* from the local
distribution of points (local PCA / density), not learned by optimization. This is the closest
prior work in spirit — anisotropy for scattered-data interpolation quality — but the mechanism
for choosing the anisotropy is entirely different.

### 1.3 Ellipsoidal RBF neural networks (jointly optimized)

> Y. Hu et al., "Molecular Sparse Representation by 3D Ellipsoid Radial Basis Function Neural
> Networks via L1 Regularization," *J. Chem. Inf. Model.*, 2020.
> <https://arxiv.org/abs/2005.05307>

> "Sparse Ellipsoidal Radial Basis Function Network for Point Cloud Surface Representation,"
> 2025. <https://arxiv.org/html/2505.02350v1>

Both jointly optimize weights, centers, shapes, and orientations of ellipsoidal RBFs — the
same set of free parameters this library trains.

**Relationship to this library:** closest prior work on the *training* axis. Differences:
they optimize with quasi-Newton methods (sOWL-QN) rather than Adam/autodiff; they target
implicit surface / density-field fitting rather than general scattered-data regression; and
they use L1 sparsity as the structural regularizer where this library uses a node-spread
penalty (see §2.1).

### 1.4 Mahalanobis-distance RBF networks

> "Using a Mahalanobis-Like Distance to Train Radial Basis Neural Networks," *IWANN 2005*.
> <https://link.springer.com/chapter/10.1007/11494669_32>

Evolves a generalized Mahalanobis-structured metric for RBF neurons using a genetic
algorithm. A broader scattered literature exists on learning RBF metrics, some of it using
exactly the Cholesky parameterization described in §1.6.

### 1.5 Compact elliptical basis functions

> R. Southern, "Compact elliptical basis functions for surface reconstruction,"
> Technical Report TR-NCCA-2011-01, Bournemouth University, 2011.
> <https://eprints.bournemouth.ac.uk/17797/>

Surface reconstruction from point clouds using compactly-supported EBFs, built top-down.
Confirms the term "elliptical basis function" was in use in graphics as well as in speech.

### 1.6 The `LLᵀ` parameterization is a standard device

Guaranteeing positive-definiteness by optimizing an unconstrained triangular factor rather
than the matrix itself is a long-standing, widely-used trick (Cholesky parameterization). It
appears throughout covariance estimation, Gaussian process hyperparameter learning, and
variational inference. **ADR-001 should be read as a sound engineering choice, not an
invention.**

### 1.7 The strongest modern parallel: 3D Gaussian Splatting

3D Gaussian Splatting (2023 onward) is mathematically the closest living relative of this
algorithm, despite coming from an unrelated field:

- thousands of primitives, each with its own **learned anisotropic covariance**
- covariance factored (rotation × diagonal scale) specifically to **guarantee
  positive-semi-definiteness under gradient descent**
- **all parameters trained end-to-end** by gradient descent on a reconstruction loss

See also DARB-Splatting, which explicitly generalizes splatting to decaying anisotropic radial
basis functions: <https://arxiv.org/pdf/2501.12369>

**Relationship to this library:** independent convergence on the same core construction.
Useful as validation that the approach is sound, and as a source of ideas — the splatting
literature has done substantial work on initialization, densification/pruning of primitives,
and fast evaluation that may transfer to node placement here.

---

## 2. What appears to be distinctive here

No direct precedent was found for the following. These are the defensible claims.

### 2.1 Metric-aware node-spread regularization (ADR-002)

The `var_weight · 1/var(dist_nodes)` term, where pairwise node distances are measured in the
**learned non-Euclidean metric** rather than in Euclidean space.

This has no analog in the surveyed work because the problem it solves does not arise there:

| Prior work | Why it doesn't need this |
|---|---|
| Mak & Kung (EM) | Centers come from clustering; collapse is structurally impossible |
| Casciola et al. | Centers are the data points; not free parameters |
| ERBFNN (2020/2025) | Uses L1 sparsity to control structure instead |
| Gaussian Splatting | Uses explicit densify/prune heuristics instead |

The dual role documented in `ALGORITHM.md` — collapse prevention *and* the primary smoothness
control, via the optimizer's compensating reduction of the distance scale — also appears to be
an original observation.

### 2.2 Arbitrary basis functions over the Mahalanobis radius

The EBF literature is effectively Gaussian-only, because it descends from GMM/EM formulations
where the per-node matrix has a probabilistic interpretation as a covariance. This library
drops that interpretation and treats Aᵢ as a pure metric, which frees it to apply *any* basis
function — including conditionally positive-definite **growing** families such as
`multiquadric` (the default) and `thin_plate` — on top of a learned per-node ellipsoid.

The anisotropic-interpolation literature (§1.2) does use non-Gaussian bases, but with
analytically-derived rather than learned metrics.

### 2.3 End-to-end gradient training for general engineering regression

Joint autodiff training of centers, ellipsoids, amplitudes, and a global linear trend, aimed
at scattered *regression* on noisy physical measurements — combined with robust losses
(Huber/Tukey, ADR-013/ADR-014), adaptive thresholds, and early stopping. The individual
ingredients are all known; the assembly targeted at engineering surface fitting is not
something the surveyed work does.

### 2.4 The tooling gap is real

No mainstream Python package offers learned anisotropic RBF interpolation:

| Package | Anisotropy support |
|---|---|
| `scipy.interpolate.RBFInterpolator` | None — isotropic only |
| `scipy.interpolate.Rbf` (legacy) | None — isotropic only |
| [`treverhines/RBF`](https://github.com/treverhines/RBF) | None — isotropic only |

The published academic implementations of §1.3 are research code tied to their specific
domains. A `fit` / `predict` library for engineers, with per-node learned ellipsoids, robust
losses, save/load, and visualization, does not otherwise exist as far as this search found.

---

## 3. How to describe this project honestly

**Do not claim:** a novel method, a new mathematical concept, or the invention of elliptical
basis functions.

**Do claim:** a novel *combination* — metric-aware node-spread regularization, arbitrary basis
function families over learned per-node ellipsoids, and robust end-to-end gradient training —
delivered as (as far as is known) the first practical open-source Python implementation of
learned anisotropic RBF interpolation for engineering data.

Historical note, for the record: the core algorithm here was developed independently around
2019–2020 without knowledge of the literature above. Independent rediscovery of an idea that
several separate fields converged on is a point in the idea's favour.

---

## 4. Open leads for a deeper search

If a rigorous prior-art review is ever needed:

- Google Scholar: "metric learning radial basis function", "anisotropic kernel interpolation",
  "adaptive Mahalanobis kernel regression"
- Forward-citation search on Mak & Kung (2000) and Casciola et al. (2006)
- USPTO / EPO patent search — several near-miss patents surfaced on RBF-with-covariance for
  signal separation and anomaly detection
- The kriging / geostatistics literature on **anisotropic variograms**, which solves a closely
  related problem with different vocabulary and was not surveyed here
