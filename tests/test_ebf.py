# -*- coding: utf-8 -*-
"""
Tests for the EBF class-based API.

Run with: python -m pytest tests/test_ebf.py
"""
import inspect

import numpy as np
import pytest

import ebf


def _r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def _make_1d_data(n=30):
    rng = np.random.RandomState(42)
    x = np.linspace(0, 2 * np.pi, n).reshape(-1, 1)
    y = np.sin(x).ravel() + 0.05 * rng.randn(n)
    return x, y


# ------------------------------------------------------------------
# Constructor
# ------------------------------------------------------------------

class TestConstructor:
    def test_defaults(self):
        m = ebf.EBF(n_nodes=5)
        assert m.n_nodes == 5
        assert m.basis == 'multiquadric'
        assert m.eps == 1e-8
        assert m._is_fitted is False
        assert m.history_ is None

    def test_custom_basis(self):
        m = ebf.EBF(n_nodes=5, basis='gaussian', eps=1e-6)
        assert m.basis == 'gaussian'
        assert m.eps == 1e-6

    def test_invalid_basis_raises(self):
        with pytest.raises(ValueError, match="Unknown basis"):
            ebf.EBF(n_nodes=5, basis='not_a_basis')


# ------------------------------------------------------------------
# Before fit — RuntimeError guards
# ------------------------------------------------------------------

class TestNotFitted:
    def test_predict_before_fit(self):
        m = ebf.EBF(n_nodes=5)
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.predict(np.zeros((3, 2)))

    def test_get_nodes_before_fit(self):
        m = ebf.EBF(n_nodes=5)
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.get_nodes()

    def test_save_before_fit(self, tmp_path):
        m = ebf.EBF(n_nodes=5)
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.save(str(tmp_path))


# ------------------------------------------------------------------
# Fit
# ------------------------------------------------------------------

class TestFit:
    def test_fit_separate_xy(self):
        x, y = _make_1d_data(n=20)
        m = ebf.EBF(n_nodes=6)
        result = m.fit(x, y, steps=500, verbose=False)
        assert result is m
        assert m._is_fitted is True

    def test_fit_combined_array(self):
        x, y = _make_1d_data(n=20)
        data = np.column_stack([x, y])
        m = ebf.EBF(n_nodes=6)
        m.fit(data, steps=500, verbose=False)
        assert m._is_fitted is True

    def test_fit_returns_self(self):
        x, y = _make_1d_data(n=20)
        m = ebf.EBF(n_nodes=6)
        assert m.fit(x, y, steps=500, verbose=False) is m

    def test_fit_bad_x_shape(self):
        m = ebf.EBF(n_nodes=5)
        with pytest.raises(ValueError):
            m.fit(np.zeros(10), np.zeros(10), steps=100, verbose=False)

    def test_fit_bad_y_shape(self):
        m = ebf.EBF(n_nodes=5)
        with pytest.raises(ValueError):
            m.fit(np.zeros((10, 2)), np.zeros((10, 1)), steps=100,
                  verbose=False)

    def test_fit_huber_auto_delta(self):
        """loss_type='huber' with the default adaptive delta (ADR-013)."""
        x, y = _make_1d_data(n=30)
        m = ebf.EBF(n_nodes=6)
        m.fit(x, y, steps=6000, loss_type='huber', seed=0, verbose=False)
        assert m._is_fitted is True
        r2 = _r2_score(y, m.predict(x))
        assert r2 > 0.90, f"Huber (auto delta) R²={r2:.3f} below threshold"

    def test_robust_loss_scale_matches_rmse(self):
        """ADR-013/014: sqrt(2*mean(rho)) keeps Huber and Tukey losses on
        the RMSE scale, so loss_threshold values carry over between loss
        types."""
        x, y = _make_1d_data(n=30)
        final = {}
        for lt in ('rmse', 'huber', 'tukey'):
            m = ebf.EBF(n_nodes=6).fit(
                x, y, steps=2000, loss_type=lt, seed=0, verbose=False)
            final[lt] = m.history_[-1, 1]
        for lt in ('huber', 'tukey'):
            assert 0.2 < final[lt] / final['rmse'] < 5.0, (
                f"{lt} loss {final[lt]:.4f} not on the RMSE scale "
                f"({final['rmse']:.4f})")

    def test_fit_rmse_explicit(self):
        x, y = _make_1d_data(n=20)
        m = ebf.EBF(n_nodes=6)
        m.fit(x, y, steps=500, loss_type='rmse', verbose=False)
        assert m._is_fitted is True

    def test_fit_smooth_weight_removed(self):
        """smooth_weight was removed (ADR-010 rejected) — passing it must fail."""
        x, y = _make_1d_data(n=20)
        m = ebf.EBF(n_nodes=6)
        with pytest.raises(TypeError):
            m.fit(x, y, steps=100, smooth_weight=0.1, verbose=False)

    def test_fit_stores_history(self):
        x, y = _make_1d_data(n=20)
        m = ebf.EBF(n_nodes=6)
        m.fit(x, y, steps=500, verbose=False)
        assert m.history_.shape == (500, 2)
        np.testing.assert_array_equal(m.history_[:, 0], np.arange(1, 501))
        assert np.all(np.isfinite(m.history_[:, 1]))
        assert m.history_[-1, 1] < m.history_[0, 1], "loss did not decrease"

    def test_fit_invalid_loss_type(self):
        x, y = _make_1d_data(n=20)
        m = ebf.EBF(n_nodes=6)
        with pytest.raises(ValueError, match="loss_type must be"):
            m.fit(x, y, steps=100, loss_type='invalid', verbose=False)

    def test_fit_huber_delta_custom(self):
        """A fixed float threshold is still supported (ADR-013)."""
        x, y = _make_1d_data(n=20)
        m = ebf.EBF(n_nodes=6)
        m.fit(x, y, steps=500, loss_type='huber', huber_delta=0.5,
              verbose=False)
        assert m._is_fitted is True

    def test_fit_huber_delta_sigma_spec(self):
        """'<k>sigma' keeps the adaptive threshold with a custom K (ADR-015)."""
        x, y = _make_1d_data(n=20)
        m = ebf.EBF(n_nodes=6)
        m.fit(x, y, steps=500, loss_type='huber', huber_delta='1.0sigma',
              verbose=False)
        assert m._is_fitted is True
        assert np.all(np.isfinite(m.predict(x)))

    def test_fit_tukey_c_sigma_spec(self):
        x, y = _make_1d_data(n=20)
        m = ebf.EBF(n_nodes=6)
        m.fit(x, y, steps=500, loss_type='tukey', tukey_c='3sigma',
              verbose=False)
        assert m._is_fitted is True
        assert np.all(np.isfinite(m.predict(x)))

    def test_fit_huber_delta_invalid(self):
        x, y = _make_1d_data(n=20)
        m = ebf.EBF(n_nodes=6)
        with pytest.raises(ValueError, match="huber_delta must be"):
            m.fit(x, y, steps=100, loss_type='huber', huber_delta='bogus',
                  verbose=False)
        with pytest.raises(ValueError, match="huber_delta must be"):
            m.fit(x, y, steps=100, loss_type='huber', huber_delta='0sigma',
                  verbose=False)
        with pytest.raises(ValueError, match="huber_delta must be"):
            m.fit(x, y, steps=100, loss_type='huber', huber_delta=-1.0,
                  verbose=False)

    def test_fit_tukey_auto_c(self):
        """loss_type='tukey' with the default adaptive rejection point
        (ADR-014) must converge on clean data."""
        x, y = _make_1d_data(n=30)
        m = ebf.EBF(n_nodes=6)
        m.fit(x, y, steps=6000, loss_type='tukey', seed=0, verbose=False)
        assert m._is_fitted is True
        r2 = _r2_score(y, m.predict(x))
        assert r2 > 0.90, f"Tukey (auto c) R²={r2:.3f} below threshold"

    def test_fit_tukey_c_custom(self):
        x, y = _make_1d_data(n=20)
        m = ebf.EBF(n_nodes=6)
        m.fit(x, y, steps=500, loss_type='tukey', tukey_c=5.0,
              verbose=False)
        assert m._is_fitted is True

    def test_fit_tukey_c_invalid(self):
        x, y = _make_1d_data(n=20)
        m = ebf.EBF(n_nodes=6)
        with pytest.raises(ValueError, match="tukey_c must be"):
            m.fit(x, y, steps=100, loss_type='tukey', tukey_c='bogus',
                  verbose=False)
        with pytest.raises(ValueError, match="tukey_c must be"):
            m.fit(x, y, steps=100, loss_type='tukey', tukey_c=0.0,
                  verbose=False)

    def test_fit_combined_too_few_cols(self):
        m = ebf.EBF(n_nodes=5)
        with pytest.raises(ValueError):
            m.fit(np.zeros((10, 1)), steps=100, verbose=False)

    def test_robust_loss_outlier_robustness(self):
        """ADR-013/014: adaptive Huber and Tukey must recover the clean
        surface from data with gross outliers better than RMSE, which
        warps toward them."""
        rng = np.random.RandomState(7)
        x = np.linspace(0, 2 * np.pi, 40).reshape(-1, 1)
        y_clean = np.sin(x).ravel()
        y = y_clean + 0.03 * rng.randn(40)
        y[10] += 0.8
        y[28] -= 0.8

        r2 = {}
        for lt in ('rmse', 'huber', 'tukey'):
            m = ebf.EBF(n_nodes=12).fit(x, y, steps=8000, loss_type=lt,
                                        var_weight=0.01, seed=0,
                                        verbose=False)
            r2[lt] = _r2_score(y_clean, m.predict(x))
        for lt in ('huber', 'tukey'):
            assert r2[lt] > 0.99, (
                f"{lt} R² vs clean truth {r2[lt]:.4f} too low")
            assert r2[lt] > r2['rmse'], (
                f"{lt} ({r2[lt]:.4f}) should beat RMSE ({r2['rmse']:.4f}) "
                "on outlier-contaminated data")


# ------------------------------------------------------------------
# Ellipsoid shape penalty (S3 / ADR-011)
# ------------------------------------------------------------------

class TestEllipsoidPenalty:
    def _fit_2d(self, steps=3000, **kwargs):
        rng = np.random.RandomState(0)
        X = rng.rand(40, 2) * 4 - 2
        y = np.sin(X[:, 0]) * np.cos(X[:, 1])
        m = ebf.EBF(n_nodes=12).fit(X, y, steps=steps, seed=0,
                                    verbose=False, **kwargs)
        return m, X, y

    @staticmethod
    def _frob_and_lmax(m):
        L = m._model.ellipsoid_factors().numpy()
        frob = np.mean(np.sum(L ** 2, axis=(1, 2)))
        A = L @ np.transpose(L, (0, 2, 1)) + 1e-6 * np.eye(L.shape[-1])
        lmax = np.mean(np.linalg.eigvalsh(A)[:, -1])
        return frob, lmax

    def test_penalty_shrinks_ellipsoids(self):
        """With the penalty on, the Frobenius norms and largest
        eigenvalues of the ellipsoid matrices must be measurably lower.
        (lambda_max is the sharpness mechanism — near-node curvature
        scales with a1*lambda(A), see ADR-010/ADR-011.  The condition
        number is scale-invariant, so a magnitude penalty does not
        bound it — that would need the Option A penalty.)"""
        m_off, X, y = self._fit_2d()
        m_on, _, _ = self._fit_2d(ellipsoid_weight=0.5)

        frob_off, lmax_off = self._frob_and_lmax(m_off)
        frob_on, lmax_on = self._frob_and_lmax(m_on)

        assert frob_on < frob_off, (
            f"Frobenius norm not reduced: {frob_on:.4f} >= {frob_off:.4f}")
        assert lmax_on < lmax_off, (
            f"Max eigenvalue not reduced: {lmax_on:.4f} >= {lmax_off:.4f}")
        assert np.all(np.isfinite(m_on.predict(X)))

    def test_mild_penalty_keeps_fit_quality(self):
        m, X, y = self._fit_2d(ellipsoid_weight=0.01, steps=8000)
        r2 = _r2_score(y, m.predict(X))
        assert r2 > 0.85, f"R²={r2:.3f} degraded too far with mild penalty"


# ------------------------------------------------------------------
# Early stopping with validation split (S2 / ADR-012)
# ------------------------------------------------------------------

class TestEarlyStopping:
    def test_fit_val_history_three_columns(self):
        rng = np.random.RandomState(5)
        x = np.linspace(0, 2 * np.pi, 60).reshape(-1, 1)
        y = np.sin(x).ravel() + 0.05 * rng.randn(60)
        m = ebf.EBF(n_nodes=6)
        m.fit(x, y, steps=250, val_fraction=0.2, patience=100,
              seed=0, verbose=False)
        assert m.history_.shape == (250, 3)
        assert np.all(np.isfinite(m.history_[:, 1]))
        eval_rows = (m.history_[:, 0] % 100 == 0) | (m.history_[:, 0] == 250)
        assert np.all(np.isfinite(m.history_[eval_rows, 2]))

    def test_fit_small_dataset_warns(self):
        x, y = _make_1d_data(n=20)
        m = ebf.EBF(n_nodes=6)
        with pytest.warns(UserWarning, match="50 points"):
            m.fit(x, y, steps=200, val_fraction=0.2, verbose=False)

    def test_fit_invalid_val_fraction(self):
        x, y = _make_1d_data(n=20)
        m = ebf.EBF(n_nodes=6)
        with pytest.raises(ValueError, match="val_fraction"):
            m.fit(x, y, steps=100, val_fraction=1.5, verbose=False)


# ------------------------------------------------------------------
# Predict / get_nodes — shapes and values
# ------------------------------------------------------------------

class TestPredictAndNodes:
    @pytest.fixture(autouse=True)
    def fitted_model(self):
        x, y = _make_1d_data(n=30)
        self.x = x
        self.y = y
        self.m = ebf.EBF(n_nodes=8).fit(x, y, steps=3000, verbose=False)

    def test_predict_shape(self):
        pred = self.m.predict(self.x)
        assert pred.shape == (30,)

    def test_predict_finite(self):
        pred = self.m.predict(self.x)
        assert np.all(np.isfinite(pred))

    def test_get_nodes_shape(self):
        nodes = self.m.get_nodes()
        assert nodes.shape == (8, 1)

    def test_get_nodes_finite(self):
        nodes = self.m.get_nodes()
        assert np.all(np.isfinite(nodes))

    def test_nodes_in_data_range(self):
        nodes = self.m.get_nodes()
        # Nodes should be roughly within the data domain (with some slack)
        assert np.all(nodes > -5.0)
        assert np.all(nodes < 15.0)


# ------------------------------------------------------------------
# Ellipsoids
# ------------------------------------------------------------------

class TestGetEllipsoids:

    def test_before_fit(self):
        m = ebf.EBF(n_nodes=5)
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.get_ellipsoids()

    def test_shape_and_properties(self):
        rng = np.random.default_rng(0)
        X = rng.random((40, 2)) * [100.0, 0.5] + [-50.0, 3.0]
        y = np.sin(X[:, 0] / 20) * 5 + X[:, 1] * 2
        m = ebf.EBF(n_nodes=5).fit(X, y, steps=300, verbose=False, seed=1)

        A = m.get_ellipsoids()
        assert A.shape == (5, 2, 2)
        assert np.all(np.isfinite(A))
        # Symmetric positive-definite (ADR-001)
        assert np.allclose(A, np.transpose(A, (0, 2, 1)))
        assert all(np.all(np.linalg.eigvalsh(a) > 0) for a in A)

    def test_matches_internal_distances(self):
        """A in original space must reproduce the model's scaled r^2.

        Guards the ``A_orig = S A_scaled S`` transform: anisotropic input
        ranges make an inverted or omitted scaling fail loudly.
        """
        from ebf.scaling import scale_data

        rng = np.random.default_rng(0)
        X = rng.random((40, 2)) * [100.0, 0.5] + [-50.0, 3.0]
        y = np.sin(X[:, 0] / 20) * 5 + X[:, 1] * 2
        m = ebf.EBF(n_nodes=5).fit(X, y, steps=300, verbose=False, seed=1)

        # r^2 from the public API, in original units
        d = X[:, None, :] - m.get_nodes()[None, :, :]
        r2_public = np.einsum('pnd,nde,pne->pn', d, m.get_ellipsoids(), d)

        # r^2 the model actually computes, in scaled units
        data = np.column_stack([X, y])
        X_scaled = scale_data(data, m._scale, m._offset)[:, :-1]
        r2_internal = m._model(X_scaled)[2].numpy()

        assert np.allclose(r2_public, r2_internal, rtol=1e-4, atol=1e-4)


# ------------------------------------------------------------------
# Quality — R-squared
# ------------------------------------------------------------------

class TestQuality:
    def test_1d_r2(self):
        x, y = _make_1d_data(n=30)
        m = ebf.EBF(n_nodes=8).fit(x, y, steps=5000, loss_type='rmse',
                                    verbose=False)
        pred = m.predict(x)
        r2 = _r2_score(y, pred)
        assert r2 > 0.95, f"1D R²={r2:.3f} below threshold"

    def test_2d_r2(self):
        rng = np.random.RandomState(0)
        X = rng.rand(40, 2) * 4 - 2
        y = np.sin(X[:, 0]) * np.cos(X[:, 1])
        m = ebf.EBF(n_nodes=12).fit(X, y, steps=8000, verbose=False)
        pred = m.predict(X)
        r2 = _r2_score(y, pred)
        assert r2 > 0.90, f"2D R²={r2:.3f} below threshold"


# ------------------------------------------------------------------
# Save / Load round-trip
# ------------------------------------------------------------------

class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        x, y = _make_1d_data(n=20)
        m = ebf.EBF(n_nodes=6).fit(x, y, steps=2000, verbose=False)
        pred_before = m.predict(x)

        file = m.save(str(tmp_path) + "/", filename='roundtrip')
        m2 = ebf.EBF.load(file)

        pred_after = m2.predict(x)
        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5)

    def test_load_restores_config(self, tmp_path):
        x, y = _make_1d_data(n=20)
        m = ebf.EBF(n_nodes=7, basis='gaussian', eps=1e-6)
        m.fit(x, y, steps=500, verbose=False)
        file = m.save(str(tmp_path) + "/", filename='cfg')

        m2 = ebf.EBF.load(file)
        assert m2.n_nodes == 7
        assert m2.basis == 'gaussian'
        assert m2.eps == 1e-6
        assert m2._is_fitted is True

    def test_save_after_load(self, tmp_path):
        """A loaded model (no optimizer) must be re-saveable."""
        x, y = _make_1d_data(n=20)
        m = ebf.EBF(n_nodes=6).fit(x, y, steps=500, verbose=False)
        pred = m.predict(x)

        file1 = m.save(str(tmp_path) + "/", filename='first')
        m2 = ebf.EBF.load(file1)
        file2 = m2.save(str(tmp_path) + "/", filename='second')

        m3 = ebf.EBF.load(file2)
        np.testing.assert_allclose(pred, m3.predict(x), rtol=1e-5)


# ------------------------------------------------------------------
# Backwards compatibility — old functional API still works
# ------------------------------------------------------------------

class TestBackwardsCompat:
    def test_run_and_run_points(self, tmp_path):
        x, y = _make_1d_data(n=20)
        data = np.column_stack([x, y])
        Scale, Offset, file = ebf.run(
            data, n_nodes=6, train_steps=2000,
            path=str(tmp_path) + "/", filename="compat")
        Out, Nodes = ebf.run_points(data[:, :-1], Scale, Offset, file)
        assert Out.shape == (20,)
        assert np.all(np.isfinite(Out))


# ------------------------------------------------------------------
# Parity — the functional and class APIs must train the same model
# ------------------------------------------------------------------

class TestAPIParity:
    def test_shared_defaults_match(self):
        """run() and EBF.fit() must agree on shared hyperparameter defaults."""
        run_params = inspect.signature(ebf.run).parameters
        fit_params = inspect.signature(ebf.EBF.fit).parameters
        for name in ('var_weight', 'ellipsoid_weight', 'loss_type',
                     'huber_delta', 'tukey_c', 'val_fraction', 'patience'):
            assert run_params[name].default == fit_params[name].default, (
                f"Default for '{name}' differs: run()="
                f"{run_params[name].default!r} vs "
                f"EBF.fit()={fit_params[name].default!r}")

    def test_same_seed_same_predictions(self, tmp_path):
        """Same data, seed, and hyperparameters -> same model from both APIs."""
        x, y = _make_1d_data(n=20)
        data = np.column_stack([x, y])

        Scale, Offset, file = ebf.run(
            data, n_nodes=6, train_steps=500, seed=0,
            path=str(tmp_path) + "/", filename="parity")
        out_run, _ = ebf.run_points(x, Scale, Offset, file)

        m = ebf.EBF(n_nodes=6).fit(x, y, steps=500, seed=0, verbose=False)
        out_fit = m.predict(x)

        np.testing.assert_allclose(out_run, out_fit, rtol=1e-4)
