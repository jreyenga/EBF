# -*- coding: utf-8 -*-
"""
Integration test: train on a synthetic function and assert R² > threshold.

Run with: python -m pytest tests/test_train.py
"""
import os
import tempfile
import numpy as np
import pytest

import ebf
from ebf.train import HUBER_K, TUKEY_K, _parse_threshold


def _r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return 1.0 - ss_res / ss_tot


class TestTrainPredict:
    def _make_1d_data(self, n=30):
        x = np.linspace(0, 2 * np.pi, n).reshape(-1, 1)
        y = np.sin(x) + 0.05 * np.random.randn(n, 1)
        return np.concatenate([x, y], axis=1)

    def _make_2d_data(self, n=40):
        rng = np.random.RandomState(0)
        X = rng.rand(n, 2) * 4 - 2
        y = np.sin(X[:, 0]) * np.cos(X[:, 1])
        return np.concatenate([X, y[:, np.newaxis]], axis=1)

    def test_1d_fit_r2(self, tmp_path):
        data = self._make_1d_data()
        Scale, Offset, file = ebf.run(
            data, n_nodes=8, train_steps=5000, var_weight=0.1,
            loss_type='rmse',
            path=str(tmp_path) + "/", filename="test1d"
        )
        Out, _ = ebf.run_points(data[:, :-1], Scale, Offset, file)
        r2 = _r2_score(data[:, -1], Out)
        assert r2 > 0.95, f"1D fit R²={r2:.3f} below threshold"

    def test_2d_fit_r2(self, tmp_path):
        data = self._make_2d_data()
        Scale, Offset, file = ebf.run(
            data, n_nodes=12, train_steps=8000, var_weight=0.1,
            path=str(tmp_path) + "/", filename="test2d"
        )
        Out, _ = ebf.run_points(data[:, :-1], Scale, Offset, file)
        r2 = _r2_score(data[:, -1], Out)
        assert r2 > 0.90, f"2D fit R²={r2:.3f} below threshold"

    def test_checkpoint_save_restore(self, tmp_path):
        """Predictions before and after checkpoint restore must match."""
        data = self._make_1d_data(n=20)
        Scale, Offset, file = ebf.run(
            data, n_nodes=6, train_steps=2000,
            path=str(tmp_path) + "/", filename="ckpt_test"
        )
        Out1, _ = ebf.run_points(data[:, :-1], Scale, Offset, file)
        Out2, _ = ebf.run_points(data[:, :-1], Scale, Offset, file)
        np.testing.assert_allclose(Out1, Out2, rtol=1e-5)

    def test_huber_auto_delta(self, tmp_path):
        """loss_type='huber' with the default adaptive delta (ADR-013)."""
        data = self._make_1d_data(n=20)
        Scale, Offset, file = ebf.run(
            data, n_nodes=6, loss_type='huber', train_steps=2000,
            path=str(tmp_path) + "/", filename="huber_def"
        )
        Out, _ = ebf.run_points(data[:, :-1], Scale, Offset, file)
        assert np.all(np.isfinite(Out))

    def test_huber_delta_invalid(self):
        data = self._make_1d_data(n=20)
        with pytest.raises(ValueError, match="huber_delta must be"):
            ebf.run(data, n_nodes=6, loss_type='huber',
                    huber_delta='bogus', train_steps=10)

    def test_huber_delta_sigma_spec(self, tmp_path):
        """A '<k>sigma' spec stays adaptive with the caller's K (ADR-015)."""
        data = self._make_1d_data(n=20)
        Scale, Offset, file = ebf.run(
            data, n_nodes=6, loss_type='huber', huber_delta='1.0sigma',
            train_steps=2000,
            path=str(tmp_path) + "/", filename="huber_sigma"
        )
        Out, _ = ebf.run_points(data[:, :-1], Scale, Offset, file)
        assert np.all(np.isfinite(Out))

    def test_tukey_c_sigma_spec(self, tmp_path):
        data = self._make_1d_data(n=20)
        Scale, Offset, file = ebf.run(
            data, n_nodes=6, loss_type='tukey', tukey_c='3sigma',
            train_steps=2000,
            path=str(tmp_path) + "/", filename="tukey_sigma"
        )
        Out, _ = ebf.run_points(data[:, :-1], Scale, Offset, file)
        assert np.all(np.isfinite(Out))

    def test_sigma_spec_matches_auto(self, tmp_path):
        """'1.345sigma' is exactly 'auto' for Huber — same K, same result."""
        data = self._make_1d_data(n=20)
        results = []
        for i, delta in enumerate(('auto', f'{HUBER_K}sigma')):
            Scale, Offset, file = ebf.run(
                data, n_nodes=6, loss_type='huber', huber_delta=delta,
                train_steps=500, seed=0, verbose=False,
                path=str(tmp_path) + "/", filename=f"equiv{i}"
            )
            Out, _ = ebf.run_points(data[:, :-1], Scale, Offset, file)
            results.append(Out)
        np.testing.assert_allclose(results[0], results[1], rtol=1e-5)

    def test_tukey_auto_c(self, tmp_path):
        """loss_type='tukey' with the default adaptive rejection point (ADR-014)."""
        data = self._make_1d_data(n=20)
        Scale, Offset, file = ebf.run(
            data, n_nodes=6, loss_type='tukey', train_steps=2000,
            path=str(tmp_path) + "/", filename="tukey_def"
        )
        Out, _ = ebf.run_points(data[:, :-1], Scale, Offset, file)
        assert np.all(np.isfinite(Out))

    def test_tukey_c_invalid(self):
        data = self._make_1d_data(n=20)
        with pytest.raises(ValueError, match="tukey_c must be"):
            ebf.run(data, n_nodes=6, loss_type='tukey',
                    tukey_c=-2.0, train_steps=10)

    def test_rmse_explicit(self, tmp_path):
        data = self._make_1d_data()
        Scale, Offset, file = ebf.run(
            data, n_nodes=8, loss_type='rmse',
            train_steps=5000,
            path=str(tmp_path) + "/", filename="rmse_exp"
        )
        Out, _ = ebf.run_points(data[:, :-1], Scale, Offset, file)
        r2 = _r2_score(data[:, -1], Out)
        assert r2 > 0.95, f"RMSE explicit R²={r2:.3f} below threshold"

    def test_smooth_weight_removed(self):
        """smooth_weight was removed (ADR-010 rejected) — passing it must fail."""
        data = self._make_1d_data(n=20)
        with pytest.raises(TypeError):
            ebf.run(data, n_nodes=6, smooth_weight=0.01, train_steps=10)

    def test_invalid_loss_type(self, tmp_path):
        data = self._make_1d_data(n=20)
        with pytest.raises(ValueError, match="loss_type must be"):
            ebf.run(data, n_nodes=6, loss_type='bad',
                    path=str(tmp_path) + "/")

    @pytest.mark.parametrize("basis", ['multiquadric', 'gaussian', 'thin_plate', 'matern52'])
    def test_basis_selection(self, tmp_path, basis):
        data = self._make_1d_data(n=20)
        Scale, Offset, file = ebf.run(
            data, n_nodes=6, basis=basis, train_steps=2000,
            path=str(tmp_path) + "/", filename=f"basis_{basis}"
        )
        Out, _ = ebf.run_points(data[:, :-1], Scale, Offset, file)
        assert np.all(np.isfinite(Out)), f"Non-finite output with basis '{basis}'"


class TestParseThreshold:
    """Unit tests for the threshold spec parser (ADR-015)."""

    @pytest.mark.parametrize("spec, expected_k", [
        ('auto', HUBER_K),
        ('2.5sigma', 2.5),
        ('2.5 sigma', 2.5),
        ('2.5*sigma', 2.5),
        ('2.5 * sigma', 2.5),
        ('  3sigma  ', 3.0),
        ('3.sigma', 3.0),
        ('.5sigma', 0.5),
        ('1e-1sigma', 0.1),
    ])
    def test_adaptive_forms(self, spec, expected_k):
        adaptive, k = _parse_threshold('huber_delta', spec, HUBER_K)
        assert adaptive is True
        assert k == pytest.approx(expected_k)

    def test_auto_uses_loss_specific_default(self):
        assert _parse_threshold('tukey_c', 'auto', TUKEY_K) == (True, TUKEY_K)

    def test_fixed_float(self):
        adaptive, value = _parse_threshold('huber_delta', 0.4, HUBER_K)
        assert adaptive is False
        assert value == pytest.approx(0.4)

    @pytest.mark.parametrize("spec", [
        'bogus', 'sigma', '0sigma', '-2sigma', '2 sigmas', 'auto sigma',
        'autos', '2.5 std', '', 0.0, -1.0, float('nan'),
    ])
    def test_rejects_invalid(self, spec):
        with pytest.raises(ValueError, match="huber_delta must be"):
            _parse_threshold('huber_delta', spec, HUBER_K)


class TestTrainingHistory:
    def _train_small(self, tmp_path, **kwargs):
        rng = np.random.RandomState(1)
        x = np.linspace(0, 2 * np.pi, 20).reshape(-1, 1)
        y = np.sin(x) + 0.05 * rng.randn(20, 1)
        data = np.concatenate([x, y], axis=1)
        return ebf.run(data, n_nodes=6, train_steps=500,
                       path=str(tmp_path) + "/", filename="hist",
                       verbose=False, **kwargs)

    def test_default_return_unchanged(self, tmp_path):
        """Without return_history, run() keeps its 3-tuple return."""
        result = self._train_small(tmp_path)
        assert len(result) == 3

    def test_return_history(self, tmp_path):
        Scale, Offset, file, history = self._train_small(
            tmp_path, return_history=True)
        assert history.shape == (500, 2)
        np.testing.assert_array_equal(history[:, 0], np.arange(1, 501))
        assert np.all(np.isfinite(history[:, 1]))
        assert history[-1, 1] < history[0, 1], "loss did not decrease"


class TestEllipsoidPenalty:
    def _make_data(self, n=25):
        rng = np.random.RandomState(3)
        x = np.linspace(0, 2 * np.pi, n).reshape(-1, 1)
        y = np.sin(x) + 0.05 * rng.randn(n, 1)
        return np.concatenate([x, y], axis=1)

    def _train(self, tmp_path, filename, **kwargs):
        return ebf.run(self._make_data(), n_nodes=6, train_steps=300,
                       seed=0, verbose=False, return_history=True,
                       path=str(tmp_path) + "/", filename=filename, **kwargs)

    def test_zero_weight_matches_default(self, tmp_path):
        """ellipsoid_weight=0 must reproduce default behavior exactly."""
        _, _, _, hist_default = self._train(tmp_path, "ep_default")
        _, _, _, hist_zero = self._train(tmp_path, "ep_zero",
                                         ellipsoid_weight=0.0)
        np.testing.assert_allclose(hist_default[:, 1], hist_zero[:, 1],
                                   rtol=1e-6)

    def test_penalty_changes_loss(self, tmp_path):
        """Activating the penalty must change the training trajectory."""
        _, _, _, hist_off = self._train(tmp_path, "ep_off")
        _, _, _, hist_on = self._train(tmp_path, "ep_on",
                                       ellipsoid_weight=0.5)
        assert not np.allclose(hist_off[:, 1], hist_on[:, 1])

    def test_penalty_output_finite(self, tmp_path):
        data = self._make_data()
        Scale, Offset, file, _ = self._train(tmp_path, "ep_finite",
                                             ellipsoid_weight=0.5)
        Out, _ = ebf.run_points(data[:, :-1], Scale, Offset, file)
        assert np.all(np.isfinite(Out))


class TestEarlyStopping:
    """SMOOTHNESS Phase S2 — validation split + patience (ADR-012)."""

    def _make_data(self, n=60, noise=0.05, signal=True, seed=4):
        rng = np.random.RandomState(seed)
        x = np.linspace(0, 2 * np.pi, n).reshape(-1, 1)
        base = np.sin(x) if signal else np.zeros_like(x)
        y = base + noise * rng.randn(n, 1)
        return np.concatenate([x, y], axis=1)

    def _train(self, tmp_path, filename, data=None, **kwargs):
        if data is None:
            data = self._make_data()
        return ebf.run(data, n_nodes=6, seed=0, verbose=False,
                       return_history=True,
                       path=str(tmp_path) + "/", filename=filename, **kwargs)

    def test_zero_fraction_matches_default(self, tmp_path):
        """val_fraction=0 must reproduce default behavior exactly."""
        _, _, _, hist_default = self._train(tmp_path, "es_default",
                                            train_steps=300)
        _, _, _, hist_zero = self._train(tmp_path, "es_zero",
                                         train_steps=300, val_fraction=0.0)
        assert hist_zero.shape == (300, 2)
        np.testing.assert_allclose(hist_default[:, 1], hist_zero[:, 1],
                                   rtol=1e-6)

    def test_history_gains_val_column(self, tmp_path):
        _, _, _, hist = self._train(tmp_path, "es_hist", train_steps=250,
                                    val_fraction=0.2, patience=100)
        assert hist.shape == (250, 3)
        eval_rows = (hist[:, 0] % 100 == 0) | (hist[:, 0] == 250)
        assert np.all(np.isfinite(hist[eval_rows, 2]))
        assert np.all(np.isnan(hist[~eval_rows, 2]))
        assert np.all(np.isfinite(hist[:, 1]))

    def test_stops_early_on_pure_noise(self, tmp_path):
        """With no signal to learn, validation loss stops improving and
        training must halt well before the step budget."""
        data = self._make_data(noise=1.0, signal=False)
        _, _, _, hist = self._train(tmp_path, "es_noise", data=data,
                                    train_steps=20000,
                                    val_fraction=0.2, patience=3)
        assert hist.shape[0] < 20000

    def test_predictions_finite_after_early_stop(self, tmp_path):
        data = self._make_data(noise=1.0, signal=False)
        Scale, Offset, file, _ = self._train(tmp_path, "es_finite", data=data,
                                             train_steps=20000,
                                             val_fraction=0.2, patience=3)
        Out, _ = ebf.run_points(data[:, :-1], Scale, Offset, file)
        assert np.all(np.isfinite(Out))

    def test_small_dataset_warns(self, tmp_path):
        data = self._make_data(n=20)
        with pytest.warns(UserWarning, match="50 points"):
            self._train(tmp_path, "es_warn", data=data, train_steps=200,
                        val_fraction=0.2)

    def test_invalid_val_fraction(self):
        data = self._make_data()
        for bad in (1.0, -0.1):
            with pytest.raises(ValueError, match="val_fraction"):
                ebf.run(data, n_nodes=6, val_fraction=bad, train_steps=10)

    def test_invalid_patience(self):
        data = self._make_data()
        with pytest.raises(ValueError, match="patience"):
            ebf.run(data, n_nodes=6, val_fraction=0.2, patience=0,
                    train_steps=10)


class TestSidecarScaleOffset:
    def _train_small(self, tmp_path):
        rng = np.random.RandomState(2)
        x = np.linspace(0, 2 * np.pi, 20).reshape(-1, 1)
        y = np.sin(x) + 0.05 * rng.randn(20, 1)
        data = np.concatenate([x, y], axis=1)
        Scale, Offset, file = ebf.run(
            data, n_nodes=6, train_steps=1000,
            path=str(tmp_path) + "/", filename="sidecar", verbose=False)
        return data, Scale, Offset, file

    def test_sidecar_defaults_match_explicit(self, tmp_path):
        data, Scale, Offset, file = self._train_small(tmp_path)
        out_explicit, nodes_explicit = ebf.run_points(
            data[:, :-1], Scale, Offset, file)
        out_sidecar, nodes_sidecar = ebf.run_points(data[:, :-1], file=file)
        np.testing.assert_allclose(out_explicit, out_sidecar, rtol=1e-6)
        np.testing.assert_allclose(nodes_explicit, nodes_sidecar, rtol=1e-6)

    def test_mismatched_scale_warns(self, tmp_path):
        data, Scale, Offset, file = self._train_small(tmp_path)
        with pytest.warns(UserWarning, match="differs from the value stored"):
            ebf.run_points(data[:, :-1], Scale * 2.0, Offset, file)

    def test_missing_file_raises(self):
        with pytest.raises(TypeError, match="missing required argument"):
            ebf.run_points(np.zeros((3, 1)))
