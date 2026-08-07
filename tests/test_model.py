# -*- coding: utf-8 -*-
"""
Unit tests for core EBF math functions.

Run with: python -m pytest tests/test_model.py
"""
import numpy as np
import pytest
import tensorflow as tf

from ebf.model import DeltaAll, NonEuclidDistance, LinearBias, EBFModel
from ebf.basis_functions import BASIS_FUNCTIONS
from ebf.scaling import compute_scale_offset, scale_data, unscale_output, unscale_nodes


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

class TestScaling:
    def test_round_trip(self):
        data = np.random.rand(20, 3) * 10 + 5
        Scale, Offset = compute_scale_offset(data)
        scaled = scale_data(data, Scale, Offset)
        np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(scaled.std(axis=0), 1.0, atol=1e-10)

    def test_unscale_output(self):
        data = np.random.rand(10, 2)
        Scale, Offset = compute_scale_offset(data)
        scaled = scale_data(data, Scale, Offset)
        recovered = unscale_output(scaled[:, -1], Scale, Offset)
        np.testing.assert_allclose(recovered, data[:, -1], atol=1e-10)

    def test_unscale_nodes(self):
        data = np.random.rand(10, 3)
        Scale, Offset = compute_scale_offset(data)
        nodes_orig = np.random.rand(5, 2)
        nodes_scaled = (nodes_orig - Offset[:-1]) * Scale[:-1]
        recovered = unscale_nodes(nodes_scaled, Scale, Offset)
        np.testing.assert_allclose(recovered, nodes_orig, atol=1e-10)


# ---------------------------------------------------------------------------
# DeltaAll
# ---------------------------------------------------------------------------

class TestDeltaAll:
    def test_shape(self):
        x1 = tf.constant(np.random.rand(5, 3).astype(np.float32))
        x2 = tf.constant(np.random.rand(4, 3).astype(np.float32))
        result = DeltaAll(x1, x2).numpy()
        assert result.shape == (5, 4, 3)

    def test_single_point(self):
        """DeltaAll([0,0], [[1,0],[0,1]]) should give [[-1,0],[0,-1]]."""
        x1 = tf.constant([[0.0, 0.0]])
        x2 = tf.constant([[1.0, 0.0], [0.0, 1.0]])
        result = DeltaAll(x1, x2).numpy()
        np.testing.assert_allclose(result[0], [[-1.0, 0.0], [0.0, -1.0]], atol=1e-6)


# ---------------------------------------------------------------------------
# NonEuclidDistance
# ---------------------------------------------------------------------------

class TestNonEuclidDistance:
    def test_shape(self):
        deltas = tf.constant(np.random.rand(5, 4, 3).astype(np.float32))
        W = tf.constant(np.eye(3, dtype=np.float32)[np.newaxis].repeat(4, axis=0))
        result = NonEuclidDistance(deltas, W, 3).numpy()
        assert result.shape == (5, 4)

    def test_nonnegative(self):
        deltas = tf.constant(np.random.randn(10, 6, 2).astype(np.float32))
        W = tf.constant(np.random.randn(6, 2, 2).astype(np.float32))
        result = NonEuclidDistance(deltas, W, 2).numpy()
        assert np.all(result >= 0)

    def test_zero_at_node(self):
        """Distance from a node to itself should be zero."""
        deltas = tf.constant(np.zeros((1, 1, 2), dtype=np.float32))
        W = tf.constant(np.eye(2, dtype=np.float32)[np.newaxis])
        result = NonEuclidDistance(deltas, W, 2).numpy()
        assert result[0, 0] < 1e-5


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------

class TestBasisFunctions:
    @pytest.mark.parametrize("name,n_params", [
        (name, info[1]) for name, info in BASIS_FUNCTIONS.items()
    ])
    def test_output_shape(self, name, n_params):
        n_points, n_nodes = 7, 4
        r2 = tf.constant(np.abs(np.random.rand(n_points, n_nodes)).astype(np.float32) + 0.01)
        a1 = tf.constant(np.random.rand(n_nodes).astype(np.float32))
        a2 = tf.constant(np.random.rand(n_nodes).astype(np.float32))
        a3 = tf.constant(np.random.rand(n_nodes).astype(np.float32))
        eps = 1e-8
        fn, _ = BASIS_FUNCTIONS[name]
        if n_params == 1:
            out = fn(r2, a1, eps).numpy()
        elif n_params == 2:
            out = fn(r2, a1, a2, eps).numpy()
        else:
            out = fn(r2, a1, a2, a3, eps).numpy()
        assert out.shape == (n_points,), f"Shape mismatch for basis '{name}'"

    @pytest.mark.parametrize("name", list(BASIS_FUNCTIONS))
    def test_finite_output(self, name):
        fn, n_params = BASIS_FUNCTIONS[name]
        r2 = tf.constant(np.abs(np.random.rand(5, 3)).astype(np.float32) + 0.01)
        a1 = tf.constant(np.ones(3, dtype=np.float32))
        a2 = tf.constant(np.ones(3, dtype=np.float32))
        a3 = tf.constant(np.ones(3, dtype=np.float32))
        if n_params == 1:
            out = fn(r2, a1, 1e-8).numpy()
        elif n_params == 2:
            out = fn(r2, a1, a2, 1e-8).numpy()
        else:
            out = fn(r2, a1, a2, a3, 1e-8).numpy()
        assert np.all(np.isfinite(out)), f"Non-finite output for basis '{name}'"


# ---------------------------------------------------------------------------
# EBFModel
# ---------------------------------------------------------------------------

class TestEBFModel:
    def test_output_shape(self):
        n_dims, n_nodes, n_points = 2, 5, 10
        model = EBFModel(n_dims, n_nodes)
        x_data = np.random.rand(n_points, n_dims).astype(np.float32)
        y, dn, d = model(x_data)
        assert y.numpy().shape == (n_points,)
        assert dn.numpy().shape == (n_nodes, n_nodes)
        assert d.numpy().shape == (n_points, n_nodes)

    @pytest.mark.parametrize("basis", list(BASIS_FUNCTIONS))
    def test_all_bases_build(self, basis):
        n_dims, n_nodes = 2, 4
        model = EBFModel(n_dims, n_nodes, basis=basis)
        x_data = np.random.rand(6, n_dims).astype(np.float32)
        y, _, _ = model(x_data)
        assert y.numpy().shape == (6,)

    def test_variable_count_by_basis(self):
        """Verify that only the required a-weight variables are created."""
        for basis, (_, n_params) in BASIS_FUNCTIONS.items():
            model = EBFModel(2, 4, basis=basis)
            assert hasattr(model, 'a1')
            assert hasattr(model, 'a2') == (n_params >= 2), f"a2 mismatch for {basis}"
            assert hasattr(model, 'a3') == (n_params >= 3), f"a3 mismatch for {basis}"

    def test_unknown_basis_raises(self):
        with pytest.raises(ValueError):
            EBFModel(2, 4, basis='not_a_real_basis')
