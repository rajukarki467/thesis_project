"""
Unit tests for Federated Spline Parameter and Delta Aggregation.

Verifies:
1. Exact-FedAvg Property: When all clients use (5, 3), aggregation under both
   'project_full' and 'project_delta' matches plain FedAvg to within 1e-6.
2. Idempotency / Zero Update: When clients perform zero local update, 'project_delta'
   leaves the server parameters exactly unchanged.
3. Shape Consistency: Aggregation across heterogeneous client configurations
   produces parameters strictly matching the server reference architecture.
4. Finite Values: Aggregation produces finite values without NaNs or Infs.
5. Delta Consistency: Linear scaling of local client deltas scales the server
   aggregated delta update linearly.
6. Sample-Size Weighted Aggregation: Verifies n_i / sum(n) weighting holds correctly
   for both homogeneous and heterogeneous client parameter shapes.
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch

from src.federated.server import aggregate_parameters
from src.models.fastkan_model import FastKANClassifier, project_spline_weights


@pytest.fixture
def reference_model():
    """Create reference FastKANClassifier with canonical (G=5, k=3)."""
    torch.manual_seed(42)
    model = FastKANClassifier(
        layers_hidden=[32, 16, 5],
        grid_size=5,
        spline_order=3,
    )
    return model


@pytest.fixture
def sample_server_parameters(reference_model):
    """Extract list of numpy ndarrays from reference model."""
    return [p.detach().cpu().numpy().copy() for p in reference_model.parameters()]


def test_exact_fedavg_property_homogeneous(sample_server_parameters):
    """
    Test that when all clients use the canonical (5, 3) grid,
    both 'project_full' and 'project_delta' equal plain FedAvg to within 1e-6.
    """
    rng = np.random.RandomState(42)
    num_clients = 4
    sample_counts = [100, 200, 150, 250]
    total_samples = sum(sample_counts)

    # Simulate client updates around server parameters
    client_results = []
    for cid in range(num_clients):
        client_p = [
            p + rng.normal(0, 0.05, size=p.shape).astype(p.dtype)
            for p in sample_server_parameters
        ]
        client_results.append((client_p, sample_counts[cid]))

    # Compute ground-truth FedAvg directly
    ground_truth = [np.zeros_like(p, dtype=np.float64) for p in sample_server_parameters]
    for params, n_samples in client_results:
        w = float(n_samples) / float(total_samples)
        for i, p in enumerate(params):
            ground_truth[i] += w * p.astype(np.float64)
    expected_fedavg = [p.astype(sample_server_parameters[i].dtype) for i, p in enumerate(ground_truth)]

    # 1. Test project_full
    res_full = aggregate_parameters(
        client_results=client_results,
        canonical_gk=(5, 3),
        server_parameters=sample_server_parameters,
        mode="project_full",
    )
    for res_p, exp_p in zip(res_full, expected_fedavg):
        max_diff = np.max(np.abs(res_p - exp_p))
        assert max_diff < 1e-6, f"project_full diff {max_diff} exceeded 1e-6"

    # 2. Test project_delta
    res_delta = aggregate_parameters(
        client_results=client_results,
        canonical_gk=(5, 3),
        server_parameters=sample_server_parameters,
        mode="project_delta",
    )
    for res_p, exp_p in zip(res_delta, expected_fedavg):
        max_diff = np.max(np.abs(res_p - exp_p))
        assert max_diff < 1e-6, f"project_delta diff {max_diff} exceeded 1e-6"


def test_idempotency_zero_update(sample_server_parameters):
    """
    Test that when clients have zero local updates (W_local == W_init),
    'project_delta' preserves the server parameters exactly, even with heterogeneous configs.
    """
    num_bases_to_gk = {5: (3, 2), 8: (5, 3), 10: (7, 3), 14: (10, 4)}
    client_configs = [(3, 2), (5, 3), (7, 3), (10, 4)]
    sample_counts = [100, 150, 200, 250]

    client_results = []
    client_initials = []
    for g, k in client_configs:
        c_init = []
        for p in sample_server_parameters:
            if p.ndim == 3:  # Spline weight
                p_proj = project_spline_weights(p, src_gk=(5, 3), dst_gk=(g, k))
                c_init.append(p_proj)
            else:
                c_init.append(p.copy())
        client_initials.append(c_init)
        # Zero local update: client final == client initial
        client_results.append((copy.deepcopy(c_init), sample_counts[len(client_results)]))

    aggregated = aggregate_parameters(
        client_results=client_results,
        canonical_gk=(5, 3),
        server_parameters=sample_server_parameters,
        client_initial_parameters=client_initials,
        client_gks=client_configs,
        mode="project_delta",
    )

    # Server parameters should be identical to within machine precision
    for agg_p, s_p in zip(aggregated, sample_server_parameters):
        max_err = np.max(np.abs(agg_p - s_p))
        assert max_err < 1e-6, f"Zero-update error {max_err} exceeded 1e-6"


def test_shape_consistency_heterogeneous(sample_server_parameters):
    """
    Test that aggregating heterogeneous client configurations produces
    parameters with shapes matching the canonical (5, 3) reference model.
    """
    rng = np.random.RandomState(99)
    client_configs = [(3, 2), (5, 3), (7, 4), (10, 2)]
    client_results = []

    for g, k in client_configs:
        c_params = []
        for p in sample_server_parameters:
            if p.ndim == 3:
                p_proj = project_spline_weights(p, src_gk=(5, 3), dst_gk=(g, k))
                c_params.append(p_proj + rng.normal(0, 0.01, size=p_proj.shape).astype(p.dtype))
            else:
                c_params.append(p + rng.normal(0, 0.01, size=p.shape).astype(p.dtype))
        client_results.append((c_params, 100))

    for mode in ["project_full", "project_delta"]:
        aggregated = aggregate_parameters(
            client_results=client_results,
            canonical_gk=(5, 3),
            server_parameters=sample_server_parameters,
            client_gks=client_configs,
            mode=mode,
        )
        assert len(aggregated) == len(sample_server_parameters)
        for agg_p, s_p in zip(aggregated, sample_server_parameters):
            assert agg_p.shape == s_p.shape
            assert agg_p.dtype == s_p.dtype


def test_finite_values(sample_server_parameters):
    """Test that aggregated parameters contain no NaN or Inf values."""
    rng = np.random.RandomState(77)
    client_configs = [(3, 2), (5, 3), (10, 4)]
    client_results = []

    for g, k in client_configs:
        c_params = []
        for p in sample_server_parameters:
            if p.ndim == 3:
                p_proj = project_spline_weights(p, src_gk=(5, 3), dst_gk=(g, k))
                c_params.append(p_proj + rng.uniform(-0.1, 0.1, size=p_proj.shape).astype(p.dtype))
            else:
                c_params.append(p + rng.uniform(-0.1, 0.1, size=p.shape).astype(p.dtype))
        client_results.append((c_params, 100))

    for mode in ["project_full", "project_delta"]:
        aggregated = aggregate_parameters(
            client_results=client_results,
            canonical_gk=(5, 3),
            server_parameters=sample_server_parameters,
            client_gks=client_configs,
            mode=mode,
        )
        for agg_p in aggregated:
            assert np.all(np.isfinite(agg_p))


def test_delta_consistency_linearity(sample_server_parameters):
    """
    Test that scaling the local delta by a scalar alpha scales the
    resulting server update by alpha linearly.
    """
    g, k = (3, 2)
    # Generate initial client weights
    c_init = []
    delta_base = []
    rng = np.random.RandomState(123)

    for p in sample_server_parameters:
        if p.ndim == 3:
            p_proj = project_spline_weights(p, src_gk=(5, 3), dst_gk=(g, k))
            c_init.append(p_proj)
            delta_base.append(rng.normal(0, 0.05, size=p_proj.shape).astype(p.dtype))
        else:
            c_init.append(p.copy())
            delta_base.append(rng.normal(0, 0.05, size=p.shape).astype(p.dtype))

    # Single client with delta
    client_res_1 = [([ini + d for ini, d in zip(c_init, delta_base)], 100)]
    agg_1 = aggregate_parameters(
        client_results=client_res_1,
        canonical_gk=(5, 3),
        server_parameters=sample_server_parameters,
        client_initial_parameters=[c_init],
        client_gks=[(g, k)],
        mode="project_delta",
    )
    diff_1 = [a - s for a, s in zip(agg_1, sample_server_parameters)]

    # Single client with 2.5 * delta
    alpha = 2.5
    client_res_2 = [([ini + alpha * d for ini, d in zip(c_init, delta_base)], 100)]
    agg_2 = aggregate_parameters(
        client_results=client_res_2,
        canonical_gk=(5, 3),
        server_parameters=sample_server_parameters,
        client_initial_parameters=[c_init],
        client_gks=[(g, k)],
        mode="project_delta",
    )
    diff_2 = [a - s for a, s in zip(agg_2, sample_server_parameters)]

    for d1, d2 in zip(diff_1, diff_2):
        np.testing.assert_allclose(d2, alpha * d1, atol=1e-5)


def test_sample_size_weighting(sample_server_parameters):
    """
    Verify sample-size-weighted aggregation (FedAvg weights n_i / sum n)
    is applied correctly when client param shapes differ.
    """
    g1, k1 = (3, 2)
    g2, k2 = (10, 4)

    c1_init = [project_spline_weights(p, (5, 3), (g1, k1)) if p.ndim == 3 else p.copy() for p in sample_server_parameters]
    c2_init = [project_spline_weights(p, (5, 3), (g2, k2)) if p.ndim == 3 else p.copy() for p in sample_server_parameters]

    # Client 1 makes zero update, Client 2 makes fixed update
    c1_final = copy.deepcopy(c1_init)
    c2_delta = [np.ones_like(p) * 0.1 for p in c2_init]
    c2_final = [ini + d for ini, d in zip(c2_init, c2_delta)]

    # Run with 100 samples for c1 and 300 samples for c2 (weight 0.75 for c2)
    res_75 = aggregate_parameters(
        client_results=[(c1_final, 100), (c2_final, 300)],
        canonical_gk=(5, 3),
        server_parameters=sample_server_parameters,
        client_initial_parameters=[c1_init, c2_init],
        client_gks=[(g1, k1), (g2, k2)],
        mode="project_delta",
    )

    # Run with 300 samples for c1 and 100 samples for c2 (weight 0.25 for c2)
    res_25 = aggregate_parameters(
        client_results=[(c1_final, 300), (c2_final, 100)],
        canonical_gk=(5, 3),
        server_parameters=sample_server_parameters,
        client_initial_parameters=[c1_init, c2_init],
        client_gks=[(g1, k1), (g2, k2)],
        mode="project_delta",
    )

    diff_75 = [r - s for r, s in zip(res_75, sample_server_parameters)]
    diff_25 = [r - s for r, s in zip(res_25, sample_server_parameters)]

    # diff_75 should be exactly 3 times diff_25 because 0.75 / 0.25 = 3.0
    for d75, d25 in zip(diff_75, diff_25):
        np.testing.assert_allclose(d75, 3.0 * d25, atol=1e-5)
