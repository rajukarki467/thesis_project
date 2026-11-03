"""
Unit tests for Heterogeneous Edge Fleet Configuration, Tiers, Masks, and Profiles.

Verifies:
1. Parameter count formula (592 * (G + k) + 613) matches FastKANClassifier for all 12 configs.
2. Hard action masking per tier: illegal actions are strictly False, allowed actions are True.
3. Projection round-trip relative L2 error is finite for all 12x12 config pairs.
4. Exact FedAvg equivalence when all 8 clients use the server reference config (difference < 1e-6).
5. Dirichlet partitions for 8 clients (alpha in {0.1, 0.5, 1.0}): non-empty, sum to 75,795, deterministic.
6. Fleet profile generation: deterministic by seed, memory budget fits all allowed configs.
7. Log-scale normalization: maps wide-ranging bandwidth and latency to [-1.0, 1.0] without NaN/Inf.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest
import torch

from src.environment.fleet import (
    ACTION_TO_INDEX,
    ACTION_TUPLES,
    ALL_GRID_SIZES,
    ALL_SPLINE_ORDERS,
    FleetConfig,
    compute_param_count,
    generate_fleet_profiles,
    get_tier_one_hot,
    normalize_log_scale,
)
from src.federated.server import aggregate_parameters
from src.models.fastkan_model import FastKANClassifier, project_spline_weights
from src.partition import dirichlet_partition


# ==============================================================================
# 1. Parameter Formula Tests (all 12 configs)
# ==============================================================================

@pytest.mark.parametrize("grid_size", ALL_GRID_SIZES)
@pytest.mark.parametrize("spline_order", ALL_SPLINE_ORDERS)
def test_parameter_count_formula_matches_fastkan(grid_size, spline_order):
    """
    Verify params = 592 * (G + k) + 613 against live FastKANClassifier([32, 16, 5])
    for all 12 configurations.
    """
    expected = compute_param_count(grid_size, spline_order)
    model = FastKANClassifier(
        layers_hidden=[32, 16, 5],
        grid_size=grid_size,
        spline_order=spline_order,
    )
    actual = model.get_parameter_count()
    assert actual == expected, (
        f"Mismatch for (G={grid_size}, k={spline_order}): actual={actual} vs expected={expected}"
    )


def test_specific_parameter_values():
    """Verify specific headline configurations explicitly."""
    assert compute_param_count(3, 2) == 3573
    assert compute_param_count(5, 3) == 5349
    assert compute_param_count(10, 4) == 8901


# ==============================================================================
# 2. Hard Action Masking Tests
# ==============================================================================

def test_tier_masks_conform_to_spec():
    fleet_cfg = FleetConfig()

    # Tier 1 (Clients 0, 1): mask G > 5 or k > 3 -> allowed G in {3, 5}, k in {2, 3} (4 configs)
    tier1_allowed = {(3, 2), (3, 3), (5, 2), (5, 3)}
    for cid in [0, 1]:
        mask = fleet_cfg.get_action_mask_for_client(cid)
        assert mask.shape == (12,)
        assert np.sum(mask) == 4
        for idx, gk in enumerate(ACTION_TUPLES):
            if gk in tier1_allowed:
                assert mask[idx] is True or mask[idx] == 1
            else:
                assert mask[idx] is False or mask[idx] == 0

    # Tier 2 (Clients 2, 3, 4): mask G > 7 -> allowed G in {5, 7}, k in {2, 3, 4} (6 configs)
    tier2_allowed = {(5, 2), (5, 3), (5, 4), (7, 2), (7, 3), (7, 4)}
    for cid in [2, 3, 4]:
        mask = fleet_cfg.get_action_mask_for_client(cid)
        assert mask.shape == (12,)
        assert np.sum(mask) == 6
        for idx, gk in enumerate(ACTION_TUPLES):
            if gk in tier2_allowed:
                assert mask[idx] is True or mask[idx] == 1
            else:
                assert mask[idx] is False or mask[idx] == 0

    # Tier 3 (Clients 5, 6): mask G < 7 or k < 3 -> allowed G in {7, 10}, k in {3, 4} (4 configs)
    tier3_allowed = {(7, 3), (7, 4), (10, 3), (10, 4)}
    for cid in [5, 6]:
        mask = fleet_cfg.get_action_mask_for_client(cid)
        assert mask.shape == (12,)
        assert np.sum(mask) == 4
        for idx, gk in enumerate(ACTION_TUPLES):
            if gk in tier3_allowed:
                assert mask[idx] is True or mask[idx] == 1
            else:
                assert mask[idx] is False or mask[idx] == 0

    # Tier 4 (Client 7): Hospital server -> only (10, 4) allowed (1 config)
    tier4_allowed = {(10, 4)}
    mask7 = fleet_cfg.get_action_mask_for_client(7)
    assert mask7.shape == (12,)
    assert np.sum(mask7) == 1
    for idx, gk in enumerate(ACTION_TUPLES):
        if gk in tier4_allowed:
            assert mask7[idx] is True or mask7[idx] == 1
        else:
            assert mask7[idx] is False or mask7[idx] == 0


def test_full_action_mask_shape():
    fleet_cfg = FleetConfig()
    full_mask = fleet_cfg.get_full_action_mask()
    assert full_mask.shape == (8 * 12,)
    # Sum of allowed actions: 4 + 4 + 6 + 6 + 6 + 4 + 4 + 1 = 35 allowed actions
    assert np.sum(full_mask) == (4*2 + 6*3 + 4*2 + 1)


# ==============================================================================
# 3. Projection Round-Trip L2 Error Tests (12x12 pairs)
# ==============================================================================

def test_projection_roundtrip_finite_all_pairs():
    torch.manual_seed(42)
    for src in ACTION_TUPLES:
        for dst in ACTION_TUPLES:
            w_orig = torch.randn(8, 16, src[0] + src[1])
            w_proj = project_spline_weights(w_orig, src_gk=src, dst_gk=dst)
            w_back = project_spline_weights(w_proj, src_gk=dst, dst_gk=src)

            assert torch.all(torch.isfinite(w_proj)), f"NaN/Inf projecting {src} -> {dst}"
            assert torch.all(torch.isfinite(w_back)), f"NaN/Inf projecting {dst} -> {src}"

            # Diagonal identity check
            if src == dst:
                diff = torch.norm(w_orig - w_back).item()
                assert diff < 1e-6, f"Identity failed for {src}"


def test_projection_roundtrip_csv_exists_and_valid():
    csv_path = Path("data/processed/metadata/spline_projection_roundtrip_l2_errors.csv")
    assert csv_path.exists(), "Projection error CSV must be saved on disk."
    df = pd.read_csv(csv_path, index_col=0)
    assert df.shape == (12, 12)
    assert not df.isna().any().any()
    assert np.all(np.diag(df.values) < 1e-6)


# ==============================================================================
# 4. Exact FedAvg Equivalence for 8 Clients
# ==============================================================================

def test_exact_fedavg_equivalence_8clients():
    """
    Verify that when all 8 clients use the server reference config (5, 3),
    aggregation matches plain FedAvg to within 1e-6.
    """
    ref_model = FastKANClassifier(layers_hidden=[32, 16, 5], grid_size=5, spline_order=3)
    server_p = [p.detach().cpu().numpy().copy() for p in ref_model.parameters()]

    rng = np.random.RandomState(42)
    sample_counts = [1000, 500, 2500, 800, 1200, 3000, 600, 1400]
    total_samples = sum(sample_counts)

    client_results = []
    for cid in range(8):
        cp = [p + rng.normal(0, 0.02, size=p.shape).astype(p.dtype) for p in server_p]
        client_results.append((cp, sample_counts[cid]))

    # Plain FedAvg
    plain = [np.zeros_like(p, dtype=np.float64) for p in server_p]
    for cp, n in client_results:
        w = float(n) / float(total_samples)
        for i, p in enumerate(cp):
            plain[i] += w * p.astype(np.float64)

    agg_full = aggregate_parameters(client_results, mode="project_full", canonical_gk=(5, 3))
    agg_delta = aggregate_parameters(
        client_results, mode="project_delta", canonical_gk=(5, 3), server_parameters=server_p
    )

    diff_full = max(np.max(np.abs(a.astype(np.float64) - p)) for a, p in zip(agg_full, plain))
    diff_delta = max(np.max(np.abs(a.astype(np.float64) - p)) for a, p in zip(agg_delta, plain))

    assert diff_full < 1e-6, f"project_full FedAvg mismatch: {diff_full:.2e}"
    assert diff_delta < 1e-6, f"project_delta FedAvg mismatch: {diff_delta:.2e}"


# ==============================================================================
# 5. Dirichlet Partitions for 8 Clients
# ==============================================================================

@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
def test_dirichlet_partition_8clients(alpha):
    y_train = np.load("data/processed/beats/y_train.npy")
    p1 = dirichlet_partition(y_train, num_clients=8, alpha=alpha, seed=42)
    p2 = dirichlet_partition(y_train, num_clients=8, alpha=alpha, seed=42)

    assert len(p1) == 8
    # Sum must equal total train beats
    total_beats = sum(len(indices) for indices in p1.values())
    assert total_beats == len(y_train)

    # No client is empty
    for cid in range(8):
        assert len(p1[cid]) > 0
        assert np.array_equal(p1[cid], p2[cid]), f"Non-deterministic partition for alpha={alpha}"


# ==============================================================================
# 6. Fleet Profile Generation Tests
# ==============================================================================

def test_fleet_profile_determinism_and_memory():
    profiles_a = generate_fleet_profiles(seed=42)
    profiles_b = generate_fleet_profiles(seed=42)

    assert len(profiles_a) == 8
    for cid in range(8):
        pa = profiles_a[cid]
        pb = profiles_b[cid]
        assert pa.tier == pb.tier
        assert pa.compute_speed_factor == pb.compute_speed_factor
        assert pa.memory_budget_mb == pb.memory_budget_mb
        assert pa.bandwidth_mbps == pb.bandwidth_mbps
        assert pa.energy_budget_rel == pb.energy_budget_rel

        # Verify all allowed configs fit client's memory
        for g, k in pa.allowed_configs:
            assert pa.fits_memory(g, k, batch_size=32), (
                f"Client {cid} ({pa.device_name}) fails memory budget for ({g}, {k})"
            )


# ==============================================================================
# 7. Telemetry Normalization & One-Hot Encoding
# ==============================================================================

def test_log_scale_normalization():
    # Bandwidth spanning 0.5 to 1000 Mbps
    bw_low = normalize_log_scale(0.5, min_val=0.5, max_val=1000.0)
    bw_high = normalize_log_scale(1000.0, min_val=0.5, max_val=1000.0)
    bw_mid = normalize_log_scale(22.36, min_val=0.5, max_val=1000.0)  # geometric mean ~sqrt(0.5*1000)

    assert pytest.approx(bw_low, abs=1e-3) == -1.0
    assert pytest.approx(bw_high, abs=1e-3) == 1.0
    assert pytest.approx(bw_mid, abs=1e-2) == 0.0

    # Ensure no NaN/Inf for edge values
    assert np.isfinite(normalize_log_scale(0.001, min_val=0.5, max_val=1000.0))
    assert np.isfinite(normalize_log_scale(5000.0, min_val=0.5, max_val=1000.0))


def test_tier_one_hot():
    for tier in [1, 2, 3, 4]:
        oh = get_tier_one_hot(tier, total_tiers=4)
        assert oh.shape == (4,)
        assert np.sum(oh) == 1.0
        assert oh[tier - 1] == 1.0
