"""
Unit tests for Telemetry Tracker and Gymnasium Federated KAN Environment.

Verifies:
1. TelemetryTracker: 5-dim state vector calculation, curvature computation, normalization bounds.
2. BandwidthSimulator: Uniform & log-normal distributions, boundary clipping [1.0, 10.0] Mbps.
3. Action mapping bijection: 12 discrete actions <-> (G, k) in {3, 5, 7, 10} x {2, 3, 4}.
4. FederatedKANEnv: Observation space Box(0, 1, (N, 5)), action space MultiDiscrete([12]*N).
5. Multi-objective step reward calculation matching Equation (7).
6. Deterministic reproducibility across resets and rollouts.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch

from src.environment.f1_env import (
    ACTION_TO_INDEX,
    ACTION_TUPLES,
    GRID_SIZES,
    NUM_ACTIONS_PER_CLIENT,
    SPLINE_ORDERS,
    FederatedKANEnv,
    action_to_gk,
    gk_to_action,
)
from src.environment.telemetry import BandwidthSimulator, ClientTelemetry, TelemetryTracker


# ==============================================================================
# 1. Bandwidth Simulator Tests
# ==============================================================================

def test_bandwidth_simulator_uniform():
    sim = BandwidthSimulator(distribution="uniform", min_mbps=1.0, max_mbps=10.0, seed=42)
    samples = sim.sample(num_clients=16)
    assert len(samples) == 16
    assert np.all(samples >= 1.0)
    assert np.all(samples <= 10.0)


def test_bandwidth_simulator_lognormal():
    sim = BandwidthSimulator(
        distribution="lognormal", min_mbps=1.0, max_mbps=10.0, lognormal_mu=1.6, lognormal_sigma=0.5, seed=42
    )
    samples = sim.sample(num_clients=20)
    assert len(samples) == 20
    assert np.all(samples >= 1.0)
    assert np.all(samples <= 10.0)


def test_bandwidth_simulator_reproducibility():
    sim1 = BandwidthSimulator(distribution="uniform", seed=123)
    sim2 = BandwidthSimulator(distribution="uniform", seed=123)
    np.testing.assert_allclose(sim1.sample(8), sim2.sample(8))


# ==============================================================================
# 2. Client Telemetry & Curvature Tests
# ==============================================================================

def test_client_telemetry_curvature():
    telemetry = ClientTelemetry(client_id="0", input_variance=1.0)
    assert telemetry.compute_curvature() == 0.0

    # 1 loss entry
    telemetry.update(local_loss=1.5, latency=0.2)
    assert telemetry.compute_curvature() == 0.0

    # 2 loss entries: |L_1 - L_0| = |1.2 - 1.5| = 0.3
    telemetry.update(local_loss=1.2, latency=0.25)
    assert pytest.approx(telemetry.compute_curvature(), abs=1e-5) == 0.3

    # 3 loss entries: |L_2 - 2*L_1 + L_0| = |0.8 - 2*(1.2) + 1.5| = |-0.1| = 0.1
    telemetry.update(local_loss=0.8, latency=0.22)
    assert pytest.approx(telemetry.compute_curvature(), abs=1e-5) == 0.1

    # 4 loss entries: uses last 3: [1.2, 0.8, 0.5] -> |0.5 - 2*(0.8) + 1.2| = |0.1| = 0.1
    telemetry.update(local_loss=0.5, latency=0.21)
    assert pytest.approx(telemetry.compute_curvature(), abs=1e-5) == 0.1


def test_client_telemetry_normalization_bounds():
    telemetry = ClientTelemetry(
        client_id="client_1",
        input_variance=0.9,
        normalization_bounds={
            "loss_max": 5.0,
            "variance_max": 2.0,
            "latency_max": 10.0,
            "bandwidth_min": 1.0,
            "bandwidth_max": 10.0,
            "curvature_max": 2.0,
        },
    )
    # Feed values within nominal range
    telemetry.update(local_loss=2.5, latency=5.0, bandwidth=5.5)
    vec = telemetry.get_state_vector(normalize=True)

    assert vec.shape == (5,)
    assert vec.dtype == np.float32
    assert 0.0 <= vec[0] <= 1.0  # loss: 2.5 / 5.0 = 0.5
    assert 0.0 <= vec[1] <= 1.0  # var: 0.9 / 2.0 = 0.45
    assert 0.0 <= vec[2] <= 1.0  # lat: 5.0 / 10.0 = 0.5
    assert 0.0 <= vec[3] <= 1.0  # bw: (5.5 - 1.0) / 9.0 = 0.5
    assert 0.0 <= vec[4] <= 1.0  # curvature

    # Test extreme overflow clipping
    telemetry.update(local_loss=10.0, latency=20.0, bandwidth=15.0)
    overflow_vec = telemetry.get_state_vector(normalize=True)
    assert np.all(overflow_vec <= 1.0)
    assert np.all(overflow_vec >= 0.0)


# ==============================================================================
# 3. Telemetry Tracker Multi-Client Tests
# ==============================================================================

def test_telemetry_tracker_global_matrix():
    client_ids = [0, 1, 2, 3]
    variances = {0: 0.8, 1: 1.0, 2: 1.2, 3: 0.9}
    tracker = TelemetryTracker(client_ids=client_ids, client_variances=variances, seed=42)

    init_state = tracker.reset(initial_losses=1.5)
    assert init_state.shape == (4, 5)
    assert np.all(init_state >= 0.0) and np.all(init_state <= 1.0)

    # Update round
    round_metrics = {
        cid: {"local_loss": 1.0 + 0.1 * cid, "execution_latency": 0.3} for cid in client_ids
    }
    updated_state = tracker.update_round(round_metrics)
    assert updated_state.shape == (4, 5)
    assert np.all(updated_state >= 0.0) and np.all(updated_state <= 1.0)

    summary = tracker.get_summary()
    assert summary["num_clients"] == 4
    assert len(summary["feature_names"]) == 5
    assert summary["total_latency"] > 0


# ==============================================================================
# 4. Action Space Mapping Tests
# ==============================================================================

def test_action_mapping_bijection():
    assert NUM_ACTIONS_PER_CLIENT == 12
    assert len(ACTION_TUPLES) == 12

    for action_idx in range(NUM_ACTIONS_PER_CLIENT):
        g, k = action_to_gk(action_idx)
        assert g in GRID_SIZES
        assert k in SPLINE_ORDERS
        recovered_idx = gk_to_action(g, k)
        assert recovered_idx == action_idx


def test_action_mapping_errors():
    with pytest.raises(ValueError):
        action_to_gk(-1)
    with pytest.raises(ValueError):
        action_to_gk(12)
    with pytest.raises(ValueError):
        gk_to_action(grid_size=99, spline_order=3)


# ==============================================================================
# 5. Gymnasium Environment Interface Tests
# ==============================================================================

@pytest.fixture
def test_env():
    """Create a lightweight instance of FederatedKANEnv for testing."""
    # Fast synthetic datasets for rapid test execution
    rng = np.random.RandomState(42)
    synthetic_clients = {
        cid: (rng.randn(64, 32).astype(np.float32), rng.randint(0, 5, size=64).astype(np.int64))
        for cid in range(4)
    }
    synthetic_test = (
        rng.randn(64, 32).astype(np.float32),
        rng.randint(0, 5, size=64).astype(np.int64),
    )

    env = FederatedKANEnv(
        client_datasets=synthetic_clients,
        test_data=synthetic_test,
        num_clients=4,
        max_rounds=3,
        lambda1=1e-6,
        lambda2=0.005,
        seed=42,
    )
    return env


def test_environment_spaces(test_env):
    assert test_env.observation_space.shape == (4, 5)
    assert test_env.observation_space.dtype == np.float32
    assert test_env.action_space.nvec.tolist() == [12, 12, 12, 12]


def test_environment_lifecycle_and_reward(test_env):
    obs, info = test_env.reset(seed=42)
    assert obs.shape == (4, 5)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    assert info["round"] == 0
    assert "global_accuracy" in info

    # Step 1: Assign actions
    actions = [0, 4, 8, 11]  # Mapped to (3, 2), (5, 3), (7, 4), (10, 4)
    next_obs, reward, terminated, truncated, step_info = test_env.step(actions)

    assert next_obs.shape == (4, 5)
    assert np.all(next_obs >= 0.0) and np.all(next_obs <= 1.0)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated

    # Verify Equation (7): R_t = ΔAcc - λ1 * Σ Params - λ2 * Σ Latency
    rc = step_info["reward_components"]
    expected_reward = rc["delta_accuracy"] - rc["param_penalty"] - rc["latency_penalty"]
    assert pytest.approx(reward, abs=1e-6) == expected_reward

    # Verify active parameter counts are positive
    assert step_info["total_parameters"] > 0
    assert step_info["total_latency"] > 0


def test_environment_termination(test_env):
    test_env.reset(seed=42)
    # max_rounds is 3
    for r in range(1, 4):
        actions = test_env.action_space.sample()
        obs, reward, terminated, truncated, info = test_env.step(actions)
        if r < 3:
            assert not terminated
        else:
            assert terminated
            assert info["round"] == 3


def test_environment_reproducibility():
    rng = np.random.RandomState(123)
    client_data = {
        cid: (rng.randn(32, 32).astype(np.float32), rng.randint(0, 5, size=32).astype(np.int64))
        for cid in range(2)
    }
    test_data = (
        rng.randn(32, 32).astype(np.float32),
        rng.randint(0, 5, size=32).astype(np.int64),
    )

    env1 = FederatedKANEnv(client_datasets=client_data, test_data=test_data, num_clients=2, seed=42)
    env2 = FederatedKANEnv(client_datasets=client_data, test_data=test_data, num_clients=2, seed=42)

    obs1, info1 = env1.reset(seed=42)
    obs2, info2 = env2.reset(seed=42)

    np.testing.assert_allclose(obs1, obs2, atol=1e-5)
    assert pytest.approx(info1["global_accuracy"], abs=1e-5) == info2["global_accuracy"]


def test_aggregation_modes_action_spaces():
    """Verify action spaces for nested_grid and compatible_actions_only."""
    rng = np.random.RandomState(42)
    client_data = {
        cid: (rng.randn(32, 32).astype(np.float32), rng.randint(0, 5, size=32).astype(np.int64))
        for cid in range(2)
    }
    test_data = (rng.randn(32, 32).astype(np.float32), rng.randint(0, 5, size=32).astype(np.int64))

    env_nested = FederatedKANEnv(
        client_datasets=client_data, test_data=test_data, num_clients=2,
        aggregation_mode="nested_grid", seed=42,
    )
    assert env_nested.action_space.nvec.tolist() == [2, 2]
    assert env_nested.action_tuples == [(5, 3), (10, 3)]

    env_compat = FederatedKANEnv(
        client_datasets=client_data, test_data=test_data, num_clients=2,
        aggregation_mode="compatible_actions_only", seed=42,
    )
    assert env_compat.action_space.nvec.tolist() == [8, 8]
    assert len(env_compat.action_tuples) == 8


def test_sticky_actions_execution():
    """Verify sticky action repeat for sticky_k rounds."""
    rng = np.random.RandomState(42)
    client_data = {
        cid: (rng.randn(32, 32).astype(np.float32), rng.randint(0, 5, size=32).astype(np.int64))
        for cid in range(2)
    }
    test_data = (rng.randn(32, 32).astype(np.float32), rng.randint(0, 5, size=32).astype(np.int64))

    env = FederatedKANEnv(
        client_datasets=client_data, test_data=test_data, num_clients=2,
        max_rounds=3, sticky_k=3, seed=42,
    )
    env.reset(seed=42)

    # Round 1: actions [1, 2]
    _, _, _, _, info1 = env.step([1, 2])
    configs1 = [(a["G"], a["k"]) for a in info1["actions"]]

    # Round 2: submit new actions [7, 8], but sticky_k=3 should keep configs from round 1
    _, _, _, _, info2 = env.step([7, 8])
    configs2 = [(a["G"], a["k"]) for a in info2["actions"]]
    assert configs2 == configs1


def test_project_delta_execution():
    """Verify end-to-end execution of project_delta mode."""
    rng = np.random.RandomState(42)
    client_data = {
        cid: (rng.randn(32, 32).astype(np.float32), rng.randint(0, 5, size=32).astype(np.int64))
        for cid in range(2)
    }
    test_data = (rng.randn(32, 32).astype(np.float32), rng.randint(0, 5, size=32).astype(np.int64))

    env = FederatedKANEnv(
        client_datasets=client_data, test_data=test_data, num_clients=2,
        max_rounds=2, aggregation_mode="project_delta", seed=42,
    )
    obs, info = env.reset(seed=42)
    assert obs.shape == (2, 5)

    obs2, reward, terminated, truncated, step_info = env.step([0, 11])
    assert obs2.shape == (2, 5)
    assert np.all(np.isfinite(obs2))
    assert isinstance(reward, float)
    assert "config_changes" in step_info

