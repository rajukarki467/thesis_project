"""
Verification and Test Suite for Dynamic Resource-Tier Allocation,
Straggler/Timeout Handling, and Spline Weight Aggregation.

Thesis Proposal: "Resource-Adaptive Federated Kolmogorov-Arnold Networks"

Tasks Verified:
1. Dynamic Resource-Tier Profiling:
   - 8 Clients: 4 Constrained Edge Devices (1-2 Mbps, high latency),
               4 High-Compute Nodes (10-20 Mbps, fast compute).
   - 5D Telemetry State accurately reflects device constraints.
2. PPO Dynamic Allocation:
   - Edge tier receives smaller G in {3, 5}, k in {2, 3}.
   - Compute tier receives larger G in {7, 10}, k in {3, 4}.
   - Assert: edge parameter allocation < compute parameter allocation.
3. Straggler, Dropout, and Timeout Handling:
   - Deadline checking: latency = Model_Size_MB / bw^i + Δτ_compute^i.
   - Timed-out clients marked as stragglers.
   - Server aggregates reporting clients (Option A: aggregate available; Option B: stale cache).
4. Canonical Spline Projection:
   - Aggregates heterogeneous weight shapes via least-squares canonical projection,
     never zero-padding, preventing crashes or weight corruption.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch

from src.controllers.ppo_controller import PPOController, TierAllocationCallback, log_allocation_by_tier
from src.environment.f1_env import FederatedKANEnv, action_to_gk, gk_to_action
from src.environment.telemetry import (
    COMPUTE_HARDWARE_PROFILE,
    EDGE_HARDWARE_PROFILE,
    ClientHardwareProfile,
    create_heterogeneous_profiles,
)
from src.federated.server import (
    DeadlineAwareFedAvg,
    aggregate_parameters,
    compute_transmission_latency,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_stragglers")


def print_round_table(
    round_idx: int,
    actions_info: List[Dict[str, Any]],
) -> None:
    """Print formatted markdown-style allocation table for a round."""
    header = (
        f"\n### Round {round_idx} Client Resource Allocation Table\n"
        f"| Client ID | Hardware Tier | Assigned G | Assigned k | Active Params | Bandwidth (Mbps) | Latency (s) | Status |\n"
        f"|:---------:|:-------------:|:----------:|:----------:|:-------------:|:----------------:|:-----------:|:------:|"
    )
    print(header)
    for act in actions_info:
        cid = act["client_id"]
        tier = act["hardware_tier"]
        g = act["G"]
        k = act["k"]
        params = act["active_params"]
        bw = act["bandwidth"]
        lat = act["latency"]
        status = act["status"]
        print(
            f"| {cid:^9d} | {tier:^13s} | {g:^10d} | {k:^10d} | {params:^13d} | {bw:^16.2f} | {lat:^11.3f} | {status:^6s} |"
        )
    print()


def test_heterogeneous_hardware_profiles_telemetry():
    """Verify that heterogeneous client profiles generate distinct, valid 5D telemetry states."""
    client_ids = list(range(8))
    profiles = create_heterogeneous_profiles(client_ids, edge_ratio=0.5)

    assert len(profiles) == 8
    # Clients 0..3 are edge tier, 4..7 are compute tier
    for cid in range(4):
        assert profiles[str(cid)].tier == "edge"
        assert profiles[str(cid)].min_bandwidth_mbps == 1.0
        assert profiles[str(cid)].max_bandwidth_mbps == 2.0
    for cid in range(4, 8):
        assert profiles[str(cid)].tier == "compute"
        assert profiles[str(cid)].min_bandwidth_mbps == 10.0
        assert profiles[str(cid)].max_bandwidth_mbps == 20.0


def test_canonical_spline_aggregation_with_missing_and_heterogeneous_updates():
    """
    Verify Task 3: When a subset of clients fails to report within the round deadline,
    the server aggregates parameter matrices of varying shapes across the active reporting clients
    using canonical reference projection rather than zero-padding, without crashing.
    """
    from src.models.fastkan_model import FastKANClassifier, project_spline_weights

    torch.manual_seed(42)
    canonical_model = FastKANClassifier(layers_hidden=[32, 16, 5], grid_size=5, spline_order=3)
    server_params = [p.detach().cpu().numpy().copy() for p in canonical_model.parameters()]

    # Simulate 3 active clients reporting heterogeneous architectures: (3, 2), (5, 3), (10, 4)
    # 5 clients dropped out due to timeouts
    active_configs = [(3, 2), (5, 3), (10, 4)]
    active_samples = [150, 200, 250]
    client_results = []
    client_initials = []

    for g, k in active_configs:
        c_p = []
        c_init = []
        for p in server_params:
            if p.ndim == 3:
                p_proj = project_spline_weights(p, src_gk=(5, 3), dst_gk=(g, k))
                # Add simulated gradient update
                c_p.append(p_proj + np.random.normal(0, 0.01, size=p_proj.shape).astype(p_proj.dtype))
                c_init.append(p_proj)
            else:
                c_p.append(p + np.random.normal(0, 0.01, size=p.shape).astype(p.dtype))
                c_init.append(p.copy())
        client_results.append((c_p, active_samples[len(client_results)]))
        client_initials.append(c_init)

    # 1. Test project_full with missing clients and heterogeneous shapes
    aggregated_full = aggregate_parameters(
        client_results=client_results,
        canonical_gk=(5, 3),
        server_parameters=server_params,
        mode="project_full",
    )
    assert len(aggregated_full) == len(server_params)
    for p_agg, p_orig in zip(aggregated_full, server_params):
        assert p_agg.shape == p_orig.shape, f"Shape mismatch: {p_agg.shape} vs {p_orig.shape}"
        assert np.all(np.isfinite(p_agg)), "Non-finite weights found after aggregation!"

    # 2. Test project_delta with missing clients and heterogeneous shapes
    aggregated_delta = aggregate_parameters(
        client_results=client_results,
        canonical_gk=(5, 3),
        server_parameters=server_params,
        client_initial_parameters=client_initials,
        client_gks=active_configs,
        mode="project_delta",
    )
    assert len(aggregated_delta) == len(server_params)
    for p_agg, p_orig in zip(aggregated_delta, server_params):
        assert p_agg.shape == p_orig.shape, f"Shape mismatch: {p_agg.shape} vs {p_orig.shape}"
        assert np.all(np.isfinite(p_agg)), "Non-finite weights found after delta aggregation!"

    # 3. Test empty results fallback (all clients dropped)
    fallback_params = aggregate_parameters(
        client_results=[],
        canonical_gk=(5, 3),
        server_parameters=server_params,
    )
    assert len(fallback_params) == len(server_params)
    for p_fb, p_orig in zip(fallback_params, server_params):
        np.testing.assert_array_equal(p_fb, p_orig)


def test_straggler_timeout_deadline_detection():
    """
    Verify Task 2: Transmission latency Model_Size_MB / bw + compute_latency
    correctly triggers straggler status when exceeding round_timeout_seconds.
    """
    round_timeout = 2.5
    # Case 1: Fast compute node (20 Mbps, 0.1s compute)
    lat_fast = compute_transmission_latency(
        model_size_mb=0.033,
        bandwidth_mbps=20.0,
        compute_latency=0.10,
    )
    assert lat_fast < round_timeout, f"Fast client latency {lat_fast:.3f}s exceeded timeout"

    # Case 2: Slow edge node with heavy model (1.0 Mbps, 3.5s compute)
    lat_slow = compute_transmission_latency(
        model_size_mb=0.033,
        bandwidth_mbps=1.0,
        compute_latency=3.5,
    )
    assert lat_slow > round_timeout, f"Slow client latency {lat_slow:.3f}s failed to trigger timeout"


def run_5_round_simulation_and_verify():
    """
    Full end-to-end execution:
    a) Simulates 8 clients (4 constrained edge devices, 4 high-performance nodes).
    b) Runs 5 federated rounds with PPO adaptation and active timeouts.
    c) Prints an allocation table per round.
    d) Asserts that high-latency clients receive significantly lower parameter allocations.
    """
    print("=" * 80)
    print("RUNNING 5-ROUND RESOURCE-ADAPTIVE FEDERATED KAN VERIFICATION")
    print("=" * 80)

    # Initialize environment with 8 clients and 5 max rounds
    env = FederatedKANEnv(
        config_path="configs/ppo.yaml",
        num_clients=8,
        max_rounds=5,
        seed=42,
    )

    # Bind PPO controller with tier-aware action prior
    ctrl = PPOController(
        env=env,
        config_path="configs/ppo.yaml",
        use_action_prior=True,
        verbose=0,
    )

    tier_callback = TierAllocationCallback(verbose=0)
    obs, info = env.reset(seed=42)

    all_round_actions: List[List[Dict[str, Any]]] = []
    edge_param_history: List[float] = []
    comp_param_history: List[float] = []
    edge_G_history: List[float] = []
    comp_G_history: List[float] = []

    print("\nSimulating 8 Clients:")
    print("  - Clients 0..3: Constrained Edge Tier (1-2 Mbps, high compute latency)")
    print("  - Clients 4..7: High-Compute Server Tier (10-20 Mbps, fast compute)")
    print(f"  - Round Timeout Deadline: {env.round_timeout_seconds:.1f} s")
    print(f"  - Straggler Fallback Mode: {env.straggler_handling}")

    for round_idx in range(1, 6):
        # Predict actions via PPO controller
        actions = ctrl.predict(obs, deterministic=True)

        # Step environment
        next_obs, reward, terminated, truncated, step_info = env.step(actions)
        obs = next_obs

        actions_info = step_info["actions"]
        all_round_actions.append(actions_info)

        # Print markdown allocation table
        print_round_table(round_idx=round_idx, actions_info=actions_info)

        # Collect metrics by tier
        edge_acts = [a for a in actions_info if a["hardware_tier"] == "edge"]
        comp_acts = [a for a in actions_info if a["hardware_tier"] == "compute"]

        edge_params = np.mean([a["active_params"] for a in edge_acts])
        comp_params = np.mean([a["active_params"] for a in comp_acts])
        edge_g = np.mean([a["G"] for a in edge_acts])
        comp_g = np.mean([a["G"] for a in comp_acts])
        edge_k = np.mean([a["k"] for a in edge_acts])
        comp_k = np.mean([a["k"] for a in comp_acts])

        edge_param_history.append(edge_params)
        comp_param_history.append(comp_params)
        edge_G_history.append(edge_g)
        comp_G_history.append(comp_g)

        print(
            f"Round {round_idx} Summary: "
            f"Edge Mean (G={edge_g:.1f}, k={edge_k:.1f}, Params={edge_params:.0f}) | "
            f"Compute Mean (G={comp_g:.1f}, k={comp_k:.1f}, Params={comp_params:.0f}) | "
            f"Global Acc: {step_info['global_accuracy']:.4f} | "
            f"Macro F1: {step_info['global_macro_f1']:.4f} | "
            f"Stragglers: {step_info.get('stragglers_count', 0)}"
        )

    # Verification Asserts (Task 4d)
    mean_edge_params = float(np.mean(edge_param_history))
    mean_comp_params = float(np.mean(comp_param_history))
    mean_edge_G = float(np.mean(edge_G_history))
    mean_comp_G = float(np.mean(comp_G_history))

    print("\n" + "=" * 80)
    print("FINAL 5-ROUND ALLOCATION VERIFICATION METRICS:")
    print(f"  Edge Tier Mean Parameters   : {mean_edge_params:.1f}")
    print(f"  Compute Tier Mean Parameters: {mean_comp_params:.1f}")
    print(f"  Edge Tier Mean Grid Size G  : {mean_edge_G:.2f}")
    print(f"  Compute Tier Mean Grid Size G: {mean_comp_G:.2f}")
    print(f"  Parameter Reduction on Edge : {(1.0 - mean_edge_params / mean_comp_params) * 100:.1f}%")
    print("=" * 80)

    # 1. Assert edge devices receive significantly lower parameter allocations
    assert mean_edge_params < mean_comp_params, (
        f"Assertion Failed: Edge parameter allocation ({mean_edge_params:.1f}) "
        f"must be lower than compute allocation ({mean_comp_params:.1f})."
    )

    # 2. Assert edge devices receive smaller grid size G
    assert mean_edge_G < mean_comp_G, (
        f"Assertion Failed: Edge G ({mean_edge_G:.2f}) must be strictly less than Compute G ({mean_comp_G:.2f})."
    )

    # 3. Assert edge allocations fall within G in {3, 5}, k in {2, 3}
    for round_acts in all_round_actions:
        for a in round_acts:
            if a["hardware_tier"] == "edge":
                assert a["G"] in [3, 5], f"Edge client assigned G={a['G']} outside {{3, 5}}"
                assert a["k"] in [2, 3], f"Edge client assigned k={a['k']} outside {{2, 3}}"
            else:
                assert a["G"] in [7, 10], f"Compute client assigned G={a['G']} outside {{7, 10}}"
                assert a["k"] in [3, 4], f"Compute client assigned k={a['k']} outside {{3, 4}}"

    # 4. Verify aggregation completed successfully and server weights remain finite
    for p in env.server_parameters:
        assert np.all(np.isfinite(p)), "Non-finite values found in global server parameters!"

    print("\n>>> ALL VERIFICATION ASSERTIONS PASSED SUCCESSFULLY! <<<\n")


def test_round_rollout_assertion():
    """Pytest test case invoking the 5-round simulation and verification."""
    run_5_round_simulation_and_verify()


if __name__ == "__main__":
    run_5_round_simulation_and_verify()
