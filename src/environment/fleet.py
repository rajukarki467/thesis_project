"""
Heterogeneous Edge Fleet Modeling and Simulation for Federated KAN.

Defines:
- 4 device tiers: Smartwatch (Tier 1), Smartphone (Tier 2), Local Clinic (Tier 3), Hospital Server (Tier 4)
- Deterministic client profile generation saved to data/processed/metadata/fleet_profiles_seed{seed}.csv
- Hard action masking per tier for (G, k) spline configurations
- Parameter count formula verification: params = 592 * (G + k) + 613 for [32, 16, 5]
- Memory budget verification (model + activations vs device RAM)
- Log-scale normalization for telemetry across orders-of-magnitude ranges
- One-hot tier encoding for RL state representations
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("fleet")

# Canonical 12 action space configurations: {3, 5, 7, 10} x {2, 3, 4}
ALL_GRID_SIZES: List[int] = [3, 5, 7, 10]
ALL_SPLINE_ORDERS: List[int] = [2, 3, 4]

ACTION_TUPLES: List[Tuple[int, int]] = [
    (g, k) for g in ALL_GRID_SIZES for k in ALL_SPLINE_ORDERS
]
ACTION_TO_INDEX: Dict[Tuple[int, int], int] = {
    gk: idx for idx, gk in enumerate(ACTION_TUPLES)
}
INDEX_TO_ACTION: Dict[int, Tuple[int, int]] = {
    idx: gk for idx, gk in enumerate(ACTION_TUPLES)
}
NUM_ACTIONS_TOTAL: int = len(ACTION_TUPLES)  # 12


def compute_param_count(grid_size: int, spline_order: int) -> int:
    """
    Exact parameter count formula for FastKANClassifier with [32, 16, 5] architecture:
        params = 592 * (G + k) + 613

    Breakdown:
    - Layer 0 (32 -> 16):
        base_weight: 16 * 32 = 512
        spline_weight: 16 * 32 * (G + k) = 512 * (G + k)
        bias: 16
    - Layer 1 (16 -> 5):
        base_weight: 5 * 16 = 80
        spline_weight: 5 * 16 * (G + k) = 80 * (G + k)
        bias: 5
    - Total: (512 + 80) * (G + k) + (512 + 80) + (16 + 5) = 592 * (G + k) + 613
    """
    return 592 * (int(grid_size) + int(spline_order)) + 613


# Client-to-Tier deterministic mapping for 8 clients
DEFAULT_CLIENT_TIER_MAP: Dict[int, int] = {
    0: 1,  # Smartwatch
    1: 1,  # Smartwatch
    2: 2,  # Smartphone
    3: 2,  # Smartphone
    4: 2,  # Smartphone
    5: 3,  # Local Clinic
    6: 3,  # Local Clinic
    7: 4,  # Hospital Server
}

TIER_DEVICE_NAMES: Dict[int, str] = {
    1: "Smartwatch",
    2: "Smartphone",
    3: "Local Clinic",
    4: "Hospital Server",
}

# Hard allowed (G, k) configs per tier
TIER_ALLOWED_CONFIGS: Dict[int, List[Tuple[int, int]]] = {
    1: [(3, 2), (3, 3), (5, 2), (5, 3)],
    2: [(5, 2), (5, 3), (5, 4), (7, 2), (7, 3), (7, 4)],
    3: [(7, 3), (7, 4), (10, 3), (10, 4)],
    4: [(10, 4)],
}

# Start config (Round 0): (5, 3) if allowed; otherwise closest allowed config by params
TIER_START_CONFIGS: Dict[int, Tuple[int, int]] = {
    1: (5, 3),  # 5,349 params (allowed)
    2: (5, 3),  # 5,349 params (allowed)
    3: (7, 3),  # 6,533 params (closest to 5,349 among {(7,3), (7,4), (10,3), (10,4)})
    4: (10, 4), # 8,901 params (sole allowed config)
}


@dataclass
class ClientFleetProfile:
    """
    Simulated physical edge device profile.
    Explicitly labeled as an emulated benchmark profile, not real hardware telemetry.
    """

    client_id: int
    tier: int
    device_name: str
    compute_speed_factor: float      # Relative to reference laptop baseline (laptop = 1.0)
    memory_budget_mb: float          # Device RAM budget allocated for model + forward pass
    bandwidth_mbps: float            # Simulated uplink/downlink throughput (Mbps)
    energy_budget_rel: float         # Relative energy budget
    dropout_prob: float              # Round dropout probability
    deadline_seconds: float          # Per-tier straggler deadline
    allowed_configs: List[Tuple[int, int]]
    start_config: Tuple[int, int]
    is_emulated_profile: bool = True

    def fits_memory(self, grid_size: int, spline_order: int, batch_size: int = 32) -> bool:
        """
        Verify whether the FastKAN model with (G, k) and training activations fit within RAM.
        Footprint:
        - Model weights (float32): params * 4 bytes
        - Optimizer states (Adam: 2x params * 4 bytes): 2 * params * 4 bytes
        - Gradients: params * 4 bytes
        - Forward activations (batch_size=32): ~100 KB
        """
        params = compute_param_count(grid_size, spline_order)
        model_bytes = params * 4
        optim_bytes = 2 * params * 4
        grad_bytes = params * 4
        # Activations: L0 bases (32 * 32 * (G+k) * 4), L0 out (32 * 16 * 4), L1 bases (32 * 16 * (G+k) * 4), L1 out (32 * 5 * 4)
        act_bytes = (
            batch_size * 32 * (grid_size + spline_order) * 4
            + batch_size * 16 * 4
            + batch_size * 16 * (grid_size + spline_order) * 4
            + batch_size * 5 * 4
        )
        total_mb = (model_bytes + optim_bytes + grad_bytes + act_bytes) / (1024.0 * 1024.0)
        return bool(total_mb <= self.memory_budget_mb)

    def estimate_round_time(
        self,
        grid_size: int,
        spline_order: int,
        measured_laptop_compute_sec: float,
    ) -> float:
        """
        Estimate round execution time (s) = local compute time * speed factor + network transmission time.
        Network transmission: Downlink + Uplink = 2 * (params * 4 * 8) / (bandwidth_mbps * 1e6)
        """
        params = compute_param_count(grid_size, spline_order)
        comm_bytes = 2 * params * 4
        comm_seconds = (comm_bytes * 8.0) / max(0.1, self.bandwidth_mbps * 1e6)
        comp_seconds = measured_laptop_compute_sec * self.compute_speed_factor
        return float(comp_seconds + comm_seconds)


class FleetConfig:
    """
    Loads and provides convenient programmatic access to configs/fleet.yaml.
    """

    def __init__(self, config_path: Union[str, Path] = "configs/fleet.yaml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            # Fallback to relative from project root
            self.config_path = PROJECT_ROOT / config_path

        with open(self.config_path, "r") as f:
            raw_cfg = yaml.safe_load(f)

        self.cfg = raw_cfg.get("fleet", {})
        self.num_clients = int(self.cfg.get("num_clients", 8))
        self.reference_mode = str(self.cfg.get("reference_mode", "fixed_5_3")).lower()
        self.server_reference_model = tuple(self.cfg.get("server_reference_model", [5, 3]))

        # Read budgets
        b_cfg = self.cfg.get("budgets", {})
        self.baseline_budget = float(b_cfg.get("baseline_budget_8clients", 42792.0))
        self.all_max_budget = float(b_cfg.get("all_max_budget_8clients", 71208.0))
        self.all_min_budget = float(b_cfg.get("all_min_budget_8clients", 28584.0))

        # Build tier metadata
        self.tiers_raw = self.cfg.get("tiers", {})
        self.client_tier_map: Dict[int, int] = {}
        for t_id, t_info in self.tiers_raw.items():
            for c_id in t_info.get("client_ids", []):
                self.client_tier_map[int(c_id)] = int(t_id)

        # Fallback to default client mapping if empty
        if not self.client_tier_map:
            self.client_tier_map = dict(DEFAULT_CLIENT_TIER_MAP)

    def get_tier_for_client(self, client_id: int) -> int:
        return self.client_tier_map.get(int(client_id), 1)

    def get_allowed_configs_for_tier(self, tier: int) -> List[Tuple[int, int]]:
        if tier in self.tiers_raw:
            raw_list = self.tiers_raw[tier].get("allowed_configs", [])
            return [tuple(cfg) for cfg in raw_list]
        return TIER_ALLOWED_CONFIGS.get(tier, [(5, 3)])

    def get_action_mask_for_client(self, client_id: int) -> np.ndarray:
        """
        Return a boolean mask of shape (12,) for the client where True means allowed.
        """
        tier = self.get_tier_for_client(client_id)
        allowed = set(self.get_allowed_configs_for_tier(tier))
        mask = np.zeros(NUM_ACTIONS_TOTAL, dtype=bool)
        for idx, gk in enumerate(ACTION_TUPLES):
            if gk in allowed:
                mask[idx] = True
        return mask

    def get_full_action_mask(self) -> np.ndarray:
        """
        Return flat boolean mask for sb3-contrib MaskablePPO over MultiDiscrete([12] * N).
        Length = N * 12.
        """
        masks = [self.get_action_mask_for_client(cid) for cid in range(self.num_clients)]
        return np.concatenate(masks, axis=0)

    def get_start_config_for_client(self, client_id: int) -> Tuple[int, int]:
        tier = self.get_tier_for_client(client_id)
        if tier in self.tiers_raw and "start_config" in self.tiers_raw[tier]:
            return tuple(self.tiers_raw[tier]["start_config"])
        return TIER_START_CONFIGS.get(tier, (5, 3))


def generate_fleet_profiles(
    seed: int = 42,
    config_path: Union[str, Path] = "configs/fleet.yaml",
    output_dir: Union[str, Path] = "data/processed/metadata",
) -> Dict[int, ClientFleetProfile]:
    """
    Generate deterministic simulated physical hardware profiles for all 8 edge clients
    drawn from each tier's configured ranges, and persist to CSV.

    Parameters
    ----------
    seed : int
        RNG seed for deterministic profile sampling.
    config_path : Union[str, Path]
        Path to configs/fleet.yaml.
    output_dir : Union[str, Path]
        Destination directory to save fleet_profiles_seed{seed}.csv.

    Returns
    -------
    Dict[int, ClientFleetProfile]
        Mapping from client_id (0..7) to ClientFleetProfile.
    """
    fleet_cfg = FleetConfig(config_path)
    rng = np.random.RandomState(seed)

    profiles: Dict[int, ClientFleetProfile] = {}
    rows = []

    for cid in range(fleet_cfg.num_clients):
        tier = fleet_cfg.get_tier_for_client(cid)
        t_data = fleet_cfg.tiers_raw.get(tier, {})
        dev_name = t_data.get("device_name", TIER_DEVICE_NAMES.get(tier, f"Tier {tier}"))
        r_ranges = t_data.get("resource_ranges", {})

        # Draw values deterministically within tier ranges
        spd_range = r_ranges.get("compute_speed_factor", [1.0, 1.0])
        mem_range = r_ranges.get("memory_budget_mb", [128.0, 128.0])
        bw_range = r_ranges.get("bandwidth_mbps", [10.0, 10.0])
        eng_range = r_ranges.get("energy_budget_rel", [100.0, 100.0])
        drop_prob = float(r_ranges.get("dropout_prob", 0.0))
        deadline = float(r_ranges.get("deadline_seconds", 3.0))

        # Sample values
        spd = float(rng.uniform(spd_range[0], spd_range[1]))
        mem = float(rng.uniform(mem_range[0], mem_range[1]))
        # Sample bandwidth log-uniformly for realistic networking distributions
        log_bw = float(rng.uniform(np.log(bw_range[0]), np.log(bw_range[1])))
        bw = float(np.exp(log_bw))
        eng = float(rng.uniform(eng_range[0], eng_range[1]))

        allowed = fleet_cfg.get_allowed_configs_for_tier(tier)
        start_cfg = fleet_cfg.get_start_config_for_client(cid)

        prof = ClientFleetProfile(
            client_id=cid,
            tier=tier,
            device_name=dev_name,
            compute_speed_factor=round(spd, 2),
            memory_budget_mb=round(mem, 1),
            bandwidth_mbps=round(bw, 2),
            energy_budget_rel=round(eng, 1),
            dropout_prob=round(drop_prob, 3),
            deadline_seconds=round(deadline, 2),
            allowed_configs=allowed,
            start_config=start_cfg,
            is_emulated_profile=True,
        )
        profiles[cid] = prof

        rows.append({
            "client_id": cid,
            "tier": tier,
            "device_name": dev_name,
            "compute_speed_factor": prof.compute_speed_factor,
            "memory_budget_mb": prof.memory_budget_mb,
            "bandwidth_mbps": prof.bandwidth_mbps,
            "energy_budget_rel": prof.energy_budget_rel,
            "dropout_prob": prof.dropout_prob,
            "deadline_seconds": prof.deadline_seconds,
            "allowed_configs": str(allowed),
            "start_config": str(start_cfg),
            "is_emulated_profile": True,
        })

    # Save to CSV
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"fleet_profiles_seed{seed}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved deterministic fleet profiles (seed {seed}) to {csv_path}")

    return profiles


# -----------------------------------------------------------------------------
# Log-Scale Normalization and One-Hot Encoding for Telemetry and RL States
# -----------------------------------------------------------------------------

def normalize_log_scale(
    value: float,
    min_val: float,
    max_val: float,
    eps: float = 1e-4,
) -> float:
    """
    Log10-scale normalization into [-1.0, 1.0] for physical quantities
    spanning orders of magnitude (e.g. bandwidth 0.5 - 1000 Mbps, latency 0.05 - 10 s):
        u = (log10(x) - log10(min)) / (log10(max) - log10(min))
        norm = 2.0 * u - 1.0
    """
    safe_val = max(eps, float(value))
    safe_min = max(eps, float(min_val))
    safe_max = max(safe_min + eps, float(max_val))

    log_v = np.log10(safe_val)
    log_min = np.log10(safe_min)
    log_max = np.log10(safe_max)

    u = (log_v - log_min) / (log_max - log_min)
    u_clipped = np.clip(u, 0.0, 1.0)
    return float(2.0 * u_clipped - 1.0)


def get_tier_one_hot(tier: int, total_tiers: int = 4) -> np.ndarray:
    """
    Generate a one-hot encoding array of shape (total_tiers,) for the given tier ID (1..4).
    """
    one_hot = np.zeros(total_tiers, dtype=np.float32)
    idx = max(0, min(total_tiers - 1, int(tier) - 1))
    one_hot[idx] = 1.0
    return one_hot
