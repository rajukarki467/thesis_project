"""
Deterministic Rule-Based Fallback and Tier-Aware Heuristic Controller for Federated KAN.

Mathematical Formulation & Architecture:
- Section 4.2 of Thesis Proposal:
    For each client i at communication round t, a client complexity score is computed:
        C_i = w1 * L_local^i + w2 * Var(x^i) + w3 * (1.0 / bw^i)

- Tier-Based Resource-Aware Allocation:
    1. Round 0:
       Every client starts at the reference config (5, 3). If (5, 3) is not permitted by the
       client's tier mask, it starts at the permitted config closest to (5, 3) by parameter count.
       (Tier 1: (5, 3), Tier 2: (5, 3), Tier 3: (7, 3), Tier 4: (10, 4)).
    2. Round t >= 1:
       Evaluates client complexity scores C_i within each device tier (not across the whole fleet,
       avoiding comparing smartwatches with hospital servers).
       - Tier 1 (Smartwatch, 2 clients): Rank-based allocation among { (3,2), (3,3), (5,2), (5,3) }.
       - Tier 2 (Smartphone, 3 clients): Rank-based allocation among { (5,2), (5,3), (5,4), (7,2), (7,3), (7,4) }.
       - Tier 3 (Local Clinic, 2 clients): Rank-based allocation among { (7,3), (7,4), (10,3), (10,4) }.
       - Tier 4 (Hospital Server, 1 client): Sole allowed config (10, 4) (zero decision freedom).
    3. Resource Feasibility & Pruning:
       Filters out any config whose estimated round time exceeds the tier deadline or whose
       memory footprint exceeds device RAM.
    4. Expected Utility & Hysteresis:
       Utility = AccuracyProxy - Cost(params, latency, energy).
       Switches configuration only if utility gain exceeds a threshold or on switching intervals,
       preventing thrashing.
    5. Final Assignment:
       Outputs the mode of the last 5 rounds, plus the best validation macro F1 config recorded,
       flagging any disagreements.
"""

from __future__ import annotations

from collections import Counter
import copy
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.fleet import (
    ACTION_TO_INDEX,
    ACTION_TUPLES,
    FleetConfig,
    compute_param_count,
    generate_fleet_profiles,
)

logger = logging.getLogger("heuristic_controller")


class HeuristicController:
    """
    Tier-Aware and Fallback Heuristic Controller for Resource-Adaptive Federated KAN.

    Parameters
    ----------
    config_path : Optional[str | Path]
        Path to PPO / heuristic YAML configuration (default 'configs/ppo.yaml').
    fleet_config_path : Optional[str | Path]
        Path to fleet configuration (default 'configs/fleet.yaml').
    use_tier_masks : bool
        Whether to enforce hard device tier masks (default False for baseline compatibility,
        set True for tiered edge fleet evaluation).
    w1 : float
        Weight for local loss (default 1.0).
    w2 : float
        Weight for empirical input variance (default 1.0).
    w3 : float
        Weight for inverse bandwidth (default 1.0).
    hysteresis_threshold : float
        Minimum utility improvement required to switch configuration (default 0.05).
    switch_interval_k : int
        Number of rounds between configuration update opportunities (default 1).
    seed : int
        RNG seed for deterministic profile sampling.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = "configs/ppo.yaml",
        fleet_config_path: Optional[Union[str, Path]] = "configs/fleet.yaml",
        use_tier_masks: bool = False,
        w1: Optional[float] = None,
        w2: Optional[float] = None,
        w3: Optional[float] = None,
        hysteresis_threshold: float = 0.05,
        switch_interval_k: int = 1,
        seed: int = 42,
    ):
        self.config_path = config_path
        self.fleet_config_path = fleet_config_path
        self.use_tier_masks = use_tier_masks
        self.seed = seed

        # Load PPO/heuristic config
        cfg = self._load_config(config_path)
        h_cfg = cfg.get("heuristic", {})
        w_cfg = h_cfg.get("weights", {})
        q_cfg = h_cfg.get("quantiles", {})
        c_cfg = h_cfg.get("configurations", {})
        t_cfg = cfg.get("telemetry", {}).get("normalization", {})

        # Scoring weights
        self.w1 = float(w1 if w1 is not None else w_cfg.get("w1", 1.0))
        self.w2 = float(w2 if w2 is not None else w_cfg.get("w2", 1.0))
        self.w3 = float(w3 if w3 is not None else w_cfg.get("w3", 1.0))

        # Legacy quantile thresholds
        self.high_percentile = float(q_cfg.get("high_percentile", 75.0))
        self.low_percentile = float(q_cfg.get("low_percentile", 25.0))

        # Fallback configs
        self.high_cfg = tuple(c_cfg.get("high", [10, 4]))
        self.med_cfg = tuple(c_cfg.get("medium", [5, 3]))
        self.low_cfg = tuple(c_cfg.get("low", [3, 2]))

        self.high_action = ACTION_TO_INDEX.get(self.high_cfg, 11)
        self.med_action = ACTION_TO_INDEX.get(self.med_cfg, 4)
        self.low_action = ACTION_TO_INDEX.get(self.low_cfg, 0)

        # Un-normalization bounds
        self.loss_max = float(t_cfg.get("loss_max", 5.0))
        self.variance_max = float(t_cfg.get("variance_max", 2.0))
        self.bandwidth_min = float(t_cfg.get("bandwidth_min", 1.0))
        self.bandwidth_max = float(t_cfg.get("bandwidth_max", 10.0))

        # Hysteresis parameters
        self.hysteresis_threshold = float(hysteresis_threshold)
        self.switch_interval_k = int(switch_interval_k)

        # Load fleet configuration
        try:
            self.fleet_config = FleetConfig(fleet_config_path)
            self.fleet_profiles = generate_fleet_profiles(seed=self.seed, config_path=fleet_config_path)
        except Exception as e:
            logger.warning(f"Could not load fleet configuration: {e}. Falling back to unconstrained mode.")
            self.fleet_config = None
            self.fleet_profiles = {}

        # History tracking for hysteresis, round traces, and final mode calculation
        self.current_round: int = 0
        self.client_action_history: Dict[int, List[int]] = {}
        self.client_config_history: Dict[int, List[Tuple[int, int]]] = {}
        self.client_val_f1_records: Dict[int, List[Tuple[Tuple[int, int], float]]] = {}

        self.reset()

    def reset(self) -> None:
        """Reset historical tracking across episodes."""
        self.current_round = 0
        n_clients = self.fleet_config.num_clients if self.fleet_config else 8
        self.client_action_history = {i: [] for i in range(n_clients)}
        self.client_config_history = {i: [] for i in range(n_clients)}
        self.client_val_f1_records = {i: [] for i in range(n_clients)}

    def _load_config(self, config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        if config_path is not None:
            path = Path(config_path)
            if not path.exists():
                path = PROJECT_ROOT / config_path
            if path.exists():
                try:
                    with open(path, "r") as f:
                        return yaml.safe_load(f) or {}
                except Exception as e:
                    logger.warning(f"Could not load config from {path}: {e}")
        return {}

    def compute_complexity_scores(
        self,
        observation: np.ndarray,
        raw_telemetry: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """
        Compute client complexity score C_i for each client i:
            C_i = w1 * L_local^i + w2 * Var(x^i) + w3 * (1.0 / bw^i)
        """
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 5 if obs.shape[0] % 5 == 0 else obs.shape[0])

        num_clients = obs.shape[0]

        if raw_telemetry is not None and len(raw_telemetry) == num_clients:
            losses = np.array([float(t.get("local_loss", 1.0)) for t in raw_telemetry])
            variances = np.array([float(t.get("input_variance", 1.0)) for t in raw_telemetry])
            bandwidths = np.array([float(t.get("bandwidth", 5.0)) for t in raw_telemetry])
        else:
            norm_loss = obs[:, 0]
            norm_var = obs[:, 1]
            norm_bw = obs[:, 3]

            losses = np.clip(norm_loss * self.loss_max, 0.0, self.loss_max)
            variances = np.clip(norm_var * self.variance_max, 0.0, self.variance_max)
            bw_range = max(1e-6, self.bandwidth_max - self.bandwidth_min)
            bandwidths = np.clip(self.bandwidth_min + norm_bw * bw_range, 0.1, 1000.0)

        safe_bw = np.maximum(bandwidths, 0.1)
        inv_bw = 1.0 / safe_bw

        complexity_scores = self.w1 * losses + self.w2 * variances + self.w3 * inv_bw
        return complexity_scores.astype(np.float32)

    def _select_tiered_action(
        self,
        client_id: int,
        tier: int,
        tier_clients: List[int],
        scores: np.ndarray,
        val_improvements: Optional[Dict[int, float]] = None,
    ) -> int:
        """
        Select config for a client within its device tier using rank-based allocation,
        resource feasibility checks, and hysteresis.
        """
        prof = self.fleet_profiles.get(client_id)
        if prof is None:
            return self.med_action

        allowed_configs = prof.allowed_configs
        if len(allowed_configs) == 1:
            if self.current_round <= 1:
                logger.info(
                    f"[Tier {tier}] Client {client_id} ({prof.device_name}) has no decision freedom: "
                    f"restricted to sole allowed config {allowed_configs[0]}."
                )
            return ACTION_TO_INDEX[allowed_configs[0]]

        # Feasible config filtering
        feasible_configs = []
        for g, k in allowed_configs:
            fits_mem = prof.fits_memory(g, k)
            est_time = prof.estimate_round_time(g, k, measured_laptop_compute_sec=0.08)
            if fits_mem and est_time <= prof.deadline_seconds:
                feasible_configs.append((g, k))

        if not feasible_configs:
            feasible_configs = [min(allowed_configs, key=lambda gk: compute_param_count(gk[0], gk[1]))]

        # Rank clients within this specific tier
        tier_scores = [scores[c] for c in tier_clients]
        sorted_tier_clients = [c for _, c in sorted(zip(tier_scores, tier_clients))]
        client_rank = sorted_tier_clients.index(client_id)
        rank_ratio = client_rank / max(1, len(tier_clients) - 1)

        sorted_configs = sorted(feasible_configs, key=lambda gk: compute_param_count(gk[0], gk[1]))
        target_idx = int(round(rank_ratio * (len(sorted_configs) - 1)))
        proposed_cfg = sorted_configs[target_idx]
        proposed_action = ACTION_TO_INDEX[proposed_cfg]

        # Hysteresis check
        if self.client_action_history[client_id]:
            prev_action = self.client_action_history[client_id][-1]
            prev_cfg = ACTION_TUPLES[prev_action]

            prev_params = compute_param_count(prev_cfg[0], prev_cfg[1])
            prop_params = compute_param_count(proposed_cfg[0], proposed_cfg[1])
            cost_diff = (prop_params - prev_params) / 5349.0

            val_gain = 0.0
            if val_improvements and client_id in val_improvements:
                val_gain = val_improvements[client_id]

            utility_gain = val_gain - 0.05 * cost_diff

            if (self.current_round % self.switch_interval_k != 0) and (utility_gain < self.hysteresis_threshold):
                return prev_action

        return proposed_action

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        raw_telemetry: Optional[List[Dict[str, Any]]] = None,
        val_improvements: Optional[Dict[int, float]] = None,
        val_f1_scores: Optional[Dict[int, float]] = None,
        round_num: Optional[int] = None,
    ) -> np.ndarray:
        """
        Select adaptive (G_i, k_i) actions for all N clients.
        """
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 5 if obs.shape[0] % 5 == 0 else obs.shape[0])

        num_clients = obs.shape[0]
        if round_num is not None:
            self.current_round = int(round_num)
        else:
            self.current_round += 1

        if val_f1_scores is not None:
            for cid, f1 in val_f1_scores.items():
                if self.client_config_history.get(cid):
                    curr_cfg = self.client_config_history[cid][-1]
                    self.client_val_f1_records[cid].append((curr_cfg, float(f1)))

        # Unconstrained fallback mode (for unit testing or baseline comparison)
        if not self.use_tier_masks or self.fleet_config is None or num_clients != 8:
            return self._predict_unconstrained(obs, raw_telemetry=raw_telemetry)

        # ---------------------------------------------------------------------
        # Tiered Mode
        # ---------------------------------------------------------------------

        # Round 0 / Initial start config:
        if self.current_round <= 1 and not any(self.client_action_history.values()):
            actions = []
            for cid in range(num_clients):
                start_cfg = self.fleet_config.get_start_config_for_client(cid)
                tier = self.fleet_config.get_tier_for_client(cid)
                act_idx = ACTION_TO_INDEX[start_cfg]
                actions.append(act_idx)
                if start_cfg != (5, 3):
                    logger.info(
                        f"[Round 0 Start] Client {cid} (Tier {tier}): Reference (5, 3) not permitted. "
                        f"Starting at closest allowed config {start_cfg} "
                        f"({compute_param_count(start_cfg[0], start_cfg[1]):,} params)."
                    )
            actions = np.array(actions, dtype=np.int64)

            for cid in range(num_clients):
                self.client_action_history[cid].append(int(actions[cid]))
                self.client_config_history[cid].append(ACTION_TUPLES[actions[cid]])
            return actions

        scores = self.compute_complexity_scores(obs, raw_telemetry=raw_telemetry)

        tier_to_clients: Dict[int, List[int]] = {}
        for cid in range(num_clients):
            t = self.fleet_config.get_tier_for_client(cid)
            tier_to_clients.setdefault(t, []).append(cid)

        actions = np.zeros(num_clients, dtype=np.int64)
        for cid in range(num_clients):
            tier = self.fleet_config.get_tier_for_client(cid)
            tier_clients = tier_to_clients.get(tier, [cid])
            act = self._select_tiered_action(
                client_id=cid,
                tier=tier,
                tier_clients=tier_clients,
                scores=scores,
                val_improvements=val_improvements,
            )
            actions[cid] = act

            self.client_action_history[cid].append(int(act))
            self.client_config_history[cid].append(ACTION_TUPLES[act])

        return actions

    def _predict_unconstrained(
        self,
        obs: np.ndarray,
        raw_telemetry: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """Unconstrained global quantile allocation for backward compatibility."""
        num_clients = obs.shape[0]
        scores = self.compute_complexity_scores(obs, raw_telemetry=raw_telemetry)

        if num_clients < 4:
            median_score = float(np.median(scores))
            actions = np.zeros(num_clients, dtype=np.int64)
            for i in range(num_clients):
                actions[i] = self.high_action if scores[i] >= median_score else self.low_action
            return actions

        ranks = np.argsort(np.argsort(scores))
        n_low = int(round(num_clients * (self.low_percentile / 100.0)))
        n_high = int(round(num_clients * ((100.0 - self.high_percentile) / 100.0)))

        actions = np.full(num_clients, fill_value=self.med_action, dtype=np.int64)
        for i in range(num_clients):
            if ranks[i] >= (num_clients - n_high):
                actions[i] = self.high_action
            elif ranks[i] < n_low:
                actions[i] = self.low_action
            else:
                actions[i] = self.med_action

        return actions

    def get_final_assignments(self) -> Dict[int, Dict[str, Any]]:
        """
        Report each client's final config assignment:
        - Mode over the last 5 rounds
        - Config yielding the best validation macro F1 under resource limits
        - Disagreement flag where they differ
        """
        results: Dict[int, Dict[str, Any]] = {}
        n_clients = self.fleet_config.num_clients if self.fleet_config else 8

        for cid in range(n_clients):
            history = self.client_config_history.get(cid, [])
            last_5 = history[-5:] if len(history) >= 5 else history
            if last_5:
                mode_cfg = Counter(last_5).most_common(1)[0][0]
            else:
                mode_cfg = self.fleet_config.get_start_config_for_client(cid) if self.fleet_config else (5, 3)

            val_records = self.client_val_f1_records.get(cid, [])
            if val_records:
                best_cfg, best_f1 = max(val_records, key=lambda item: item[1])
            else:
                best_cfg, best_f1 = mode_cfg, 0.0

            disagree = (mode_cfg != best_cfg)
            results[cid] = {
                "client_id": cid,
                "tier": self.fleet_config.get_tier_for_client(cid) if self.fleet_config else 1,
                "mode_last_5": mode_cfg,
                "mode_params": compute_param_count(mode_cfg[0], mode_cfg[1]),
                "best_val_f1_config": best_cfg,
                "best_val_f1": float(best_f1),
                "best_params": compute_param_count(best_cfg[0], best_cfg[1]),
                "disagreement": bool(disagree),
            }

        return results

    def explain_decision(
        self,
        observation: np.ndarray,
        raw_telemetry: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate structured breakdown of controller decision."""
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 5)

        num_clients = obs.shape[0]
        scores = self.compute_complexity_scores(obs, raw_telemetry=raw_telemetry)
        actions = self.predict(obs, raw_telemetry=raw_telemetry)

        breakdown = []
        for cid in range(num_clients):
            act = int(actions[cid])
            cfg = ACTION_TUPLES[act]

            if not self.use_tier_masks or self.fleet_config is None:
                # Legacy quantile tier string
                if act == self.high_action:
                    tier_str = "high"
                elif act == self.low_action:
                    tier_str = "low"
                else:
                    tier_str = "medium"
            else:
                tier_str = f"Tier {self.fleet_config.get_tier_for_client(cid)}"

            breakdown.append({
                "client_id": cid,
                "tier": tier_str,
                "complexity_score": float(scores[cid]),
                "action_idx": act,
                "grid_size": cfg[0],
                "spline_order": cfg[1],
                "params": compute_param_count(cfg[0], cfg[1]),
            })
        return breakdown
