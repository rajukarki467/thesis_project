"""
High-Speed Analytical Surrogate Gymnasium Environment for Federated KAN RL Training.

Reference: Step E2 of the Diagnostic and Repair Work Plan.
Full federated training rounds with 8 clients take ~5.8s per step on CPU,
making 20,000-50,000 environment steps computationally intractable (~32-80 hours).
This surrogate environment provides a high-fidelity analytical simulation that models:
1. Exact active parameter counts: P(G, k) = 613 + 592 * (G + k).
2. Network transmission latency and local computation time from simulated bandwidths.
3. Telemetry dynamics (loss trajectory, second-difference curvature, client variance).
4. Asymmetric parameter budget constraints relative to fixed (5,3) fleet (B_ref = 42,792).
5. Calibrated validation metric progression with transient down-weighting.

Interface matches FederatedKANEnv identically:
- Observation Space: Box(low=0.0, high=1.0, shape=(N, 5), dtype=np.float32)
- Action Space: MultiDiscrete([len(action_tuples)] * N)
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml

from src.environment.f1_env import (
    ACTION_TO_INDEX,
    ACTION_TUPLES,
    COMPATIBLE_ACTION_TUPLES,
    NESTED_ACTION_TUPLES,
    NUM_ACTIONS_PER_CLIENT,
    action_to_gk,
    gk_to_action,
)
from src.environment.telemetry import BandwidthSimulator, TelemetryTracker
from src.partition import dirichlet_partition
from src.utils.reproducibility import set_seed


def calculate_active_parameters(g: int, k: int) -> int:
    """Exact parameter count for [32, 16, 5] FastKAN with (G, k)."""
    return 613 + 592 * (int(g) + int(k))


class SurrogateFederatedKANEnv(gym.Env):
    """
    Analytical Surrogate Environment for Federated KAN PPO Meta-Controller Training.
    """

    metadata = {"render_modes": ["ansi", "human"], "name": "SurrogateFederatedKANEnv-v0"}

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = "configs/ppo.yaml",
        aggregation_mode: Optional[str] = None,
        sticky_k: int = 1,
        target_budget_ratio: Optional[float] = None,
        accuracy_metric: Optional[str] = None,
        num_clients: Optional[int] = None,
        max_rounds: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.config = self._load_config(config_path)

        env_cfg = self.config.get("environment", {})
        reward_cfg = self.config.get("reward", {})
        fed_cfg = self.config.get("federated", {})
        agg_cfg = self.config.get("aggregation", {})
        telemetry_cfg = self.config.get("telemetry", {})

        self.num_clients = int(num_clients if num_clients is not None else env_cfg.get("num_clients", 8))
        self.max_rounds = int(max_rounds if max_rounds is not None else env_cfg.get("max_rounds", 10))
        self.alpha = float(env_cfg.get("alpha", 0.5))
        self.global_seed = int(seed if seed is not None else env_cfg.get("seed", 42))
        self.sticky_k = int(sticky_k if sticky_k is not None else env_cfg.get("sticky_k", 1))

        # Aggregation mode
        self.aggregation_mode = str(
            aggregation_mode
            if aggregation_mode is not None
            else agg_cfg.get("mode", fed_cfg.get("aggregation_mode", "project_delta"))
        ).lower()

        if self.aggregation_mode == "nested_grid":
            self.action_tuples = list(NESTED_ACTION_TUPLES)
        elif self.aggregation_mode == "compatible_actions_only":
            self.action_tuples = list(COMPATIBLE_ACTION_TUPLES)
        else:
            self.action_tuples = list(ACTION_TUPLES)

        self.num_actions_per_client = len(self.action_tuples)
        self.ref_grid_size = int(env_cfg.get("reference_grid_size", 5))
        self.ref_spline_order = int(env_cfg.get("reference_spline_order", 3))

        # Reward formulation
        self.baseline_params_per_client = 5349.0
        self.baseline_budget = float(self.num_clients * self.baseline_params_per_client)
        if target_budget_ratio is not None:
            self.target_budget_ratio = float(target_budget_ratio)
        else:
            self.target_budget_ratio = float(reward_cfg.get("target_budget_ratio", 1.0))
        self.target_budget = float(self.baseline_budget * self.target_budget_ratio)

        self.scale_acc = float(reward_cfg.get("scale_acc", 100.0))
        self.beta1 = float(reward_cfg.get("beta1", reward_cfg.get("lambda1", 2.0)))
        self.beta2 = float(reward_cfg.get("beta2", reward_cfg.get("lambda2", 0.1)))
        self.beta3 = float(reward_cfg.get("beta3", 0.02))
        self.baseline_latency = float(reward_cfg.get("baseline_latency", 0.5))
        self.reward_clip_min = float(reward_cfg.get("reward_clip_min", -5.0))
        self.reward_clip_max = float(reward_cfg.get("reward_clip_max", 5.0))
        self.delta_clip = float(reward_cfg.get("delta_clip", 0.04))

        if accuracy_metric is not None:
            self.accuracy_metric = str(accuracy_metric).lower()
        else:
            self.accuracy_metric = str(reward_cfg.get("accuracy_metric", "macro_f1")).lower()

        self.transient_downweight = bool(reward_cfg.get("transient_downweight", True))
        self.alpha_abs = float(reward_cfg.get("alpha_abs", 0.5))

        # Observation and Action Spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_clients, 5), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(
            [self.num_actions_per_client] * self.num_clients
        )

        self.client_ids = list(range(self.num_clients))

        # Client data variances (synthesize or load empirical)
        rng = np.random.RandomState(self.global_seed)
        self.client_variances = {
            cid: float(np.clip(rng.normal(0.85, 0.25), 0.2, 1.8))
            for cid in self.client_ids
        }

        norm_bounds = telemetry_cfg.get("normalization", {})
        bw_sim_cfg = telemetry_cfg.get("bandwidth_simulation", {})
        self.telemetry_tracker = TelemetryTracker(
            client_ids=self.client_ids,
            client_variances=self.client_variances,
            normalization_bounds=norm_bounds,
            bandwidth_cfg=bw_sim_cfg,
            seed=self.global_seed,
        )

        # Internal state
        self.current_round: int = 0
        self.metric_prev: float = 0.20
        self.prev_actions: Optional[np.ndarray] = None
        self.client_base_losses: Dict[int, float] = {}

    def _load_config(self, config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        if config_path is not None:
            path = Path(config_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path.exists():
                try:
                    with open(path, "r") as f:
                        return yaml.safe_load(f) or {}
                except Exception:
                    pass
        return {}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.global_seed = int(seed)
            set_seed(self.global_seed)
            self.telemetry_tracker.set_seed(self.global_seed)

        self.current_round = 0
        self.metric_prev = 0.20  # Random guessing macro F1 on 5 classes

        # Randomize initial client losses per episode
        rng = np.random.RandomState(seed if seed is not None else self.global_seed)
        self.client_base_losses = {
            cid: float(rng.uniform(1.6, 2.4)) for cid in self.client_ids
        }

        initial_obs = self.telemetry_tracker.reset(
            initial_losses=float(np.mean(list(self.client_base_losses.values()))),
            seed=seed,
        )

        base_act = 0 if len(self.action_tuples) != NUM_ACTIONS_PER_CLIENT else gk_to_action(self.ref_grid_size, self.ref_spline_order)
        self.prev_actions = np.full(self.num_clients, base_act, dtype=np.int64)

        info = {
            "round": 0,
            "val_macro_f1": self.metric_prev,
            "val_accuracy": 0.20,
            "telemetry_summary": self.telemetry_tracker.get_summary(),
        }
        return initial_obs, info

    def step(
        self,
        actions: Union[np.ndarray, Sequence[int]],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.int64).flatten()
        if len(actions) != self.num_clients:
            raise ValueError(f"Action count mismatch: expected {self.num_clients}, got {len(actions)}.")

        if self.sticky_k > 1 and self.prev_actions is not None and ((self.current_round % self.sticky_k) != 0):
            actions = self.prev_actions.copy()

        round_num = self.current_round + 1
        client_params_list: List[int] = []
        client_latencies_list: List[float] = []
        chosen_configs: List[Tuple[int, int]] = []
        round_metrics: Dict[int, Dict[str, Any]] = {}

        # Simulate execution for each client
        for cid in self.client_ids:
            act_idx = int(actions[cid])
            if 0 <= act_idx < len(self.action_tuples):
                g_i, k_i = self.action_tuples[act_idx]
            elif 0 <= act_idx < len(ACTION_TUPLES):
                candidate_gk = ACTION_TUPLES[act_idx]
                if candidate_gk in self.action_tuples:
                    g_i, k_i = candidate_gk
                else:
                    g_i, k_i = min(
                        self.action_tuples,
                        key=lambda gk: abs(gk[0] * gk[1] - candidate_gk[0] * candidate_gk[1]),
                    )
            else:
                raise ValueError(f"Action index {act_idx} out of range [0, {len(self.action_tuples)}).")

            chosen_configs.append((g_i, k_i))
            p_count = calculate_active_parameters(g_i, k_i)
            client_params_list.append(p_count)

            # Simulated execution latency: compute time + transmission time
            # Compute scales with bases (G + k); transmission scales with model bytes / bandwidth
            t_compute = 0.08 * ((g_i + k_i) / 8.0)
            bw = self.telemetry_tracker.clients[str(cid)].latest_bandwidth
            t_comm = (2.0 * p_count * 4.0 * 8.0) / (max(1.0, bw) * 1e6)
            exec_lat = float(t_compute + t_comm)
            client_latencies_list.append(exec_lat)

            # Local loss evolution: decay with rounds, faster decay with higher (G, k)
            decay_rate = 0.15 + 0.03 * ((g_i + k_i) / 8.0)
            self.client_base_losses[cid] = float(
                max(0.1, self.client_base_losses[cid] * np.exp(-decay_rate))
            )

            round_metrics[cid] = {
                "local_loss": self.client_base_losses[cid],
                "execution_latency": exec_lat,
            }

        # Simulated global validation macro F1 trajectory
        # Under project_delta: steady convergence to ~0.465, improved by balanced allocations
        mean_bases = np.mean([g + k for g, k in chosen_configs])
        delta_m = (0.47 - self.metric_prev) * (0.28 + 0.02 * (mean_bases / 8.0))
        metric_curr = float(np.clip(self.metric_prev + delta_m, 0.20, 0.48))
        raw_delta = float(metric_curr - self.metric_prev)

        # Transient down-weighting: down-weight round 1 and 2
        if self.transient_downweight:
            if round_num == 1:
                round_weight = 0.1
            elif round_num == 2:
                round_weight = 0.5
            else:
                round_weight = 1.0
        else:
            round_weight = 1.0

        effective_delta = raw_delta * round_weight
        delta_acc_clipped = float(np.clip(effective_delta, -self.delta_clip, self.delta_clip))

        abs_level_bonus = float(self.alpha_abs * max(0.0, metric_curr - 0.20))
        acc_term = float(self.scale_acc * delta_acc_clipped + abs_level_bonus)

        sum_params = float(sum(client_params_list))
        sum_latency = float(sum(client_latencies_list))
        mean_params = float(np.mean(client_params_list))
        mean_latency = float(np.mean(client_latencies_list))

        # Asymmetric budget penalty relative to B_ref
        relative_param_cost = (
            float((sum_params - self.target_budget) / self.baseline_budget)
            if self.baseline_budget > 0
            else 0.0
        )
        if relative_param_cost > 0:
            param_penalty = float(
                self.beta1 * (relative_param_cost + 4.0 * (relative_param_cost ** 2))
            )
        else:
            param_penalty = float(self.beta1 * relative_param_cost)

        latency_penalty = float(self.beta2 * (mean_latency / max(1e-6, self.baseline_latency)))

        if self.prev_actions is None:
            base_act = 0 if len(self.action_tuples) != NUM_ACTIONS_PER_CLIENT else gk_to_action(self.ref_grid_size, self.ref_spline_order)
            self.prev_actions = np.full(self.num_clients, base_act, dtype=np.int64)

        smoothness_norm = float(np.sum(np.abs(actions - self.prev_actions)))
        smoothness_penalty = float(self.beta3 * smoothness_norm)
        config_changes = int(np.sum(actions != self.prev_actions))

        raw_reward = float(acc_term - param_penalty - latency_penalty - smoothness_penalty)
        reward = float(np.clip(raw_reward, self.reward_clip_min, self.reward_clip_max))

        self.prev_actions = actions.copy()
        self.metric_prev = metric_curr
        self.current_round = round_num

        new_obs = self.telemetry_tracker.update_round(round_metrics)
        assert np.all(np.isfinite(new_obs))
        new_obs = np.clip(new_obs, 0.0, 1.0)

        terminated = bool(self.current_round >= self.max_rounds)
        truncated = False

        savings_pct = (
            float((self.baseline_budget - sum_params) / self.baseline_budget * 100.0)
            if self.baseline_budget > 0
            else 0.0
        )

        info: Dict[str, Any] = {
            "round": self.current_round,
            "val_macro_f1": metric_curr,
            "val_accuracy": metric_curr * 1.8,
            "delta_accuracy": raw_delta,
            "config_changes": config_changes,
            "total_parameters": sum_params,
            "mean_parameters_per_client": mean_params,
            "total_latency": sum_latency,
            "mean_latency": mean_latency,
            "reward_components": {
                "acc_term": acc_term,
                "delta_accuracy": acc_term - smoothness_penalty,
                "raw_delta_acc": raw_delta,
                "delta_acc_clipped": delta_acc_clipped,
                "param_penalty": param_penalty,
                "relative_param_cost": relative_param_cost,
                "parameter_savings_pct": savings_pct,
                "latency_penalty": latency_penalty,
                "smoothness_penalty": smoothness_penalty,
                "raw_reward": raw_reward,
                "total_reward": reward,
            },
            "actions": [
                {"client_id": cid, "action_idx": int(actions[cid]), "G": g, "k": k}
                for cid, (g, k) in enumerate(chosen_configs)
            ],
            "telemetry_summary": self.telemetry_tracker.get_summary(),
        }
        return new_obs, reward, terminated, truncated, info
