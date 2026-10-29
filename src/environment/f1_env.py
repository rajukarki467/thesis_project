"""
Gymnasium Federated Reinforcement Learning Environment for Resource-Adaptive KANs.

Mathematical Formulation (Section 3.2.4 & Section 3.2.5 of the Thesis Proposal):
- State/Observation Space:
    S_t = [S_1^t, S_2^t, ..., S_N^t]^T in R^{N x 5}
    Normalized to Box(low=0.0, high=1.0, shape=(N, 5), dtype=np.float32)
    where each S_i^t = [L_local^i, Var(x^i), Δτ^i, bw^i, κ^i].

- Action Space:
    a_t = (a_1^t, a_2^t, ..., a_N^t) in MultiDiscrete([12] * N)
    where each a_i^t maps to (G_i^t, k_i^t) in {3, 5, 7, 10} x {2, 3, 4} (12 configurations).

- Multi-Objective Step Reward (Equation 7):
    R_t = ΔAcc_global - λ1 * Σ_{i=1}^N Params(G_i^t, k_i^t) - λ2 * Σ_{i=1}^N Δτ_i^t
    where ΔAcc_global = Acc_t - Acc_{t-1}.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import yaml

from src.environment.telemetry import TelemetryTracker
from src.federated.client import FlowerECGClient
from src.federated.server import CentralizedEvaluator, aggregate_parameters
from src.models.fastkan_model import FastKANClassifier
from src.partition import dirichlet_partition
from src.utils.reproducibility import set_seed

logger = logging.getLogger("federated_env")

# 12 Discrete Action Configurations: {3, 5, 7, 10} x {2, 3, 4}
GRID_SIZES: List[int] = [3, 5, 7, 10]
SPLINE_ORDERS: List[int] = [2, 3, 4]

ACTION_TUPLES: List[Tuple[int, int]] = [
    (g, k) for g in GRID_SIZES for k in SPLINE_ORDERS
]
ACTION_TO_INDEX: Dict[Tuple[int, int], int] = {
    gk: idx for idx, gk in enumerate(ACTION_TUPLES)
}
NUM_ACTIONS_PER_CLIENT: int = len(ACTION_TUPLES)  # 12


def action_to_gk(action_idx: int) -> Tuple[int, int]:
    """Convert integer action index [0..11] to (grid_size, spline_order)."""
    if not 0 <= action_idx < NUM_ACTIONS_PER_CLIENT:
        raise ValueError(
            f"Action index {action_idx} out of range [0, {NUM_ACTIONS_PER_CLIENT})."
        )
    return ACTION_TUPLES[action_idx]


def gk_to_action(grid_size: int, spline_order: int) -> int:
    """Convert (grid_size, spline_order) pair to integer action index."""
    pair = (int(grid_size), int(spline_order))
    if pair not in ACTION_TO_INDEX:
        raise ValueError(f"Configuration (G={grid_size}, k={spline_order}) not in {ACTION_TUPLES}.")
    return ACTION_TO_INDEX[pair]


class FederatedKANEnv(gym.Env):
    """
    OpenAI Gymnasium Environment for Resource-Adaptive Federated Kolmogorov-Arnold Networks.

    Each step corresponds to one communication round of federated training where the agent
    assigns (G_i, k_i) spline edge configurations to all N participating clients based on
    their runtime telemetry (loss, variance, latency, bandwidth, loss curvature).

    Parameters
    ----------
    config_path : Optional[str | Path]
        Path to YAML configuration file (default 'configs/ppo.yaml').
    client_datasets : Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]]
        Pre-split client training datasets. If None, loaded from disk/partitioned.
    test_data : Optional[Tuple[np.ndarray, np.ndarray]]
        Held-out evaluation dataset (X_test, y_test). If None, loaded from disk.
    num_clients : int
        Number of participating federated edge clients (default 8).
    max_rounds : int
        Maximum communication rounds per episode (default 10).
    lambda1 : float
        Parameter penalty scaling coefficient in Equation (7) (default 0.20 when normalized, 1e-6 otherwise).
    lambda2 : float
        Latency penalty scaling coefficient in Equation (7) (default 0.005).
    alpha : float
        Dirichlet non-IID skew parameter for client partitioning (default 0.5).
    normalize_params : bool
        Whether to normalize parameter cost against reference baseline budget (default True).
    baseline_budget : float
        Reference parameter budget (default num_clients * 5349.0).
    device : Optional[str | torch.device]
        Computation device ('cpu' or 'cuda').
    seed : int
        Random seed for deterministic initialization.
    """

    metadata = {"render_modes": ["ansi", "human"], "name": "FederatedKANEnv-v0"}

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = "configs/ppo.yaml",
        client_datasets: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        num_clients: Optional[int] = None,
        max_rounds: Optional[int] = None,
        lambda1: Optional[float] = None,
        lambda2: Optional[float] = None,
        alpha: Optional[float] = None,
        normalize_params: Optional[bool] = None,
        baseline_budget: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
        seed: int = 42,
    ):
        super().__init__()

        # 1. Load configuration from YAML
        self.config = self._load_config(config_path)

        env_cfg = self.config.get("environment", {})
        reward_cfg = self.config.get("reward", {})
        fed_cfg = self.config.get("federated", {})
        telemetry_cfg = self.config.get("telemetry", {})

        self.num_clients = int(num_clients if num_clients is not None else env_cfg.get("num_clients", 8))
        self.max_rounds = int(max_rounds if max_rounds is not None else env_cfg.get("max_rounds", 10))
        self.alpha = float(alpha if alpha is not None else env_cfg.get("alpha", 0.5))
        self.global_seed = int(seed if seed is not None else env_cfg.get("seed", 42))

        # Reward hyperparameters (Calibrated Normalized Equation 7)
        if normalize_params is not None:
            self.normalize_params = bool(normalize_params)
        else:
            self.normalize_params = bool(reward_cfg.get("normalize_params", True))

        ref_params_per_client = float(reward_cfg.get("baseline_params_per_client", 5349.0))
        if baseline_budget is not None:
            self.baseline_budget = float(baseline_budget)
        elif "baseline_budget" in reward_cfg and self.num_clients == env_cfg.get("num_clients", 8):
            self.baseline_budget = float(reward_cfg["baseline_budget"])
        else:
            self.baseline_budget = float(self.num_clients * ref_params_per_client)

        self.scale_acc = float(reward_cfg.get("scale_acc", 100.0))
        if lambda1 is not None:
            self.beta1 = float(lambda1)
        else:
            self.beta1 = float(reward_cfg.get("beta1", reward_cfg.get("lambda1", 1.5)))

        if lambda2 is not None:
            self.beta2 = float(lambda2)
        else:
            self.beta2 = float(reward_cfg.get("beta2", reward_cfg.get("lambda2", 0.1)))

        self.beta3 = float(reward_cfg.get("beta3", 0.02))
        self.lambda1 = self.beta1
        self.lambda2 = self.beta2
        self.baseline_latency = float(reward_cfg.get("baseline_latency", 0.5))
        self.reward_clip_min = float(reward_cfg.get("reward_clip_min", -5.0))
        self.reward_clip_max = float(reward_cfg.get("reward_clip_max", 5.0))
        self.delta_clip = float(reward_cfg.get("delta_clip", 0.04))
        self.accuracy_metric = reward_cfg.get("accuracy_metric", "accuracy")

        # Federated training hyperparameters
        self.local_epochs = int(fed_cfg.get("local_epochs", 1))
        self.batch_size = int(fed_cfg.get("batch_size", 32))
        self.learning_rate = float(fed_cfg.get("learning_rate", 0.001))
        self.optimizer_type = fed_cfg.get("optimizer", "adam")
        self.weight_decay = float(fed_cfg.get("weight_decay", 1e-4))
        self.proximal_mu = float(fed_cfg.get("proximal_mu", 0.0))

        # Device selection
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # FastKAN architecture parameters
        self.layers_hidden = env_cfg.get("reference_hidden", [32, 16, 5])
        self.ref_grid_size = int(env_cfg.get("reference_grid_size", 5))
        self.ref_spline_order = int(env_cfg.get("reference_spline_order", 3))

        # 2. Define Observation and Action Spaces
        # Observation Space: Matrix S_t in [0.0, 1.0]^{N x 5}
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_clients, 5),
            dtype=np.float32,
        )

        # Action Space: MultiDiscrete choice of 12 (G, k) pairs across N clients
        self.action_space = spaces.MultiDiscrete(
            [NUM_ACTIONS_PER_CLIENT] * self.num_clients
        )

        # 3. Load or generate client partitions and evaluation data
        self.client_ids = list(range(self.num_clients))
        self.client_datasets, self.test_data = self._setup_datasets(
            client_datasets=client_datasets, test_data=test_data
        )

        # Compute empirical input variances per client
        self.client_variances: Dict[int, float] = {}
        for cid, (x_c, _) in self.client_datasets.items():
            if len(x_c) > 1:
                self.client_variances[cid] = float(np.mean(np.var(x_c, axis=0)))
            else:
                self.client_variances[cid] = 1.0

        # 4. Initialize Telemetry Tracker
        norm_bounds = telemetry_cfg.get("normalization", {})
        bw_sim_cfg = telemetry_cfg.get("bandwidth_simulation", {})
        self.telemetry_tracker = TelemetryTracker(
            client_ids=self.client_ids,
            client_variances=self.client_variances,
            normalization_bounds=norm_bounds,
            bandwidth_cfg=bw_sim_cfg,
            seed=self.global_seed,
        )

        # 5. Initialize Evaluator
        ref_model = FastKANClassifier(
            layers_hidden=self.layers_hidden,
            grid_size=self.ref_grid_size,
            spline_order=self.ref_spline_order,
        )
        self.evaluator = CentralizedEvaluator(
            model=ref_model,
            test_data=self.test_data,
            batch_size=64,
            device=self.device,
        )

        # Internal state tracking
        self.current_round: int = 0
        self.acc_prev: float = 0.0
        self.f1_prev: float = 0.0
        self.server_parameters: List[np.ndarray] = []
        self.client_models: Dict[int, FastKANClassifier] = {}
        self.clients: Dict[int, FlowerECGClient] = {}
        self.prev_actions: Optional[np.ndarray] = None

    def _load_config(self, config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """Load YAML configuration or return sensible defaults."""
        if config_path is not None:
            path = Path(config_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path.exists():
                try:
                    with open(path, "r") as f:
                        return yaml.safe_load(f) or {}
                except Exception as e:
                    logger.warning(f"Could not load config from {path}: {e}")
        return {}

    def _setup_datasets(
        self,
        client_datasets: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]],
        test_data: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
        """Load preprocessed datasets from disk or generate synthetic splits."""
        if client_datasets is not None and test_data is not None:
            return client_datasets, test_data

        beats_dir = PROJECT_ROOT / "data" / "processed" / "beats"
        partitions_dir = PROJECT_ROOT / "data" / "partitions"

        # Check if preprocessed numpy arrays exist
        x_tr_path = beats_dir / "X_train.npy"
        y_tr_path = beats_dir / "y_train.npy"
        x_te_path = beats_dir / "X_test.npy"
        y_te_path = beats_dir / "y_test.npy"

        if x_tr_path.exists() and y_tr_path.exists() and x_te_path.exists() and y_te_path.exists():
            X_train = np.load(x_tr_path)
            y_train = np.load(y_tr_path)
            X_test = np.load(x_te_path)
            y_test = np.load(y_te_path)
            loaded_test = (X_test, y_test)

            # Check for existing partition indices
            target_pdir = partitions_dir / f"clients_{self.num_clients}_alpha_{self.alpha}"
            alt_pdir = partitions_dir / f"alpha_{self.alpha}"
            pdir = target_pdir if target_pdir.exists() else alt_pdir

            loaded_clients: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            if pdir.exists() and len(list(pdir.glob("client_*.npy"))) == self.num_clients:
                for cid in self.client_ids:
                    indices = np.load(pdir / f"client_{cid}.npy")
                    loaded_clients[cid] = (X_train[indices], y_train[indices])
            else:
                # Generate deterministically
                partitions = dirichlet_partition(
                    y=y_train,
                    num_clients=self.num_clients,
                    alpha=self.alpha,
                    seed=self.global_seed,
                )
                for cid in self.client_ids:
                    idx = partitions[cid]
                    loaded_clients[cid] = (X_train[idx], y_train[idx])

            return loaded_clients, loaded_test

        # Fallback to fast synthetic data for isolated testing
        logger.warning("Preprocessed data not found on disk. Generating synthetic ECG datasets.")
        rng = np.random.RandomState(self.global_seed)
        n_features = self.layers_hidden[0]
        n_classes = self.layers_hidden[-1]

        mock_test_x = rng.randn(200, n_features).astype(np.float32)
        mock_test_y = rng.randint(0, n_classes, size=200).astype(np.int64)
        synthetic_test = (mock_test_x, mock_test_y)

        synthetic_clients = {}
        for cid in self.client_ids:
            c_x = rng.randn(150, n_features).astype(np.float32)
            c_y = rng.randint(0, n_classes, size=150).astype(np.int64)
            synthetic_clients[cid] = (c_x, c_y)

        return synthetic_clients, synthetic_test

    def set_seed(self, seed: Optional[int] = None) -> None:
        """Enforce strict deterministic reproducibility across all generators."""
        if seed is None:
            seed = self.global_seed
        self.global_seed = int(seed)
        set_seed(self.global_seed)
        self.telemetry_tracker.set_seed(self.global_seed)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the federated learning environment to round 0.

        Parameters
        ----------
        seed : Optional[int]
            Random seed for this episode.
        options : Optional[Dict[str, Any]]
            Additional configuration overrides.

        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            (initial_observation, initial_info)
        """
        super().reset(seed=seed)
        if seed is not None:
            self.set_seed(seed)

        self.current_round = 0

        # Initialize reference global model with reference (G_ref, k_ref)
        server_model = FastKANClassifier(
            layers_hidden=self.layers_hidden,
            grid_size=self.ref_grid_size,
            spline_order=self.ref_spline_order,
        ).to(self.device)

        self.server_parameters = [
            val.detach().cpu().numpy() for _, val in server_model.state_dict().items()
        ]

        # Initial zero-round evaluation on test data
        init_loss, init_metrics = self.evaluator.evaluate(self.server_parameters)
        self.acc_prev = float(init_metrics.get("accuracy", 0.0))
        self.f1_prev = float(init_metrics.get("macro_f1", 0.0))

        # Instantiate clients with initial reference architecture
        self.clients = {}
        for cid in self.client_ids:
            c_model = FastKANClassifier(
                layers_hidden=self.layers_hidden,
                grid_size=self.ref_grid_size,
                spline_order=self.ref_spline_order,
            ).to(self.device)

            self.clients[cid] = FlowerECGClient(
                client_id=cid,
                model=c_model,
                train_data=self.client_datasets[cid],
                batch_size=self.batch_size,
                epochs=self.local_epochs,
                learning_rate=self.learning_rate,
                optimizer_type=self.optimizer_type,
                weight_decay=self.weight_decay,
                proximal_mu=self.proximal_mu,
                device=self.device,
            )

        # Reset telemetry tracker
        initial_obs = self.telemetry_tracker.reset(
            initial_losses=float(init_loss), seed=seed
        )

        # Initialize previous actions for action smoothness regularization: - β3 * ||a_t - a_{t-1}||_1
        base_act = gk_to_action(self.ref_grid_size, self.ref_spline_order)
        self.prev_actions = np.full(self.num_clients, base_act, dtype=np.int64)

        info = {
            "round": 0,
            "global_accuracy": self.acc_prev,
            "global_macro_f1": self.f1_prev,
            "global_loss": float(init_loss),
            "telemetry_summary": self.telemetry_tracker.get_summary(),
        }

        return initial_obs, info

    def step(
        self,
        actions: Union[np.ndarray, Sequence[int]],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one communication round of resource-adaptive federated learning.

        Parameters
        ----------
        actions : np.ndarray or Sequence[int]
            Array of length N, where each element is an action index in [0..11]
            specifying (G_i, k_i) for client i.

        Returns
        -------
        observation : np.ndarray
            Updated telemetry matrix S^{t+1} of shape (N, 5) with values in [0.0, 1.0].
        reward : float
            Scalar step reward computed via Equation (7).
        terminated : bool
            Whether the episode reached max_rounds.
        truncated : bool
            Whether the episode was truncated prematurely (False).
        info : Dict[str, Any]
            Diagnostic metrics, reward decomposition, and telemetry statistics.
        """
        actions = np.asarray(actions, dtype=np.int64).flatten()
        if len(actions) != self.num_clients:
            raise ValueError(
                f"Action count mismatch: expected {self.num_clients}, got {len(actions)}."
            )

        round_num = self.current_round + 1
        client_updates: List[Tuple[List[np.ndarray], int]] = []
        round_metrics: Dict[int, Dict[str, Any]] = {}

        client_params_list: List[int] = []
        client_latencies_list: List[float] = []
        chosen_configs: List[Tuple[int, int]] = []

        # 1. Dispatch heterogeneous (G_i, k_i) configurations and train locally
        for cid in self.client_ids:
            act_idx = int(actions[cid])
            g_i, k_i = action_to_gk(act_idx)
            chosen_configs.append((g_i, k_i))

            client = self.clients[cid]

            # Reconfigure client model to server reference weights, then interpolate to (G_i, k_i)
            # A: Ensure client model is on server reference grid
            for l_idx in range(len(client.model.layers)):
                client.model.set_layer_gk(
                    layer_idx=l_idx,
                    grid_size=self.ref_grid_size,
                    spline_order=self.ref_spline_order,
                )
            client.set_parameters(self.server_parameters)

            # B: Dynamic B-spline interpolation to target action (G_i, k_i)
            for l_idx in range(len(client.model.layers)):
                client.model.set_layer_gk(layer_idx=l_idx, grid_size=g_i, spline_order=k_i)

            # Active parameters under action (G_i, k_i)
            active_params = int(client.model.get_parameter_count())
            client_params_list.append(active_params)

            # C: Local training round
            fit_config = {
                "local_epochs": self.local_epochs,
                "proximal_mu": self.proximal_mu,
                "server_round": round_num,
            }
            client_params, n_samples, metrics = client.fit(
                parameters=client.get_parameters(), config=fit_config
            )

            client_latencies_list.append(float(metrics["execution_latency"]))
            round_metrics[cid] = metrics

            # D: Interpolate updated client weights back to reference (G_ref, k_ref) for server aggregation
            for l_idx in range(len(client.model.layers)):
                client.model.set_layer_gk(
                    layer_idx=l_idx,
                    grid_size=self.ref_grid_size,
                    spline_order=self.ref_spline_order,
                )
            resampled_params = client.get_parameters()
            client_updates.append((resampled_params, n_samples))

        # 2. Server Aggregation (Weighted FedAvg on Reference Grid)
        self.server_parameters = aggregate_parameters(
            client_updates, canonical_gk=(self.ref_grid_size, self.ref_spline_order)
        )

        # 3. Global Evaluation on Held-out Patient Data
        test_loss, test_metrics = self.evaluator.evaluate(self.server_parameters)
        acc_curr = float(test_metrics.get("accuracy", 0.0))
        f1_curr = float(test_metrics.get("macro_f1", 0.0))

        # 4. Multi-Objective Reward Calculation (Calibrated Dimensionless Formulation)
        # R_t = 100.0 * (Acc_t - Acc_{t-1}) - β1 * (Mean_Params_t / Baseline_Params) - β2 * (Mean_Latency_t / Baseline_Latency) - β3 * ||a_t - a_{t-1}||_1
        delta_acc = float(acc_curr - self.acc_prev)
        # Safe clipping for numerical gradient stability
        delta_acc_clipped = float(np.clip(delta_acc, -self.delta_clip, self.delta_clip))
        acc_term = float(self.scale_acc * delta_acc_clipped)

        sum_params = float(sum(client_params_list))
        sum_latency = float(sum(client_latencies_list))
        mean_params = float(np.mean(client_params_list))
        mean_latency = float(np.mean(client_latencies_list))

        # Normalized parameter penalty: Choosing configurations above 42,792 incurs an immediate negative penalty
        relative_param_cost = float((sum_params - self.baseline_budget) / self.baseline_budget) if self.baseline_budget > 0 else 0.0
        if relative_param_cost > 0:
            param_penalty = float(self.beta1 * relative_param_cost * 1.5)  # Immediate negative penalty for over-parameterization
        else:
            param_penalty = float(self.beta1 * relative_param_cost)  # Compression incentive when below baseline

        latency_penalty = float(self.beta2 * (mean_latency / max(1e-6, self.baseline_latency)))

        # Action smoothness regularization: - β3 * ||a_t - a_{t-1}||_1
        if self.prev_actions is None:
            base_act = gk_to_action(self.ref_grid_size, self.ref_spline_order)
            self.prev_actions = np.full(self.num_clients, base_act, dtype=np.int64)
        smoothness_norm = float(np.sum(np.abs(actions - self.prev_actions)))
        smoothness_penalty = float(self.beta3 * smoothness_norm)

        raw_reward = float(acc_term - param_penalty - latency_penalty - smoothness_penalty)
        reward = float(np.clip(raw_reward, self.reward_clip_min, self.reward_clip_max))

        # Update previous actions, accuracy, and round counter
        self.prev_actions = actions.copy()
        self.acc_prev = acc_curr
        self.f1_prev = f1_curr
        self.current_round = round_num

        # 5. Update Telemetry and Collect New Observation S^{t+1}
        new_obs = self.telemetry_tracker.update_round(round_metrics)

        # 6. Check Termination
        terminated = bool(self.current_round >= self.max_rounds)
        truncated = False

        savings_pct = (
            float((self.baseline_budget - sum_params) / self.baseline_budget * 100.0)
            if self.baseline_budget > 0
            else 0.0
        )

        # Diagnostic info dictionary
        info: Dict[str, Any] = {
            "round": self.current_round,
            "global_accuracy": acc_curr,
            "global_macro_f1": f1_curr,
            "global_loss": float(test_loss),
            "delta_accuracy": delta_acc,
            "total_parameters": sum_params,
            "mean_parameters_per_client": float(np.mean(client_params_list)),
            "total_latency": sum_latency,
            "mean_latency": float(np.mean(client_latencies_list)),
            "reward_components": {
                "acc_term": acc_term,
                "delta_accuracy": acc_term - smoothness_penalty,
                "raw_delta_acc": delta_acc,
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

    def render(self, mode: str = "ansi") -> Optional[str]:
        """Render the current environment state."""
        summary = (
            f"[Round {self.current_round:02d}/{self.max_rounds:02d}] "
            f"Acc: {self.acc_prev:.4f} | F1: {self.f1_prev:.4f} | "
            f"Clients: {self.num_clients}"
        )
        if mode == "human":
            print(summary)
            return None
        return summary


def run_sanity_check(
    config_path: str = "configs/ppo.yaml",
    num_steps: int = 2,
    seed: int = 42,
) -> None:
    """
    Standalone sanity check CLI: Executes a 2-step random action rollout.
    """
    print("=" * 78)
    print("FEDERATED KAN GYMNASIUM ENVIRONMENT SANITY CHECK")
    print("=" * 78)
    print(f"Config path : {config_path}")
    print(f"Rollout seed: {seed}")
    print(f"Num steps   : {num_steps}")

    env = FederatedKANEnv(config_path=config_path, seed=seed)

    print("\nEnvironment Metadata:")
    print(f"  Observation Space : {env.observation_space}")
    print(f"  Action Space      : {env.action_space}")
    print(f"  Number of Clients : {env.num_clients}")
    print(f"  Max Rounds        : {env.max_rounds}")
    print(f"  Lambda 1 (Params) : {env.lambda1}")
    print(f"  Lambda 2 (Latency): {env.lambda2}")

    # Reset
    obs, info = env.reset(seed=seed)
    print("\n[RESET] Initial Environment State:")
    print(f"  Initial State Shape  : {obs.shape} (dtype: {obs.dtype})")
    print(f"  Observation Range    : [{obs.min():.4f}, {obs.max():.4f}]")
    print(f"  Initial Test Accuracy: {info['global_accuracy']:.4f}")
    print(f"  Initial Test Macro F1: {info['global_macro_f1']:.4f}")

    assert obs.shape == (env.num_clients, 5), f"Unexpected shape {obs.shape}"
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0), "Observation out of bounds [0, 1]!"

    # Perform rollout
    total_reward = 0.0
    for step_idx in range(1, num_steps + 1):
        print(f"\n--- STEP {step_idx}/{num_steps} ---")
        # Sample random discrete actions
        actions = env.action_space.sample()
        print(f"  Sampled Actions (Indices): {actions.tolist()}")
        configs = [action_to_gk(int(a)) for a in actions]
        print(f"  Mapped (G, k) Configs    : {configs}")

        t_start = time.perf_counter()
        next_obs, reward, terminated, truncated, step_info = env.step(actions)
        elapsed = time.perf_counter() - t_start
        total_reward += reward

        print(f"  Step Wall-Clock Duration : {elapsed:.3f} s")
        print(f"  New Observation Shape    : {next_obs.shape}")
        print(f"  Reward (Equation 7)      : {reward:+.6f}")
        rc = step_info["reward_components"]
        print(
            f"    * ΔAcc_global        : {rc['delta_accuracy']:+.6f}\n"
            f"    * -λ1 * Cost(Params) : -{rc['param_penalty']:.6f} (Σ Params: {step_info['total_parameters']:.0f}, Savings: {rc.get('parameter_savings_pct', 0.0):+.1f}%)\n"
            f"    * -λ2 * Σ Latency    : -{rc['latency_penalty']:.6f} (Σ Latency: {step_info['total_latency']:.3f}s)"
        )
        print(
            f"  Global Test Accuracy     : {step_info['global_accuracy']:.4f} "
            f"(Δ: {step_info['delta_accuracy']:+.4f})"
        )
        print(f"  Global Test Macro F1     : {step_info['global_macro_f1']:.4f}")
        print(f"  Terminated: {terminated} | Truncated: {truncated}")

        assert next_obs.shape == (env.num_clients, 5)
        assert np.all(next_obs >= 0.0) and np.all(next_obs <= 1.0)

    print("\n" + "=" * 78)
    print(f"SANITY CHECK COMPLETE: 2-step rollout succeeded! Total Reward: {total_reward:+.6f}")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated KAN Gymnasium Environment Sanity Check")
    parser.add_argument("--config", type=str, default="configs/ppo.yaml", help="Path to PPO config file")
    parser.add_argument("--steps", type=int, default=2, help="Number of steps to roll out")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")
    args = parser.parse_args()

    run_sanity_check(config_path=args.config, num_steps=args.steps, seed=args.seed)
