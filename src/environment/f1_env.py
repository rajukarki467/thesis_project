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
from src.models.fastkan_model import FastKANClassifier, project_spline_weights
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

# Knot-nested configurations relative to canonical (5, 3):
# G in {5, 10} with k=3 (exact knot insertion / subdivision)
NESTED_ACTION_TUPLES: List[Tuple[int, int]] = [
    (5, 3), (10, 3)
]

# Compatible configurations: configs whose round-trip error vs (5, 3) is <= 0.01 (1%)
# Measured in Step B2: excludes all G=3 and (5, 2)
COMPATIBLE_ACTION_TUPLES: List[Tuple[int, int]] = [
    (5, 3), (5, 4), (7, 2), (7, 3), (7, 4), (10, 2), (10, 3), (10, 4)
]


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
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        eval_split: Optional[str] = None,
        aggregation_mode: Optional[str] = None,
        proximal_mu: Optional[float] = None,
        sticky_k: int = 1,
        target_budget_ratio: Optional[float] = None,
        accuracy_metric: Optional[str] = None,
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
        agg_cfg = self.config.get("aggregation", {})
        telemetry_cfg = self.config.get("telemetry", {})

        self.num_clients = int(num_clients if num_clients is not None else env_cfg.get("num_clients", 8))
        self.max_rounds = int(max_rounds if max_rounds is not None else env_cfg.get("max_rounds", 10))
        self.alpha = float(alpha if alpha is not None else env_cfg.get("alpha", 0.5))
        self.global_seed = int(seed if seed is not None else env_cfg.get("seed", 42))

        # Sticky action variant (actions repeat for sticky_k rounds)
        self.sticky_k = int(sticky_k if sticky_k is not None else env_cfg.get("sticky_k", 1))

        # Evaluation split for reward (default: 'val' to prevent test leakage; 'test' supported)
        self.eval_split = str(
            eval_split if eval_split is not None else reward_cfg.get("eval_split", "val")
        ).lower()

        # Aggregation mode: 'project_full', 'project_delta', 'nested_grid', 'compatible_actions_only'
        self.aggregation_mode = str(
            aggregation_mode
            if aggregation_mode is not None
            else agg_cfg.get("mode", fed_cfg.get("aggregation_mode", "project_full"))
        ).lower()

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

        # Target budget ratio relative to baseline budget (B_ref = 42,792 for 8 clients)
        if target_budget_ratio is not None:
            self.target_budget_ratio = float(target_budget_ratio)
        else:
            self.target_budget_ratio = float(reward_cfg.get("target_budget_ratio", 1.0))
        self.target_budget = float(self.baseline_budget * self.target_budget_ratio)

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

        if accuracy_metric is not None:
            self.accuracy_metric = str(accuracy_metric).lower()
        else:
            self.accuracy_metric = str(reward_cfg.get("accuracy_metric", "macro_f1")).lower()

        # Transient down-weighting: down-weight round 1 and 2 initialization transients
        self.transient_downweight = bool(reward_cfg.get("transient_downweight", True))
        self.alpha_abs = float(reward_cfg.get("alpha_abs", 0.5))

        # Federated training hyperparameters
        self.local_epochs = int(fed_cfg.get("local_epochs", 1))
        self.batch_size = int(fed_cfg.get("batch_size", 32))
        self.learning_rate = float(fed_cfg.get("learning_rate", 0.001))
        self.optimizer_type = fed_cfg.get("optimizer", "adam")
        self.weight_decay = float(fed_cfg.get("weight_decay", 1e-4))
        self.proximal_mu = float(proximal_mu if proximal_mu is not None else fed_cfg.get("proximal_mu", 0.0))

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
        # Select action set based on aggregation mode
        if self.aggregation_mode == "nested_grid":
            self.action_tuples = list(NESTED_ACTION_TUPLES)
        elif self.aggregation_mode == "compatible_actions_only":
            self.action_tuples = list(COMPATIBLE_ACTION_TUPLES)
        else:
            self.action_tuples = list(ACTION_TUPLES)

        self.num_actions_per_client = len(self.action_tuples)

        # Observation Space: Matrix S_t in [0.0, 1.0]^{N x 5}
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_clients, 5),
            dtype=np.float32,
        )

        # Action Space: MultiDiscrete choice across N clients
        self.action_space = spaces.MultiDiscrete(
            [self.num_actions_per_client] * self.num_clients
        )

        # 3. Load or generate client partitions and evaluation data
        self.client_ids = list(range(self.num_clients))
        self.client_datasets, self.test_data, self.val_data = self._setup_datasets(
            client_datasets=client_datasets, test_data=test_data, val_data=val_data
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

        # 5. Initialize Evaluators (val evaluator for reward, test evaluator strictly for evaluation)
        ref_model = FastKANClassifier(
            layers_hidden=self.layers_hidden,
            grid_size=self.ref_grid_size,
            spline_order=self.ref_spline_order,
        )
        self.test_evaluator = CentralizedEvaluator(
            model=ref_model,
            test_data=self.test_data,
            batch_size=64,
            device=self.device,
        )
        self.val_evaluator = CentralizedEvaluator(
            model=ref_model,
            test_data=self.val_data,
            batch_size=64,
            device=self.device,
        )
        self.evaluator = self.val_evaluator if self.eval_split == "val" else self.test_evaluator

        # Internal state tracking
        self.current_round: int = 0
        self.acc_prev: float = 0.0
        self.f1_prev: float = 0.0
        self.bacc_prev: float = 0.0
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
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[
        Dict[int, Tuple[np.ndarray, np.ndarray]],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        """Load preprocessed datasets from disk or generate synthetic splits."""
        if client_datasets is not None and test_data is not None:
            v_data = val_data if val_data is not None else test_data
            return client_datasets, test_data, v_data

        beats_dir = PROJECT_ROOT / "data" / "processed" / "beats"
        partitions_dir = PROJECT_ROOT / "data" / "partitions"

        # Check if preprocessed numpy arrays exist
        x_tr_path = beats_dir / "X_train.npy"
        y_tr_path = beats_dir / "y_train.npy"
        x_te_path = beats_dir / "X_test.npy"
        y_te_path = beats_dir / "y_test.npy"
        x_va_path = beats_dir / "X_val.npy"
        y_va_path = beats_dir / "y_val.npy"

        if x_tr_path.exists() and y_tr_path.exists() and x_te_path.exists() and y_te_path.exists():
            X_train = np.load(x_tr_path)
            y_train = np.load(y_tr_path)
            X_test = np.load(x_te_path)
            y_test = np.load(y_te_path)
            loaded_test = (X_test, y_test)

            if val_data is not None:
                loaded_val = val_data
            elif x_va_path.exists() and y_va_path.exists():
                X_val = np.load(x_va_path)
                y_val = np.load(y_va_path)
                loaded_val = (X_val, y_val)
            else:
                loaded_val = loaded_test

            # Check for existing partition indices with seed support
            seed_pdir = partitions_dir / f"clients_{self.num_clients}_alpha_{self.alpha}_seed_{self.global_seed}"
            target_pdir = partitions_dir / f"clients_{self.num_clients}_alpha_{self.alpha}"
            alt_pdir = partitions_dir / f"alpha_{self.alpha}"

            loaded_clients: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            if seed_pdir.exists() and len(list(seed_pdir.glob("client_*.npy"))) == self.num_clients:
                pdir = seed_pdir
            elif self.global_seed == 42 and target_pdir.exists() and len(list(target_pdir.glob("client_*.npy"))) == self.num_clients:
                pdir = target_pdir
            elif self.global_seed == 42 and alt_pdir.exists() and len(list(alt_pdir.glob("client_*.npy"))) == self.num_clients:
                pdir = alt_pdir
            else:
                pdir = None

            if pdir is not None:
                for cid in self.client_ids:
                    indices = np.load(pdir / f"client_{cid}.npy")
                    loaded_clients[cid] = (X_train[indices], y_train[indices])
            else:
                # Generate deterministically and persist
                partitions = dirichlet_partition(
                    y=y_train,
                    num_clients=self.num_clients,
                    alpha=self.alpha,
                    seed=self.global_seed,
                )
                seed_pdir.mkdir(parents=True, exist_ok=True)
                for cid in self.client_ids:
                    idx = partitions[cid]
                    loaded_clients[cid] = (X_train[idx], y_train[idx])
                    np.save(seed_pdir / f"client_{cid}.npy", idx)

            return loaded_clients, loaded_test, loaded_val

        # Fallback to fast synthetic data for isolated testing
        logger.warning("Preprocessed data not found on disk. Generating synthetic ECG datasets.")
        rng = np.random.RandomState(self.global_seed)
        n_features = self.layers_hidden[0]
        n_classes = self.layers_hidden[-1]

        mock_test_x = rng.randn(200, n_features).astype(np.float32)
        mock_test_y = rng.randint(0, n_classes, size=200).astype(np.int64)
        synthetic_test = (mock_test_x, mock_test_y)
        synthetic_val = (mock_test_x, mock_test_y)

        synthetic_clients = {}
        for cid in self.client_ids:
            c_x = rng.randn(150, n_features).astype(np.float32)
            c_y = rng.randint(0, n_classes, size=150).astype(np.int64)
            synthetic_clients[cid] = (c_x, c_y)

        return synthetic_clients, synthetic_test, synthetic_val

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

        # Initial zero-round evaluation on validation split (or test split if eval_split == 'test')
        init_loss, init_metrics = self.evaluator.evaluate(self.server_parameters)
        self.acc_prev = float(init_metrics.get("accuracy", 0.0))
        self.f1_prev = float(init_metrics.get("macro_f1", 0.0))
        self.bacc_prev = float(init_metrics.get("balanced_accuracy", self.acc_prev))

        # Zero leakage: test evaluation strictly for reporting, never influences reward
        init_test_loss, init_test_metrics = (
            self.test_evaluator.evaluate(self.server_parameters)
            if self.eval_split == "val"
            else (init_loss, init_metrics)
        )

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

        # Initialize cumulative communication metrics
        self.cum_params = 0
        self.cum_bytes = 0
        self.cum_MB = 0.0
        self.cum_comm_time_s = 0.0
        self.cum_comm_time_sum_s = 0.0

        info = {
            "round": 0,
            "global_accuracy": float(init_test_metrics["accuracy"]),
            "global_macro_f1": float(init_test_metrics["macro_f1"]),
            "global_balanced_accuracy": float(init_test_metrics.get("balanced_accuracy", init_test_metrics["accuracy"])),
            "global_loss": float(init_test_loss),
            "val_accuracy": self.acc_prev,
            "val_macro_f1": self.f1_prev,
            "val_balanced_accuracy": self.bacc_prev,
            "val_loss": float(init_loss),
            "params_up": 0,
            "params_down": 0,
            "params_round": 0,
            "bytes_round": 0,
            "cum_params": 0,
            "cum_bytes": 0,
            "cum_MB": 0.0,
            "est_comm_time_s": 0.0,
            "est_comm_time_sum_s": 0.0,
            "cum_comm_time_s": 0.0,
            "config_histogram": {},
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

        # Handle sticky actions (actions repeat for sticky_k rounds)
        if self.sticky_k > 1 and self.prev_actions is not None and ((self.current_round % self.sticky_k) != 0):
            actions = self.prev_actions.copy()

        round_num = self.current_round + 1
        client_updates: List[Tuple[List[np.ndarray], int]] = []
        round_metrics: Dict[int, Dict[str, Any]] = {}

        client_params_list: List[int] = []
        client_latencies_list: List[float] = []
        chosen_configs: List[Tuple[int, int]] = []

        # 1. Dispatch heterogeneous (G_i, k_i) configurations and train locally
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
                raise ValueError(
                    f"Action index {act_idx} out of range [0, {len(self.action_tuples)})."
                )

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

            # Record initial local parameters at (g_i, k_i) before training
            init_local_params = [p.copy() for p in client.get_parameters()]

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

            use_delta = self.aggregation_mode in ("project_delta", "nested_grid", "compatible_actions_only")
            if use_delta:
                # D1 (project_delta): Calculate delta at (g_i, k_i), project delta to (ref_grid_size, ref_spline_order)
                delta_params = []
                for p_fin, p_ini in zip(client_params, init_local_params):
                    delta = p_fin - p_ini
                    if delta.ndim == 3 and (g_i, k_i) != (self.ref_grid_size, self.ref_spline_order):
                        d_proj = project_spline_weights(
                            delta,
                            src_gk=(g_i, k_i),
                            dst_gk=(self.ref_grid_size, self.ref_spline_order),
                            grid_range=client.model.grid_range,
                        )
                        delta_params.append(d_proj)
                    else:
                        delta_params.append(delta)
                client_updates.append((delta_params, n_samples))
            else:
                # D2 (project_full): Interpolate updated client weights back to reference (G_ref, k_ref) for server aggregation
                for l_idx in range(len(client.model.layers)):
                    client.model.set_layer_gk(
                        layer_idx=l_idx,
                        grid_size=self.ref_grid_size,
                        spline_order=self.ref_spline_order,
                    )
                resampled_params = client.get_parameters()
                client_updates.append((resampled_params, n_samples))

        # 2. Server Aggregation
        use_delta = self.aggregation_mode in ("project_delta", "nested_grid", "compatible_actions_only")
        if use_delta:
            total_samples = sum(n for _, n in client_updates)
            delta_avg = [np.zeros_like(p, dtype=np.float64) for p in self.server_parameters]
            for delta_p, n_s in client_updates:
                w = float(n_s) / float(total_samples)
                for i, dp in enumerate(delta_p):
                    delta_avg[i] += w * dp.astype(np.float64)
            self.server_parameters = [
                (sp.astype(np.float64) + da).astype(sp.dtype)
                for sp, da in zip(self.server_parameters, delta_avg)
            ]
        else:
            self.server_parameters = aggregate_parameters(
                client_updates, canonical_gk=(self.ref_grid_size, self.ref_spline_order)
            )

        # 3. Global Evaluation (validation for reward, test strictly for evaluation)
        eval_loss, eval_metrics = self.evaluator.evaluate(self.server_parameters)
        acc_curr = float(eval_metrics.get("accuracy", 0.0))
        f1_curr = float(eval_metrics.get("macro_f1", 0.0))
        bacc_curr = float(eval_metrics.get("balanced_accuracy", acc_curr))

        # Zero leakage: test evaluation strictly for reporting, never influences reward
        test_loss, test_metrics = (
            self.test_evaluator.evaluate(self.server_parameters)
            if self.eval_split == "val"
            else (eval_loss, eval_metrics)
        )

        # 4. Multi-Objective Reward Calculation (Calibrated Dimensionless Formulation)
        if self.accuracy_metric == "macro_f1":
            metric_curr = f1_curr
            metric_prev = self.f1_prev
        elif self.accuracy_metric == "balanced_accuracy":
            metric_curr = bacc_curr
            metric_prev = self.bacc_prev
        else:
            metric_curr = acc_curr
            metric_prev = self.acc_prev

        raw_delta = float(metric_curr - metric_prev)

        # Transient down-weighting: down-weight round 1 and 2 initialization transients
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

        # Small absolute level bonus to provide steady gradient towards high accuracy
        abs_level_bonus = float(self.alpha_abs * max(0.0, metric_curr - 0.20))
        acc_term = float(self.scale_acc * delta_acc_clipped + abs_level_bonus)

        sum_params = float(sum(client_params_list))
        sum_latency = float(sum(client_latencies_list))
        mean_params = float(np.mean(client_params_list))
        mean_latency = float(np.mean(client_latencies_list))

        # Budget constraint relative to fixed (5,3) fleet (B_ref = 42,792)
        relative_param_cost = (
            float((sum_params - self.target_budget) / self.baseline_budget)
            if self.baseline_budget > 0
            else 0.0
        )
        if relative_param_cost > 0:
            # Asymmetric penalty: heavily punish exceeding budget (linear + quadratic)
            param_penalty = float(
                self.beta1 * (relative_param_cost + 4.0 * (relative_param_cost ** 2))
            )
        else:
            # Compression incentive when below budget (param_penalty < 0 => reward bonus)
            param_penalty = float(self.beta1 * relative_param_cost)

        latency_penalty = float(self.beta2 * (mean_latency / max(1e-6, self.baseline_latency)))

        # Action smoothness regularization: - β3 * ||a_t - a_{t-1}||_1
        if self.prev_actions is None:
            base_act = 0 if len(self.action_tuples) != NUM_ACTIONS_PER_CLIENT else gk_to_action(self.ref_grid_size, self.ref_spline_order)
            self.prev_actions = np.full(self.num_clients, base_act, dtype=np.int64)

        smoothness_norm = float(np.sum(np.abs(actions - self.prev_actions)))
        smoothness_penalty = float(self.beta3 * smoothness_norm)
        config_changes = int(np.sum(actions != self.prev_actions))

        raw_reward = float(acc_term - param_penalty - latency_penalty - smoothness_penalty)
        reward = float(np.clip(raw_reward, self.reward_clip_min, self.reward_clip_max))

        # Update previous actions, accuracy, and round counter
        self.prev_actions = actions.copy()
        self.acc_prev = acc_curr
        self.f1_prev = f1_curr
        self.bacc_prev = bacc_curr
        self.current_round = round_num

        # 5. Update Telemetry and Collect New Observation S^{t+1}
        new_obs = self.telemetry_tracker.update_round(round_metrics)
        assert np.all(np.isfinite(new_obs)), f"Non-finite values encountered in observation: {new_obs}"
        new_obs = np.clip(new_obs, 0.0, 1.0)

        # 6. Check Termination
        terminated = bool(self.current_round >= self.max_rounds)
        truncated = False

        savings_pct = (
            float((self.baseline_budget - sum_params) / self.baseline_budget * 100.0)
            if self.baseline_budget > 0
            else 0.0
        )

        # Standardized Communication Tracking (Downlink projected + Uplink delta, float32)
        params_up = int(sum_params)
        params_down = int(sum_params)
        params_round = int(params_up + params_down)
        bytes_round = int(params_round * 4)
        self.cum_params += params_round
        self.cum_bytes += bytes_round
        self.cum_MB = float(self.cum_bytes / (1024.0 ** 2))

        # Per-client communication time calculation (slowest bottleneck and total sum)
        comm_times = []
        for cid in self.client_ids:
            bw_i = float(self.telemetry_tracker.clients[str(cid)].latest_bandwidth)
            c_bytes = float(client_params_list[cid] * 2 * 4)  # Uplink + Downlink bytes
            c_time = (c_bytes * 8.0) / (bw_i * 1e6)          # bytes * 8 bits / (Mbps * 1e6)
            comm_times.append(c_time)

        est_comm_time_s = float(max(comm_times)) if comm_times else 0.0
        est_comm_time_sum_s = float(sum(comm_times)) if comm_times else 0.0
        self.cum_comm_time_s += est_comm_time_s
        self.cum_comm_time_sum_s += est_comm_time_sum_s

        # Configuration Histogram (fraction of clients per (G,k) config)
        config_hist: Dict[str, float] = {}
        for g, k in ACTION_TUPLES:
            cnt = sum(1 for (cg, ck) in chosen_configs if cg == g and ck == k)
            if cnt > 0:
                config_hist[f"({g},{k})"] = round(cnt / float(self.num_clients), 4)

        # Diagnostic info dictionary
        info: Dict[str, Any] = {
            "round": self.current_round,
            "global_accuracy": float(test_metrics["accuracy"]),
            "global_macro_f1": float(test_metrics["macro_f1"]),
            "global_balanced_accuracy": float(test_metrics.get("balanced_accuracy", test_metrics["accuracy"])),
            "global_loss": float(test_loss),
            "val_accuracy": acc_curr,
            "val_macro_f1": f1_curr,
            "val_balanced_accuracy": bacc_curr,
            "val_loss": float(eval_loss),
            "delta_accuracy": raw_delta,
            "effective_delta_accuracy": effective_delta,
            "config_changes": config_changes,
            "total_parameters": sum_params,
            "mean_parameters_per_client": float(np.mean(client_params_list)),
            "total_latency": sum_latency,
            "mean_latency": float(np.mean(client_latencies_list)),
            "params_up": params_up,
            "params_down": params_down,
            "params_round": params_round,
            "bytes_round": bytes_round,
            "cum_params": self.cum_params,
            "cum_bytes": self.cum_bytes,
            "cum_MB": self.cum_MB,
            "est_comm_time_s": est_comm_time_s,
            "est_comm_time_sum_s": est_comm_time_sum_s,
            "cum_comm_time_s": self.cum_comm_time_s,
            "config_histogram": config_hist,
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
