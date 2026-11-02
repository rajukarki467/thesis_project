"""
Gymnasium Environment for Resource-Adaptive Federated Kolmogorov-Arnold Networks (KANs).

Reinforcement Learning Controller Environment:
Mitigates policy collapse (aggressive parameter shrinking at the expense of accuracy) via:
1. Strict Accuracy-Constrained Reward Formulation:
   - If current global accuracy drops below previous round's accuracy minus a 0.5% tolerance threshold:
       Apply exponential penalty: -10.0 * exp(|acc_diff|)  (or -10.0 * abs(acc_diff) if linear mode)
   - Positive accuracy improvement scaled higher: acc_diff * 100.0 (performance is primary target)
   - Parameter compression ratio CR = (B_ref - Params) / B_ref treated as secondary booster (+w_boost * CR),
     strictly gated so compression is never rewarded if the accuracy constraint is violated.
2. 5-Dimensional Client State Vector S_i^t in [-1.0, 1.0]:
   S_i^t = [L_local^i, Var(x^i), Δτ^i, bw^i, H(y^i)]
   where H(y^i) is Shannon entropy of client local class labels (quantifying non-IID data skew),
   and all 5 components are normalized strictly between [-1.0, 1.0].
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from src.environment.f1_env import (
    ACTION_TO_INDEX,
    ACTION_TUPLES,
    COMPATIBLE_ACTION_TUPLES,
    GRID_SIZES,
    NESTED_ACTION_TUPLES,
    NUM_ACTIONS_PER_CLIENT,
    SPLINE_ORDERS,
    FederatedKANEnv as BaseFederatedKANEnv,
    action_to_gk,
    gk_to_action,
)
from src.environment.telemetry import (
    ClientHardwareProfile,
    ClientTelemetry,
    TelemetryTracker,
    compute_label_entropy,
    create_heterogeneous_profiles,
)

logger = logging.getLogger("rl_controller.env")


def extract_client_state_vector(
    local_loss: float,
    input_variance: float,
    latency: float,
    bandwidth: float,
    label_entropy: float,
    bounds: Optional[Dict[str, float]] = None,
    num_classes: int = 5,
    normalize: bool = True,
) -> np.ndarray:
    """
    Extract and normalize the 5-dimensional client state vector S_i^t:
        S_i^t = [L_local^i, Var(x^i), Δτ^i, bw^i, H(y^i)]

    Parameters
    ----------
    local_loss : float
        Cross-entropy loss on local training batch.
    input_variance : float
        Empirical variance of input features Var(x^i).
    latency : float
        Wall-clock step latency in seconds.
    bandwidth : float
        Channel bandwidth in Mbps.
    label_entropy : float
        Shannon entropy of local client training labels H(y^i).
    bounds : Optional[Dict[str, float]]
        Normalization bounds for min-max scaling.
    num_classes : int
        Number of task classes (for maximum entropy ln(C)).
    normalize : bool
        If True, maps features into [-1.0, 1.0]; otherwise returns raw physical values.

    Returns
    -------
    np.ndarray
        1D array of shape (5,), dtype np.float32 bounded in [-1.0, 1.0] when normalize=True.
    """
    raw = np.array(
        [
            float(np.nan_to_num(local_loss, nan=1.0, posinf=5.0, neginf=0.0)),
            float(np.nan_to_num(input_variance, nan=1.0, posinf=2.0, neginf=0.0)),
            float(np.nan_to_num(latency, nan=0.5, posinf=10.0, neginf=0.0)),
            float(np.nan_to_num(bandwidth, nan=5.0, posinf=20.0, neginf=1.0)),
            float(np.nan_to_num(label_entropy, nan=0.0, posinf=5.0, neginf=0.0)),
        ],
        dtype=np.float32,
    )

    if not normalize:
        return raw

    b = bounds or {}
    loss_min = float(b.get("loss_min", 0.0))
    loss_max = float(b.get("loss_max", 5.0))
    u_loss = np.clip((raw[0] - loss_min) / max(1e-6, loss_max - loss_min), 0.0, 1.0)
    norm_loss = 2.0 * u_loss - 1.0

    var_min = float(b.get("variance_min", 0.0))
    var_max = float(b.get("variance_max", 2.0))
    u_var = np.clip((raw[1] - var_min) / max(1e-6, var_max - var_min), 0.0, 1.0)
    norm_var = 2.0 * u_var - 1.0

    lat_min = float(b.get("latency_min", 0.0))
    lat_max = float(b.get("latency_max", 10.0))
    u_lat = np.clip((raw[2] - lat_min) / max(1e-6, lat_max - lat_min), 0.0, 1.0)
    norm_lat = 2.0 * u_lat - 1.0

    bw_min = float(b.get("bandwidth_min", 1.0))
    bw_max = float(b.get("bandwidth_max", 20.0))
    u_bw = np.clip((raw[3] - bw_min) / max(1e-6, bw_max - bw_min), 0.0, 1.0)
    norm_bw = 2.0 * u_bw - 1.0

    ent_min = float(b.get("entropy_min", 0.0))
    ent_max = float(b.get("entropy_max", np.log(max(2, num_classes))))
    u_ent = np.clip((raw[4] - ent_min) / max(1e-6, ent_max - ent_min), 0.0, 1.0)
    norm_ent = 2.0 * u_ent - 1.0

    out = np.array([norm_loss, norm_var, norm_lat, norm_bw, norm_ent], dtype=np.float32)
    out = np.clip(out, -1.0, 1.0)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


def compute_accuracy_constrained_reward(
    acc_curr: float,
    acc_prev: float,
    sum_params: float,
    baseline_budget: float,
    mean_latency: float,
    baseline_latency: float = 0.5,
    tolerance_threshold: float = 0.005,
    scale_acc: float = 100.0,
    acc_penalty_mode: str = "exponential",
    compression_booster_weight: float = 0.20,
    beta1: float = 2.0,
    beta2: float = 0.1,
    smoothness_penalty: float = 0.0,
    reward_clip_min: float = -5.0,
    reward_clip_max: float = 5.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate the strict accuracy-constrained multi-objective step reward:

    1. Accuracy Constraint:
       acc_diff = acc_curr - acc_prev
       - If acc_diff < -tolerance_threshold (0.5% drop):
           Exponential penalty: -10.0 * exp(|acc_diff|)
       - If acc_diff > 0:
           Scaled positive improvement: acc_diff * scale_acc (100.0)
       - If -tolerance_threshold <= acc_diff <= 0:
           acc_diff * scale_acc (within tolerance)

    2. Secondary Compression Booster:
       compression_ratio = max(0.0, (baseline_budget - sum_params) / baseline_budget)
       - If sum_params > baseline_budget:
           param_penalty = beta1 * (rel_cost + 2 * rel_cost^2)
           compression_booster = 0.0
       - If sum_params <= baseline_budget:
           param_penalty = 0.0
           compression_booster = compression_booster_weight * compression_ratio (gated on accuracy!)

    3. Latency & Smoothness Regularization:
       latency_penalty = beta2 * (mean_latency / baseline_latency)

    Returns
    -------
    Tuple[float, Dict[str, Any]]
        (reward, reward_components_dict)
    """
    acc_diff = float(acc_curr - acc_prev)

    # 1. Strict Accuracy Constraint
    if acc_diff < -tolerance_threshold:
        if str(acc_penalty_mode).lower() == "linear":
            acc_term = -10.0 * abs(acc_diff)
        else:
            acc_term = -10.0 * float(np.exp(abs(acc_diff)))
        accuracy_collapsed = True
    elif acc_diff > 0:
        acc_term = float(acc_diff * scale_acc)
        accuracy_collapsed = False
    else:
        acc_term = float(acc_diff * scale_acc)
        accuracy_collapsed = False

    # 2. Parameter Compression as Secondary Booster
    relative_param_cost = (
        float((sum_params - baseline_budget) / baseline_budget)
        if baseline_budget > 0
        else 0.0
    )
    compression_ratio = max(0.0, -relative_param_cost)

    if relative_param_cost > 0:
        param_penalty = float(beta1 * (relative_param_cost + 2.0 * (relative_param_cost ** 2)))
        compression_booster = 0.0
    else:
        param_penalty = 0.0
        if accuracy_collapsed:
            # Gated: Never boost parameter shrinking when accuracy dropped below tolerance!
            compression_booster = 0.0
        else:
            compression_booster = float(compression_booster_weight * compression_ratio)

    # 3. Latency penalty
    latency_penalty = float(beta2 * (mean_latency / max(1e-6, baseline_latency)))

    raw_reward = float(
        acc_term + compression_booster - param_penalty - latency_penalty - smoothness_penalty
    )
    reward = float(np.clip(raw_reward, reward_clip_min, reward_clip_max))

    components = {
        "acc_term": acc_term,
        "delta_accuracy": acc_term - smoothness_penalty + compression_booster,
        "raw_delta_acc": acc_diff,
        "acc_diff": acc_diff,
        "accuracy_collapsed": accuracy_collapsed,
        "compression_ratio": compression_ratio,
        "compression_booster": compression_booster,
        "param_penalty": param_penalty,
        "relative_param_cost": relative_param_cost,
        "latency_penalty": latency_penalty,
        "smoothness_penalty": smoothness_penalty,
        "raw_reward": raw_reward,
        "total_reward": reward,
    }
    return reward, components


class FederatedKANEnv(BaseFederatedKANEnv):
    """
    RL Controller Environment for Resource-Adaptive Federated Kolmogorov-Arnold Networks.

    Subclasses BaseFederatedKANEnv and enforces:
    1. Observation Space in [-1.0, 1.0]^{N x 5}:
       S_i^t = [L_local^i, Var(x^i), Δτ^i, bw^i, H(y^i)]
       Includes local label Shannon entropy H(y^i) to quantify non-IID data skew.
    2. Strict Accuracy-Constrained Step Reward:
       - Exponential penalty (-10.0 * exp(|acc_diff|)) on accuracy drops exceeding 0.5% tolerance.
       - Positive accuracy improvement scaled higher (acc_diff * 100.0).
       - Parameter compression ratio treated as secondary booster, gated on accuracy preservation.
    """

    metadata = {"render_modes": ["ansi", "human"], "name": "FederatedKANEnv-v1"}

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
        super().__init__(
            config_path=config_path,
            client_datasets=client_datasets,
            test_data=test_data,
            val_data=val_data,
            eval_split=eval_split,
            aggregation_mode=aggregation_mode,
            proximal_mu=proximal_mu,
            sticky_k=sticky_k,
            target_budget_ratio=target_budget_ratio,
            accuracy_metric=accuracy_metric,
            num_clients=num_clients,
            max_rounds=max_rounds,
            lambda1=lambda1,
            lambda2=lambda2,
            alpha=alpha,
            normalize_params=normalize_params,
            baseline_budget=baseline_budget,
            device=device,
            seed=seed,
        )

        # Enforce [-1.0, 1.0] observation space
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_clients, 5),
            dtype=np.float32,
        )


# Alias for explicit controller naming
RLControllerEnv = FederatedKANEnv

__all__ = [
    "FederatedKANEnv",
    "RLControllerEnv",
    "GRID_SIZES",
    "SPLINE_ORDERS",
    "ACTION_TUPLES",
    "ACTION_TO_INDEX",
    "NUM_ACTIONS_PER_CLIENT",
    "action_to_gk",
    "gk_to_action",
    "compute_label_entropy",
    "compute_accuracy_constrained_reward",
    "extract_client_state_vector",
]
