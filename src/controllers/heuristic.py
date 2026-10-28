"""
Deterministic Rule-Based Fallback Heuristic Controller for Federated KAN.

Mathematical Formulation (Section 4.2 of Thesis Proposal):
For each client i at communication round t, a client complexity score is computed:
    C_i = w1 * L_local^i + w2 * Var(x^i) + w3 * (1.0 / bw^i)

Quantile-based allocation assigns (G_i, k_i) spline edge configurations:
- Top 25% complexity / high resource clients: Fine configuration (G=10, k=4)
- Middle 50% complexity clients: Medium configuration (G=5, k=3)
- Bottom 25% complexity / constrained clients: Coarse configuration (G=3, k=2)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from src.environment.f1_env import gk_to_action

logger = logging.getLogger("heuristic_controller")


class HeuristicController:
    """
    Deterministic Rule-Based Fallback Controller.

    Evaluates client telemetry (local loss, empirical input variance, and available bandwidth)
    to compute a composite complexity score C_i, and assigns adaptive (G, k) parameters
    using quantile-based resource allocation.

    Parameters
    ----------
    config_path : Optional[str | Path]
        Path to YAML config (default 'configs/ppo.yaml').
    w1 : float
        Weight for local cross-entropy loss L_local (default 1.0).
    w2 : float
        Weight for empirical input variance Var(x) (default 1.0).
    w3 : float
        Weight for reciprocal link bandwidth 1 / bw (default 1.0).
    high_cfg : Tuple[int, int]
        Fine resolution configuration for top quantile (default (10, 4)).
    med_cfg : Tuple[int, int]
        Medium resolution configuration for median quantile (default (5, 3)).
    low_cfg : Tuple[int, int]
        Coarse resolution configuration for bottom quantile (default (3, 2)).
    high_percentile : float
        Percentile cutoff for fine configuration (default 75.0, top 25%).
    low_percentile : float
        Percentile cutoff for coarse configuration (default 25.0, bottom 25%).
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = "configs/ppo.yaml",
        w1: Optional[float] = None,
        w2: Optional[float] = None,
        w3: Optional[float] = None,
        high_cfg: Optional[Tuple[int, int]] = None,
        med_cfg: Optional[Tuple[int, int]] = None,
        low_cfg: Optional[Tuple[int, int]] = None,
        high_percentile: Optional[float] = None,
        low_percentile: Optional[float] = None,
    ):
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

        # Quantile thresholds
        self.high_percentile = float(
            high_percentile if high_percentile is not None else q_cfg.get("high_percentile", 75.0)
        )
        self.low_percentile = float(
            low_percentile if low_percentile is not None else q_cfg.get("low_percentile", 25.0)
        )

        # Spline (G, k) configurations
        self.high_cfg = tuple(high_cfg if high_cfg is not None else c_cfg.get("high", [10, 4]))
        self.med_cfg = tuple(med_cfg if med_cfg is not None else c_cfg.get("medium", [5, 3]))
        self.low_cfg = tuple(low_cfg if low_cfg is not None else c_cfg.get("low", [3, 2]))

        # Map to action indices
        self.high_action = gk_to_action(self.high_cfg[0], self.high_cfg[1])
        self.med_action = gk_to_action(self.med_cfg[0], self.med_cfg[1])
        self.low_action = gk_to_action(self.low_cfg[0], self.low_cfg[1])

        # Telemetry normalization bounds for un-normalization
        self.loss_max = float(t_cfg.get("loss_max", 5.0))
        self.variance_max = float(t_cfg.get("variance_max", 2.0))
        self.bandwidth_min = float(t_cfg.get("bandwidth_min", 1.0))
        self.bandwidth_max = float(t_cfg.get("bandwidth_max", 10.0))

    def _load_config(self, config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """Load YAML configuration or return empty dict."""
        if config_path is not None:
            path = Path(config_path)
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

        Parameters
        ----------
        observation : np.ndarray
            State matrix of shape (N, 5) or flattened (N * 5,).
            Observation columns: [loss, variance, latency, bandwidth, curvature].
        raw_telemetry : Optional[List[Dict[str, Any]]]
            Optional list of raw telemetry dictionaries per client.

        Returns
        -------
        np.ndarray
            1D array of length N containing complexity scores C_i.
        """
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 5)

        num_clients = obs.shape[0]

        if raw_telemetry is not None and len(raw_telemetry) == num_clients:
            losses = np.array([float(t.get("local_loss", 1.0)) for t in raw_telemetry])
            variances = np.array([float(t.get("input_variance", 1.0)) for t in raw_telemetry])
            bandwidths = np.array([float(t.get("bandwidth", 5.0)) for t in raw_telemetry])
        else:
            # Un-normalize features from observation matrix
            norm_loss = obs[:, 0]
            norm_var = obs[:, 1]
            norm_bw = obs[:, 3]

            losses = norm_loss * self.loss_max
            variances = norm_var * self.variance_max
            bw_range = max(1e-6, self.bandwidth_max - self.bandwidth_min)
            bandwidths = self.bandwidth_min + norm_bw * bw_range

        # Clip bandwidth to prevent division by zero
        safe_bw = np.maximum(bandwidths, 1e-4)
        inv_bw = 1.0 / safe_bw

        # Compute complexity score
        complexity_scores = (
            self.w1 * losses + self.w2 * variances + self.w3 * inv_bw
        )
        return complexity_scores.astype(np.float32)

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        raw_telemetry: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """
        Select adaptive (G_i, k_i) actions for all N clients based on quantile scoring.

        Parameters
        ----------
        observation : np.ndarray
            Observation matrix of shape (N, 5).
        deterministic : bool
            Maintained for compatibility with Stable-Baselines3 API.
        raw_telemetry : Optional[List[Dict[str, Any]]]
            Optional un-normalized telemetry metrics per client.

        Returns
        -------
        np.ndarray
            1D array of shape (N,) containing action indices in [0..11].
        """
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 5)

        num_clients = obs.shape[0]
        scores = self.compute_complexity_scores(obs, raw_telemetry=raw_telemetry)

        # Handle small client counts (e.g. N < 4)
        if num_clients < 4:
            median_score = float(np.median(scores))
            actions = np.zeros(num_clients, dtype=np.int64)
            for i in range(num_clients):
                if scores[i] >= median_score:
                    actions[i] = self.high_action
                else:
                    actions[i] = self.low_action
            return actions

        # Quantile thresholds
        q_high = float(np.percentile(scores, self.high_percentile))
        q_low = float(np.percentile(scores, self.low_percentile))

        # Sort ranks for strict quantile partition
        # top 25%: highest complexity -> fine configuration
        # bottom 25%: lowest complexity -> coarse configuration
        # middle 50%: medium configuration
        ranks = np.argsort(np.argsort(scores))  # 0 is lowest score, N-1 is highest score
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

    def explain_decision(
        self,
        observation: np.ndarray,
        raw_telemetry: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate detailed explanation of complexity scoring and action assignments.
        """
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 5)

        num_clients = obs.shape[0]
        scores = self.compute_complexity_scores(obs, raw_telemetry=raw_telemetry)
        actions = self.predict(obs, raw_telemetry=raw_telemetry)

        ranks = np.argsort(np.argsort(scores))

        breakdown = []
        for cid in range(num_clients):
            act = int(actions[cid])
            if act == self.high_action:
                tier = "high"
                cfg = self.high_cfg
            elif act == self.low_action:
                tier = "low"
                cfg = self.low_cfg
            else:
                tier = "medium"
                cfg = self.med_cfg

            breakdown.append({
                "client_id": cid,
                "complexity_score": float(scores[cid]),
                "rank": int(ranks[cid]),
                "tier": tier,
                "action_idx": act,
                "grid_size": cfg[0],
                "spline_order": cfg[1],
            })
        return breakdown
