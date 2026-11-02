"""
Client-Side and Server-Side Telemetry Aggregation Tracker for Federated KAN.

Mathematical Formulation (Section 3.2.4 & 3.2.5):
For each client i at communication round t, the 5-dimensional state vector is:
    S_i^t = [L_local^i, Var(x^i), Δτ^i, bw^i, κ^i]
where:
    - L_local^i: Most recent local training cross-entropy loss.
    - Var(x^i): Variance of input features computed over client local batch data.
    - Δτ^i: Wall-clock latency of the local training step in seconds.
    - bw^i: Simulated available bandwidth in Mbps (drawn from uniform or lognormal).
    - κ^i: Curvature proxy derived from the second-difference statistics of the loss trajectory:
           κ_t^i = |L_t^i - 2 * L_{t-1}^i + L_{t-2}^i|  (for t >= 2)
           κ_1^i = |L_1^i - L_0^i|                     (for t = 1)
           κ_0^i = 0.0                                 (for t = 0)

All features are normalized into standard numerical ranges [0.0, 1.0] for stable RL policy inputs.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("telemetry")


@dataclass
class ClientHardwareProfile:
    """
    Hardware profile capturing client physical compute and network constraints.

    Parameters
    ----------
    tier : str
        'edge' (constrained device) or 'compute' (high-performance node).
    min_bandwidth_mbps : float
        Minimum downlink/uplink channel throughput in Mbps.
    max_bandwidth_mbps : float
        Maximum downlink/uplink channel throughput in Mbps.
    compute_time_multiplier : float
        Multiplier scaling computation time relative to baseline benchmark.
    base_latency : float
        Base hardware latency in seconds.
    description : str
        Human-readable description of device category.
    """

    tier: str = "edge"
    min_bandwidth_mbps: float = 1.0
    max_bandwidth_mbps: float = 2.0
    compute_time_multiplier: float = 2.5
    base_latency: float = 1.2
    description: str = "Constrained Edge Device (1-2 Mbps, high latency)"


EDGE_HARDWARE_PROFILE = ClientHardwareProfile(
    tier="edge",
    min_bandwidth_mbps=1.0,
    max_bandwidth_mbps=2.0,
    compute_time_multiplier=2.5,
    base_latency=1.2,
    description="Constrained Edge Device (1-2 Mbps, high latency)",
)

COMPUTE_HARDWARE_PROFILE = ClientHardwareProfile(
    tier="compute",
    min_bandwidth_mbps=10.0,
    max_bandwidth_mbps=20.0,
    compute_time_multiplier=0.2,
    base_latency=0.1,
    description="High-Compute Node (10-20 Mbps, fast compute)",
)


def create_heterogeneous_profiles(
    client_ids: Sequence[Union[int, str]],
    edge_ratio: float = 0.5,
) -> Dict[str, ClientHardwareProfile]:
    """
    Generate realistic heterogeneous hardware profiles across simulated clients.
    By default: 50% edge tier (1-2 Mbps, slow compute), 50% server/high-compute tier (10-20 Mbps, fast compute).

    Parameters
    ----------
    client_ids : Sequence[Union[int, str]]
        List of client identifiers.
    edge_ratio : float
        Fraction of clients assigned to constrained edge tier (default 0.5).

    Returns
    -------
    Dict[str, ClientHardwareProfile]
        Mapping from client_id string to hardware profile.
    """
    c_ids = [str(cid) for cid in client_ids]
    n_clients = len(c_ids)
    n_edge = int(round(n_clients * edge_ratio))
    profiles: Dict[str, ClientHardwareProfile] = {}
    for idx, cid in enumerate(c_ids):
        if idx < n_edge:
            profiles[cid] = copy.deepcopy(EDGE_HARDWARE_PROFILE)
        else:
            profiles[cid] = copy.deepcopy(COMPUTE_HARDWARE_PROFILE)
    return profiles


class BandwidthSimulator:
    """
    Synthetic Network Bandwidth Simulator for Edge Clients.

    Simulates client downlink/uplink channel conditions drawn from
    uniform or log-normal throughput distributions, with optional per-client
    hardware profile tier constraints.

    Parameters
    ----------
    distribution : str
        'uniform' or 'lognormal' (default 'uniform').
    min_mbps : float
        Lower bound of throughput in Mbps (default 1.0).
    max_mbps : float
        Upper bound of throughput in Mbps (default 10.0).
    lognormal_mu : float
        Location parameter for log-normal distribution (default 1.6, median ~5.0 Mbps).
    lognormal_sigma : float
        Scale parameter for log-normal distribution (default 0.5).
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        distribution: str = "uniform",
        min_mbps: float = 1.0,
        max_mbps: float = 10.0,
        lognormal_mu: float = 1.6,
        lognormal_sigma: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.distribution = distribution.lower()
        self.min_mbps = float(min_mbps)
        self.max_mbps = float(max_mbps)
        self.lognormal_mu = float(lognormal_mu)
        self.lognormal_sigma = float(lognormal_sigma)
        self.rng = np.random.RandomState(seed)

    def set_seed(self, seed: Optional[int] = None) -> None:
        """Reset the internal RNG seed."""
        self.rng = np.random.RandomState(seed)

    def sample(
        self,
        num_clients: int = 1,
        profiles: Optional[Sequence[ClientHardwareProfile]] = None,
    ) -> np.ndarray:
        """
        Sample bandwidth values for participating clients.

        Parameters
        ----------
        num_clients : int
            Number of clients to sample for.
        profiles : Optional[Sequence[ClientHardwareProfile]]
            Optional hardware profiles per client specifying individual min/max bandwidth.

        Returns
        -------
        np.ndarray
            Array of bandwidth values in Mbps, clipped to bounds.
        """
        if profiles is not None and len(profiles) == num_clients:
            sampled = []
            for prof in profiles:
                min_b = float(prof.min_bandwidth_mbps)
                max_b = float(prof.max_bandwidth_mbps)
                if self.distribution == "lognormal":
                    mid = float(np.sqrt(max(1e-3, min_b * max_b)))
                    raw = float(self.rng.lognormal(mean=np.log(mid), sigma=0.15))
                else:
                    raw = float(self.rng.uniform(low=min_b, high=max_b))
                sampled.append(float(np.clip(raw, min_b, max_b)))
            return np.array(sampled, dtype=np.float32)

        if self.distribution == "lognormal":
            raw = self.rng.lognormal(
                mean=self.lognormal_mu, sigma=self.lognormal_sigma, size=num_clients
            )
        else:
            raw = self.rng.uniform(
                low=self.min_mbps, high=self.max_mbps, size=num_clients
            )
        return np.clip(raw, self.min_mbps, self.max_mbps).astype(np.float32)


class ClientTelemetry:
    """
    Client-level telemetry tracker.

    Tracks loss trajectory, empirical input variance, step latency,
    channel bandwidth, loss curvature, and hardware constraints for a single client.

    Parameters
    ----------
    client_id : int or str
        Unique identifier of the client.
    input_variance : float
        Empirical variance of input features Var(x^i).
    normalization_bounds : Optional[Dict[str, float]]
        Scaling parameters for normalizing features into [0, 1].
    hardware_profile : Optional[ClientHardwareProfile]
        Physical device constraints (bandwidth range, compute latency multiplier).
    """

    def __init__(
        self,
        client_id: Union[int, str],
        input_variance: float = 1.0,
        normalization_bounds: Optional[Dict[str, float]] = None,
        hardware_profile: Optional[ClientHardwareProfile] = None,
    ):
        self.client_id = str(client_id)
        self.input_variance = float(input_variance)
        self.hardware_profile = (
            copy.deepcopy(hardware_profile)
            if hardware_profile is not None
            else copy.deepcopy(EDGE_HARDWARE_PROFILE)
        )

        self.loss_history: List[float] = []
        self.latency_history: List[float] = []
        self.bandwidth_history: List[float] = []

        # Current state components
        self.latest_loss: float = 1.0
        self.latest_latency: float = 0.5
        self.latest_bandwidth: float = 5.0
        self.latest_curvature: float = 0.0

        # Runtime status and straggler flag
        self.is_straggler: bool = False
        self.status: str = "active"

        # Default normalization bounds: [0, 1] range mapping
        self.bounds = {
            "loss_max": 5.0,
            "variance_max": 2.0,
            "latency_max": 10.0,
            "bandwidth_min": 1.0,
            "bandwidth_max": 20.0,
            "curvature_max": 2.0,
        }
        if normalization_bounds is not None:
            self.bounds.update(normalization_bounds)

    def update(
        self,
        local_loss: float,
        latency: float,
        bandwidth: Optional[float] = None,
        input_variance: Optional[float] = None,
        is_straggler: bool = False,
        status: str = "active",
    ) -> None:
        """
        Record new telemetry values for the current communication round.

        Parameters
        ----------
        local_loss : float
            Cross-entropy loss computed during local training.
        latency : float
            Execution wall-clock latency in seconds.
        bandwidth : Optional[float]
            Available link bandwidth in Mbps.
        input_variance : Optional[float]
            Updated input variance (if changed).
        is_straggler : bool
            Whether client exceeded round timeout deadline.
        status : str
            Operational status ('active', 'straggler', 'dropped').
        """
        self.latest_loss = float(np.nan_to_num(local_loss, nan=1.0, posinf=5.0, neginf=0.0))
        self.latest_latency = float(np.nan_to_num(latency, nan=0.5, posinf=10.0, neginf=0.0))
        if input_variance is not None:
            self.input_variance = float(np.nan_to_num(input_variance, nan=1.0, posinf=2.0, neginf=0.0))
        if bandwidth is not None:
            self.latest_bandwidth = float(np.nan_to_num(bandwidth, nan=5.0, posinf=20.0, neginf=1.0))

        self.is_straggler = bool(is_straggler)
        self.status = str(status)

        self.loss_history.append(self.latest_loss)
        self.latency_history.append(self.latest_latency)
        self.bandwidth_history.append(self.latest_bandwidth)

        # Compute curvature proxy κ^i from loss trajectory
        self.latest_curvature = self.compute_curvature()

    def compute_curvature(self) -> float:
        """
        Compute the second-difference loss curvature proxy κ^i:
            κ_t^i = |L_t - 2 * L_{t-1} + L_{t-2}|
        For t = 1: κ_1^i = |L_1 - L_0|
        For t = 0: κ_0^i = 0.0
        """
        n = len(self.loss_history)
        if n >= 3:
            l_t = self.loss_history[-1]
            l_t_minus_1 = self.loss_history[-2]
            l_t_minus_2 = self.loss_history[-3]
            val = abs(l_t - 2.0 * l_t_minus_1 + l_t_minus_2)
        elif n == 2:
            val = abs(self.loss_history[-1] - self.loss_history[-2])
        else:
            val = 0.0

        if np.isnan(val) or np.isinf(val):
            return 0.0
        return float(val)

    def get_state_vector(self, normalize: bool = True) -> np.ndarray:
        """
        Extract the 5-dimensional state vector S_i^t:
            S_i^t = [L_local^i, Var(x^i), Δτ^i, bw^i, κ^i]

        Parameters
        ----------
        normalize : bool
            If True, scales features into [0.0, 1.0]; otherwise returns raw physical values.

        Returns
        -------
        np.ndarray
            1D array of shape (5,), dtype np.float32.
        """
        raw = np.array(
            [
                float(np.nan_to_num(self.latest_loss, nan=1.0, posinf=5.0, neginf=0.0)),
                float(np.nan_to_num(self.input_variance, nan=1.0, posinf=2.0, neginf=0.0)),
                float(np.nan_to_num(self.latest_latency, nan=0.5, posinf=10.0, neginf=0.0)),
                float(np.nan_to_num(self.latest_bandwidth, nan=5.0, posinf=20.0, neginf=1.0)),
                float(np.nan_to_num(self.latest_curvature, nan=0.0, posinf=2.0, neginf=0.0)),
            ],
            dtype=np.float32,
        )

        if not normalize:
            return raw

        b = self.bounds
        # Safe min-max normalization to [0.0, 1.0] with clipping
        loss_min = float(b.get("loss_min", 0.0))
        loss_max = float(b.get("loss_max", 5.0))
        norm_loss = np.clip((raw[0] - loss_min) / max(1e-6, loss_max - loss_min), 0.0, 1.0)

        var_min = float(b.get("variance_min", 0.0))
        var_max = float(b.get("variance_max", 2.0))
        norm_var = np.clip((raw[1] - var_min) / max(1e-6, var_max - var_min), 0.0, 1.0)

        lat_min = float(b.get("latency_min", 0.0))
        lat_max = float(b.get("latency_max", 10.0))
        norm_lat = np.clip((raw[2] - lat_min) / max(1e-6, lat_max - lat_min), 0.0, 1.0)

        bw_min = float(b.get("bandwidth_min", 1.0))
        bw_max = float(b.get("bandwidth_max", 20.0))
        norm_bw = np.clip((raw[3] - bw_min) / max(1e-6, bw_max - bw_min), 0.0, 1.0)

        curv_min = float(b.get("curvature_min", 0.0))
        curv_max = float(b.get("curvature_max", 2.0))
        norm_curv = np.clip((raw[4] - curv_min) / max(1e-6, curv_max - curv_min), 0.0, 1.0)

        out = np.array([norm_loss, norm_var, norm_lat, norm_bw, norm_curv], dtype=np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)

    def reset(self, initial_loss: float = 1.0, initial_bandwidth: float = 5.0) -> None:
        """Reset history and telemetry state."""
        init_loss = float(np.nan_to_num(initial_loss, nan=1.0, posinf=5.0, neginf=0.0))
        init_bw = float(np.nan_to_num(initial_bandwidth, nan=5.0, posinf=20.0, neginf=1.0))
        self.loss_history = [init_loss]
        self.latency_history = [0.5]
        self.bandwidth_history = [init_bw]
        self.latest_loss = init_loss
        self.latest_latency = 0.5
        self.latest_bandwidth = init_bw
        self.latest_curvature = 0.0
        self.is_straggler = False
        self.status = "active"


class TelemetryTracker:
    """
    Server-side Aggregator of Multi-Client Telemetry.

    Coordinates telemetry for all N participating clients, generates the
    global telemetry state matrix S^t in R^{N x 5}, tracks system dynamics,
    and maintains heterogeneous device hardware profiles.

    Parameters
    ----------
    client_ids : Sequence[int | str]
        Identifiers for participating clients.
    client_variances : Optional[Dict[int | str, float]]
        Pre-computed empirical input feature variances for each client.
    normalization_bounds : Optional[Dict[str, float]]
        Configuration bounds for telemetry normalization into [0.0, 1.0].
    bandwidth_cfg : Optional[Dict[str, Any]]
        Configuration parameters for synthetic bandwidth simulation.
    hardware_profiles : Optional[Dict[Union[int, str], ClientHardwareProfile]]
        Explicit mapping of client_id to ClientHardwareProfile. If None,
        50% edge tier and 50% compute tier profiles are assigned automatically.
    seed : Optional[int]
        RNG seed for deterministic bandwidth and noise generation.
    """

    FEATURE_NAMES = [
        "local_loss",
        "input_variance",
        "latency",
        "bandwidth",
        "curvature",
    ]

    def __init__(
        self,
        client_ids: Sequence[Union[int, str]],
        client_variances: Optional[Dict[Union[int, str], float]] = None,
        normalization_bounds: Optional[Dict[str, float]] = None,
        bandwidth_cfg: Optional[Dict[str, Any]] = None,
        hardware_profiles: Optional[Dict[Union[int, str], ClientHardwareProfile]] = None,
        seed: Optional[int] = None,
    ):
        self.client_ids = [str(cid) for cid in client_ids]
        self.num_clients = len(self.client_ids)
        self.normalization_bounds = normalization_bounds or {}

        # Bandwidth simulator
        bw_cfg = bandwidth_cfg or {}
        self.bw_simulator = BandwidthSimulator(
            distribution=bw_cfg.get("distribution", "uniform"),
            min_mbps=bw_cfg.get("min_mbps", 1.0),
            max_mbps=bw_cfg.get("max_mbps", 20.0 if "max_mbps" not in bw_cfg else bw_cfg["max_mbps"]),
            lognormal_mu=bw_cfg.get("lognormal_mu", 1.6),
            lognormal_sigma=bw_cfg.get("lognormal_sigma", 0.5),
            seed=seed,
        )

        # Initialize hardware profiles
        if hardware_profiles is not None:
            self.hardware_profiles = {str(k): copy.deepcopy(v) for k, v in hardware_profiles.items()}
            # Ensure every client has a profile
            for cid in self.client_ids:
                if cid not in self.hardware_profiles:
                    self.hardware_profiles[cid] = copy.deepcopy(EDGE_HARDWARE_PROFILE)
        else:
            edge_ratio = float(bw_cfg.get("edge_ratio", 0.5))
            self.hardware_profiles = create_heterogeneous_profiles(
                self.client_ids, edge_ratio=edge_ratio
            )

        # Initialize per-client telemetry trackers
        self.clients: Dict[str, ClientTelemetry] = {}
        profiles_list = [self.hardware_profiles[cid] for cid in self.client_ids]
        initial_bw = self.bw_simulator.sample(self.num_clients, profiles=profiles_list)

        for idx, cid in enumerate(self.client_ids):
            var_val = 1.0
            if client_variances is not None and (cid in client_variances or int(cid) in client_variances):
                var_val = float(client_variances.get(cid, client_variances.get(int(cid), 1.0)))
            self.clients[cid] = ClientTelemetry(
                client_id=cid,
                input_variance=var_val,
                normalization_bounds=self.normalization_bounds,
                hardware_profile=self.hardware_profiles[cid],
            )
            self.clients[cid].latest_bandwidth = float(initial_bw[idx])

    def get_client_profile(self, client_id: Union[int, str]) -> ClientHardwareProfile:
        """Retrieve hardware profile for a client."""
        cid = str(client_id)
        if cid not in self.hardware_profiles:
            raise KeyError(f"Client '{cid}' not found in hardware profiles.")
        return self.hardware_profiles[cid]

    def get_tier_clients(self, tier: str) -> List[str]:
        """Retrieve client IDs belonging to a specific hardware tier ('edge' or 'compute')."""
        target_tier = tier.lower()
        return [
            cid for cid, prof in self.hardware_profiles.items()
            if prof.tier.lower() == target_tier
        ]

    def set_seed(self, seed: Optional[int] = None) -> None:
        """Reset internal RNG seed."""
        self.bw_simulator.set_seed(seed)

    def reset(
        self,
        initial_losses: Optional[Union[float, Dict[str, float]]] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Reset all client telemetry histories and re-sample link bandwidths.

        Parameters
        ----------
        initial_losses : Optional[float | Dict[str, float]]
            Initial cross-entropy loss values for clients (default 1.5).
        seed : Optional[int]
            RNG seed.

        Returns
        -------
        np.ndarray
            Initial normalized state matrix S_0 in R^{N x 5}.
        """
        if seed is not None:
            self.set_seed(seed)

        profiles_list = [self.clients[cid].hardware_profile for cid in self.client_ids]
        bw_samples = self.bw_simulator.sample(self.num_clients, profiles=profiles_list)

        for idx, cid in enumerate(self.client_ids):
            init_loss = 1.5
            if isinstance(initial_losses, dict):
                init_loss = initial_losses.get(cid, 1.5)
            elif isinstance(initial_losses, (int, float)):
                init_loss = float(initial_losses)

            self.clients[cid].reset(
                initial_loss=init_loss,
                initial_bandwidth=float(bw_samples[idx]),
            )

        return self.get_global_state_matrix(normalize=True)

    def update_round(
        self,
        round_metrics: Dict[Union[int, str], Dict[str, Any]],
        sample_new_bandwidth: bool = True,
    ) -> np.ndarray:
        """
        Update telemetry using round metrics reported by clients.

        Parameters
        ----------
        round_metrics : Dict[int | str, Dict[str, Any]]
            Dictionary mapping client_id to metric dictionary containing:
            'local_loss', 'execution_latency', 'input_variance', 'is_straggler', and 'status'.
        sample_new_bandwidth : bool
            Whether to sample dynamic bandwidth for the upcoming round.

        Returns
        -------
        np.ndarray
            Updated normalized telemetry state matrix S^t in R^{N x 5}.
        """
        profiles_list = [self.clients[cid].hardware_profile for cid in self.client_ids]
        bw_samples = (
            self.bw_simulator.sample(self.num_clients, profiles=profiles_list)
            if sample_new_bandwidth
            else None
        )

        for idx, cid in enumerate(self.client_ids):
            # Look up metrics matching client ID string or int
            c_metrics = round_metrics.get(cid, round_metrics.get(int(cid), {}))

            loss = float(c_metrics.get("local_loss", self.clients[cid].latest_loss))
            lat = float(c_metrics.get("execution_latency", self.clients[cid].latest_latency))
            var = c_metrics.get("input_variance", None)
            bw = float(bw_samples[idx]) if bw_samples is not None else None
            is_straggler = bool(c_metrics.get("is_straggler", False))
            status = str(c_metrics.get("status", "active"))

            self.clients[cid].update(
                local_loss=loss,
                latency=lat,
                bandwidth=bw,
                input_variance=var,
                is_straggler=is_straggler,
                status=status,
            )

        return self.get_global_state_matrix(normalize=True)

    def get_global_state_matrix(self, normalize: bool = True) -> np.ndarray:
        """
        Compile the full telemetry matrix across all N clients.

        Returns
        -------
        np.ndarray
            Matrix of shape (num_clients, 5) with dtype np.float32.
            If normalize=True, all entries are strictly within [0.0, 1.0].
        """
        matrix = np.zeros((self.num_clients, 5), dtype=np.float32)
        for idx, cid in enumerate(self.client_ids):
            matrix[idx] = self.clients[cid].get_state_vector(normalize=normalize)
        return np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=0.0)

    def get_client_state(self, client_id: Union[int, str], normalize: bool = True) -> np.ndarray:
        """Get 5-dimensional state vector for a single client."""
        cid = str(client_id)
        if cid not in self.clients:
            raise KeyError(f"Client '{cid}' not registered in telemetry tracker.")
        return self.clients[cid].get_state_vector(normalize=normalize)

    def get_summary(self) -> Dict[str, Any]:
        """
        Retrieve diagnostic summary dictionary of current telemetry.
        """
        raw_matrix = self.get_global_state_matrix(normalize=False)
        norm_matrix = self.get_global_state_matrix(normalize=True)

        return {
            "num_clients": self.num_clients,
            "feature_names": self.FEATURE_NAMES,
            "raw_state": raw_matrix.tolist(),
            "normalized_state": norm_matrix.tolist(),
            "hardware_tiers": {cid: self.clients[cid].hardware_profile.tier for cid in self.client_ids},
            "straggler_status": {cid: self.clients[cid].is_straggler for cid in self.client_ids},
            "mean_loss": float(np.mean(raw_matrix[:, 0])),
            "mean_variance": float(np.mean(raw_matrix[:, 1])),
            "mean_latency": float(np.mean(raw_matrix[:, 2])),
            "mean_bandwidth": float(np.mean(raw_matrix[:, 3])),
            "mean_curvature": float(np.mean(raw_matrix[:, 4])),
            "total_latency": float(np.sum(raw_matrix[:, 2])),
        }
