"""
Global Federated Server and Aggregator for Flower (flwr) ECG Heartbeat Classification.

Features:
- Global parameter aggregation using FedAvg and FedProx strategies.
- Evaluation on held-out patient-level test data (X_test, y_test).
- Evaluates test accuracy, macro F1, micro F1, and per-class metrics.
- Aggregates client telemetry (local loss, execution latency, input variance).
- Measures per-round and cumulative communication volume (MB).
"""

from __future__ import annotations

from collections import OrderedDict
import copy
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import flwr as flwr
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedProx
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.federated.client import FlowerECGClient
from src.models.fastkan_model import project_spline_weights
from src.utils.metrics import compute_metrics

logger = logging.getLogger("federated_server")


def calculate_model_size_bytes(model: nn.Module) -> int:
    """Calculate the memory footprint in bytes of model trainable parameters (float32)."""
    return sum(p.numel() * 4 for p in model.parameters() if p.requires_grad)


def calculate_round_communication_bytes(
    num_participating_clients: int,
    model: nn.Module,
) -> int:
    """
    Calculate the total network communication volume (Downlink + Uplink) in bytes.
        Total Bytes = 2 * num_clients * model_param_bytes
    """
    model_bytes = calculate_model_size_bytes(model)
    return 2 * num_participating_clients * model_bytes


class CentralizedEvaluator:
    """
    Evaluator for the global model on the held-out patient-level test dataset.
    """

    def __init__(
        self,
        model: nn.Module,
        test_data: Tuple[np.ndarray, np.ndarray],
        batch_size: int = 64,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = copy.deepcopy(model).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        X_test, y_test = test_data
        X_test = np.asarray(X_test, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.int64)

        self.test_dataset = TensorDataset(
            torch.from_numpy(X_test), torch.from_numpy(y_test)
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

    def evaluate(
        self,
        parameters: Sequence[np.ndarray],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate given parameter list on the held-out test split.

        Returns
        -------
        test_loss : float
        metrics : Dict[str, Any]
            Dictionary containing accuracy, macro_f1, micro_f1, and class breakdown.
        """
        # Load weights
        state_dict_keys = list(self.model.state_dict().keys())
        new_state_dict = OrderedDict({
            k: torch.tensor(v, device=self.device)
            for k, v in zip(state_dict_keys, parameters)
        })
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item() * len(batch_y)

                preds = torch.argmax(outputs, dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_targets.extend(batch_y.cpu().numpy().tolist())

        n_samples = max(1, len(all_targets))
        mean_loss = float(total_loss / n_samples)
        metrics = compute_metrics(all_targets, all_preds)
        metrics["test_loss"] = mean_loss

        return mean_loss, metrics


def aggregate_parameters(
    client_results: Sequence[Tuple[List[np.ndarray], int]],
    canonical_gk: Optional[Tuple[int, int]] = (5, 3),
    grid_range: float = 1.0,
    server_parameters: Optional[List[np.ndarray]] = None,
    client_initial_parameters: Optional[Sequence[List[np.ndarray]]] = None,
    client_gks: Optional[Sequence[Tuple[int, int]]] = None,
    mode: str = "project_full",
) -> List[np.ndarray]:
    """
    Weighted FedAvg aggregation of client model parameters with support for
    heterogeneous (G_i, k_i) spline edge parameters:
        theta_{t+1} = sum_k (n_k / sum_j n_j) * theta_k

    Supports two aggregation modes:
    1. 'project_full' (default):
       Client parameters are projected to the canonical reference grid (default (5, 3))
       via least-squares B-spline projection, then averaged with sample-size weights.
    2. 'project_delta':
       Clients compute parameter deltas relative to their initial projection
       (W_local - W_projected_init). The server projects only the deltas onto the
       canonical grid and adds the weighted average to the global server parameters.
       This preserves untouched reference weights and eliminates round-trip error.

    When all clients share identical parameter shapes (fixed G, k), aggregation
    in both modes exactly equals standard FedAvg without precision loss (< 1e-6).
    """
    total_samples = sum(num_samples for _, num_samples in client_results)
    if total_samples == 0:
        raise ValueError("Cannot aggregate: total client samples is 0.")

    num_layers = len(client_results[0][0])
    target_gk = canonical_gk if canonical_gk is not None else (5, 3)
    target_bases = target_gk[0] + target_gk[1]

    num_bases_to_gk = {
        5: (3, 2), 6: (3, 3), 7: (5, 2), 8: (5, 3), 9: (7, 2),
        10: (7, 3), 11: (7, 4), 12: (10, 2), 13: (10, 3), 14: (10, 4),
        32: (30, 2),
    }

    # Fast path: Check if any parameter tensor has differing shapes across clients
    has_shape_mismatch = any(
        any(params[i].shape != client_results[0][0][i].shape for params, _ in client_results)
        for i in range(num_layers)
    )

    mode = str(mode).lower()

    # When all shapes match and mode is project_full (or server_parameters not provided for delta):
    if not has_shape_mismatch and (mode != "project_delta" or server_parameters is None):
        first_params = client_results[0][0]
        aggregated = [np.zeros_like(p, dtype=np.float64) for p in first_params]
        for params, n_samples in client_results:
            weight = float(n_samples) / float(total_samples)
            for i, p in enumerate(params):
                aggregated[i] += weight * p.astype(np.float64)
        return [p.astype(first_params[i].dtype) for i, p in enumerate(aggregated)]

    # If project_delta mode is requested:
    if mode == "project_delta":
        if server_parameters is None and client_initial_parameters is None:
            # If neither provided, client_results are treated as pre-computed deltas
            ref_params = client_results[0][0]
            aggregated_delta = [np.zeros_like(p, dtype=np.float64) for p in ref_params]
            for params, n_samples in client_results:
                weight = float(n_samples) / float(total_samples)
                for i, p in enumerate(params):
                    aggregated_delta[i] += weight * p.astype(np.float64)
            return [p.astype(ref_params[i].dtype) for i, p in enumerate(aggregated_delta)]

        # Determine reference parameters
        ref_params = server_parameters if server_parameters is not None else client_initial_parameters[0]
        projected_deltas: List[Tuple[List[np.ndarray], int]] = []

        for idx, (params, n_samples) in enumerate(client_results):
            # Obtain initial parameters for this client
            if client_initial_parameters is not None:
                init_p = client_initial_parameters[idx]
            else:
                # If client shape matches server, init_p is server_parameters
                # If client shape differs, project server_parameters to client gk
                client_gk = client_gks[idx] if client_gks is not None else None
                init_p = []
                for sp, cp in zip(server_parameters, params):
                    if sp.shape != cp.shape and cp.ndim == 3:
                        src_b = cp.shape[-1]
                        c_gk = client_gk if client_gk is not None else num_bases_to_gk.get(src_b, (max(3, src_b - 3), 3))
                        p_init = project_spline_weights(
                            sp, src_gk=target_gk, dst_gk=c_gk, grid_range=grid_range
                        )
                        init_p.append(p_init)
                    else:
                        init_p.append(sp)

            # Compute delta and project delta to target_gk
            proj_delta = []
            for p_curr, p_init in zip(params, init_p):
                delta = p_curr - p_init
                if delta.ndim == 3 and delta.shape[-1] != target_bases:
                    src_b = delta.shape[-1]
                    c_gk = client_gks[idx] if client_gks is not None else num_bases_to_gk.get(src_b, (max(3, src_b - 3), 3))
                    d_proj = project_spline_weights(
                        delta, src_gk=c_gk, dst_gk=target_gk, grid_range=grid_range
                    )
                    proj_delta.append(d_proj)
                else:
                    proj_delta.append(delta)
            projected_deltas.append((proj_delta, n_samples))

        # Weighted aggregation of deltas
        aggregated_delta = [np.zeros_like(p, dtype=np.float64) for p in ref_params]
        for delta_p, n_samples in projected_deltas:
            weight = float(n_samples) / float(total_samples)
            for i, dp in enumerate(delta_p):
                aggregated_delta[i] += weight * dp.astype(np.float64)

        # Add delta to server parameters
        return [
            (ref_params[i].astype(np.float64) + aggregated_delta[i]).astype(ref_params[i].dtype)
            for i in range(num_layers)
        ]

    # Full parameter projection path (mode == 'project_full'):
    projected_clients: List[Tuple[List[np.ndarray], int]] = []
    for idx, (params, n_samples) in enumerate(client_results):
        proj_params = []
        for p in params:
            if p.ndim == 3 and p.shape[-1] != target_bases:
                src_bases = p.shape[-1]
                src_gk = client_gks[idx] if client_gks is not None else num_bases_to_gk.get(src_bases, (max(3, src_bases - 3), 3))
                p_proj = project_spline_weights(
                    p, src_gk=src_gk, dst_gk=target_gk, grid_range=grid_range
                )
                proj_params.append(p_proj)
            else:
                proj_params.append(p)
        projected_clients.append((proj_params, n_samples))

    first_params = projected_clients[0][0]
    aggregated = [np.zeros_like(p, dtype=np.float64) for p in first_params]
    for params, n_samples in projected_clients:
        weight = float(n_samples) / float(total_samples)
        for i, p in enumerate(params):
            aggregated[i] += weight * p.astype(np.float64)

    return [p.astype(first_params[i].dtype) for i, p in enumerate(aggregated)]


def run_federated_simulation(
    model: nn.Module,
    client_datasets: Dict[Union[int, str], Tuple[np.ndarray, np.ndarray]],
    test_data: Tuple[np.ndarray, np.ndarray],
    val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    num_rounds: int = 10,
    strategy_name: str = "fedavg",
    proximal_mu: float = 0.0,
    local_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    optimizer_type: str = "adam",
    weight_decay: float = 1e-4,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = True,
    checkpoint_metric: str = "macro_f1",
) -> Dict[str, Any]:
    """
    Execute complete end-to-end Federated Learning simulation.

    Supports:
    - Strategy: 'fedavg' or 'fedprox'
    - Telemetry tracking across clients
    - Communication volume calculation (MB)
    - Centralized validation evaluation per round (Loss, Accuracy, Macro F1, Balanced Accuracy)
    - Checkpoint selection on validation split only, touching test split for reporting
    - Centralized test evaluation per round (Loss, Accuracy, Macro F1, Balanced Accuracy)

    Returns
    -------
    Dict[str, Any]
        Round-by-round metrics history, best validation metrics, and final test summary.
    """
    client_ids = sorted(client_datasets.keys())
    num_clients = len(client_ids)
    strategy_name = strategy_name.lower()
    actual_mu = proximal_mu if strategy_name == "fedprox" else 0.0

    test_evaluator = CentralizedEvaluator(model=model, test_data=test_data, device=device)
    val_evaluator = (
        CentralizedEvaluator(model=model, test_data=val_data, device=device)
        if val_data is not None
        else None
    )

    # Initialize global parameters from template model
    current_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Initial zero-round evaluation
    init_loss, init_metrics = test_evaluator.evaluate(current_parameters)
    init_val_loss, init_val_metrics = (
        val_evaluator.evaluate(current_parameters) if val_evaluator is not None else (None, {})
    )

    if verbose:
        val_str = (
            f" | Val Acc: {init_val_metrics.get('accuracy', 0.0):.4f} | Val F1: {init_val_metrics.get('macro_f1', 0.0):.4f}"
            if val_evaluator is not None
            else ""
        )
        logger.info(
            f"[Round 00/{num_rounds:02d}] Initial Global Model - "
            f"Test Loss: {init_loss:.4f} | Test Acc: {init_metrics['accuracy']:.4f} | Test Macro F1: {init_metrics['macro_f1']:.4f}{val_str}"
        )

    # Instantiate local clients
    clients: Dict[Union[int, str], FlowerECGClient] = {}
    for cid in client_ids:
        clients[cid] = FlowerECGClient(
            client_id=cid,
            model=copy.deepcopy(model),
            train_data=client_datasets[cid],
            batch_size=batch_size,
            epochs=local_epochs,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type,
            weight_decay=weight_decay,
            proximal_mu=actual_mu,
            device=device,
        )

    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_up_per_round = num_clients * p_count
    params_down_per_round = num_clients * p_count
    params_per_round = params_up_per_round + params_down_per_round
    bytes_per_round = params_per_round * 4
    mb_per_round = bytes_per_round / (1024.0 ** 2)

    history: List[Dict[str, Any]] = []
    cumulative_comm_mb = 0.0
    cum_params = 0
    cum_bytes = 0
    cum_comm_time_s = 0.0
    cum_comm_time_sum_s = 0.0

    best_val_score = float(init_val_metrics.get(checkpoint_metric, init_metrics.get(checkpoint_metric, -1.0)))
    best_round = 0
    best_parameters = copy.deepcopy(current_parameters)
    best_val_metrics = copy.deepcopy(init_val_metrics)

    # Record initial state
    init_record = {
        "round": 0,
        "test_loss": float(init_loss),
        "test_accuracy": float(init_metrics["accuracy"]),
        "test_balanced_accuracy": float(init_metrics.get("balanced_accuracy", init_metrics["accuracy"])),
        "test_macro_f1": float(init_metrics["macro_f1"]),
        "test_micro_f1": float(init_metrics["micro_f1"]),
        "communication_volume_mb": 0.0,
        "cumulative_communication_mb": 0.0,
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
        "mean_client_latency": 0.0,
        "mean_client_loss": 0.0,
        "mean_input_variance": float(np.mean([c.input_variance for c in clients.values()])),
        "per_class_metrics": init_metrics["class_metrics"],
        "confusion_matrix": init_metrics.get("confusion_matrix", []),
    }
    if val_evaluator is not None:
        init_record.update({
            "val_loss": float(init_val_loss),
            "val_accuracy": float(init_val_metrics.get("accuracy", 0.0)),
            "val_balanced_accuracy": float(init_val_metrics.get("balanced_accuracy", 0.0)),
            "val_macro_f1": float(init_val_metrics.get("macro_f1", 0.0)),
        })
    history.append(init_record)

    # Federated Rounds Loop
    for r in range(1, num_rounds + 1):
        round_fit_results: List[Tuple[List[np.ndarray], int]] = []
        round_telemetries: List[Dict[str, Any]] = []

        config = {
            "local_epochs": local_epochs,
            "proximal_mu": actual_mu,
            "server_round": r,
        }

        # Local client updates
        for cid in client_ids:
            client = clients[cid]
            updated_params, n_samples, metrics = client.fit(
                parameters=current_parameters, config=config
            )
            round_fit_results.append((updated_params, n_samples))
            round_telemetries.append(metrics)

        # Server aggregation
        current_parameters = aggregate_parameters(round_fit_results)
        cumulative_comm_mb += mb_per_round

        # Global validation evaluation (primary for checkpoint selection)
        val_score_for_ckpt = 0.0
        if val_evaluator is not None:
            val_loss, val_metrics = val_evaluator.evaluate(current_parameters)
            val_score_for_ckpt = float(val_metrics.get(checkpoint_metric, val_metrics.get("accuracy", 0.0)))

        # Global test evaluation (logged per round for convergence curves)
        test_loss, test_metrics = test_evaluator.evaluate(current_parameters)

        # Checkpoint selection on validation split only
        if val_evaluator is not None:
            if val_score_for_ckpt >= best_val_score:
                best_val_score = val_score_for_ckpt
                best_round = r
                best_parameters = copy.deepcopy(current_parameters)
                best_val_metrics = copy.deepcopy(val_metrics)
        else:
            best_parameters = copy.deepcopy(current_parameters)
            best_round = r

        mean_latency = float(np.mean([m["execution_latency"] for m in round_telemetries]))
        mean_local_loss = float(np.mean([m["local_loss"] for m in round_telemetries]))
        mean_inp_var = float(np.mean([m["input_variance"] for m in round_telemetries]))

        cum_params += params_per_round
        cum_bytes += bytes_per_round
        cum_MB = float(cum_bytes / (1024.0 ** 2))

        # Simulated communication times across clients
        from src.environment.telemetry import BandwidthSimulator
        bw_sim = BandwidthSimulator(seed=r * 100 + num_clients)
        bw_clients = bw_sim.sample(num_clients)
        comm_times = []
        for cid_idx in range(num_clients):
            c_bytes = float(2 * p_count * 4)
            c_time = (c_bytes * 8.0) / (bw_clients[cid_idx] * 1e6)
            comm_times.append(c_time)
        est_comm_time_s = float(max(comm_times)) if comm_times else 0.0
        est_comm_time_sum_s = float(sum(comm_times)) if comm_times else 0.0
        cum_comm_time_s += est_comm_time_s
        cum_comm_time_sum_s += est_comm_time_sum_s

        round_record = {
            "round": r,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_balanced_accuracy": float(test_metrics.get("balanced_accuracy", test_metrics["accuracy"])),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "test_micro_f1": float(test_metrics["micro_f1"]),
            "communication_volume_mb": float(mb_per_round),
            "cumulative_communication_mb": float(cumulative_comm_mb),
            "params_up": params_up_per_round,
            "params_down": params_down_per_round,
            "params_round": params_per_round,
            "bytes_round": bytes_per_round,
            "cum_params": cum_params,
            "cum_bytes": cum_bytes,
            "cum_MB": cum_MB,
            "est_comm_time_s": est_comm_time_s,
            "est_comm_time_sum_s": est_comm_time_sum_s,
            "cum_comm_time_s": cum_comm_time_s,
            "config_histogram": {"(5,3)": 1.0} if hasattr(model, "layers") else {},
            "mean_client_latency": mean_latency,
            "mean_client_loss": mean_local_loss,
            "mean_input_variance": mean_inp_var,
            "per_class_metrics": test_metrics["class_metrics"],
            "confusion_matrix": test_metrics.get("confusion_matrix", []),
        }
        if val_evaluator is not None:
            round_record.update({
                "val_loss": float(val_loss),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_balanced_accuracy": float(val_metrics.get("balanced_accuracy", val_metrics["accuracy"])),
                "val_macro_f1": float(val_metrics["macro_f1"]),
            })
        history.append(round_record)

        if verbose:
            val_log = (
                f" | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f}"
                if val_evaluator is not None
                else ""
            )
            logger.info(
                f"[{strategy_name.upper()} | Round {r:02d}/{num_rounds:02d}] "
                f"Test Acc: {test_metrics['accuracy']:.4f} | Test F1: {test_metrics['macro_f1']:.4f}"
                f"{val_log} | Latency: {mean_latency:.3f}s | Comm: {cumulative_comm_mb:.2f} MB"
            )

    # Evaluate best validation checkpoint on test data for final unbiased reporting
    best_test_loss, best_test_metrics = test_evaluator.evaluate(best_parameters)

    return {
        "strategy": strategy_name,
        "proximal_mu": actual_mu,
        "num_rounds": num_rounds,
        "num_clients": num_clients,
        "final_parameters": current_parameters,
        "final_metrics": history[-1],
        "best_round": best_round,
        "best_parameters": best_parameters,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "history": history,
    }
