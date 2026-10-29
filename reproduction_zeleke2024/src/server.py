"""
Federated Server and Aggregator Implementation.
Replicating Zeleke & Bochicchio (IEEE BigData 2024):
"Federated Kolmogorov-Arnold Networks for Health Data Analysis: A Study Using ECG Signal"

Server Specifications:
- Exact FedAvg aggregation: theta_global = sum_{k=1}^K (n_k / N) * theta_k
- Evaluates global model on the 20% hold-out test set per communication round.
- Metrics calculated:
  * Accuracy (%)
  * Macro F1-score (%)
  * Weighted F1-score (%)
  * Macro Precision (%)
  * Macro Recall (%)
  * Per-class precision, recall, F1-scores
  * Confusion matrix
- Orchestrates federated simulation across 10 communication rounds.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from reproduction_zeleke2024.src.client import FederatedClient

logger = logging.getLogger("federated_server")

CLASS_NAMES = [
    "Normal (N)",
    "Supraventricular (S)",
    "Ventricular (V)",
    "Fusion (F)",
    "Unclassified (Q)",
]


def aggregate_fedavg(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
) -> Dict[str, torch.Tensor]:
    """
    FedAvg parameter aggregation weighted by client sample counts:
        theta_{global} = sum_{k=1}^K (n_k / N) * theta_k
    """
    total_samples = sum(n_samples for _, n_samples in client_updates)
    if total_samples <= 0:
        raise ValueError("Total sample count across clients must be positive.")

    first_state = client_updates[0][0]
    aggregated_state: Dict[str, torch.Tensor] = {}

    for key, template_val in first_state.items():
        if template_val.dtype in (torch.int32, torch.int64, torch.bool):
            # Non-floating parameter (e.g. buffers)
            aggregated_state[key] = template_val.clone()
            continue

        weighted_sum = torch.zeros_like(template_val, dtype=torch.float32)
        for state_dict, n_k in client_updates:
            weight = float(n_k) / float(total_samples)
            weighted_sum += weight * state_dict[key].to(dtype=torch.float32)

        aggregated_state[key] = weighted_sum.to(dtype=template_val.dtype)

    return aggregated_state


def evaluate_global_model(
    model: nn.Module,
    test_data: Tuple[np.ndarray, np.ndarray],
    batch_size: int = 128,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Centralized evaluation of the global model on the hold-out test set (20% split).

    Returns
    -------
    metrics : Dict[str, Any]
        Comprehensive evaluation dictionary containing accuracy, macro F1,
        weighted F1, per-class metrics, and confusion matrix.
    """
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()

    X_test, y_test = test_data
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(
        torch.from_numpy(np.asarray(X_test, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_test, dtype=np.int64)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)

            logits = model(bx)
            loss = criterion(logits, by)
            total_loss += loss.item() * bx.size(0)

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(by.cpu().numpy().tolist())

    total_samples = len(all_targets)
    avg_loss = float(total_loss / max(1, total_samples))

    y_true = np.asarray(all_targets, dtype=np.int64)
    y_pred = np.asarray(all_preds, dtype=np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)

    class_metrics: Dict[str, Dict[str, float]] = {}
    for idx, c_name in enumerate(CLASS_NAMES):
        f1_val = float(per_class_f1[idx]) if idx < len(per_class_f1) else 0.0
        rec_val = float(per_class_rec[idx]) if idx < len(per_class_rec) else 0.0
        prec_val = float(per_class_prec[idx]) if idx < len(per_class_prec) else 0.0
        class_metrics[c_name] = {
            "f1": f1_val,
            "precision": prec_val,
            "recall": rec_val,
        }

    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))).tolist()

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "class_metrics": class_metrics,
        "confusion_matrix": conf_mat,
    }


class FederatedServer:
    """
    Orchestrates the Federated Learning simulation matching Zeleke & Bochicchio (2024).

    Parameters
    ----------
    global_model : nn.Module
        Template PyTorch model (KAN or MLP).
    test_data : Tuple[np.ndarray, np.ndarray]
        (X_test, y_test) hold-out test set (20% split).
    clients : Dict[int, FederatedClient]
        Dictionary of simulated edge client instances.
    device : Optional[Union[str, torch.device]]
        Computation device.
    """

    def __init__(
        self,
        global_model: nn.Module,
        test_data: Tuple[np.ndarray, np.ndarray],
        clients: Dict[int, FederatedClient],
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.global_model = copy.deepcopy(global_model).to(self.device)
        self.test_data = test_data
        self.clients = clients
        self.client_ids = sorted(clients.keys())
        self.num_clients = len(self.client_ids)

        self.history: List[Dict[str, Any]] = []

    def run_simulation(
        self,
        num_rounds: int = 10,
        local_epochs: int = 30,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute full federated training rounds.

        Parameters
        ----------
        num_rounds : int
            Total global communication rounds (paper default: 10).
        local_epochs : int
            Number of local epochs per client per round (paper default: 30).
        verbose : bool
            Whether to print progress logs.

        Returns
        -------
        summary : Dict[str, Any]
            Complete simulation history, final metrics, and convergence records.
        """
        # Initial zero-round evaluation
        init_metrics = evaluate_global_model(
            self.global_model, self.test_data, device=self.device
        )
        if verbose:
            logger.info(
                f"[Round 00/{num_rounds:02d}] Initial Model - "
                f"Loss: {init_metrics['loss']:.4f} | Acc: {init_metrics['accuracy'] * 100:.2f}% | "
                f"Macro F1: {init_metrics['macro_f1'] * 100:.2f}%"
            )

        self.history.append({
            "round": 0,
            "test_loss": init_metrics["loss"],
            "test_accuracy": init_metrics["accuracy"],
            "test_macro_f1": init_metrics["macro_f1"],
            "test_weighted_f1": init_metrics["weighted_f1"],
            "test_macro_precision": init_metrics["macro_precision"],
            "test_macro_recall": init_metrics["macro_recall"],
            "round_duration": 0.0,
            "mean_client_loss": 0.0,
            "mean_client_acc": 0.0,
            "class_metrics": init_metrics["class_metrics"],
            "confusion_matrix": init_metrics["confusion_matrix"],
        })

        # Federated Communication Rounds Loop
        for r in range(1, num_rounds + 1):
            round_start = time.time()
            client_updates: List[Tuple[Dict[str, torch.Tensor], int]] = []
            client_losses: List[float] = []
            client_accs: List[float] = []

            current_global_state = {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}

            # 1. Local Training on each client
            for cid in self.client_ids:
                client = self.clients[cid]
                local_state, n_k, metrics = client.fit(
                    global_state_dict=current_global_state,
                    epochs=local_epochs,
                )
                client_updates.append((local_state, n_k))
                client_losses.append(metrics["local_loss"])
                client_accs.append(metrics["local_accuracy"])

            # 2. FedAvg Parameter Aggregation
            aggregated_state = aggregate_fedavg(client_updates)
            self.global_model.load_state_dict(aggregated_state)

            # 3. Centralized Test Set Evaluation
            test_metrics = evaluate_global_model(
                self.global_model, self.test_data, device=self.device
            )

            round_duration = time.time() - round_start
            mean_c_loss = float(np.mean(client_losses))
            mean_c_acc = float(np.mean(client_accs))

            round_record = {
                "round": r,
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
                "test_macro_precision": test_metrics["macro_precision"],
                "test_macro_recall": test_metrics["macro_recall"],
                "round_duration": round_duration,
                "mean_client_loss": mean_c_loss,
                "mean_client_acc": mean_c_acc,
                "class_metrics": test_metrics["class_metrics"],
                "confusion_matrix": test_metrics["confusion_matrix"],
            }
            self.history.append(round_record)

            if verbose:
                logger.info(
                    f"[Round {r:02d}/{num_rounds:02d}] "
                    f"Test Acc: {test_metrics['accuracy'] * 100:.2f}% | "
                    f"Macro F1: {test_metrics['macro_f1'] * 100:.2f}% | "
                    f"Loss: {test_metrics['loss']:.4f} | "
                    f"Client Acc: {mean_c_acc * 100:.2f}% | "
                    f"Time: {round_duration:.2f}s"
                )

        final_record = self.history[-1]
        return {
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "num_clients": self.num_clients,
            "final_metrics": final_record,
            "history": self.history,
        }
