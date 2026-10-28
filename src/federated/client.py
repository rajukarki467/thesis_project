"""
Flower NumPyClient for Federated ECG Arrhythmia Heartbeat Classification.

Features:
- Subclasses flwr.client.NumPyClient.
- Local training with Adam or SGD optimizer and CrossEntropyLoss.
- Local validation evaluation on held-out client split.
- Telemetry calculation: local loss, execution latency, empirical input variance.
- FedProx proximal regularization term (mu / 2) * ||theta - theta_t||^2.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import flwr as flwr
from flwr.client import NumPyClient
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.federated.fedprox import FedProxCriterion
from src.utils.metrics import compute_metrics

logger = logging.getLogger("federated_client")


class FlowerECGClient(NumPyClient):
    """
    Flower federated learning client for ECG beat classification.

    Parameters
    ----------
    client_id : int or str
        Unique identifier for this client.
    model : nn.Module
        PyTorch model (FastKANClassifier or MLPClassifier).
    train_data : Tuple[np.ndarray, np.ndarray]
        (X, y) tuple of client training samples.
    val_data : Optional[Tuple[np.ndarray, np.ndarray]]
        (X_val, y_val) tuple for local client validation. If None, derived automatically.
    val_ratio : float
        Fraction of local client data reserved for validation if val_data is None (default 0.2).
    batch_size : int
        Mini-batch size for local training (default 32).
    epochs : int
        Number of local training epochs per federated round (default 3).
    learning_rate : float
        Learning rate (default 1e-3).
    optimizer_type : str
        'adam' or 'sgd' (default 'adam').
    momentum : float
        Momentum factor for SGD (default 0.9).
    weight_decay : float
        L2 weight decay regularization (default 1e-4).
    proximal_mu : float
        FedProx proximal penalty parameter mu (default 0.0, i.e. standard FedAvg).
    device : torch.device or str
        Computation device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        client_id: Union[int, str],
        model: nn.Module,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        val_ratio: float = 0.20,
        batch_size: int = 32,
        epochs: int = 3,
        learning_rate: float = 0.001,
        optimizer_type: str = "adam",
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        proximal_mu: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.client_id = str(client_id)
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)

        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.optimizer_type = optimizer_type.lower()
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.proximal_mu = float(proximal_mu)

        # Prepare local datasets
        X_all, y_all = train_data
        X_all = np.asarray(X_all, dtype=np.float32)
        y_all = np.asarray(y_all, dtype=np.int64)

        # Telemetry: Empirical input variance across features
        # Var(X_local) = mean_d Var(X_{:, d})
        if len(X_all) > 1:
            self.input_variance = float(np.mean(np.var(X_all, axis=0)))
        else:
            self.input_variance = 0.0

        if val_data is not None:
            X_tr, y_tr = X_all, y_all
            X_v, y_v = np.asarray(val_data[0], dtype=np.float32), np.asarray(val_data[1], dtype=np.int64)
        else:
            # Deterministically split client data into train and val
            n_samples = len(X_all)
            n_val = int(round(n_samples * val_ratio))
            # Ensure at least 1 training sample
            n_val = min(n_val, max(0, n_samples - 1))

            if n_val > 0:
                X_tr, X_v = X_all[:-n_val], X_all[-n_val:]
                y_tr, y_v = y_all[:-n_val], y_all[-n_val:]
            else:
                X_tr, y_tr = X_all, y_all
                X_v, y_v = X_all, y_all

        self.train_dataset = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        self.val_dataset = TensorDataset(torch.from_numpy(X_v), torch.from_numpy(y_v))

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        # Loss function with FedProx regularization support
        self.criterion = FedProxCriterion(base_criterion=nn.CrossEntropyLoss(), mu=self.proximal_mu)

    def get_parameters(self, config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        """Extract model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: Sequence[np.ndarray]) -> None:
        """Load global model parameters from NumPy arrays into the PyTorch model."""
        state_dict_keys = list(self.model.state_dict().keys())
        if len(parameters) != len(state_dict_keys):
            raise ValueError(
                f"Parameter length mismatch: expected {len(state_dict_keys)}, got {len(parameters)}."
            )
        new_state_dict = OrderedDict({
            k: torch.tensor(v, device=self.device)
            for k, v in zip(state_dict_keys, parameters)
        })
        self.model.load_state_dict(new_state_dict, strict=True)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Construct client optimizer (Adam or SGD)."""
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if self.optimizer_type == "sgd":
            return torch.optim.SGD(
                trainable,
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            return torch.optim.Adam(
                trainable,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Execute local client training round.

        Calculates telemetry:
        - local_loss: Average training loss
        - execution_latency: Wall-clock latency of training round in seconds
        - input_variance: Empirical variance of client inputs
        """
        start_time = time.perf_counter()

        # Update local model weights with server parameters
        self.set_parameters(parameters)

        # Configure hyperparameters from config if provided
        epochs = int(config.get("local_epochs", self.epochs))
        proximal_mu = float(config.get("proximal_mu", self.proximal_mu))
        self.criterion.mu = proximal_mu

        # Snapshot server weights theta_t for FedProx regularization
        if self.criterion.mu > 0.0:
            self.criterion.set_global_parameters(self.model)

        self.model.train()
        optimizer = self._build_optimizer()

        total_loss_accum = 0.0
        total_batches = 0

        for epoch in range(epochs):
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss, base_loss_val, _ = self.criterion(outputs, batch_y, self.model)

                loss.backward()
                # Gradient clipping for numerical stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss_accum += base_loss_val
                total_batches += 1

        latency = time.perf_counter() - start_time
        avg_local_loss = float(total_loss_accum / max(1, total_batches))

        # Evaluate on local validation set
        val_loss, val_acc, val_f1 = self._evaluate_local()

        # Return updated parameters, sample count, and telemetry metrics
        metrics = {
            "client_id": self.client_id,
            "local_loss": avg_local_loss,
            "execution_latency": float(latency),
            "input_variance": float(self.input_variance),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "val_macro_f1": float(val_f1),
            "num_samples": len(self.train_dataset),
        }

        return self.get_parameters(), len(self.train_dataset), metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate server parameters on the client's local validation split."""
        self.set_parameters(parameters)
        val_loss, val_acc, val_f1 = self._evaluate_local()

        metrics = {
            "client_id": self.client_id,
            "accuracy": float(val_acc),
            "macro_f1": float(val_f1),
            "input_variance": float(self.input_variance),
        }
        return float(val_loss), len(self.val_dataset), metrics

    def _evaluate_local(self) -> Tuple[float, float, float]:
        """Internal helper to compute loss, accuracy, and macro F1 on local val split."""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item() * len(batch_y)

                preds = torch.argmax(outputs, dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_targets.extend(batch_y.cpu().numpy().tolist())

        n_samples = max(1, len(all_targets))
        mean_loss = total_loss / n_samples
        metrics = compute_metrics(all_targets, all_preds)

        return float(mean_loss), float(metrics["accuracy"]), float(metrics["macro_f1"])
