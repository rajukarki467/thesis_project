"""
Federated Client Implementation for ECG Arrhythmia Classification.
Replicating Zeleke & Bochicchio (IEEE BigData 2024):
"Federated Kolmogorov-Arnold Networks for Health Data Analysis: A Study Using ECG Signal"

Client Specifications:
- Local training on private non-IID data partition.
- Adam optimizer with learning rate eta = 1e-5 (0.00001).
- 30 local epochs per federated round.
- Cross-entropy loss on 5 AAMI heartbeat classes.
- Returns updated local model parameters theta_k and sample count n_k.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("federated_client")


class FederatedClient:
    """
    Federated Learning Client executing local edge training.

    Parameters
    ----------
    client_id : int
        Unique client identifier.
    model : nn.Module
        Template model architecture (KAN or MLP).
    train_data : Tuple[np.ndarray, np.ndarray]
        (X, y) tuple of local client training data.
    batch_size : int
        Mini-batch size (paper default: 32).
    local_epochs : int
        Number of local training epochs per communication round (paper default: 30).
    learning_rate : float
        Learning rate (paper default: 1e-5).
    optimizer_type : str
        Optimizer algorithm (paper default: 'adam').
    weight_decay : float
        L2 regularization parameter (default: 0.0).
    device : Optional[Union[str, torch.device]]
        Computation device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_data: Tuple[np.ndarray, np.ndarray],
        batch_size: int = 32,
        local_epochs: int = 30,
        learning_rate: float = 1e-5,
        optimizer_type: str = "adam",
        weight_decay: float = 0.0,
        class_weight: Optional[str] = "balanced",
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.client_id = client_id
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = copy.deepcopy(model).to(self.device)

        self.batch_size = int(batch_size)
        self.local_epochs = int(local_epochs)
        self.learning_rate = float(learning_rate)
        self.optimizer_type = optimizer_type.lower()
        self.weight_decay = float(weight_decay)

        X_train, y_train = train_data
        self.num_samples = len(y_train)

        # Build PyTorch DataLoader
        dataset = TensorDataset(
            torch.from_numpy(np.asarray(X_train, dtype=np.float32)),
            torch.from_numpy(np.asarray(y_train, dtype=np.int64)),
        )
        self.loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, max(1, self.num_samples)),
            shuffle=True,
            drop_last=False,
        )

        # Class-balanced loss weighting for clinical arrhythmia class imbalance
        if class_weight == "balanced" and len(y_train) > 0:
            counts = np.bincount(y_train, minlength=5).astype(np.float32)
            total = len(y_train)
            n_classes = 5
            # Smooth inverse frequency
            weights = total / (n_classes * np.maximum(counts, 1.0))
            # Clip extreme ratios for training stability
            weights = np.clip(weights, 0.1, 10.0)
            weights = weights / np.mean(weights)
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def fit(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        epochs: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, float]]:
        """
        Execute local client training initialized with global server parameters.

        Parameters
        ----------
        global_state_dict : Dict[str, torch.Tensor]
            Global parameters broadcast by the server.
        epochs : Optional[int]
            Number of local epochs (overrides default if provided).

        Returns
        -------
        local_state_dict : Dict[str, torch.Tensor]
            Updated local model parameters.
        num_samples : int
            Number of samples used in local training.
        metrics : Dict[str, float]
            Local execution metrics (loss, accuracy, duration).
        """
        num_epochs = epochs if epochs is not None else self.local_epochs

        # Load global model weights
        self.model.load_state_dict(global_state_dict)
        self.model.train()

        # Build optimizer
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")

        start_time = time.time()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for epoch in range(num_epochs):
            for batch_x, batch_y in self.loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_x.size(0)
                preds = logits.argmax(dim=-1)
                correct_preds += (preds == batch_y).sum().item()
                total_preds += batch_x.size(0)

        duration = time.time() - start_time
        avg_loss = float(running_loss / max(1, total_preds))
        local_acc = float(correct_preds / max(1, total_preds))

        # Copy state dict to CPU
        local_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        metrics = {
            "local_loss": avg_loss,
            "local_accuracy": local_acc,
            "training_duration": duration,
        }

        return local_state, self.num_samples, metrics
