"""
Multilayer Perceptron (MLP) Baseline Classifier for ECG Heartbeat Classification.

Designed with matched parameter budget to Kolmogorov-Arnold Networks (KAN)
for rigorous, fair empirical evaluation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """
    Multilayer Perceptron (MLP) Classifier.

    Parameters
    ----------
    input_dim : int
        Input feature dimension (default 32 for resampled ECG beats).
    hidden_dims : Sequence[int]
        Hidden layer dimensions, e.g. [64, 48].
    num_classes : int
        Number of output classes (default 5 for AAMI classes: N, SVEB, VEB, F, Q).
    dropout : float
        Dropout probability between hidden layers (default 0.1).
    activation : str
        Activation function ('relu', 'silu', 'gelu', default 'relu').
    use_batch_norm : bool
        Whether to include Batch Normalization layers (default False).
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dims: Sequence[int] = (64, 48),
        num_classes: int = 5,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm

        # Resolve activation function
        act_map = {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }
        act_cls = act_map.get(activation.lower(), nn.ReLU)

        # Build feedforward layers
        dims = [input_dim] + self.hidden_dims + [num_classes]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            # Add activation and dropout for all hidden layers (except final output logits)
            if i < len(dims) - 2:
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(act_cls())
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))

        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize linear layers using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of resampled ECG beats with shape (batch_size, 32).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, num_classes).
        """
        return self.network(x)

    def get_parameter_count(self, detailed: bool = False) -> Union[int, Dict[str, Any]]:
        """
        Return the total number of trainable parameters.

        Parameters
        ----------
        detailed : bool
            If True, returns a per-layer parameter breakdown dictionary.

        Returns
        -------
        int or Dict[str, Any]
            Parameter count.
        """
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if detailed:
            breakdown = []
            for name, module in self.network.named_children():
                if isinstance(module, nn.Linear):
                    breakdown.append({
                        "name": name,
                        "in_features": module.in_features,
                        "out_features": module.out_features,
                        "weight_params": module.weight.numel(),
                        "bias_params": module.bias.numel() if module.bias is not None else 0,
                        "total": sum(p.numel() for p in module.parameters() if p.requires_grad),
                    })
            return {
                "total_parameters": total,
                "layers": breakdown,
            }
        return total


def build_matched_mlp(
    target_params: int = 5349,
    input_dim: int = 32,
    num_classes: int = 5,
    num_hidden_layers: int = 2,
    dropout: float = 0.1,
    activation: str = "relu",
) -> MLPClassifier:
    """
    Construct an MLPClassifier whose parameter count closely matches target_params.

    Solves for the optimal uniform or tapered hidden dimension H such that:
        Total Parameters(MLP) ≈ target_params

    Parameters
    ----------
    target_params : int
        Target number of trainable parameters (e.g. 5,349 for default FastKAN).
    input_dim : int
        Input feature dimension (32).
    num_classes : int
        Number of output classes (5).
    num_hidden_layers : int
        Number of hidden layers (default 2).
    dropout : float
        Dropout rate.
    activation : str
        Activation function.

    Returns
    -------
    MLPClassifier
        Constructed MLP matched to parameter budget.
    """
    best_diff = float("inf")
    best_hidden: List[int] = [64, 48]

    # Search space for 2-layer architectures
    if num_hidden_layers == 2:
        for h1 in range(16, 128):
            for h2 in range(16, h1 + 1):
                # Total params = (input_dim*h1 + h1) + (h1*h2 + h2) + (h2*num_classes + num_classes)
                params = (input_dim * h1 + h1) + (h1 * h2 + h2) + (h2 * num_classes + num_classes)
                diff = abs(params - target_params)
                if diff < best_diff:
                    best_diff = diff
                    best_hidden = [h1, h2]
    elif num_hidden_layers == 1:
        # Total params = (input_dim*h + h) + (h*num_classes + num_classes) = (input_dim + num_classes + 1)*h + num_classes
        h = max(1, int(round((target_params - num_classes) / (input_dim + num_classes + 1))))
        best_hidden = [h]
    else:
        best_hidden = [64, 48]

    return MLPClassifier(
        input_dim=input_dim,
        hidden_dims=best_hidden,
        num_classes=num_classes,
        dropout=dropout,
        activation=activation,
    )
