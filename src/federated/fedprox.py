"""
FedProx Proximal Regularization Module for Non-IID Federated Learning.

Implements the proximal loss term:
    L_prox = (mu / 2) * ||theta - theta_t||^2

where theta_t is the global server model parameter vector at round t,
and theta is the local client parameter vector.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


def compute_proximal_term(
    model: nn.Module,
    global_parameters: Sequence[torch.Tensor],
) -> torch.Tensor:
    """
    Compute the L2 proximal regularization term ||theta - theta_t||^2 across all
    trainable parameters.

    Parameters
    ----------
    model : nn.Module
        Current local client model containing parameters theta.
    global_parameters : Sequence[torch.Tensor]
        Detached snapshot of global model parameters theta_t received from the server.

    Returns
    -------
    torch.Tensor
        Scalar tensor representing sum_i ||theta_i - theta_{t, i}||_2^2.
    """
    prox_term = torch.tensor(0.0, device=next(model.parameters()).device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if len(trainable_params) != len(global_parameters):
        raise ValueError(
            f"Parameter count mismatch: local model has {len(trainable_params)} trainable "
            f"tensors, but global parameters list has {len(global_parameters)}."
        )

    for p_local, p_global in zip(trainable_params, global_parameters):
        prox_term = prox_term + torch.sum((p_local - p_global.to(p_local.device)) ** 2)

    return prox_term


class FedProxCriterion(nn.Module):
    """
    FedProx Regularized Loss Criterion.

    Combines standard classification loss (e.g. CrossEntropyLoss) with the
    proximal penalty term:
        Total Loss = L_base + (mu / 2) * ||theta - theta_t||^2

    Parameters
    ----------
    base_criterion : nn.Module
        Base loss function (default: nn.CrossEntropyLoss()).
    mu : float
        Proximal regularization strength (default: 0.01).
    """

    def __init__(
        self,
        base_criterion: Optional[nn.Module] = None,
        mu: float = 0.01,
    ):
        super().__init__()
        self.base_criterion = base_criterion if base_criterion is not None else nn.CrossEntropyLoss()
        self.mu = float(mu)
        self.global_parameters: Optional[List[torch.Tensor]] = None

    def set_global_parameters(self, model: nn.Module) -> None:
        """
        Store a detached, cloned copy of the server model weights theta_t at the
        start of the local client round.
        """
        self.global_parameters = [
            p.clone().detach().to(p.device)
            for p in model.parameters()
            if p.requires_grad
        ]

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Compute total loss = base loss + proximal regularization.

        Parameters
        ----------
        outputs : torch.Tensor
            Logits from the model forward pass.
        targets : torch.Tensor
            Ground truth class labels.
        model : nn.Module
            Local model instance.

        Returns
        -------
        total_loss : torch.Tensor
            Differentiable total loss tensor for backpropagation.
        base_loss_val : float
            Scalar value of the base classification loss (CrossEntropy).
        prox_loss_val : float
            Scalar value of the proximal penalty term.
        """
        base_loss = self.base_criterion(outputs, targets)

        if self.mu > 0.0 and self.global_parameters is not None:
            prox_term = compute_proximal_term(model, self.global_parameters)
            prox_loss = (self.mu / 2.0) * prox_term
            total_loss = base_loss + prox_loss
            return total_loss, base_loss.item(), prox_loss.item()

        return base_loss, base_loss.item(), 0.0
