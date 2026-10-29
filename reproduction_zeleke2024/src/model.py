"""
Kolmogorov-Arnold Network (KAN) and Baseline MLP Architectures.
Replicating Zeleke & Bochicchio (IEEE BigData 2024):
"Federated Kolmogorov-Arnold Networks for Health Data Analysis: A Study Using ECG Signal"

Key Paper Specifications:
- Spline-based activation functions on edges.
- Grid size G=30.
- Quadratic Spline order k=2.
- Layer architectures:
  * Default: Single hidden layer (64 neurons): [32, 64, 5].
  * Multi-layer comparison: 2 hidden layers [32, 64, 32, 5] and 3 hidden layers [32, 64, 32, 16, 5] (Fig 3).
- Parameter-matched baseline MLP model (91.65% target).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):
    """
    A single Kolmogorov-Arnold Network edge layer using B-spline basis expansion.

    phi_{j, i}(x_i) = w_b * silu(x_i) + sum_{m=0}^{G+k-1} c_{j, i, m} * B_m(x_i)

    Parameters
    ----------
    in_features : int
        Number of input dimensions.
    out_features : int
        Number of output dimensions.
    grid_size : int
        Number of grid intervals G (paper default: 30).
    spline_order : int
        Spline polynomial degree k (paper default: 2, quadratic).
    grid_range : float
        Interval boundary [-grid_range, grid_range] for B-spline evaluation (default: 1.0).
    use_base_update : bool
        Whether to include the linear residual branch w_b * silu(x) (default: True).
    use_bias : bool
        Whether to include an additive bias term (default: True).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 30,
        spline_order: int = 2,
        grid_range: float = 1.0,
        use_base_update: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.grid_range = float(grid_range)
        self.use_base_update = use_base_update
        self.use_bias = use_bias

        # Construct uniform knot vector for B-splines
        self._build_grid()

        # Base linear branch: w_b * silu(x)
        if self.use_base_update:
            self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        else:
            self.register_parameter("base_weight", None)

        # Spline coefficients: (out_features, in_features, G + k)
        num_bases = self.grid_size + self.spline_order
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, num_bases))

        # Additive bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def _build_grid(self) -> None:
        """Construct uniform B-spline knot vector including boundary margin knots."""
        h = (2.0 * self.grid_range) / self.grid_size
        total_knots = self.grid_size + 2 * self.spline_order + 1
        grid = torch.linspace(
            -self.grid_range - self.spline_order * h,
            self.grid_range + self.spline_order * h,
            total_knots,
            dtype=torch.float32,
        )
        self.register_buffer("grid", grid, persistent=False)

    def reset_parameters(self) -> None:
        """Initialize parameters with variance-scaled weights for stable training."""
        if self.base_weight is not None:
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * 0.1)

        std = 1.0 / math.sqrt(self.in_features * (self.grid_size + self.spline_order))
        nn.init.normal_(self.spline_weight, mean=0.0, std=std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def compute_b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate degree-k B-spline basis functions using Cox-de Boor recursion.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features).

        Returns
        -------
        torch.Tensor
            Evaluated basis tensor of shape (batch_size, in_features, G + k).
        """
        x_clamped = torch.clamp(x, -self.grid_range, self.grid_range).unsqueeze(-1)
        grid = self.grid.to(dtype=x.dtype, device=x.device)

        # 0-degree step functions
        bases = ((x_clamped >= grid[:-1]) & (x_clamped < grid[1:])).to(x.dtype)

        # Cox-de Boor recursive degree elevation up to spline_order k
        for p in range(1, self.spline_order + 1):
            denom1 = grid[p:-1] - grid[:-(p + 1)]
            denom2 = grid[p + 1:] - grid[1:-p]
            bases = (
                (x_clamped - grid[:-(p + 1)]) / denom1 * bases[:, :, :-1]
                + (grid[p + 1:] - x_clamped) / denom2 * bases[:, :, 1:]
            )

        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining base activation and spline basis interpolation."""
        batch_size = x.shape[0]

        # 1. Base linear residual branch: w_b * silu(x)
        if self.base_weight is not None:
            base_out = F.linear(F.silu(x), self.base_weight)
        else:
            base_out = 0.0

        # 2. Spline branch
        bases = self.compute_b_splines(x)  # (batch_size, in_features, G + k)
        spline_out = F.linear(
            bases.contiguous().view(batch_size, -1),
            self.spline_weight.contiguous().view(self.out_features, -1),
        )

        out = base_out + spline_out
        if self.bias is not None:
            out = out + self.bias
        return out

    def get_parameter_count(self) -> int:
        """Return total active trainable parameters in this layer."""
        count = self.spline_weight.numel()
        if self.base_weight is not None:
            count += self.base_weight.numel()
        if self.bias is not None:
            count += self.bias.numel()
        return count


class ZelekeKAN(nn.Module):
    """
    Modular Kolmogorov-Arnold Network matching Zeleke & Bochicchio (2024).

    Parameters
    ----------
    layers_hidden : Sequence[int]
        Layer dimensions from input to output, e.g. [32, 64, 5] (default 1 hidden layer)
        or [32, 64, 32, 5] (2 hidden layers).
    grid_size : int
        Number of grid intervals G (default: 30).
    spline_order : int
        Spline polynomial degree k (default: 2, quadratic).
    grid_range : float
        Grid boundary [-grid_range, grid_range] (default: 1.0).
    """

    def __init__(
        self,
        layers_hidden: Sequence[int] = (32, 64, 5),
        grid_size: int = 30,
        spline_order: int = 2,
        grid_range: float = 1.0,
        use_base_update: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        self.layers_hidden = list(layers_hidden)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.grid_range = float(grid_range)

        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(self.layers_hidden[:-1], self.layers_hidden[1:]):
            self.layers.append(
                KANLayer(
                    in_features=in_dim,
                    out_features=out_dim,
                    grid_size=self.grid_size,
                    spline_order=self.spline_order,
                    grid_range=self.grid_range,
                    use_base_update=use_base_update,
                    use_bias=use_bias,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

    def get_parameter_count(self) -> int:
        """Return total trainable parameters in the model."""
        return sum(layer.get_parameter_count() for layer in self.layers)


class MLPBaseline(nn.Module):
    """
    Standard Multi-Layer Perceptron (MLP) baseline model for comparative benchmarking.

    Parameters
    ----------
    input_dim : int
        Number of input features (default: 32).
    hidden_dims : Sequence[int]
        Dimensions of hidden layers (default: [64, 32]).
    num_classes : int
        Number of output classes (default: 5).
    dropout : float
        Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dims: Sequence[int] = (64, 32),
        num_classes: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.num_classes = num_classes

        dims = [input_dim] + list(hidden_dims) + [num_classes]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_parameter_count(self) -> int:
        """Return total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_matched_mlp(
    target_params: int,
    input_dim: int = 32,
    num_classes: int = 5,
    num_hidden_layers: int = 2,
    dropout: float = 0.0,
) -> MLPBaseline:
    """
    Construct an MLP baseline whose parameter count closely matches the target KAN parameter budget.
    """
    best_diff = float("inf")
    best_hidden: List[int] = [64, 32]

    # Search for matching hidden dimensions
    if num_hidden_layers == 1:
        for h in range(16, 2048):
            params = (input_dim * h + h) + (h * num_classes + num_classes)
            diff = abs(params - target_params)
            if diff < best_diff:
                best_diff = diff
                best_hidden = [h]
    else:
        for h1 in range(32, 512, 8):
            for h2 in range(16, 512, 8):
                params = (
                    (input_dim * h1 + h1)
                    + (h1 * h2 + h2)
                    + (h2 * num_classes + num_classes)
                )
                diff = abs(params - target_params)
                if diff < best_diff:
                    best_diff = diff
                    best_hidden = [h1, h2]

    return MLPBaseline(
        input_dim=input_dim,
        hidden_dims=best_hidden,
        num_classes=num_classes,
        dropout=dropout,
    )


def build_model(
    config: Dict[str, Any],
    model_type: str = "kan",
    grid_size: Optional[int] = None,
    spline_order: Optional[int] = None,
) -> nn.Module:
    """
    Model factory function instantiating either KAN or baseline MLP based on configuration.
    """
    model_type = model_type.lower()
    in_dim = config.get("dataset", {}).get("target_features", 32)
    num_classes = 5

    if model_type == "kan":
        kan_cfg = config.get("kan", {})
        g = grid_size if grid_size is not None else kan_cfg.get("grid_size", 30)
        k = spline_order if spline_order is not None else kan_cfg.get("spline_order", 2)
        hidden_dims = kan_cfg.get("hidden_dims", [64])
        grid_range = kan_cfg.get("grid_range", 1.0)
        use_base = kan_cfg.get("use_base_update", True)
        use_bias = kan_cfg.get("use_bias", True)

        full_dims = [in_dim] + list(hidden_dims) + [num_classes]
        return ZelekeKAN(
            layers_hidden=full_dims,
            grid_size=g,
            spline_order=k,
            grid_range=grid_range,
            use_base_update=use_base,
            use_bias=use_bias,
        )

    elif model_type == "mlp":
        mlp_cfg = config.get("mlp", {})
        matched = mlp_cfg.get("matched_params", True)
        hidden_dims = mlp_cfg.get("hidden_dims", [64, 32])
        dropout = mlp_cfg.get("dropout", 0.0)

        if matched:
            # Match parameter count to KAN (G=30, k=2, [32, 64, 5] -> 78,213 params)
            kan_ref = build_model(config, model_type="kan", grid_size=grid_size, spline_order=spline_order)
            target_params = kan_ref.get_parameter_count()
            return build_matched_mlp(
                target_params=target_params,
                input_dim=in_dim,
                num_classes=num_classes,
                dropout=dropout,
            )
        else:
            return MLPBaseline(
                input_dim=in_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                dropout=dropout,
            )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected 'kan' or 'mlp'.")
