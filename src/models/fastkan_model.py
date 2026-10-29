"""
Fast Kolmogorov-Arnold Network (FastKAN) Classifier for ECG Arrhythmia Classification.

Features:
- Modular KAN architecture mapping 32-sample ECG vectors to AAMI classes.
- Edge layers with B-spline parameterization over G grid intervals and degree k.
- Dynamically configurable (G, k) parameters with G in {3, 5, 7, 10} and k in {2, 3, 4}.
- Accurate active parameter and coefficient counting via get_parameter_count().
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_GRID_SIZES = {3, 5, 7, 10, 30}
SUPPORTED_SPLINE_ORDERS = {2, 3, 4}


def compute_spline_bases(
    x: Union[torch.Tensor, np.ndarray],
    grid_size: int,
    spline_order: int,
    grid_range: float = 1.0,
) -> torch.Tensor:
    """
    Evaluate degree-k B-spline basis functions using Cox-de Boor recursion.

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        Sample points, shape (P,) or (P, 1).
    grid_size : int
        Number of grid intervals G.
    spline_order : int
        Spline degree k.
    grid_range : float
        Domain boundary [-grid_range, grid_range] (default 1.0).

    Returns
    -------
    torch.Tensor
        Evaluated basis matrix of shape (P, G + k).
    """
    h = (2.0 * float(grid_range)) / float(grid_size)
    total_knots = int(grid_size) + 2 * int(spline_order) + 1

    device = x.device if isinstance(x, torch.Tensor) else None
    grid = torch.linspace(
        -grid_range - spline_order * h,
        grid_range + spline_order * h,
        total_knots,
        dtype=torch.float32,
        device=device,
    )

    if not isinstance(x, torch.Tensor):
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
    else:
        x_tensor = x

    x_clamped = torch.clamp(x_tensor, -grid_range, grid_range)
    if x_clamped.ndim == 1:
        x_clamped = x_clamped.unsqueeze(-1)

    grid = grid.to(dtype=x_clamped.dtype, device=x_clamped.device)
    # 0-degree step functions
    bases = ((x_clamped >= grid[:-1]) & (x_clamped < grid[1:])).to(x_clamped.dtype)

    # Cox-de Boor recursive degree elevation
    for p in range(1, spline_order + 1):
        denom1 = grid[p:-1] - grid[:-(p + 1)]
        denom2 = grid[p + 1:] - grid[1:-p]
        bases = (
            (x_clamped - grid[:-(p + 1)]) / denom1 * bases[:, :-1]
            + (grid[p + 1:] - x_clamped) / denom2 * bases[:, 1:]
        )
    return bases


def project_spline_weights(
    spline_weight: Union[torch.Tensor, np.ndarray],
    src_gk: Tuple[int, int],
    dst_gk: Tuple[int, int],
    grid_range: float = 1.0,
    num_samples: int = 200,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Project B-spline coefficients from source configuration (G_src, k_src) to
    destination configuration (G_dst, k_dst) via uniform sample evaluation and
    least-squares spline projection.

    Parameters
    ----------
    spline_weight : torch.Tensor or np.ndarray
        Spline coefficients tensor with shape (..., M_src) where M_src = G_src + k_src.
    src_gk : Tuple[int, int]
        (grid_size, spline_order) of the source basis.
    dst_gk : Tuple[int, int]
        (grid_size, spline_order) of the destination basis.
    grid_range : float
        Domain boundary [-grid_range, grid_range].
    num_samples : int
        Number of uniform evaluation points across [-grid_range, grid_range].

    Returns
    -------
    torch.Tensor or np.ndarray
        Projected spline coefficients of shape (..., M_dst) where M_dst = G_dst + k_dst.
    """
    g_s, k_s = int(src_gk[0]), int(src_gk[1])
    g_d, k_d = int(dst_gk[0]), int(dst_gk[1])

    is_numpy = isinstance(spline_weight, np.ndarray)
    if is_numpy:
        orig_dtype = spline_weight.dtype
        w_tensor = torch.from_numpy(spline_weight.astype(np.float32))
    else:
        orig_dtype = spline_weight.dtype
        w_tensor = spline_weight.to(torch.float32)

    # Exact identity when source matches destination
    if (g_s, k_s) == (g_d, k_d):
        return spline_weight.copy() if is_numpy else spline_weight.clone()

    device = w_tensor.device
    x_eval = torch.linspace(-grid_range, grid_range, num_samples, dtype=torch.float32, device=device)

    # Basis evaluation at sample points
    b_src = compute_spline_bases(x_eval, g_s, k_s, grid_range=grid_range)  # (P, M_s)
    b_dst = compute_spline_bases(x_eval, g_d, k_d, grid_range=grid_range)  # (P, M_d)

    # Least squares projection matrix: M_proj = pinv(b_dst) @ b_src in R^{M_d x M_s}
    proj_matrix = torch.linalg.pinv(b_dst) @ b_src  # (M_d, M_s)

    # w_tensor has shape (..., M_s), multiply by proj_matrix.T -> (..., M_d)
    w_proj = torch.matmul(w_tensor, proj_matrix.T)

    if is_numpy:
        return w_proj.detach().cpu().numpy().astype(orig_dtype)
    return w_proj.to(dtype=orig_dtype)


class FastKANLayer(nn.Module):
    """
    A single edge layer of a Kolmogorov-Arnold Network using B-spline basis expansion.

    Each input-output edge (i -> j) has an activation function:
        phi_{j, i}(x_i) = w_b * silu(x_i) + sum_{m=0}^{G+k-1} c_{j, i, m} * B_m(x_i)

    Parameters
    ----------
    in_features : int
        Number of input dimensions.
    out_features : int
        Number of output dimensions.
    grid_size : int
        Number of grid intervals G in {3, 5, 7, 10}.
    spline_order : int
        Spline polynomial degree k in {2, 3, 4}.
    grid_range : float
        Interval boundary [-grid_range, grid_range] for B-spline evaluation (default 1.0).
    use_base_update : bool
        Whether to include the linear residual branch w_b * silu(x) (default True).
    use_bias : bool
        Whether to include an additive bias term (default True).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: float = 1.0,
        use_base_update: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_range = float(grid_range)
        self.use_base_update = use_base_update
        self.use_bias = use_bias

        self._validate_gk(grid_size, spline_order)
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Initialize grid buffer
        self._build_grid()

        # Base linear branch parameters
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

    @staticmethod
    def _validate_gk(grid_size: int, spline_order: int) -> None:
        if grid_size not in SUPPORTED_GRID_SIZES:
            raise ValueError(
                f"grid_size (G) must be in {SUPPORTED_GRID_SIZES}, got {grid_size}."
            )
        if spline_order not in SUPPORTED_SPLINE_ORDERS:
            raise ValueError(
                f"spline_order (k) must be in {SUPPORTED_SPLINE_ORDERS}, got {spline_order}."
            )

    def _build_grid(self) -> None:
        """Construct uniform B-spline knot vector including margin knots."""
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

        # Initialize spline coefficients with small Gaussian noise
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
        # Clamp inputs within the grid range for numerical stability
        x_clamped = torch.clamp(x, -self.grid_range, self.grid_range).unsqueeze(-1)
        grid = self.grid.to(dtype=x.dtype, device=x.device)

        # 0-degree step functions
        bases = ((x_clamped >= grid[:-1]) & (x_clamped < grid[1:])).to(x.dtype)

        # Cox-de Boor recursive degree elevation
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

    def update_grid(self, new_grid_size: int, new_spline_order: int) -> None:
        """
        Dynamically update (G, k) parameters of this layer.
        Interpolates existing spline weights onto the new grid using
        canonical B-spline basis evaluation and least-squares projection.
        """
        self._validate_gk(new_grid_size, new_spline_order)
        if new_grid_size == self.grid_size and new_spline_order == self.spline_order:
            return

        old_gk = (self.grid_size, self.spline_order)
        new_gk = (new_grid_size, new_spline_order)

        self.grid_size = new_grid_size
        self.spline_order = new_spline_order
        self._build_grid()

        # Project spline weights via least-squares basis projection
        with torch.no_grad():
            old_w = self.spline_weight.data  # (out_features, in_features, old_num_bases)
            new_w = project_spline_weights(
                old_w, src_gk=old_gk, dst_gk=new_gk, grid_range=self.grid_range
            )
            self.spline_weight = nn.Parameter(new_w)

    def get_spline_parameter_count(self) -> int:
        """Return active count of B-spline coefficients."""
        return self.spline_weight.numel()

    def get_parameter_count(self) -> int:
        """Return total active trainable parameters in this layer."""
        count = self.spline_weight.numel()
        if self.base_weight is not None:
            count += self.base_weight.numel()
        if self.bias is not None:
            count += self.bias.numel()
        return count


class FastKANClassifier(nn.Module):
    """
    Modular Kolmogorov-Arnold Network (KAN) for ECG Arrhythmia Heartbeat Classification.

    Parameters
    ----------
    layers_hidden : Sequence[int]
        Layer dimensions from input to output, e.g. [32, 16, 5].
    grid_size : int or Sequence[int]
        Number of grid intervals G in {3, 5, 7, 10} for each layer (default 5).
    spline_order : int or Sequence[int]
        Spline degree k in {2, 3, 4} for each layer (default 3).
    grid_range : float
        Grid domain boundary (default 1.0).
    dropout : float
        Dropout probability between layers (default 0.0).
    use_base_update : bool
        Whether to include base residual branch (default True).
    use_bias : bool
        Whether to include bias parameters (default True).
    """

    def __init__(
        self,
        layers_hidden: Sequence[int] = (32, 16, 5),
        grid_size: Union[int, Sequence[int]] = 5,
        spline_order: Union[int, Sequence[int]] = 3,
        grid_range: float = 1.0,
        dropout: float = 0.0,
        use_base_update: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        self.layers_hidden = list(layers_hidden)
        self.grid_range = grid_range
        num_layers = len(self.layers_hidden) - 1

        if num_layers < 1:
            raise ValueError("layers_hidden must have at least 2 dimensions (input and output).")

        # Handle grid_size and spline_order per layer
        if isinstance(grid_size, int):
            self.grid_sizes = [grid_size] * num_layers
        else:
            self.grid_sizes = list(grid_size)
            if len(self.grid_sizes) != num_layers:
                raise ValueError(f"Expected {num_layers} grid sizes, got {len(self.grid_sizes)}.")

        if isinstance(spline_order, int):
            self.spline_orders = [spline_order] * num_layers
        else:
            self.spline_orders = list(spline_order)
            if len(self.spline_orders) != num_layers:
                raise ValueError(f"Expected {num_layers} spline orders, got {len(self.spline_orders)}.")

        # Build KAN layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(num_layers):
            in_dim = self.layers_hidden[i]
            out_dim = self.layers_hidden[i + 1]
            g = self.grid_sizes[i]
            k = self.spline_orders[i]

            layer = FastKANLayer(
                in_features=in_dim,
                out_features=out_dim,
                grid_size=g,
                spline_order=k,
                grid_range=grid_range,
                use_base_update=use_base_update,
                use_bias=use_bias,
            )
            self.layers.append(layer)
            self.dropouts.append(nn.Dropout(p=dropout) if dropout > 0 and i < num_layers - 1 else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over sequential KAN layers.

        Parameters
        ----------
        x : torch.Tensor
            Batch of resampled ECG beats with shape (batch_size, 32).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        h = x
        for layer, drop in zip(self.layers, self.dropouts):
            h = layer(h)
            h = drop(h)
        return h

    def set_layer_gk(self, layer_idx: int, grid_size: int, spline_order: int) -> None:
        """
        Dynamically update (G, k) parameters for a specific edge layer.

        Parameters
        ----------
        layer_idx : int
            Index of the layer (0 <= layer_idx < num_layers).
        grid_size : int
            New G in {3, 5, 7, 10}.
        spline_order : int
            New k in {2, 3, 4}.
        """
        if not 0 <= layer_idx < len(self.layers):
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {len(self.layers)}).")
        self.layers[layer_idx].update_grid(grid_size, spline_order)
        self.grid_sizes[layer_idx] = grid_size
        self.spline_orders[layer_idx] = spline_order

    def get_parameter_count(self, detailed: bool = False) -> Union[int, Dict[str, Any]]:
        """
        Calculate active coefficient and parameter counts.

        Parameters
        ----------
        detailed : bool
            If True, returns a breakdown dictionary; otherwise returns integer count.

        Returns
        -------
        int or Dict[str, Any]
            Active coefficient count.
        """
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        spline_coeffs = sum(layer.get_spline_parameter_count() for layer in self.layers)
        base_weights = total - spline_coeffs

        if detailed:
            layer_breakdown = []
            for idx, layer in enumerate(self.layers):
                layer_breakdown.append({
                    "layer": idx,
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                    "grid_size": layer.grid_size,
                    "spline_order": layer.spline_order,
                    "spline_coefficients": layer.get_spline_parameter_count(),
                    "total_layer_parameters": layer.get_parameter_count(),
                })
            return {
                "total_parameters": total,
                "spline_coefficients": spline_coeffs,
                "base_parameters": base_weights,
                "layers": layer_breakdown,
            }
        return total
