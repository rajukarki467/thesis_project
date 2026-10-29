"""
Unit tests for Zeleke & Bochicchio (2024) reproduction package.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch
import torch.nn as nn

from reproduction_zeleke2024.src.client import FederatedClient
from reproduction_zeleke2024.src.model import (
    KANLayer,
    MLPBaseline,
    ZelekeKAN,
    build_matched_mlp,
    build_model,
)
from reproduction_zeleke2024.src.server import (
    aggregate_fedavg,
    evaluate_global_model,
)
from reproduction_zeleke2024.src.wavelet_preprocessing import (
    apply_wavelet_denoising_db5,
    partition_clients_dirichlet,
    resample_beat_linear,
    zscore_normalize,
)


def test_wavelet_denoising_db5():
    """Verify 9-level db5 wavelet denoising suppresses drift and reconstructs original length."""
    t = np.linspace(0, 100, 36000)
    baseline_drift = 2.0 * np.sin(0.1 * 2 * np.pi * t)
    ecg_signal = np.sin(1.2 * 2 * np.pi * t) + 0.5 * np.sin(15 * 2 * np.pi * t)
    noisy_signal = baseline_drift + ecg_signal + 0.2 * np.random.randn(len(t))

    denoised = apply_wavelet_denoising_db5(
        noisy_signal, level=9, wavelet="db5", suppress_baseline=True, suppress_hf_noise=True
    )
    assert denoised.shape == noisy_signal.shape
    assert not np.isnan(denoised).any()
    # Baseline drift should be significantly suppressed
    assert np.var(denoised) < np.var(noisy_signal)


def test_beat_normalization_and_resampling():
    """Verify beat z-score normalization and linear interpolation to n=32."""
    raw_beat = np.random.randn(360) * 3.5 + 10.0
    norm_beat = zscore_normalize(raw_beat)
    assert np.isclose(np.mean(norm_beat), 0.0, atol=1e-6)
    assert np.isclose(np.std(norm_beat), 1.0, atol=1e-6)

    resampled = resample_beat_linear(norm_beat, target_length=32)
    assert resampled.shape == (32,)
    assert not np.isnan(resampled).any()


def test_kan_layer_and_cox_de_boor():
    """Verify KANLayer quadratic B-spline basis expansion (G=30, k=2)."""
    layer = KANLayer(in_features=32, out_features=64, grid_size=30, spline_order=2)
    assert layer.grid.shape[0] == 30 + 2 * 2 + 1  # 35 knots
    x = torch.randn(8, 32)
    bases = layer.compute_b_splines(x)
    assert bases.shape == (8, 32, 32)  # (batch, in_features, G + k = 32)

    out = layer(x)
    assert out.shape == (8, 64)
    assert layer.get_parameter_count() > 0


def test_zeleke_kan_model():
    """Verify ZelekeKAN architecture and parameter counts."""
    model = ZelekeKAN(layers_hidden=[32, 64, 5], grid_size=30, spline_order=2)
    x = torch.randn(4, 32)
    out = model(x)
    assert out.shape == (4, 5)
    # Layer 1: 64*32*32 (spline) + 64*32 (base) + 64 (bias) = 67,648
    # Layer 2: 5*64*32 (spline) + 5*64 (base) + 5 (bias) = 10,565
    assert model.get_parameter_count() == 78213


def test_matched_mlp_baseline():
    """Verify parameter matching for fair comparative benchmark."""
    target_params = 78213
    mlp = build_matched_mlp(target_params=target_params, input_dim=32, num_classes=5)
    mlp_params = mlp.get_parameter_count()
    delta = abs(mlp_params - target_params)
    assert delta / target_params < 0.05  # within 5%
    x = torch.randn(4, 32)
    out = mlp(x)
    assert out.shape == (4, 5)


def test_fedavg_aggregation():
    """Verify exact weighted FedAvg aggregation."""
    model1 = nn.Linear(4, 2, bias=False)
    model2 = nn.Linear(4, 2, bias=False)

    w1 = torch.ones(2, 4) * 2.0
    w2 = torch.ones(2, 4) * 4.0
    model1.weight.data = w1
    model2.weight.data = w2

    updates = [
        (model1.state_dict(), 100),
        (model2.state_dict(), 300),
    ]
    # Weighted avg: (100 * 2.0 + 300 * 4.0) / 400 = (200 + 1200) / 400 = 3.5
    agg = aggregate_fedavg(updates)
    assert torch.allclose(agg["weight"], torch.ones(2, 4) * 3.5)


def test_client_fit_and_server_evaluate():
    """Verify client local training and server evaluation."""
    model = nn.Linear(32, 5)
    X = np.random.randn(64, 32).astype(np.float32)
    y = np.random.randint(0, 5, size=(64,)).astype(np.int64)

    client = FederatedClient(client_id=0, model=model, train_data=(X, y), batch_size=16, local_epochs=1)
    state, n_samples, metrics = client.fit(model.state_dict())
    assert n_samples == 64
    assert "local_loss" in metrics

    eval_metrics = evaluate_global_model(model, (X, y), batch_size=32)
    assert "accuracy" in eval_metrics
    assert "macro_f1" in eval_metrics
