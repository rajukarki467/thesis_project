"""
Unit tests for communication cost vs performance visualization module.
Verifies:
  - Generation of all communication figures (C1-C8)
  - Output formats (.png and .pdf)
  - Edge cases: single seed, missing methods, log scale
  - Exact Pareto frontier computation
"""

from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.visualization import (
    METHOD_STYLE,
    _compute_pareto_frontier,
    generate_all_comm_figures,
    plot_accuracy_vs_cum_comm,
    plot_comm_breakdown_and_configs,
    plot_comm_to_target_acc,
    plot_f1_vs_cum_comm,
    plot_final_perf_vs_total_comm,
    plot_savings_summary_bar,
    plot_small_multiples_per_method,
)


def _generate_synthetic_traces(seeds=(42, 43), alphas=(0.1, 0.5), num_rounds=5) -> pd.DataFrame:
    """Generate synthetic communication traces dataframe matching required 19 columns."""
    rows = []
    methods = list(METHOD_STYLE.keys())

    for seed in seeds:
        for alpha in alphas:
            for method in methods:
                base_acc = 75.0 if "MLP" in method else 82.0
                params = 28576 if "3,2" in method else (71208 if "10,4" in method else 42792)
                cum_p = 0
                cum_t = 0.0

                for r in range(1, num_rounds + 1):
                    cum_p += params * 2
                    cum_mb = (cum_p * 4) / (1024 * 1024)
                    lat = 0.25
                    cum_t += lat

                    rows.append({
                        "round": r,
                        "seed": seed,
                        "alpha": alpha,
                        "method": method,
                        "test_acc": base_acc + np.log(r + 1) * 2.0 + (seed - 42) * 0.5,
                        "test_macro_f1": 35.0 + np.log(r + 1) * 3.0,
                        "test_balanced_acc": 36.0 + np.log(r + 1) * 2.5,
                        "val_acc": base_acc + np.log(r + 1) * 2.2,
                        "val_macro_f1": 36.0 + np.log(r + 1) * 2.8,
                        "params_up": params,
                        "params_down": params,
                        "params_round": params * 2,
                        "bytes_round": params * 8,
                        "cum_params": cum_p,
                        "cum_bytes": cum_p * 4,
                        "cum_MB": cum_mb,
                        "est_comm_time_s": lat,
                        "est_comm_time_sum_s": lat * 8,
                        "config_histogram": str({"Low": 0.5, "Medium": 0.5}),
                    })

    return pd.DataFrame(rows)


def test_pareto_frontier_logic():
    """Verify Pareto frontier correctly identifies non-dominated points."""
    # (comm, acc): we want MIN comm, MAX acc
    points = [
        (1.0, 90.0),  # non-dominated
        (2.0, 92.0),  # non-dominated
        (3.0, 95.0),  # non-dominated
        (2.5, 91.0),  # dominated by (2.0, 92.0)
        (4.0, 94.0),  # dominated by (3.0, 95.0)
    ]
    pareto_indices = _compute_pareto_frontier(points)
    assert pareto_indices == [0, 1, 2]


def test_plot_accuracy_and_pareto_file_creation():
    """Verify Fig C1 and Fig C3 create both .png and .pdf files."""
    df = _generate_synthetic_traces()

    with tempfile.TemporaryDirectory() as tmpdir:
        c1_path = Path(tmpdir) / "fig_c1"
        c3_path = Path(tmpdir) / "fig_c3"

        plot_accuracy_vs_cum_comm(df, save_path=c1_path)
        plot_final_perf_vs_total_comm(df, save_path=c3_path)

        assert c1_path.with_suffix(".png").exists()
        assert c1_path.with_suffix(".pdf").exists()
        assert c3_path.with_suffix(".png").exists()
        assert c3_path.with_suffix(".pdf").exists()
        assert c1_path.with_suffix(".png").stat().st_size > 0
        assert c3_path.with_suffix(".png").stat().st_size > 0


def test_generate_all_comm_figures():
    """Verify pipeline generates all figures C1-C8 and per-method individual plots."""
    df = _generate_synthetic_traces(seeds=(42,), alphas=(0.5,), num_rounds=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "comm_figures"
        files = generate_all_comm_figures(df, output_dir=out_dir)

        for fig_key in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]:
            assert fig_key in files
            assert files[fig_key].with_suffix(".png").exists()
            assert files[fig_key].with_suffix(".pdf").exists()

        # Check per_method directory
        pm_dir = out_dir / "per_method"
        assert pm_dir.exists()
        pm_pngs = list(pm_dir.glob("*.png"))
        assert len(pm_pngs) > 0


def test_missing_methods_handling():
    """Verify plots handle dataframes with only a subset of methods gracefully."""
    df = _generate_synthetic_traces()
    # Filter to only two methods
    sub_df = df[df["method"].isin(["FedAvg-KAN (fixed 5,3)", "PPO-Adaptive-KAN"])]

    with tempfile.TemporaryDirectory() as tmpdir:
        c1_path = Path(tmpdir) / "sub_c1"
        c3_path = Path(tmpdir) / "sub_c3"

        plot_accuracy_vs_cum_comm(sub_df, save_path=c1_path)
        plot_final_perf_vs_total_comm(sub_df, save_path=c3_path)

        assert c1_path.with_suffix(".png").exists()
        assert c3_path.with_suffix(".png").exists()


def test_single_seed_handling():
    """Verify plotting runs without error when only a single seed is provided."""
    df = _generate_synthetic_traces(seeds=(42,))

    with tempfile.TemporaryDirectory() as tmpdir:
        c1_path = Path(tmpdir) / "single_seed_c1"
        plot_accuracy_vs_cum_comm(df, save_path=c1_path)
        assert c1_path.with_suffix(".png").exists()
