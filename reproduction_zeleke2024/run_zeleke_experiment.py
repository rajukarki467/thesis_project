#!/usr/bin/env python3
"""
CLI Reproduction Runner for Zeleke & Bochicchio (IEEE BigData 2024):
"Federated Kolmogorov-Arnold Networks for Health Data Analysis: A Study Using ECG Signal"

Features:
- Reads configuration from reproduction_zeleke2024/configs/zeleke_baseline.yaml.
- Supports both KAN (G=30, k=2) and parameter-matched MLP baseline.
- Configurable CLI flags: --model, --grid_size, --spline_order, --rounds, --epochs, --dry-run.
- Saves round-by-round metrics to JSON, CSV, and convergence plots in reproduction_zeleke2024/results/.
- Compares experimental outcomes against the published paper targets (93.73% Acc / 92.90% F1).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure reproduction directory and repo root are accessible
REPRO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = REPRO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))

import matplotlib
if os.environ.get("DISPLAY") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from reproduction_zeleke2024.src.client import FederatedClient
from reproduction_zeleke2024.src.model import build_model
from reproduction_zeleke2024.src.server import FederatedServer
from reproduction_zeleke2024.src.wavelet_preprocessing import load_and_preprocess_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_zeleke_experiment")


def set_seed(seed: int = 42) -> None:
    """Set global seeds for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def plot_convergence(
    history: List[Dict[str, Any]],
    model_name: str,
    output_path: Path,
    target_acc: float = 93.73,
    target_f1: float = 92.90,
) -> None:
    """Generate and save convergence curves comparing to paper benchmarks."""
    rounds = [r["round"] for r in history]
    accs = [r["test_accuracy"] * 100.0 for r in history]
    f1s = [r["test_macro_f1"] * 100.0 for r in history]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    line1 = ax1.plot(rounds, accs, marker="o", color="#1f77b4", linewidth=2.0, label="Test Accuracy (%)")
    ax1.set_xlabel("Federated Communication Round", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Test Accuracy (%)", color="#1f77b4", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(min(50.0, min(accs) - 5), 100.0)

    ax2 = ax1.twinx()
    line2 = ax2.plot(rounds, f1s, marker="s", color="#2ca02c", linewidth=2.0, linestyle="--", label="Macro F1-Score (%)")
    ax2.set_ylabel("Macro F1-Score (%)", color="#2ca02c", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(min(40.0, min(f1s) - 5), 100.0)

    # Reference target lines from paper
    if target_acc is not None:
        ax1.axhline(target_acc, color="#1f77b4", linestyle=":", alpha=0.7, label=f"Paper Target Acc ({target_acc:.2f}%)")
    if target_f1 is not None:
        ax2.axhline(target_f1, color="#2ca02c", linestyle=":", alpha=0.7, label=f"Paper Target F1 ({target_f1:.2f}%)")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", frameon=True)

    plt.title(
        f"Zeleke & Bochicchio (2024) Reproduction: {model_name.upper()} Convergence",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Convergence figure saved to {output_path}")


def print_comparison_summary(
    model_type: str,
    final_metrics: Dict[str, Any],
    targets: Dict[str, float],
    total_params: int,
    num_rounds: int,
    local_epochs: int,
    num_clients: int,
) -> None:
    """Print structured terminal summary comparing experimental results to paper benchmarks."""
    achieved_acc = final_metrics["test_accuracy"] * 100.0
    achieved_f1 = final_metrics["test_macro_f1"] * 100.0
    achieved_prec = final_metrics["test_macro_precision"] * 100.0
    achieved_rec = final_metrics["test_macro_recall"] * 100.0

    if model_type.lower() == "kan":
        target_acc = targets.get("kan_test_accuracy", 93.73)
        target_f1 = targets.get("kan_macro_f1", 92.90)
    else:
        target_acc = targets.get("mlp_test_accuracy", 91.65)
        target_f1 = None

    diff_acc = achieved_acc - target_acc
    diff_f1 = (achieved_f1 - target_f1) if target_f1 is not None else None

    print("\n" + "=" * 78)
    print("  ZELEKE & BOCHICCHIO (IEEE BigData 2024) BASELINE REPRODUCTION SUMMARY")
    print("=" * 78)
    print(f"  Model Architecture    : {model_type.upper()}")
    print(f"  Total Trainable Params: {total_params:,}")
    print(f"  Federated Setup       : {num_clients} clients, {num_rounds} rounds, {local_epochs} local epochs/round")
    print("-" * 78)
    print(f"  Metric                {'Achieved':<16} {'Paper Target':<16} {'Delta':<16}")
    print("-" * 78)
    acc_target_str = f"{target_acc:.2f}%"
    diff_acc_str = f"{diff_acc:+.2f}%"
    print(f"  Test Accuracy         {achieved_acc:.2f}%{' '*9} {acc_target_str:<16} {diff_acc_str:<16}")

    if target_f1 is not None:
        f1_target_str = f"{target_f1:.2f}%"
        diff_f1_str = f"{diff_f1:+.2f}%"
        print(f"  Macro F1-Score        {achieved_f1:.2f}%{' '*9} {f1_target_str:<16} {diff_f1_str:<16}")
    else:
        print(f"  Macro F1-Score        {achieved_f1:.2f}%{' '*9} {'N/A':<16} {'N/A':<16}")

    print(f"  Macro Precision       {achieved_prec:.2f}%")
    print(f"  Macro Recall          {achieved_rec:.2f}%")
    print(f"  Final Test Loss       {final_metrics['test_loss']:.4f}")
    print("-" * 78)
    print("  Per-Class Performance Breakdown:")
    for c_name, c_dict in final_metrics.get("class_metrics", {}).items():
        print(
            f"    - {c_name:<24}: F1={c_dict['f1'] * 100:.2f}% | "
            f"Prec={c_dict['precision'] * 100:.2f}% | Rec={c_dict['recall'] * 100:.2f}%"
        )
    print("=" * 78 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replicate Zeleke & Bochicchio (IEEE BigData 2024) Federated KAN benchmarks on MIT-BIH ECG."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPRO_ROOT / "configs" / "zeleke_baseline.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["kan", "mlp"],
        default="kan",
        help="Model architecture: 'kan' (default) or 'mlp'.",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=None,
        help="Grid size G for KAN splines (default: 30 from paper).",
    )
    parser.add_argument(
        "--spline_order",
        type=int,
        default=None,
        help="Spline polynomial order k (default: 2 for quadratic splines).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of global communication rounds (default: 10).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of local client epochs per round (default: 30).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Mini-batch size for client local training (default: 32).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick verification mode: runs 2 clients, 2 rounds, and 1 local epoch on a small subset.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputing the wavelet preprocessing instead of loading cache.",
    )

    args = parser.parse_args()

    # 1. Load Configuration
    config = load_yaml_config(args.config)
    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    # 2. Resolve Parameters (CLI overrides YAML)
    model_type = args.model.lower()
    grid_size = args.grid_size if args.grid_size is not None else config.get("kan", {}).get("grid_size", 30)
    spline_order = args.spline_order if args.spline_order is not None else config.get("kan", {}).get("spline_order", 2)
    num_rounds = 2 if args.dry_run else (args.rounds if args.rounds is not None else config.get("federated", {}).get("num_rounds", 10))
    local_epochs = 1 if args.dry_run else (args.epochs if args.epochs is not None else config.get("federated", {}).get("local_epochs", 30))
    batch_size = args.batch_size if args.batch_size is not None else config.get("federated", {}).get("batch_size", 32)
    learning_rate = float(config.get("federated", {}).get("learning_rate", 1.0e-5))
    weight_decay = float(config.get("federated", {}).get("weight_decay", 0.0))

    results_dir = Path(config.get("results_dir", str(REPRO_ROOT / "results")))
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting Zeleke & Bochicchio (2024) Reproduction Experiment")
    logger.info(f"Model: {model_type.upper()} | G={grid_size}, k={spline_order} | LR={learning_rate}")
    logger.info(f"Rounds: {num_rounds} | Local Epochs: {local_epochs} | Dry-run: {args.dry_run}")
    logger.info("=" * 60)

    # 3. Load or Process Dataset
    X_train, y_train, X_test, y_test, client_partitions = load_and_preprocess_dataset(
        config=config,
        force_recompute=args.force_recompute,
        dry_run=args.dry_run,
    )
    num_clients = len(client_partitions)
    logger.info(f"Dataset ready. Train beats: {len(y_train)} | Test beats: {len(y_test)} | Clients: {num_clients}")

    # 4. Instantiate Global Model
    global_model = build_model(
        config=config,
        model_type=model_type,
        grid_size=grid_size,
        spline_order=spline_order,
    )
    total_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    logger.info(f"Model instantiated: {model_type.upper()} with {total_params:,} trainable parameters.")

    class_weight = config.get("federated", {}).get("class_weight", "balanced")
    # 5. Instantiate Local Clients
    clients: Dict[int, FederatedClient] = {}
    for cid, indices in client_partitions.items():
        client_train_data = (X_train[indices], y_train[indices])
        clients[cid] = FederatedClient(
            client_id=cid,
            model=global_model,
            train_data=client_train_data,
            batch_size=batch_size,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            class_weight=class_weight,
        )

    # 6. Run Federated Simulation
    server = FederatedServer(
        global_model=global_model,
        test_data=(X_test, y_test),
        clients=clients,
    )

    simulation_results = server.run_simulation(
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        verbose=True,
    )

    # 7. Save Artifacts (JSON, CSV, Plots)
    run_prefix = f"zeleke_{model_type}"
    if args.dry_run:
        run_prefix += "_dryrun"

    # Save metrics JSON
    json_path = results_dir / f"{run_prefix}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(simulation_results, f, indent=2)
    logger.info(f"Simulation metrics written to {json_path}")

    # Save rounds CSV
    csv_path = results_dir / f"{run_prefix}_rounds.csv"
    rounds_df = pd.DataFrame(simulation_results["history"])
    rounds_df.to_csv(csv_path, index=False)
    logger.info(f"Per-round history written to {csv_path}")

    # Save convergence plot
    plot_path = results_dir / f"{run_prefix}_convergence.png"
    targets = config.get("paper_targets", {})
    t_acc = targets.get("kan_test_accuracy", 93.73) if model_type == "kan" else targets.get("mlp_test_accuracy", 91.65)
    t_f1 = targets.get("kan_macro_f1", 92.90) if model_type == "kan" else None
    plot_convergence(
        history=simulation_results["history"],
        model_name=model_type,
        output_path=plot_path,
        target_acc=t_acc,
        target_f1=t_f1,
    )

    # 8. Print Summary Comparison
    print_comparison_summary(
        model_type=model_type,
        final_metrics=simulation_results["final_metrics"],
        targets=targets,
        total_params=total_params,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        num_clients=num_clients,
    )


if __name__ == "__main__":
    main()
