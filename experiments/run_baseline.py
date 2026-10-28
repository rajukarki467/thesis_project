"""
Baseline Experiment Runner: FedAvg-MLP, FedAvg-KAN, and FedProx-KAN across Dirichlet Alphas.

Features:
- Full federated learning simulation across FedAvg-MLP, FedAvg-KAN, and FedProx-KAN.
- Evaluates across Dirichlet non-IID alphas in {0.1, 0.5, 1.0}.
- Strictly fair evaluation: MLP baseline matched in parameter budget to KAN.
- Centralized test evaluation on patient-level test split (Accuracy, Macro F1, Micro F1).
- Telemetry & communication volume tracking (round and cumulative MB).
- Saves all metrics, summaries, and comparison plots to results/baselines/.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
if os.environ.get("DISPLAY") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml

from src.federated.server import run_federated_simulation
from src.models.fastkan_model import FastKANClassifier
from src.models.mlp import MLPClassifier, build_matched_mlp
from src.partition import dirichlet_partition
from src.utils.reproducibility import set_seed

logger = logging.getLogger("baseline_experiments")


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_or_create_partitions(
    y_train: np.ndarray,
    num_clients: int,
    alpha: float,
    partitions_dir: Path,
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    """
    Load saved client partition indices for a given alpha, or generate them if absent.
    """
    config_dir = partitions_dir / f"clients_{num_clients}_alpha_{alpha}"
    alt_dir = partitions_dir / f"alpha_{alpha}"

    target_dir = config_dir if config_dir.exists() else alt_dir
    client_files = list(target_dir.glob("client_*.npy")) if target_dir.exists() else []

    if len(client_files) == num_clients:
        logger.info(f"Loaded existing partition indices from {target_dir}")
        partitions = {}
        for cid in range(num_clients):
            c_path = target_dir / f"client_{cid}.npy"
            partitions[cid] = np.load(c_path)
        return partitions

    # Otherwise generate deterministically
    logger.info(f"Generating new Dirichlet partition for alpha={alpha}, N={num_clients}...")
    partitions = dirichlet_partition(
        y=y_train,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
    )
    # Save indices
    config_dir.mkdir(parents=True, exist_ok=True)
    for cid, idx in partitions.items():
        np.save(config_dir / f"client_{cid}.npy", idx)
    return partitions


def build_models(
    kan_cfg: Dict[str, Any],
    mlp_cfg: Dict[str, Any],
) -> Tuple[FastKANClassifier, MLPClassifier]:
    """
    Construct FastKAN and parameter-budget matched MLP baseline models.
    """
    layers_hidden = kan_cfg.get("layers_hidden", [32, 16, 5])
    grid_size = kan_cfg.get("grid_size", 5)
    spline_order = kan_cfg.get("spline_order", 3)
    grid_range = kan_cfg.get("grid_range", 1.0)
    kan_dropout = kan_cfg.get("dropout", 0.0)

    # 1. Instantiate FastKAN model
    kan_model = FastKANClassifier(
        layers_hidden=layers_hidden,
        grid_size=grid_size,
        spline_order=spline_order,
        grid_range=grid_range,
        dropout=kan_dropout,
    )
    kan_params = kan_model.get_parameter_count()

    # 2. Build equivalent MLP baseline matched in parameter budget
    mlp_hidden = mlp_cfg.get("hidden_dims", None)
    mlp_dropout = mlp_cfg.get("dropout", 0.1)

    if mlp_hidden is not None and len(mlp_hidden) > 0:
        mlp_model = MLPClassifier(
            input_dim=layers_hidden[0],
            hidden_dims=mlp_hidden,
            num_classes=layers_hidden[-1],
            dropout=mlp_dropout,
        )
    else:
        mlp_model = build_matched_mlp(
            target_params=kan_params,
            input_dim=layers_hidden[0],
            num_classes=layers_hidden[-1],
            num_hidden_layers=2,
            dropout=mlp_dropout,
        )

    mlp_params = mlp_model.get_parameter_count()
    logger.info(
        f"Model Parameter Budgets: FastKAN = {kan_params} parameters | Matched MLP = {mlp_params} parameters "
        f"(Delta = {abs(kan_params - mlp_params)})"
    )

    return kan_model, mlp_model


def plot_baseline_results(
    all_experiments: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Generate publication-ready comparison plots across rounds, alphas, and algorithms.
    """
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Combine round records
    rows = []
    for exp in all_experiments:
        alg = exp["algorithm"]
        alpha = exp["alpha"]
        for r_entry in exp["history"]:
            rows.append({
                "Algorithm": alg,
                "Alpha": f"α = {alpha}",
                "Round": r_entry["round"],
                "Test Accuracy": r_entry["test_accuracy"],
                "Test Macro F1": r_entry["test_macro_f1"],
                "Cumulative Comm (MB)": r_entry["cumulative_communication_mb"],
            })

    df = pd.DataFrame(rows)

    # 1. Macro F1 vs Rounds
    g = sns.relplot(
        data=df,
        x="Round",
        y="Test Macro F1",
        hue="Algorithm",
        col="Alpha",
        kind="line",
        marker="o",
        height=4,
        aspect=1.2,
        palette="tab10",
    )
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Federated Learning Macro F1-Score Across Rounds (MIT-BIH Test Split)", fontsize=13, fontweight="bold")
    g.savefig(fig_dir / "macro_f1_vs_rounds_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Accuracy vs Rounds
    g = sns.relplot(
        data=df,
        x="Round",
        y="Test Accuracy",
        hue="Algorithm",
        col="Alpha",
        kind="line",
        marker="s",
        height=4,
        aspect=1.2,
        palette="tab10",
    )
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Federated Learning Accuracy Across Rounds (MIT-BIH Test Split)", fontsize=13, fontweight="bold")
    g.savefig(fig_dir / "accuracy_vs_rounds_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Communication Efficiency: Comm (MB) vs Macro F1
    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=df[df["Round"] > 0],
        x="Cumulative Comm (MB)",
        y="Test Macro F1",
        hue="Algorithm",
        style="Alpha",
        markers=True,
        dashes=False,
    )
    plt.title("Communication Volume vs Macro F1 Performance", fontsize=12, fontweight="bold")
    plt.xlabel("Cumulative Network Communication (MB)", fontsize=11)
    plt.ylabel("Test Macro F1", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "communication_efficiency_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved baseline comparison figures to {fig_dir}")


def run_all_baselines(
    config_path: str | Path = "configs/paper_baseline.yaml",
    alphas: Optional[Sequence[float]] = None,
    num_rounds: Optional[int] = None,
    local_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    output_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Execute complete FL loops across FedAvg-MLP, FedAvg-KAN, and FedProx-KAN for alphas.
    """
    cfg = load_config(config_path)

    global_seed = seed if seed is not None else cfg.get("experiment", {}).get("seed", 42)
    set_seed(global_seed)
    logger.info(f"Initialized global random seed: {global_seed}")

    eval_alphas = list(alphas) if alphas is not None else cfg.get("experiment", {}).get("alphas", [0.1, 0.5, 1.0])
    rounds = num_rounds or cfg.get("experiment", {}).get("num_rounds", 10)
    epochs = local_epochs or cfg.get("client", {}).get("local_epochs", 3)
    bs = batch_size or cfg.get("client", {}).get("batch_size", 32)
    lr = cfg.get("client", {}).get("learning_rate", 0.001)
    weight_decay = cfg.get("client", {}).get("weight_decay", 1e-4)
    optimizer_type = cfg.get("client", {}).get("optimizer", "adam")

    out_path = Path(output_dir or cfg.get("output_dir", "results/baselines"))
    out_path.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg.get("data", {})
    processed_dir = PROJECT_ROOT / Path(data_cfg.get("processed_dir", "data/processed/beats"))
    partitions_dir = PROJECT_ROOT / Path(data_cfg.get("partitions_dir", "data/partitions"))

    # Load preprocessed dataset
    logger.info("Loading preprocessed ECG heartbeat datasets...")
    X_train = np.load(processed_dir / "X_train.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    test_data = (X_test, y_test)

    logger.info(
        f"Loaded {len(X_train)} training beats, {len(X_test)} held-out test beats (feature dim: {X_train.shape[1]})."
    )

    kan_cfg = cfg.get("models", {}).get("kan", {})
    mlp_cfg = cfg.get("models", {}).get("mlp", {})

    algorithms_cfg = cfg.get("algorithms", [
        {"name": "FedAvg-MLP", "model": "mlp", "strategy": "fedavg", "proximal_mu": 0.0},
        {"name": "FedAvg-KAN", "model": "kan", "strategy": "fedavg", "proximal_mu": 0.0},
        {"name": "FedProx-KAN", "model": "kan", "strategy": "fedprox", "proximal_mu": 0.01},
    ])

    num_clients = cfg.get("experiment", {}).get("num_clients", 8)

    all_experiments: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    # Outer loop: Dirichlet alpha levels
    for alpha in eval_alphas:
        logger.info(f"\n{'='*70}\nSTARTING EXPERIMENTS FOR DIRICHLET ALPHA = {alpha}\n{'='*70}")

        # Load or create partitions for this alpha
        partitions = load_or_create_partitions(
            y_train=y_train,
            num_clients=num_clients,
            alpha=alpha,
            partitions_dir=partitions_dir,
            seed=global_seed,
        )

        # Slice client datasets
        client_datasets = {
            cid: (X_train[indices], y_train[indices])
            for cid, indices in partitions.items()
        }

        # Inner loop: Federated algorithms
        for algo in algorithms_cfg:
            algo_name = algo["name"]
            model_type = algo["model"]
            strategy = algo["strategy"]
            mu = float(algo.get("proximal_mu", 0.0))

            logger.info(
                f"\n>>> Running Algorithm: {algo_name} | Strategy: {strategy.upper()} "
                f"| Model: {model_type.upper()} | Alpha: {alpha} | Mu: {mu}"
            )

            # Build fresh models
            set_seed(global_seed)
            kan_m, mlp_m = build_models(kan_cfg, mlp_cfg)
            active_model = kan_m if model_type == "kan" else mlp_m
            param_count = active_model.get_parameter_count()

            # Run FL simulation
            sim_result = run_federated_simulation(
                model=active_model,
                client_datasets=client_datasets,
                test_data=test_data,
                num_rounds=rounds,
                strategy_name=strategy,
                proximal_mu=mu,
                local_epochs=epochs,
                batch_size=bs,
                learning_rate=lr,
                optimizer_type=optimizer_type,
                weight_decay=weight_decay,
                verbose=True,
            )

            history = sim_result["history"]
            final_metric = sim_result["final_metrics"]

            # Save per-round metrics to CSV
            metrics_df = pd.DataFrame(history)
            csv_filename = f"metrics_{algo_name}_alpha_{alpha}.csv"
            metrics_df.to_csv(out_path / csv_filename, index=False)

            # Save detailed summary to JSON
            summary_record = {
                "algorithm": algo_name,
                "strategy": strategy,
                "model_type": model_type,
                "alpha": alpha,
                "proximal_mu": mu,
                "parameters_count": param_count,
                "num_rounds": rounds,
                "num_clients": num_clients,
                "final_test_loss": final_metric["test_loss"],
                "final_test_accuracy": final_metric["test_accuracy"],
                "final_test_macro_f1": final_metric["test_macro_f1"],
                "final_test_micro_f1": final_metric["test_micro_f1"],
                "total_communication_mb": final_metric["cumulative_communication_mb"],
                "mean_client_latency_sec": float(np.mean([h["mean_client_latency"] for h in history[1:]])),
                "mean_input_variance": final_metric["mean_input_variance"],
                "per_class_f1": {
                    c: final_metric["per_class_metrics"][c]["f1"]
                    for c in final_metric["per_class_metrics"]
                },
            }
            json_filename = f"summary_{algo_name}_alpha_{alpha}.json"
            with open(out_path / json_filename, "w") as f:
                json.dump(summary_record, f, indent=2)

            all_experiments.append({
                "algorithm": algo_name,
                "alpha": alpha,
                "history": history,
                "summary": summary_record,
            })

            summary_rows.append({
                "Algorithm": algo_name,
                "Alpha": alpha,
                "Model": model_type.upper(),
                "Active Parameters": param_count,
                "Final Accuracy": final_metric["test_accuracy"],
                "Final Macro F1": final_metric["test_macro_f1"],
                "Final Micro F1": final_metric["test_micro_f1"],
                "Comm (MB)": final_metric["cumulative_communication_mb"],
                "Avg Latency (s)": summary_record["mean_client_latency_sec"],
            })

    # Save overall comparison summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_path / "baseline_comparison_summary.csv", index=False)
    with open(out_path / "baseline_comparison_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    logger.info(f"\n{'='*70}\nALL BASELINE EXPERIMENTS COMPLETED SUCCESSFULLY!\n{'='*70}")
    print("\nSummary Comparison Table:")
    print(summary_df.to_string(index=False))

    # Generate comparison figures
    plot_baseline_results(all_experiments, out_path)

    return summary_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run federated baseline benchmark: FedAvg-MLP, FedAvg-KAN, and FedProx-KAN."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/paper_baseline.yaml",
        help="Path to experiment YAML configuration.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=None,
        help="Dirichlet concentration parameters to evaluate (e.g. 0.1 0.5 1.0).",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=None,
        help="Number of federated rounds.",
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=None,
        help="Local training epochs per client per round.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Local batch size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for metrics and results.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    run_all_baselines(
        config_path=args.config,
        alphas=args.alphas,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
