"""
Experiment Runner: Rule-Based Heuristic Adaptive KAN across Dirichlet Alphas.

Features:
- Runs federated rounds using the deterministic quantile-based HeuristicController.
- Evaluates across Dirichlet non-IID alphas in {0.1, 0.5, 1.0}.
- Evaluates global model on held-out patient-level test split (18,175 beats).
- Tracks accuracy, macro F1, parameter counts, latency, and chosen (G_i, k_i) traces.
- Saves results to results/adaptive_kan/heuristic_results_{alpha}.csv and JSON summary.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from src.controllers.heuristic import HeuristicController
from src.environment.f1_env import FederatedKANEnv, action_to_gk
from src.utils.reproducibility import set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_heuristic")


def run_heuristic_simulation(
    alpha: float,
    num_rounds: int = 10,
    config_path: str = "configs/ppo.yaml",
    output_dir: str = "results/adaptive_kan",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Execute federated simulation using HeuristicController for a specific Dirichlet alpha.

    Parameters
    ----------
    alpha : float
        Dirichlet non-IID skew parameter.
    num_rounds : int
        Number of federated communication rounds.
    config_path : str
        Path to PPO / heuristic YAML configuration.
    output_dir : str
        Directory to save CSV and JSON results.
    seed : int
        RNG seed for deterministic execution.
    verbose : bool
        Whether to print round logs.

    Returns
    -------
    Dict[str, Any]
        Summary dictionary containing final metrics and round trajectory.
    """
    out_dir = PROJECT_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*75}\nSTARTING HEURISTIC EXPERIMENT FOR ALPHA = {alpha} ({num_rounds} ROUNDS)\n{'='*75}")

    set_seed(seed)

    # 1. Instantiate Federated Environment and Heuristic Controller
    env = FederatedKANEnv(
        config_path=config_path,
        alpha=alpha,
        max_rounds=num_rounds,
        seed=seed,
    )
    controller = HeuristicController(config_path=config_path)

    # 2. Reset Environment
    obs, info = env.reset(seed=seed)
    initial_acc = float(info.get("global_accuracy", 0.0))
    initial_f1 = float(info.get("global_macro_f1", 0.0))
    initial_loss = float(info.get("global_loss", 0.0))

    if verbose:
        logger.info(
            f"[Round 00/{num_rounds:02d}] Initial Global Model - "
            f"Loss: {initial_loss:.4f} | Acc: {initial_acc:.4f} | Macro F1: {initial_f1:.4f}"
        )

    records: List[Dict[str, Any]] = []
    # Record round 0
    records.append({
        "round": 0,
        "alpha": alpha,
        "test_loss": initial_loss,
        "test_accuracy": initial_acc,
        "test_macro_f1": initial_f1,
        "delta_accuracy": 0.0,
        "step_reward": 0.0,
        "total_parameters": env.num_clients * 5349,
        "mean_parameters_per_client": 5349.0,
        "total_latency_seconds": 0.0,
        "mean_latency_seconds": 0.0,
        "step_duration_seconds": 0.0,
        "client_configs": "[(5, 3)] * 8",
        "client_actions": "[4] * 8",
    })

    total_time_start = time.perf_counter()

    # 3. Federated Rounds Loop
    for r in range(1, num_rounds + 1):
        step_start = time.perf_counter()

        # Controller selects actions based on current telemetry state S^t
        actions = controller.predict(obs)

        # Step environment
        obs, reward, terminated, truncated, step_info = env.step(actions)
        step_duration = time.perf_counter() - step_start

        test_acc = float(step_info["global_accuracy"])
        test_f1 = float(step_info["global_macro_f1"])
        test_loss = float(step_info["global_loss"])
        delta_acc = float(step_info["delta_accuracy"])
        tot_params = float(step_info["total_parameters"])
        mean_params = float(step_info["mean_parameters_per_client"])
        tot_lat = float(step_info["total_latency"])
        mean_lat = float(step_info["mean_latency"])

        cfg_list = [(a["G"], a["k"]) for a in step_info["actions"]]
        act_indices = [a["action_idx"] for a in step_info["actions"]]

        record = {
            "round": r,
            "alpha": alpha,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
            "delta_accuracy": delta_acc,
            "step_reward": float(reward),
            "total_parameters": tot_params,
            "mean_parameters_per_client": mean_params,
            "total_latency_seconds": tot_lat,
            "mean_latency_seconds": mean_lat,
            "step_duration_seconds": float(step_duration),
            "client_configs": str(cfg_list),
            "client_actions": str(act_indices),
        }
        records.append(record)

        if verbose:
            logger.info(
                f"[Heuristic | Alpha {alpha} | Round {r:02d}/{num_rounds:02d}] "
                f"Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | Macro F1: {test_f1:.4f} | "
                f"Reward: {reward:+.4f} | Params: {tot_params:.0f} | Latency: {tot_lat:.2f}s"
            )

    total_duration = time.perf_counter() - total_time_start

    # 4. Save CSV and JSON
    df = pd.DataFrame(records)
    csv_path = out_dir / f"heuristic_results_{alpha}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved round-by-round metrics to {csv_path}")

    summary = {
        "controller": "HeuristicController",
        "alpha": alpha,
        "num_rounds": num_rounds,
        "num_clients": env.num_clients,
        "seed": seed,
        "total_execution_time_seconds": total_duration,
        "initial_accuracy": initial_acc,
        "final_accuracy": records[-1]["test_accuracy"],
        "initial_macro_f1": initial_f1,
        "final_macro_f1": records[-1]["test_macro_f1"],
        "final_loss": records[-1]["test_loss"],
        "mean_parameters_per_round": float(df["total_parameters"].iloc[1:].mean()),
        "mean_latency_per_round": float(df["total_latency_seconds"].iloc[1:].mean()),
        "total_reward": float(df["step_reward"].sum()),
    }

    json_path = out_dir / f"heuristic_summary_{alpha}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Rule-Based Heuristic Adaptive KAN Baseline")
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0],
        help="Dirichlet alpha skew parameters to evaluate",
    )
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of federated communication rounds")
    parser.add_argument("--config", type=str, default="configs/ppo.yaml", help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, default="results/adaptive_kan", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--dry-run", action="store_true", help="Quick 2-round dry-run validation")
    args = parser.parse_args()

    alphas = args.alpha if isinstance(args.alpha, list) else [args.alpha]
    num_rounds = 2 if args.dry_run else args.num_rounds

    if args.dry_run:
        logger.info("Executing DRY-RUN validation (2 rounds per alpha)...")

    summaries = []
    for a in alphas:
        summary = run_heuristic_simulation(
            alpha=a,
            num_rounds=num_rounds,
            config_path=args.config,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        summaries.append(summary)

    # Master summary table
    summary_df = pd.DataFrame(summaries)
    master_csv = Path(args.output_dir) / "heuristic_master_summary.csv"
    summary_df.to_csv(master_csv, index=False)
    logger.info(f"\nAll heuristic experiments finished! Master summary saved to {master_csv}\n")
    print(summary_df[["alpha", "final_accuracy", "final_macro_f1", "mean_parameters_per_round", "total_reward"]])


if __name__ == "__main__":
    main()
