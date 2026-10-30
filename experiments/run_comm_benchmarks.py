"""
Benchmark runner to generate standardized per-round communication traces for every method.

Logs per-round metrics:
round, seed, alpha, method, test_acc, test_macro_f1, test_balanced_acc, val_acc, val_macro_f1,
params_up, params_down, params_round, bytes_round, cum_params, cum_bytes, cum_MB,
est_comm_time_s, est_comm_time_sum_s, config_histogram
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from src.controllers.heuristic import HeuristicController
from src.controllers.ppo_controller import PPOController
from src.environment.f1_env import (
    ACTION_TUPLES,
    FederatedKANEnv,
    gk_to_action,
)
from src.environment.telemetry import BandwidthSimulator
from src.federated.server import run_federated_simulation
from src.models.fastkan_model import FastKANClassifier
from src.models.mlp import MLPClassifier, build_matched_mlp
from src.utils.reproducibility import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("comm_benchmark")

ALL_METHODS = [
    "FedAvg-MLP",
    "FedAvg-KAN (fixed 5,3)",
    "FedProx-KAN",
    "Heuristic-KAN",
    "PPO-Adaptive-KAN",
    "Uniform-Fleet-(3,2)",
    "Uniform-Fleet-(10,4)",
    "Random-config KAN",
]


def run_single_comm_simulation(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run one federated simulation for 1 method, 1 alpha, 1 seed, over num_rounds.
    Returns a list of per-round trace dictionaries.
    """
    method_name = task["method_name"]
    alpha = float(task["alpha"])
    seed = int(task["seed"])
    num_rounds = int(task["num_rounds"])
    config_path = task.get("config_path", "configs/ppo.yaml")
    ppo_path = task.get("ppo_path", "results/adaptive_kan/checkpoints/ppo_learned.zip")

    set_seed(seed)
    torch.set_num_threads(2)
    rng = np.random.RandomState(seed)

    # 1. FedAvg-MLP baseline
    if method_name == "FedAvg-MLP":
        # Create env temporarily to extract identical dataset partitions
        temp_env = FederatedKANEnv(config_path=config_path, alpha=alpha, max_rounds=num_rounds, seed=seed)
        mlp_model = build_matched_mlp(target_params=5349, input_dim=32, num_classes=5, num_hidden_layers=2, dropout=0.1)

        sim_results = run_federated_simulation(
            model=mlp_model,
            client_datasets=temp_env.client_datasets,
            test_data=temp_env.test_data,
            val_data=temp_env.val_data,
            num_rounds=num_rounds,
            strategy_name="fedavg",
            local_epochs=1,
            batch_size=32,
            learning_rate=0.001,
            verbose=False,
        )

        trace_records = []
        for r_entry in sim_results["history"]:
            r = int(r_entry["round"])
            if r == 0:
                continue  # Standardized trace logs rounds 1..num_rounds
            trace_records.append({
                "round": r,
                "seed": seed,
                "alpha": alpha,
                "method": method_name,
                "test_acc": round(float(r_entry["test_accuracy"]) * 100.0, 4),
                "test_macro_f1": round(float(r_entry["test_macro_f1"]) * 100.0, 4),
                "test_balanced_acc": round(float(r_entry.get("test_balanced_accuracy", r_entry["test_accuracy"])) * 100.0, 4),
                "val_acc": round(float(r_entry.get("val_accuracy", r_entry["test_accuracy"])) * 100.0, 4),
                "val_macro_f1": round(float(r_entry.get("val_macro_f1", r_entry["test_macro_f1"])) * 100.0, 4),
                "params_up": int(r_entry["params_up"]),
                "params_down": int(r_entry["params_down"]),
                "params_round": int(r_entry["params_round"]),
                "bytes_round": int(r_entry["bytes_round"]),
                "cum_params": int(r_entry["cum_params"]),
                "cum_bytes": int(r_entry["cum_bytes"]),
                "cum_MB": round(float(r_entry["cum_MB"]), 4),
                "est_comm_time_s": round(float(r_entry["est_comm_time_s"]), 4),
                "est_comm_time_sum_s": round(float(r_entry.get("est_comm_time_sum_s", r_entry["est_comm_time_s"])), 4),
                "config_histogram": json.dumps(r_entry.get("config_histogram", {})),
            })
        return trace_records

    # 2. All KAN-based methods (Fixed, Prox, Adaptive, Random)
    agg_mode = "project_delta" if (method_name != "FedAvg-KAN (fixed 5,3)" and "full" not in method_name) else "project_full"
    prox_mu = 0.01 if method_name == "FedProx-KAN" else 0.0

    env = FederatedKANEnv(
        config_path=config_path,
        alpha=alpha,
        max_rounds=num_rounds,
        aggregation_mode=agg_mode,
        proximal_mu=prox_mu,
        seed=seed,
    )
    obs, info = env.reset(seed=seed)

    # Setup controller
    if method_name == "Heuristic-KAN":
        ctrl = HeuristicController(config_path=config_path)
    elif method_name == "PPO-Adaptive-KAN":
        ctrl = PPOController(config_path=config_path, verbose=0)
        actual_ppo_path = PROJECT_ROOT / ppo_path
        ctrl.load(str(actual_ppo_path))
    else:
        ctrl = None

    trace_records = []
    for r in range(1, num_rounds + 1):
        if method_name in ["FedAvg-KAN (fixed 5,3)", "FedProx-KAN"]:
            actions = [gk_to_action(5, 3)] * env.num_clients
        elif method_name == "Uniform-Fleet-(3,2)":
            actions = [gk_to_action(3, 2)] * env.num_clients
        elif method_name == "Uniform-Fleet-(10,4)":
            actions = [gk_to_action(10, 4)] * env.num_clients
        elif method_name == "Random-config KAN":
            actions = [int(rng.randint(0, len(ACTION_TUPLES))) for _ in range(env.num_clients)]
        elif method_name == "Heuristic-KAN":
            actions = ctrl.predict(obs)
        elif method_name == "PPO-Adaptive-KAN":
            actions = ctrl.predict(obs, deterministic=True)
        else:
            actions = [gk_to_action(5, 3)] * env.num_clients

        obs, reward, terminated, truncated, step_info = env.step(actions)

        trace_records.append({
            "round": r,
            "seed": seed,
            "alpha": alpha,
            "method": method_name,
            "test_acc": round(float(step_info["global_accuracy"]) * 100.0, 4),
            "test_macro_f1": round(float(step_info["global_macro_f1"]) * 100.0, 4),
            "test_balanced_acc": round(float(step_info["global_balanced_accuracy"]) * 100.0, 4),
            "val_acc": round(float(step_info["val_accuracy"]) * 100.0, 4),
            "val_macro_f1": round(float(step_info["val_macro_f1"]) * 100.0, 4),
            "params_up": int(step_info["params_up"]),
            "params_down": int(step_info["params_down"]),
            "params_round": int(step_info["params_round"]),
            "bytes_round": int(step_info["bytes_round"]),
            "cum_params": int(step_info["cum_params"]),
            "cum_bytes": int(step_info["cum_bytes"]),
            "cum_MB": round(float(step_info["cum_MB"]), 4),
            "est_comm_time_s": round(float(step_info["est_comm_time_s"]), 4),
            "est_comm_time_sum_s": round(float(step_info.get("est_comm_time_sum_s", step_info["est_comm_time_s"])), 4),
            "config_histogram": json.dumps(step_info.get("config_histogram", {})),
        })

    return trace_records


def main():
    parser = argparse.ArgumentParser(description="Standardized Communication Trace Generator")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--methods", type=str, nargs="+", default=ALL_METHODS)
    parser.add_argument("--config", type=str, default="configs/ppo.yaml")
    parser.add_argument("--ppo_checkpoint", type=str, default="results/adaptive_kan/checkpoints/ppo_learned.zip")
    parser.add_argument("--output_file", type=str, default="results/ablations/comm_round_traces.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    effective_seeds = [42] if args.dry_run else args.seeds
    effective_alphas = [0.5] if args.dry_run else args.alphas
    effective_rounds = 2 if args.dry_run else args.num_rounds
    methods_to_run = args.methods

    logger.info(
        f"\n{'='*78}\n"
        f"STARTING STANDARDIZED COMMUNICATION TRACE GENERATOR\n"
        f"Methods ({len(methods_to_run)}): {methods_to_run}\n"
        f"Alphas: {effective_alphas} | Seeds: {effective_seeds} | Rounds: {effective_rounds}\n"
        f"{'='*78}"
    )

    tasks = []
    for a in effective_alphas:
        for m in methods_to_run:
            for s in effective_seeds:
                tasks.append({
                    "method_name": m,
                    "alpha": a,
                    "seed": s,
                    "num_rounds": effective_rounds,
                    "config_path": args.config,
                    "ppo_path": args.ppo_checkpoint,
                })

    start_time = time.perf_counter()
    num_workers = min(args.num_workers, len(tasks))
    logger.info(f"Dispatching {len(tasks)} simulations across {num_workers} parallel workers...")

    all_round_rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for records in executor.map(run_single_comm_simulation, tasks):
            all_round_rows.extend(records)
            if records:
                r_last = records[-1]
                logger.info(
                    f"Finished: {r_last['method']} | Alpha: {r_last['alpha']} | Seed: {r_last['seed']} | "
                    f"R{r_last['round']} Test Acc: {r_last['test_acc']:.2f}% | "
                    f"Params/Rnd: {r_last['params_up']} | Cum MB: {r_last['cum_MB']:.2f}"
                )

    duration = time.perf_counter() - start_time
    logger.info(f"All {len(tasks)} simulations completed in {duration:.2f} s ({duration/60:.2f} min).")

    df_traces = pd.DataFrame(all_round_rows)
    out_path = PROJECT_ROOT / args.output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_traces.to_csv(out_path, index=False)
    logger.info(f"Saved master standardized communication traces -> {out_path} ({len(df_traces)} rows)")

    # Also update individual per-method CSVs for backward compatibility
    for a in effective_alphas:
        for m in methods_to_run:
            sub = df_traces[(df_traces["alpha"] == a) & (df_traces["method"] == m)]
            if len(sub) == 0:
                continue

            # Route to respective results folder
            if "MLP" in m:
                p_csv = PROJECT_ROOT / f"results/baselines/metrics_FedAvg-MLP_alpha_{a}.csv"
            elif "FedAvg-KAN" in m:
                p_csv = PROJECT_ROOT / f"results/baselines/metrics_FedAvg-KAN_alpha_{a}.csv"
            elif "FedProx" in m:
                p_csv = PROJECT_ROOT / f"results/baselines/metrics_FedProx-KAN_alpha_{a}.csv"
            elif "Heuristic" in m:
                p_csv = PROJECT_ROOT / f"results/adaptive_kan/heuristic_results_{a}.csv"
            elif "PPO" in m:
                p_csv = PROJECT_ROOT / f"results/adaptive_kan/ppo_results_{a}.csv"
            elif "3,2" in m:
                p_csv = PROJECT_ROOT / f"results/adaptive_kan/uniform_32_results_{a}.csv"
            elif "10,4" in m:
                p_csv = PROJECT_ROOT / f"results/adaptive_kan/uniform_104_results_{a}.csv"
            elif "Random" in m:
                p_csv = PROJECT_ROOT / f"results/adaptive_kan/random_results_{a}.csv"
            else:
                p_csv = PROJECT_ROOT / f"results/ablations/{m.replace(' ', '_')}_{a}.csv"

            p_csv.parent.mkdir(parents=True, exist_ok=True)
            sub.to_csv(p_csv, index=False)


if __name__ == "__main__":
    main()
