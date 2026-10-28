"""
End-to-End PPO Training and Evaluation Script for Resource-Adaptive Federated KAN.

Features:
- Trains the PPO meta-controller on the Gymnasium FederatedKANEnv.
- Saves policy checkpoints to results/adaptive_kan/checkpoints/ppo_model.zip.
- Evaluates the trained PPO policy across Dirichlet non-IID alphas in {0.1, 0.5, 1.0}.
- Tracks controller overhead (decision latency in ms, peak RAM/VRAM footprint) per Section 3.9.4.
- Saves evaluated round metrics and chosen (G_i, k_i) traces to results/adaptive_kan/ppo_results_{alpha}.csv.
- Supports --mode {train, eval, train_eval}, --alpha {0.1, 0.5, 1.0}, and --dry-run validation flags.
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
import psutil
import torch
import yaml

from src.controllers.ppo_controller import PPOController, PPOLoggingCallback
from src.environment.f1_env import FederatedKANEnv, action_to_gk
from src.utils.reproducibility import set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_ppo")


def get_process_memory_mb() -> float:
    """Return process resident set size (RSS) in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)


def get_gpu_memory_mb() -> float:
    """Return peak GPU allocated memory in MB if CUDA is available, else 0.0."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    return 0.0


def train_ppo(
    env: FederatedKANEnv,
    timesteps: int = 100,
    config_path: str = "configs/ppo.yaml",
    checkpoint_path: str = "results/adaptive_kan/checkpoints/ppo_model.zip",
    log_dir: str = "logs/ppo",
    seed: int = 42,
) -> PPOController:
    """
    Train PPO meta-controller on the federated Gymnasium environment.
    """
    logger.info(f"\n{'='*75}\nINITIALIZING PPO TRAINING ({timesteps} TIMESTEPS)\n{'='*75}")
    set_seed(seed)

    # In dry-run or short training, ensure n_steps fits timesteps
    n_steps = min(32, timesteps)
    batch_size = min(16, n_steps)

    controller = PPOController(
        env=env,
        config_path=config_path,
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=1,
    )

    callback = PPOLoggingCallback(log_dir=log_dir)
    train_start = time.perf_counter()
    controller.train(total_timesteps=timesteps, callback=callback)
    train_duration = time.perf_counter() - train_start

    logger.info(f"PPO training finished in {train_duration:.2f} s. Saving checkpoint to {checkpoint_path}...")
    controller.save(checkpoint_path)
    return controller


def evaluate_ppo(
    controller: PPOController,
    alpha: float,
    num_rounds: int = 10,
    config_path: str = "configs/ppo.yaml",
    output_dir: str = "results/adaptive_kan",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate trained PPO policy against held-out patient test split for a specific alpha.
    Tracks controller latency and memory footprint to evaluate decision overhead.
    """
    out_dir = PROJECT_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*75}\nEVALUATING PPO POLICY FOR ALPHA = {alpha} ({num_rounds} ROUNDS)\n{'='*75}")
    set_seed(seed)

    eval_env = FederatedKANEnv(
        config_path=config_path,
        alpha=alpha,
        max_rounds=num_rounds,
        seed=seed,
    )

    # Initial reset
    obs, info = eval_env.reset(seed=seed)
    initial_acc = float(info.get("global_accuracy", 0.0))
    initial_f1 = float(info.get("global_macro_f1", 0.0))
    initial_loss = float(info.get("global_loss", 0.0))

    if verbose:
        logger.info(
            f"[Round 00/{num_rounds:02d}] Initial Global Model - "
            f"Loss: {initial_loss:.4f} | Acc: {initial_acc:.4f} | Macro F1: {initial_f1:.4f}"
        )

    records: List[Dict[str, Any]] = []
    controller_latencies_ms: List[float] = []

    # Round 0 record
    records.append({
        "round": 0,
        "alpha": alpha,
        "test_loss": initial_loss,
        "test_accuracy": initial_acc,
        "test_macro_f1": initial_f1,
        "delta_accuracy": 0.0,
        "step_reward": 0.0,
        "total_parameters": eval_env.num_clients * 5349,
        "mean_parameters_per_client": 5349.0,
        "total_latency_seconds": 0.0,
        "mean_latency_seconds": 0.0,
        "controller_latency_ms": 0.0,
        "client_configs": "[(5, 3)] * 8",
        "client_actions": "[4] * 8",
        "ram_usage_mb": get_process_memory_mb(),
        "gpu_usage_mb": get_gpu_memory_mb(),
    })

    eval_start_time = time.perf_counter()

    for r in range(1, num_rounds + 1):
        # Measure controller decision latency (Section 3.9.4)
        t_decision_start = time.perf_counter()
        actions = controller.predict(obs, deterministic=True)
        decision_latency_ms = (time.perf_counter() - t_decision_start) * 1000.0
        controller_latencies_ms.append(decision_latency_ms)

        # Step federated simulation
        obs, reward, terminated, truncated, step_info = eval_env.step(actions)

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
            "controller_latency_ms": float(decision_latency_ms),
            "client_configs": str(cfg_list),
            "client_actions": str(act_indices),
            "ram_usage_mb": float(get_process_memory_mb()),
            "gpu_usage_mb": float(get_gpu_memory_mb()),
        }
        records.append(record)

        if verbose:
            logger.info(
                f"[PPO | Alpha {alpha} | Round {r:02d}/{num_rounds:02d}] "
                f"Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | Macro F1: {test_f1:.4f} | "
                f"Reward: {reward:+.4f} | Params: {tot_params:.0f} | Latency: {tot_lat:.2f}s | "
                f"PPO Decision: {decision_latency_ms:.2f}ms"
            )

    eval_duration = time.perf_counter() - eval_start_time

    # Save CSV and JSON
    df = pd.DataFrame(records)
    csv_path = out_dir / f"ppo_results_{alpha}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved PPO evaluation round metrics to {csv_path}")

    summary = {
        "controller": "PPOController",
        "alpha": alpha,
        "num_rounds": num_rounds,
        "num_clients": eval_env.num_clients,
        "seed": seed,
        "total_eval_duration_seconds": eval_duration,
        "initial_accuracy": initial_acc,
        "final_accuracy": records[-1]["test_accuracy"],
        "initial_macro_f1": initial_f1,
        "final_macro_f1": records[-1]["test_macro_f1"],
        "final_loss": records[-1]["test_loss"],
        "mean_parameters_per_round": float(df["total_parameters"].iloc[1:].mean()),
        "mean_training_latency_per_round": float(df["total_latency_seconds"].iloc[1:].mean()),
        "mean_controller_overhead_ms": float(np.mean(controller_latencies_ms)),
        "peak_ram_mb": float(df["ram_usage_mb"].max()),
        "peak_gpu_mb": float(df["gpu_usage_mb"].max()),
        "total_reward": float(df["step_reward"].sum()),
    }

    json_path = out_dir / f"ppo_summary_{alpha}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved PPO summary to {json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="PPO Meta-Controller Training and Evaluation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "train_eval"],
        default="train_eval",
        help="Execution mode: train, eval, or train_eval",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0],
        help="Dirichlet alpha skew parameters to evaluate",
    )
    parser.add_argument("--timesteps", type=int, default=200, help="Total PPO training timesteps")
    parser.add_argument("--eval_rounds", type=int, default=10, help="Federated evaluation rounds per alpha")
    parser.add_argument("--config", type=str, default="configs/ppo.yaml", help="Path to YAML config")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/adaptive_kan/checkpoints/ppo_model.zip",
        help="Path to save or load PPO checkpoint",
    )
    parser.add_argument("--output_dir", type=str, default="results/adaptive_kan", help="Output directory")
    parser.add_argument("--log_dir", type=str, default="logs/ppo", help="PPO training log directory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--dry-run", action="store_true", help="Quick dry-run validation (16 timesteps, 2 rounds)")
    args = parser.parse_args()

    alphas = args.alpha if isinstance(args.alpha, list) else [args.alpha]
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    output_dir = PROJECT_ROOT / args.output_dir

    if args.dry_run:
        logger.info("Executing DRY-RUN validation mode...")
        timesteps = 16
        eval_rounds = 2
    else:
        timesteps = args.timesteps
        eval_rounds = args.eval_rounds

    controller = None

    # 1. Training Phase
    if args.mode in ["train", "train_eval"]:
        train_env = FederatedKANEnv(
            config_path=args.config,
            alpha=0.5,
            max_rounds=eval_rounds,
            seed=args.seed,
        )
        controller = train_ppo(
            env=train_env,
            timesteps=timesteps,
            config_path=args.config,
            checkpoint_path=str(checkpoint_path),
            log_dir=args.log_dir,
            seed=args.seed,
        )

    # 2. Evaluation Phase
    if args.mode in ["eval", "train_eval"]:
        if controller is None:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"PPO checkpoint not found at {checkpoint_path}. Train model first.")
            logger.info(f"Loading trained PPO model from {checkpoint_path}...")
            controller = PPOController(config_path=args.config, verbose=0)
            controller.load(str(checkpoint_path))

        summaries = []
        for a in alphas:
            summary = evaluate_ppo(
                controller=controller,
                alpha=a,
                num_rounds=eval_rounds,
                config_path=args.config,
                output_dir=str(output_dir),
                seed=args.seed,
            )
            summaries.append(summary)

        # Master summary table
        master_df = pd.DataFrame(summaries)
        master_csv = output_dir / "ppo_master_summary.csv"
        master_df.to_csv(master_csv, index=False)
        logger.info(f"\nAll PPO evaluations finished! Master summary saved to {master_csv}\n")
        print(
            master_df[[
                "alpha",
                "final_accuracy",
                "final_macro_f1",
                "mean_parameters_per_round",
                "mean_controller_overhead_ms",
                "total_reward",
            ]]
        )


if __name__ == "__main__":
    main()
