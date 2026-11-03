"""
Experiment Runner: Rule-Based Heuristic Adaptive KAN across Dirichlet Alphas.

Features:
- Runs federated rounds using the deterministic quantile-based / tier-aware HeuristicController.
- Supports heterogeneous edge fleet masks via --fleet configs/fleet.yaml.
- Evaluates across Dirichlet non-IID alphas in {0.1, 0.5, 1.0}.
- Evaluates global model on held-out patient-level test split (18,175 beats).
- Tracks accuracy, macro F1, parameter counts, latency, and chosen (G_i, k_i) traces.
- Final per-client assignment: mode of last 5 rounds, best validation config, disagreement flags.
- Saves results to results/adaptive_kan/heuristic_results_{alpha}.csv and JSON summary.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import logging
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
from src.environment.fleet import FleetConfig, compute_param_count
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
    fleet_config_path: Optional[str] = "configs/fleet.yaml",
    reference_mode: str = "fixed_5_3",
    output_dir: str = "results/adaptive_kan",
    seed: int = 42,
    verbose: bool = True,
    tier_report: bool = True,
) -> Dict[str, Any]:
    """
    Execute federated simulation using HeuristicController for a specific Dirichlet alpha.
    """
    out_dir = PROJECT_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*75}\nSTARTING HEURISTIC EXPERIMENT FOR ALPHA = {alpha} ({num_rounds} ROUNDS)\n{'='*75}")

    set_seed(seed)

    use_tier_masks = bool(fleet_config_path is not None)
    fleet_cfg = FleetConfig(fleet_config_path) if use_tier_masks else None

    # Reference grid and spline order based on reference_mode
    if reference_mode == "canonical_max":
        ref_g, ref_k = (10, 4)
    else:
        ref_g, ref_k = (5, 3)

    # 1. Instantiate Federated Environment and Heuristic Controller
    env = FederatedKANEnv(
        config_path=config_path,
        alpha=alpha,
        max_rounds=num_rounds,
        seed=seed,
    )
    # Configure environment reference model if canonical_max requested
    if reference_mode == "canonical_max":
        env.ref_grid_size = 10
        env.ref_spline_order = 4

    controller = HeuristicController(
        config_path=config_path,
        fleet_config_path=fleet_config_path,
        use_tier_masks=use_tier_masks,
        seed=seed,
    )

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
    start_configs = [fleet_cfg.get_start_config_for_client(cid) if fleet_cfg else (5, 3) for cid in range(env.num_clients)]
    start_params_tot = sum(compute_param_count(g, k) for g, k in start_configs)
    records.append({
        "round": 0,
        "alpha": alpha,
        "test_loss": initial_loss,
        "test_accuracy": initial_acc,
        "test_macro_f1": initial_f1,
        "delta_accuracy": 0.0,
        "step_reward": 0.0,
        "total_parameters": start_params_tot,
        "mean_parameters_per_client": start_params_tot / env.num_clients,
        "total_latency_seconds": 0.0,
        "mean_latency_seconds": 0.0,
        "step_duration_seconds": 0.0,
        "client_configs": str(start_configs),
        "client_actions": str([controller.predict(obs, round_num=0)[i] for i in range(env.num_clients)]),
    })

    total_time_start = time.perf_counter()

    # 3. Federated Rounds Loop
    for r in range(1, num_rounds + 1):
        step_start = time.perf_counter()

        # Controller selects actions based on current telemetry state S^t
        actions = controller.predict(obs, round_num=r)

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

    # 4. Extract Final Assignments per client (Mode of last 5 rounds, best val F1)
    final_assignments = controller.get_final_assignments()

    # 5. Save CSV and JSON
    df = pd.DataFrame(records)
    csv_path = out_dir / f"heuristic_results_{alpha}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved round-by-round metrics to {csv_path}")

    final_assign_path = out_dir / f"heuristic_final_assignments_alpha_{alpha}.json"
    # Format assignments for json serialization
    formatted_assign = {}
    for cid, info in final_assignments.items():
        formatted_assign[int(cid)] = {
            "client_id": int(cid),
            "tier": int(info["tier"]),
            "device_name": fleet_cfg.tiers_raw[info["tier"]]["device_name"] if fleet_cfg else f"Tier {info['tier']}",
            "mode_last_5": list(info["mode_last_5"]),
            "mode_params": int(info["mode_params"]),
            "best_val_f1_config": list(info["best_val_f1_config"]),
            "best_params": int(info["best_params"]),
            "disagreement": bool(info["disagreement"]),
        }

    with open(final_assign_path, "w") as f:
        json.dump(formatted_assign, f, indent=2)
    logger.info(f"Saved per-client final assignments to {final_assign_path}")

    summary = {
        "controller": "HeuristicController",
        "alpha": alpha,
        "num_rounds": num_rounds,
        "num_clients": env.num_clients,
        "reference_mode": reference_mode,
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
        "final_client_assignments": formatted_assign,
    }

    json_path = out_dir / f"heuristic_summary_{alpha}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {json_path}")

    if tier_report:
        print("\n" + "=" * 80)
        print(f"HEURISTIC CONTROLLER PER-CLIENT FINAL ASSIGNMENTS (Alpha = {alpha})")
        print("=" * 80)
        for cid, info in formatted_assign.items():
            flag = " [DISAGREEMENT]" if info["disagreement"] else " [CONVERGED]"
            print(
                f"Client {cid:2d} | Tier {info['tier']} ({info['device_name']:<15}): "
                f"Mode Last 5 = ({info['mode_last_5'][0]},{info['mode_last_5'][1]}) [{info['mode_params']} params] | "
                f"Best Val F1 = ({info['best_val_f1_config'][0]},{info['best_val_f1_config'][1]}){flag}"
            )
        print("=" * 80 + "\n")

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
    parser.add_argument("--fleet", type=str, default="configs/fleet.yaml", help="Path to fleet YAML config")
    parser.add_argument("--reference_mode", type=str, default="fixed_5_3", choices=["fixed_5_3", "canonical_max"], help="Server reference mode")
    parser.add_argument("--tier_report", action="store_true", default=True, help="Print per-client tier report")
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
            fleet_config_path=args.fleet,
            reference_mode=args.reference_mode,
            output_dir=args.output_dir,
            seed=args.seed,
            tier_report=args.tier_report,
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
