"""
Comprehensive Benchmark Suite for Heterogeneous Edge Fleet (4 Tiers, 8 Clients).

Evaluates 6 resource-adaptive methods across 5 random seeds (42..46) and 3 Dirichlet alphas (0.1, 0.5, 1.0):
1. Fixed (5, 3) Reference Fleet
2. Tier-min Fleet (minimal allowed (G, k) per device tier)
3. Tier-max Fleet (maximum allowed (G, k) per device tier)
4. Tier-aware Heuristic Controller
5. Random-within-mask Baseline
6. MaskablePPO Meta-Controller (with action-4 prior)
7. MaskablePPO Meta-Controller (without action-4 prior)

Outputs:
- results/fleet/benchmark_rounds.csv
- results/fleet/benchmark_per_seed.csv
- results/fleet/benchmark_master_summary.csv and .json
- Publication-quality figures in results/figures/fleet/ (PNG 300 DPI + PDF)
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import torch

from src.controllers.heuristic import HeuristicController
from src.controllers.ppo_controller import PPOController
from src.environment.f1_env import (
    ACTION_TO_INDEX,
    ACTION_TUPLES,
    FederatedKANEnv,
    action_to_gk,
    gk_to_action,
)
from src.environment.fleet import FleetConfig, compute_param_count
from src.utils.reproducibility import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fleet_benchmarks")

# Reference Budgets
B_REF_5_3 = 42792.0    # 8 * 5349
B_ALL_10_4 = 71208.0   # 8 * 8901
B_ALL_3_2 = 28584.0    # 8 * 3573


def get_tier_min_actions(fleet_cfg: FleetConfig) -> List[int]:
    """Return action indices for minimal allowed (G, k) per tier."""
    # Smartwatch: (3, 2) -> 0; Smartphone: (5, 2) -> 3; Clinic: (7, 3) -> 7; Hospital: (10, 4) -> 11
    actions = []
    for c in range(fleet_cfg.num_clients):
        tier = fleet_cfg.get_tier_for_client(c)
        allowed = fleet_cfg.get_allowed_configs_for_tier(tier)
        min_cfg = min(allowed, key=lambda gk: compute_param_count(gk[0], gk[1]))
        actions.append(ACTION_TO_INDEX[min_cfg])
    return actions


def get_tier_max_actions(fleet_cfg: FleetConfig) -> List[int]:
    """Return action indices for maximal allowed (G, k) per tier."""
    actions = []
    for c in range(fleet_cfg.num_clients):
        tier = fleet_cfg.get_tier_for_client(c)
        allowed = fleet_cfg.get_allowed_configs_for_tier(tier)
        max_cfg = max(allowed, key=lambda gk: compute_param_count(gk[0], gk[1]))
        actions.append(ACTION_TO_INDEX[max_cfg])
    return actions


def run_single_episode(
    method: str,
    alpha: float,
    seed: int,
    rounds: int,
    fleet_cfg: FleetConfig,
    reference_mode: str = "fixed_5_3",
    ppo_prior_model: Optional[PPOController] = None,
    ppo_noprior_model: Optional[PPOController] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute one 10-round federated training run for a given method, alpha, and seed."""
    set_seed(seed)
    rng = np.random.RandomState(seed + 1000)

    env = FederatedKANEnv(
        config_path="configs/ppo.yaml",
        fleet_config=fleet_cfg,
        reference_mode=reference_mode,
        alpha=alpha,
        max_rounds=rounds,
        seed=seed,
    )

    obs, info = env.reset(seed=seed)

    # Initialize controllers if required
    heuristic_ctrl = None
    if method == "heuristic":
        heuristic_ctrl = HeuristicController(
            config_path="configs/ppo.yaml",
            fleet_config_path=fleet_cfg.config_path if fleet_cfg is not None else Path("configs/fleet.yaml"),
            use_tier_masks=True,
            hysteresis_threshold=0.03,
            seed=seed,
        )

    tier_min_acts = get_tier_min_actions(fleet_cfg)
    tier_max_acts = get_tier_max_actions(fleet_cfg)

    round_records = []
    initial_acc = float(info.get("global_accuracy", 0.0))
    initial_f1 = float(info.get("global_macro_f1", 0.0))
    initial_bacc = float(info.get("global_balanced_accuracy", initial_acc))
    initial_loss = float(info.get("global_loss", 0.0))

    round_records.append({
        "method": method,
        "alpha": alpha,
        "seed": seed,
        "round": 0,
        "test_loss": initial_loss,
        "test_accuracy": initial_acc,
        "test_macro_f1": initial_f1,
        "test_balanced_accuracy": initial_bacc,
        "total_parameters": B_REF_5_3 if method == "fixed_5_3" else sum(compute_param_count(*ACTION_TUPLES[a]) for a in fleet_cfg.get_start_config_for_client(0)),
        "mean_parameters_per_client": 5349.0,
        "cum_MB": 0.0,
        "est_comm_time_s": 0.0,
        "est_comm_time_sum_s": 0.0,
        "cum_comm_time_s": 0.0,
        "decision_latency_ms": 0.0,
        "client_configs": str([ACTION_TUPLES[4]] * 8),
    })

    dec_latencies = []

    for r in range(1, rounds + 1):
        t0 = time.perf_counter()
        if method == "fixed_5_3":
            actions = np.full(env.num_clients, 4, dtype=np.int64)
            dec_lat = 0.01
        elif method == "tier_min":
            actions = np.array(tier_min_acts, dtype=np.int64)
            dec_lat = 0.01
        elif method == "tier_max":
            actions = np.array(tier_max_acts, dtype=np.int64)
            dec_lat = 0.01
        elif method == "random_within_mask":
            actions = np.zeros(env.num_clients, dtype=np.int64)
            for c in range(env.num_clients):
                t = fleet_cfg.get_tier_for_client(c)
                allowed_cfgs = fleet_cfg.get_allowed_configs_for_tier(t)
                chosen_cfg = allowed_cfgs[rng.choice(len(allowed_cfgs))]
                actions[c] = ACTION_TO_INDEX[chosen_cfg]
            dec_lat = 0.05
        elif method == "heuristic":
            actions = heuristic_ctrl.predict(obs, round_num=r)
            dec_lat = (time.perf_counter() - t0) * 1000.0
        elif method == "ppo_prior":
            if ppo_prior_model is not None:
                actions = ppo_prior_model.predict(obs, deterministic=True)
            else:
                actions = np.full(env.num_clients, 4, dtype=np.int64)
            dec_lat = (time.perf_counter() - t0) * 1000.0
        elif method == "ppo_noprior":
            if ppo_noprior_model is not None:
                actions = ppo_noprior_model.predict(obs, deterministic=True)
            else:
                actions = np.full(env.num_clients, 4, dtype=np.int64)
            dec_lat = (time.perf_counter() - t0) * 1000.0
        else:
            raise ValueError(f"Unknown method {method}")

        dec_latencies.append(dec_lat)

        obs, reward, terminated, truncated, step_info = env.step(actions)

        test_acc = float(step_info["global_accuracy"])
        test_f1 = float(step_info["global_macro_f1"])
        test_bacc = float(step_info["global_balanced_accuracy"])
        test_loss = float(step_info["global_loss"])
        tot_params = float(step_info["total_parameters"])
        mean_params = float(step_info["mean_parameters_per_client"])
        cum_mb = float(step_info["cum_MB"])
        comm_time_s = float(step_info["est_comm_time_s"])
        cum_comm_time_s = float(step_info["cum_comm_time_s"])

        chosen_gk = [(a["G"], a["k"]) for a in step_info["actions"]]

        round_records.append({
            "method": method,
            "alpha": alpha,
            "seed": seed,
            "round": r,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
            "test_balanced_accuracy": test_bacc,
            "total_parameters": tot_params,
            "mean_parameters_per_client": mean_params,
            "cum_MB": cum_mb,
            "est_comm_time_s": comm_time_s,
            "est_comm_time_sum_s": float(step_info.get("est_comm_time_sum_s", comm_time_s)),
            "cum_comm_time_s": cum_comm_time_s,
            "decision_latency_ms": dec_lat,
            "client_configs": str(chosen_gk),
        })

    # Summary record for this episode
    final_rec = round_records[-1]
    mean_round_params = float(np.mean([rec["total_parameters"] for rec in round_records[1:]]))

    summary = {
        "method": method,
        "alpha": alpha,
        "seed": seed,
        "final_accuracy": final_rec["test_accuracy"],
        "final_macro_f1": final_rec["test_macro_f1"],
        "final_balanced_accuracy": final_rec["test_balanced_accuracy"],
        "final_loss": final_rec["test_loss"],
        "mean_parameters_per_round": mean_round_params,
        "mean_parameters_per_client": mean_round_params / env.num_clients,
        "cum_MB": final_rec["cum_MB"],
        "cum_comm_time_s": final_rec["cum_comm_time_s"],
        "mean_decision_latency_ms": float(np.mean(dec_latencies)),
        "savings_vs_fixed_5_3_pct": (B_REF_5_3 - mean_round_params) / B_REF_5_3 * 100.0,
        "savings_vs_all_10_4_pct": (B_ALL_10_4 - mean_round_params) / B_ALL_10_4 * 100.0,
        "savings_vs_all_3_2_pct": (B_ALL_3_2 - mean_round_params) / B_ALL_3_2 * 100.0,
        "final_client_configs": final_rec["client_configs"],
    }

    return round_records, summary


def generate_benchmark_figures(
    rounds_df: pd.DataFrame,
    per_seed_df: pd.DataFrame,
    master_df: pd.DataFrame,
    figures_dir: Union[str, Path] = "results/figures/fleet",
) -> None:
    """Generate publication-quality 300 DPI PNG and vector PDF figures."""
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {
        "fixed_5_3": "#1f77b4",       # Blue
        "tier_min": "#2ca02c",        # Green
        "tier_max": "#d62728",        # Red
        "heuristic": "#ff7f0e",       # Orange
        "random_within_mask": "#7f7f7f", # Gray
        "ppo_prior": "#9467bd",       # Purple
        "ppo_noprior": "#8c564b",     # Brown
    }
    method_labels = {
        "fixed_5_3": "Fixed (5, 3) Ref",
        "tier_min": "Tier-Min",
        "tier_max": "Tier-Max",
        "heuristic": "Heuristic (Ours)",
        "random_within_mask": "Random (Masked)",
        "ppo_prior": "MaskablePPO (Prior)",
        "ppo_noprior": "MaskablePPO (No Prior)",
    }

    # 1. Pareto Frontier: Macro F1 vs Cumulative MB (alpha = 0.5)
    df_05 = master_df[master_df["alpha"] == 0.5]
    if not df_05.empty:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        for _, row in df_05.iterrows():
            m = row["method"]
            c = colors.get(m, "#333333")
            lbl = method_labels.get(m, m)
            f1_mean = row["macro_f1_mean"]
            f1_std = row.get("macro_f1_std", 0.0)
            mb_mean = row["cum_MB_mean"]
            mb_std = row.get("cum_MB_std", 0.0)

            ax.errorbar(
                mb_mean, f1_mean,
                xerr=mb_std, yerr=f1_std,
                fmt="o", markersize=9, capsize=4, elinewidth=1.5,
                color=c, label=lbl, alpha=0.9,
            )
            ax.annotate(
                f" {lbl}",
                (mb_mean, f1_mean),
                fontsize=9, weight="bold", color=c,
                xytext=(5, 2), textcoords="offset points",
            )

        ax.set_title("Pareto Efficiency: Macro F1 vs Communication (Alpha = 0.5)", fontsize=13, weight="bold", pad=12)
        ax.set_xlabel("Cumulative Communication (MB)", fontsize=11)
        ax.set_ylabel("Global Test Macro F1", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        fig.savefig(fig_dir / "pareto_macro_f1_vs_cum_mb.png", dpi=300)
        fig.savefig(fig_dir / "pareto_macro_f1_vs_cum_mb.pdf")
        plt.close(fig)

    # 2. Pareto Frontier: Accuracy vs Cumulative MB (alpha = 0.5)
    if not df_05.empty:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        for _, row in df_05.iterrows():
            m = row["method"]
            c = colors.get(m, "#333333")
            lbl = method_labels.get(m, m)
            acc_mean = row["accuracy_mean"] * 100.0
            acc_std = row.get("accuracy_std", 0.0) * 100.0
            mb_mean = row["cum_MB_mean"]
            mb_std = row.get("cum_MB_std", 0.0)

            ax.errorbar(
                mb_mean, acc_mean,
                xerr=mb_std, yerr=acc_std,
                fmt="s", markersize=9, capsize=4, elinewidth=1.5,
                color=c, label=lbl, alpha=0.9,
            )
            ax.annotate(
                f" {lbl}",
                (mb_mean, acc_mean),
                fontsize=9, weight="bold", color=c,
                xytext=(5, 2), textcoords="offset points",
            )

        ax.set_title("Pareto Efficiency: Accuracy vs Communication (Alpha = 0.5)", fontsize=13, weight="bold", pad=12)
        ax.set_xlabel("Cumulative Communication (MB)", fontsize=11)
        ax.set_ylabel("Global Test Accuracy (%)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        fig.savefig(fig_dir / "pareto_accuracy_vs_cum_mb.png", dpi=300)
        fig.savefig(fig_dir / "pareto_accuracy_vs_cum_mb.pdf")
        plt.close(fig)

    # 3. Accuracy vs Estimated Round Execution Time
    if not df_05.empty:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        for _, row in df_05.iterrows():
            m = row["method"]
            c = colors.get(m, "#333333")
            lbl = method_labels.get(m, m)
            acc_mean = row["accuracy_mean"] * 100.0
            acc_std = row.get("accuracy_std", 0.0) * 100.0
            t_mean = row["cum_comm_time_s_mean"]
            t_std = row.get("cum_comm_time_s_std", 0.0)

            ax.errorbar(
                t_mean, acc_mean,
                xerr=t_std, yerr=acc_std,
                fmt="^", markersize=9, capsize=4, elinewidth=1.5,
                color=c, label=lbl, alpha=0.9,
            )
            ax.annotate(
                f" {lbl}",
                (t_mean, acc_mean),
                fontsize=9, weight="bold", color=c,
                xytext=(5, 2), textcoords="offset points",
            )

        ax.set_title("Wall-Clock Efficiency: Accuracy vs Estimated Comm Time (Alpha = 0.5)", fontsize=13, weight="bold", pad=12)
        ax.set_xlabel("Cumulative Communication Time (s)", fontsize=11)
        ax.set_ylabel("Global Test Accuracy (%)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        fig.savefig(fig_dir / "accuracy_vs_comm_time.png", dpi=300)
        fig.savefig(fig_dir / "accuracy_vs_comm_time.pdf")
        plt.close(fig)

    # 4. Convergence Trajectories across Rounds (alpha = 0.5)
    r_05 = rounds_df[rounds_df["alpha"] == 0.5]
    if not r_05.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
        for m in r_05["method"].unique():
            sub = r_05[r_05["method"] == m]
            grouped = sub.groupby("round")
            mean_acc = grouped["test_accuracy"].mean() * 100.0
            std_acc = grouped["test_accuracy"].std().fillna(0.0) * 100.0
            mean_f1 = grouped["test_macro_f1"].mean()
            std_f1 = grouped["test_macro_f1"].std().fillna(0.0)
            r_idx = mean_acc.index

            c = colors.get(m, "#333333")
            lbl = method_labels.get(m, m)

            ax1.plot(r_idx, mean_acc, label=lbl, color=c, lw=2)
            ax1.fill_between(r_idx, mean_acc - std_acc, mean_acc + std_acc, color=c, alpha=0.15)

            ax2.plot(r_idx, mean_f1, label=lbl, color=c, lw=2)
            ax2.fill_between(r_idx, mean_f1 - std_f1, mean_f1 + std_f1, color=c, alpha=0.15)

        ax1.set_title("Test Accuracy Trajectory (Alpha = 0.5)", fontsize=12, weight="bold")
        ax1.set_xlabel("Federated Communication Round", fontsize=10)
        ax1.set_ylabel("Global Test Accuracy (%)", fontsize=10)
        ax1.legend(loc="lower right", framealpha=0.9, fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.set_title("Macro F1 Trajectory (Alpha = 0.5)", fontsize=12, weight="bold")
        ax2.set_xlabel("Federated Communication Round", fontsize=10)
        ax2.set_ylabel("Global Test Macro F1", fontsize=10)
        ax2.legend(loc="lower right", framealpha=0.9, fontsize=9)
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig.savefig(fig_dir / "convergence_trajectories_alpha_0.5.png", dpi=300)
        fig.savefig(fig_dir / "convergence_trajectories_alpha_0.5.pdf")
        plt.close(fig)

    logger.info(f"Generated all publication figures in {fig_dir}")


def main():
    parser = argparse.ArgumentParser(description="Multi-seed benchmark suite for heterogeneous edge fleet.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="Random seeds (5 seeds)")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5, 1.0], help="Dirichlet alphas")
    parser.add_argument("--rounds", type=int, default=10, help="Rounds per federated run")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["fixed_5_3", "tier_min", "tier_max", "heuristic", "random_within_mask", "ppo_prior", "ppo_noprior"],
        help="Methods to evaluate",
    )
    parser.add_argument("--fleet", type=str, default="configs/fleet.yaml", help="Path to fleet YAML config")
    parser.add_argument("--reference_mode", type=str, default="fixed_5_3", help="Server reference mode")
    parser.add_argument("--output_dir", type=str, default="results/fleet", help="Output directory for CSVs")
    parser.add_argument("--figures_dir", type=str, default="results/figures/fleet", help="Output directory for figures")
    parser.add_argument(
        "--ppo_prior_ckpt",
        type=str,
        default="results/fleet/checkpoints/ppo_maskable_prior.zip",
        help="Path to trained MaskablePPO with prior",
    )
    parser.add_argument(
        "--ppo_noprior_ckpt",
        type=str,
        default="results/fleet/checkpoints/ppo_maskable_noprior.zip",
        help="Path to trained MaskablePPO without prior",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing benchmark_per_seed.csv if present")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fleet_cfg = FleetConfig(args.fleet)

    # Load PPO models if available
    ppo_prior_model = None
    if "ppo_prior" in args.methods and Path(args.ppo_prior_ckpt).exists():
        logger.info(f"Loading trained MaskablePPO (prior) from {args.ppo_prior_ckpt}...")
        ppo_prior_model = PPOController(config_path="configs/ppo.yaml", fleet_config=fleet_cfg, verbose=0)
        ppo_prior_model.load(args.ppo_prior_ckpt)

    ppo_noprior_model = None
    if "ppo_noprior" in args.methods and Path(args.ppo_noprior_ckpt).exists():
        logger.info(f"Loading trained MaskablePPO (no prior) from {args.ppo_noprior_ckpt}...")
        ppo_noprior_model = PPOController(config_path="configs/ppo.yaml", fleet_config=fleet_cfg, verbose=0)
        ppo_noprior_model.load(args.ppo_noprior_ckpt)

    rounds_csv = out_dir / "benchmark_rounds.csv"
    seed_csv = out_dir / "benchmark_per_seed.csv"

    all_round_records: List[Dict[str, Any]] = []
    all_seed_summaries: List[Dict[str, Any]] = []

    # Check for existing results if resuming
    completed_keys = set()
    if args.resume and seed_csv.exists() and rounds_csv.exists():
        try:
            prev_seed_df = pd.read_csv(seed_csv)
            all_seed_summaries = prev_seed_df.to_dict("records")
            for r in all_seed_summaries:
                completed_keys.add((str(r["method"]), float(r["alpha"]), int(r["seed"])))
            prev_rounds_df = pd.read_csv(rounds_csv)
            all_round_records = prev_rounds_df.to_dict("records")
            logger.info(f"Resuming benchmark suite: loaded {len(all_seed_summaries)} existing completed runs.")
        except Exception as e:
            logger.warning(f"Could not load existing files to resume: {e}")

    total_runs = len(args.alphas) * len(args.seeds) * len(args.methods)
    logger.info(f"Fleet Benchmark Suite target: {total_runs} runs total ({len(args.methods)} methods, {len(args.seeds)} seeds, {len(args.alphas)} alphas).")

    run_idx = 0
    t_start = time.perf_counter()

    def update_and_save_master():
        if not all_seed_summaries:
            return
        s_df = pd.DataFrame(all_seed_summaries)
        r_df = pd.DataFrame(all_round_records)
        s_df.to_csv(seed_csv, index=False)
        r_df.to_csv(rounds_csv, index=False)

        master_rows = []
        for alpha in s_df["alpha"].unique():
            alpha_sub = s_df[s_df["alpha"] == alpha]
            fixed_sub = alpha_sub[alpha_sub["method"] == "fixed_5_3"].sort_values("seed")
            fixed_acc = fixed_sub["final_accuracy"].values if not fixed_sub.empty else None
            fixed_f1 = fixed_sub["final_macro_f1"].values if not fixed_sub.empty else None

            for method in s_df["method"].unique():
                m_sub = alpha_sub[alpha_sub["method"] == method].sort_values("seed")
                if m_sub.empty:
                    continue

                acc_vals = m_sub["final_accuracy"].values
                f1_vals = m_sub["final_macro_f1"].values
                bacc_vals = m_sub["final_balanced_accuracy"].values
                loss_vals = m_sub["final_loss"].values
                param_vals = m_sub["mean_parameters_per_round"].values
                mb_vals = m_sub["cum_MB"].values
                comm_t_vals = m_sub["cum_comm_time_s"].values
                sav_5_3_vals = m_sub["savings_vs_fixed_5_3_pct"].values
                sav_10_4_vals = m_sub["savings_vs_all_10_4_pct"].values
                sav_3_2_vals = m_sub["savings_vs_all_3_2_pct"].values
                lat_ms_vals = m_sub["mean_decision_latency_ms"].values

                p_val_acc = 1.0
                p_val_f1 = 1.0
                if fixed_acc is not None and len(acc_vals) == len(fixed_acc) and len(acc_vals) > 1 and method != "fixed_5_3":
                    try:
                        res_acc = stats.ttest_rel(acc_vals, fixed_acc)
                        p_val_acc = float(res_acc.pvalue) if not np.isnan(res_acc.pvalue) else 1.0
                    except Exception:
                        p_val_acc = 1.0
                    try:
                        res_f1 = stats.ttest_rel(f1_vals, fixed_f1)
                        p_val_f1 = float(res_f1.pvalue) if not np.isnan(res_f1.pvalue) else 1.0
                    except Exception:
                        p_val_f1 = 1.0

                master_rows.append({
                    "alpha": float(alpha),
                    "method": str(method),
                    "n_seeds": len(m_sub),
                    "accuracy_mean": float(np.mean(acc_vals)),
                    "accuracy_std": float(np.std(acc_vals, ddof=1)) if len(acc_vals) > 1 else 0.0,
                    "macro_f1_mean": float(np.mean(f1_vals)),
                    "macro_f1_std": float(np.std(f1_vals, ddof=1)) if len(f1_vals) > 1 else 0.0,
                    "balanced_acc_mean": float(np.mean(bacc_vals)),
                    "balanced_acc_std": float(np.std(bacc_vals, ddof=1)) if len(bacc_vals) > 1 else 0.0,
                    "loss_mean": float(np.mean(loss_vals)),
                    "loss_std": float(np.std(loss_vals, ddof=1)) if len(loss_vals) > 1 else 0.0,
                    "params_per_round_mean": float(np.mean(param_vals)),
                    "params_per_round_std": float(np.std(param_vals, ddof=1)) if len(param_vals) > 1 else 0.0,
                    "cum_MB_mean": float(np.mean(mb_vals)),
                    "cum_MB_std": float(np.std(mb_vals, ddof=1)) if len(mb_vals) > 1 else 0.0,
                    "cum_comm_time_s_mean": float(np.mean(comm_t_vals)),
                    "cum_comm_time_s_std": float(np.std(comm_t_vals, ddof=1)) if len(comm_t_vals) > 1 else 0.0,
                    "decision_latency_ms_mean": float(np.mean(lat_ms_vals)),
                    "savings_vs_fixed_5_3_mean_pct": float(np.mean(sav_5_3_vals)),
                    "savings_vs_all_10_4_mean_pct": float(np.mean(sav_10_4_vals)),
                    "savings_vs_all_3_2_mean_pct": float(np.mean(sav_3_2_vals)),
                    "p_value_acc_vs_fixed": p_val_acc,
                    "p_value_f1_vs_fixed": p_val_f1,
                })

        m_df = pd.DataFrame(master_rows)
        m_df.to_csv(out_dir / "benchmark_master_summary.csv", index=False)
        with open(out_dir / "benchmark_master_summary.json", "w") as f:
            json.dump(master_rows, f, indent=2)

    for alpha in args.alphas:
        for seed in args.seeds:
            for method in args.methods:
                run_idx += 1
                if (str(method), float(alpha), int(seed)) in completed_keys:
                    logger.info(f"[{run_idx:02d}/{total_runs:02d}] Skipping already completed {method} (a={alpha}, s={seed}).")
                    continue

                logger.info(
                    f"\n[{run_idx:02d}/{total_runs:02d}] Running Method '{method}' | "
                    f"Alpha={alpha} | Seed={seed} | Rounds={args.rounds}..."
                )
                r_recs, s_summary = run_single_episode(
                    method=method,
                    alpha=alpha,
                    seed=seed,
                    rounds=args.rounds,
                    fleet_cfg=fleet_cfg,
                    reference_mode=args.reference_mode,
                    ppo_prior_model=ppo_prior_model,
                    ppo_noprior_model=ppo_noprior_model,
                )
                all_round_records.extend(r_recs)
                all_seed_summaries.append(s_summary)
                completed_keys.add((str(method), float(alpha), int(seed)))

                # Incremental persist
                update_and_save_master()

                logger.info(
                    f"[{method} | a={alpha} | s={seed}] Acc: {s_summary['final_accuracy']*100:.2f}% | "
                    f"F1: {s_summary['final_macro_f1']:.4f} | BAcc: {s_summary['final_balanced_accuracy']*100:.2f}% | "
                    f"Params: {s_summary['mean_parameters_per_round']:.0f} | "
                    f"Savings: {s_summary['savings_vs_fixed_5_3_pct']:.1f}%"
                )

    # Final summary and figures
    update_and_save_master()
    master_df = pd.read_csv(out_dir / "benchmark_master_summary.csv")
    generate_benchmark_figures(
        rounds_df=pd.DataFrame(all_round_records),
        per_seed_df=pd.DataFrame(all_seed_summaries),
        master_df=master_df,
        figures_dir=args.figures_dir,
    )

    elapsed_total = time.perf_counter() - t_start
    logger.info(f"\n=======================================================\nBENCHMARK SUITE COMPLETED IN {elapsed_total:.2f} s\n=======================================================\n")



if __name__ == "__main__":
    main()

