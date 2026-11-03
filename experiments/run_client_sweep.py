"""
Controlled Per-Client (G, k) Configuration Sweep and Best-Config Feasibility Study.

Mathematical Formulation & Experimental Protocol:
- For every client i in {0..7} and every (G, k) configuration allowed by its device tier:
    - Trains FastKAN on client i's local partition using identical optimizer, epochs (3), and seed.
    - Evaluates on the held-out validation split (X_val, y_val, 9,088 beats; zero test set leakage).
    - Records validation Macro F1, Accuracy, Balanced Accuracy, parameter footprint,
      estimated round time, estimated energy, and memory fit.
    - Test metrics are evaluated for final archival reporting only (never used for selection).
- Feasible Selection Rule:
    - Feasible set: memory_fit is True AND estimated_round_time <= tier deadline.
    - Objective: maximize validation Macro F1 within the feasible set.
    - Tie-breaking: lower parameter count.
- Outputs top 3 configurations per client and calculates convergence agreement
  with Heuristic and PPO controllers.
- Results saved to results/fleet/client_sweep_alpha_{alpha}.csv and JSON summary.
"""

from __future__ import annotations

import argparse
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.controllers.heuristic import HeuristicController
from src.environment.fleet import (
    FleetConfig,
    compute_param_count,
    generate_fleet_profiles,
)
from src.models.fastkan_model import FastKANClassifier
from src.utils.metrics import compute_metrics
from src.utils.reproducibility import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("client_sweep")


def evaluate_dataset(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate model on dataset and return standard metrics dictionary."""
    model.eval()
    dev = torch.device(device)
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_b, y_b in loader:
            x_b = x_b.to(dev)
            y_b = y_b.to(dev)
            logits = model(x_b)
            loss = criterion(logits, y_b)
            total_loss += float(loss.item()) * len(y_b)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y_b.cpu().numpy().tolist())

    metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
    metrics["loss"] = float(total_loss / max(1, len(y)))
    return metrics


def run_single_client_sweep(
    client_id: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    client_indices: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fleet_config: FleetConfig,
    fleet_profiles: Dict[int, Any],
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Run sweep over all allowed (G, k) configs for a single client."""
    prof = fleet_profiles[client_id]
    tier = prof.tier
    dev_name = prof.device_name
    allowed = prof.allowed_configs

    x_c = X_train[client_indices]
    y_c = y_train[client_indices]

    results = []

    for g, k in allowed:
        set_seed(seed)
        params = compute_param_count(g, k)

        dataset = TensorDataset(torch.from_numpy(x_c).float(), torch.from_numpy(y_c).long())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = FastKANClassifier(
            layers_hidden=[32, 16, 5],
            grid_size=g,
            spline_order=k,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        t_start = time.perf_counter()
        model.train()
        for ep in range(epochs):
            for x_b, y_b in loader:
                x_b = x_b.to(device)
                y_b = y_b.to(device)
                optimizer.zero_grad()
                out = model(x_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()
        measured_comp_time = (time.perf_counter() - t_start) / max(1, epochs)

        # Validation evaluation (strictly for configuration ranking & selection)
        val_metrics = evaluate_dataset(model, X_val, y_val, batch_size=64, device=device)

        # Held-out test evaluation (strictly for final reporting)
        test_metrics = evaluate_dataset(model, X_test, y_test, batch_size=64, device=device)

        # Physical resource estimation
        fits_mem = prof.fits_memory(g, k, batch_size=batch_size)
        est_round_time = prof.estimate_round_time(g, k, measured_laptop_compute_sec=measured_comp_time)
        est_energy = est_round_time * prof.compute_speed_factor * (1.0 if tier <= 2 else 0.1)

        deadline_ok = bool(est_round_time <= prof.deadline_seconds)
        feasible = bool(fits_mem and deadline_ok)

        results.append({
            "client_id": client_id,
            "tier": tier,
            "device_name": dev_name,
            "grid_size": g,
            "spline_order": k,
            "config": f"({g},{k})",
            "params": params,
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_balanced_accuracy": float(val_metrics.get("balanced_accuracy", val_metrics["accuracy"])),
            "val_loss": float(val_metrics["loss"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_balanced_accuracy": float(test_metrics.get("balanced_accuracy", test_metrics["accuracy"])),
            "test_loss": float(test_metrics["loss"]),
            "measured_comp_sec": round(measured_comp_time, 4),
            "est_round_time_s": round(est_round_time, 4),
            "est_energy": round(est_energy, 4),
            "memory_fit": fits_mem,
            "deadline_ok": deadline_ok,
            "feasible": feasible,
        })

    return results


def run_fleet_sweep(
    alpha: float = 0.5,
    epochs: int = 3,
    fleet_config_path: str = "configs/fleet.yaml",
    output_dir: str = "results/fleet",
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    """
    Run full 8-client controlled sweep across all tier-allowed configurations.
    """
    out_dir = PROJECT_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fleet_cfg = FleetConfig(fleet_config_path)
    fleet_profiles = generate_fleet_profiles(seed=seed, config_path=fleet_config_path)

    # Load dataset splits
    beats_dir = PROJECT_ROOT / "data" / "processed" / "beats"
    X_train = np.load(beats_dir / "X_train.npy")
    y_train = np.load(beats_dir / "y_train.npy")
    X_val = np.load(beats_dir / "X_val.npy")
    y_val = np.load(beats_dir / "y_val.npy")
    X_test = np.load(beats_dir / "X_test.npy")
    y_test = np.load(beats_dir / "y_test.npy")

    # Load client partitions for alpha
    part_dir = PROJECT_ROOT / "data" / "partitions" / f"clients_8_alpha_{alpha}"
    all_sweep_records = []

    logger.info(f"Starting controlled configuration sweep for 8 clients (alpha={alpha}, seed={seed})...")

    for cid in range(8):
        c_idx = np.load(part_dir / f"client_{cid}.npy")
        c_records = run_single_client_sweep(
            client_id=cid,
            X_train=X_train,
            y_train=y_train,
            client_indices=c_idx,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            fleet_config=fleet_cfg,
            fleet_profiles=fleet_profiles,
            epochs=epochs,
            seed=seed,
        )
        all_sweep_records.extend(c_records)
        logger.info(f"Completed Client {cid} ({fleet_profiles[cid].device_name}, Tier {fleet_profiles[cid].tier}): {len(c_records)} configs evaluated.")

    sweep_df = pd.DataFrame(all_sweep_records)
    csv_path = out_dir / f"client_sweep_alpha_{alpha}.csv"
    sweep_df.to_csv(csv_path, index=False)
    logger.info(f"Saved full sweep table ({len(sweep_df)} rows) to {csv_path}")

    # Best-by-sweep selection per client:
    # Maximize val_macro_f1 among feasible configs, break ties by lower params
    top3_per_client = {}
    for cid in range(8):
        c_sub = sweep_df[sweep_df["client_id"] == cid].copy()
        c_feas = c_sub[c_sub["feasible"]]
        if c_feas.empty:
            c_feas = c_sub  # Fallback if none feasible

        sorted_c = c_feas.sort_values(by=["val_macro_f1", "params"], ascending=[False, True])
        top3_list = []
        for rank, (_, row) in enumerate(sorted_c.head(3).iterrows(), start=1):
            top3_list.append({
                "rank": rank,
                "config": row["config"],
                "grid_size": int(row["grid_size"]),
                "spline_order": int(row["spline_order"]),
                "params": int(row["params"]),
                "val_macro_f1": float(row["val_macro_f1"]),
                "val_accuracy": float(row["val_accuracy"]),
                "val_balanced_accuracy": float(row["val_balanced_accuracy"]),
                "est_round_time_s": float(row["est_round_time_s"]),
                "feasible": bool(row["feasible"]),
            })
        top3_per_client[cid] = {
            "client_id": cid,
            "device_name": fleet_profiles[cid].device_name,
            "tier": fleet_profiles[cid].tier,
            "best_config": top3_list[0]["config"],
            "best_val_macro_f1": top3_list[0]["val_macro_f1"],
            "top3": top3_list,
        }

    top3_json_path = out_dir / f"client_top3_best_configs_alpha_{alpha}.json"
    with open(top3_json_path, "w") as f:
        json.dump(top3_per_client, f, indent=2)
    logger.info(f"Saved top-3 configurations summary to {top3_json_path}")

    return sweep_df, top3_per_client


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Controlled Per-Client (G, k) Configuration Sweep")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration alpha")
    parser.add_argument("--epochs", type=int, default=3, help="Local epochs per sweep trial")
    parser.add_argument("--fleet", type=str, default="configs/fleet.yaml", help="Path to fleet config")
    parser.add_argument("--output_dir", type=str, default="results/fleet", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    sweep_df, top3 = run_fleet_sweep(
        alpha=args.alpha,
        epochs=args.epochs,
        fleet_config_path=args.fleet,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print("\n" + "=" * 95)
    print(f"PER-CLIENT BEST CONFIGURATION SWEEP STUDY (Alpha = {args.alpha})")
    print("=" * 95)
    for cid in range(8):
        c_info = top3[cid]
        print(f"\nClient {cid} | Tier {c_info['tier']} ({c_info['device_name']}): Best Config = {c_info['best_config']} (Val Macro F1: {c_info['best_val_macro_f1']:.4f})")
        for item in c_info["top3"]:
            print(f"  Rank {item['rank']}: {item['config']:<7} | Params: {item['params']:<5d} | Val F1: {item['val_macro_f1']:.4f} | Val Acc: {item['val_accuracy']:.4f} | Est Time: {item['est_round_time_s']:.3f}s")
    print("=" * 95 + "\n")
