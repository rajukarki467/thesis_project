"""
Federated Non-IID Dirichlet Dataset Partitioning for ECG Heartbeat Classification.

Tasks:
- Sample class distribution for N clients via Dirichlet(alpha, ..., alpha) for alpha in [0.1, 0.5, 1.0].
- Partition samples per client and save metadata to data/processed/metadata/ and indices to data/partitions/.
- Add CLI entrypoints with argparse reading from configs/dataset.yaml.
- Ensure deterministic execution via a global seed parameter.
- Save intermediate verification plots into results/figures/eda/.
"""

from __future__ import annotations

import argparse
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

import numpy as np
import pandas as pd
import yaml

from src.preprocessing import AAMI_CLASSES
from src.utils.reproducibility import set_seed
from src.visualization import (
    plot_client_distribution,
    plot_client_sample_counts,
    plot_multi_alpha_comparison,
)

logger = logging.getLogger("partition")


# ---------------------------------------------------------------------
# Dirichlet Partitioning Algorithm
# ---------------------------------------------------------------------

def dirichlet_partition(
    y: Sequence[int] | np.ndarray,
    num_clients: int = 8,
    alpha: float = 0.5,
    seed: int = 42,
    min_samples_per_client: int = 10,
    max_retries: int = 100,
) -> Dict[int, np.ndarray]:
    """
    Partition dataset sample indices among N clients according to a Dirichlet distribution
    over class allocations:
        p_c ~ Dirichlet(alpha * 1_N)

    Parameters
    ----------
    y : Sequence[int] or np.ndarray
        Class labels (integer array or list).
    num_clients : int
        Number of simulated federated clients (N).
    alpha : float
        Dirichlet concentration parameter (e.g. 0.1, 0.5, 1.0).
        Lower alpha produces higher non-IID label skew across clients.
    seed : int
        Random seed for deterministic sampling.
    min_samples_per_client : int
        Minimum number of samples required per client across all classes.
    max_retries : int
        Maximum retries if any client receives fewer than min_samples_per_client.

    Returns
    -------
    client_indices : Dict[int, np.ndarray]
        Dictionary mapping client_id (0..N-1) to an array of integer indices.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    num_samples = len(y)

    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if num_samples < num_clients * min_samples_per_client:
        raise ValueError(
            f"Dataset length ({num_samples}) is too small for {num_clients} clients with min_samples={min_samples_per_client}."
        )

    current_seed = seed
    attempt = 0

    while attempt < max_retries:
        rng = np.random.default_rng(current_seed)
        client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

        for c in classes:
            class_indices = np.where(y == c)[0]
            rng.shuffle(class_indices)
            n_class_samples = len(class_indices)

            # Sample class proportions for all N clients from Dirichlet(alpha)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))

            # Discretize proportions to integer sample counts
            split_points = (np.cumsum(proportions) * n_class_samples).astype(int)
            split_points[-1] = n_class_samples

            start = 0
            for client_id, end in enumerate(split_points):
                assigned = class_indices[start:end]
                client_indices[client_id].extend(assigned.tolist())
                start = end

        # Check client sample size constraints
        sizes = [len(client_indices[i]) for i in range(num_clients)]
        if min(sizes) >= min_samples_per_client:
            # Deterministically shuffle each client's partition
            final_partitions = {}
            for client_id in range(num_clients):
                client_arr = np.array(client_indices[client_id], dtype=np.int64)
                rng.shuffle(client_arr)
                final_partitions[client_id] = client_arr

            # Verify total count matches
            total_allocated = sum(len(p) for p in final_partitions.values())
            assert total_allocated == num_samples, (
                f"Index mismatch: {total_allocated} allocated vs {num_samples} total."
            )

            return final_partitions

        attempt += 1
        current_seed += 1

    raise RuntimeError(
        f"Failed to find a valid Dirichlet partition with min_samples_per_client={min_samples_per_client} "
        f"after {max_retries} attempts."
    )


# ---------------------------------------------------------------------
# Class Distribution Analysis
# ---------------------------------------------------------------------

def client_class_distribution(
    y: Sequence[int] | np.ndarray,
    client_indices: Dict[int, Sequence[int]],
    class_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Compute class counts and proportions for each client.

    Parameters
    ----------
    y : Sequence[int] or np.ndarray
        Ground-truth class labels for the partitioned dataset.
    client_indices : Dict[int, Sequence[int]]
        Client ID to sample indices mapping.
    class_names : Optional[Sequence[str]]
        Names for each class (defaults to standard AAMI classes).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['client', 'total_samples', <class_0_pct>, ...]
    """
    y = np.asarray(y)
    unique_classes = np.unique(y)

    if class_names is None:
        class_names = [AAMI_CLASSES[i] if i < len(AAMI_CLASSES) else f"class_{i}" for i in unique_classes]

    records = []
    for client_id in sorted(client_indices.keys()):
        indices = client_indices[client_id]
        client_labels = y[indices]
        total = len(client_labels)

        row = {
            "client": client_id,
            "total_samples": total,
        }

        # Calculate counts and proportions for each class
        for c_id, c_name in zip(unique_classes, class_names):
            cnt = int(np.sum(client_labels == c_id))
            pct = float(cnt / total) if total > 0 else 0.0
            row[f"{c_name}_count"] = cnt
            row[c_name] = pct

        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Partition Execution and Saving Pipeline
# ---------------------------------------------------------------------

def run_federated_partitions(
    y: np.ndarray,
    num_clients: int = 8,
    alphas: Sequence[float] = (0.1, 0.5, 1.0),
    seed: int = 42,
    min_samples_per_client: int = 10,
    partitions_dir: Union[str, Path] = "data/partitions",
    metadata_dir: Union[str, Path] = "data/processed/metadata",
    eda_dir: Optional[Union[str, Path]] = "results/figures/eda",
) -> Dict[float, Dict[str, Any]]:
    """
    Execute Dirichlet partitioning for multiple alpha values, save indices and metadata,
    and generate verification plots.

    Parameters
    ----------
    y : np.ndarray
        Labels of dataset to partition (typically y_train).
    num_clients : int
        Number of federated clients (default 8).
    alphas : Sequence[float]
        Dirichlet concentration parameters to evaluate (default [0.1, 0.5, 1.0]).
    seed : int
        Random seed for deterministic sampling.
    min_samples_per_client : int
        Minimum samples per client.
    partitions_dir : Union[str, Path]
        Directory to store partition indices.
    metadata_dir : Union[str, Path]
        Directory to store partition metadata CSV/JSON.
    eda_dir : Optional[Union[str, Path]]
        Directory to store verification EDA plots.

    Returns
    -------
    Dict[float, Dict[str, Any]]
        Results dictionary keyed by alpha containing partitions and distribution data.
    """
    partitions_dir = Path(partitions_dir)
    metadata_dir = Path(metadata_dir)
    partitions_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if eda_dir:
        eda_path = Path(eda_dir)
        eda_path.mkdir(parents=True, exist_ok=True)
    else:
        eda_path = None

    class_names = [AAMI_CLASSES[i] if i < len(AAMI_CLASSES) else f"class_{i}" for i in np.unique(y)]
    all_results = {}
    distributions_by_alpha: Dict[float, pd.DataFrame] = {}

    overall_meta = {
        "num_clients": num_clients,
        "seed": seed,
        "alphas": list(alphas),
        "total_samples": len(y),
        "classes": class_names,
        "partitions": {},
    }

    for alpha in alphas:
        logger.info(f"Generating Dirichlet partition for alpha = {alpha} ({num_clients} clients)...")

        # 1. Sample Dirichlet partition
        client_indices = dirichlet_partition(
            y=y,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
            min_samples_per_client=min_samples_per_client,
        )

        # 2. Compute class distribution
        dist_df = client_class_distribution(y, client_indices, class_names=class_names)

        # 3. Save partition indices (.npy)
        # Standard config folder: clients_{num_clients}_alpha_{alpha}
        config_name = f"clients_{num_clients}_alpha_{alpha}"
        config_dir = partitions_dir / config_name
        config_dir.mkdir(parents=True, exist_ok=True)

        # Also support alpha_{alpha} folder for direct compatibility
        alpha_dir = partitions_dir / f"alpha_{alpha}"
        alpha_dir.mkdir(parents=True, exist_ok=True)

        # Root data/alpha_{alpha} folder if requested
        root_alpha_dir = PROJECT_ROOT / "data" / f"alpha_{alpha}"
        root_alpha_dir.mkdir(parents=True, exist_ok=True)

        for client_id, indices in client_indices.items():
            np.save(config_dir / f"client_{client_id}.npy", indices)
            np.save(alpha_dir / f"client_{client_id}.npy", indices)
            np.save(root_alpha_dir / f"client_{client_id}.npy", indices)

        # 4. Save metadata CSV and JSON
        meta_csv_path = metadata_dir / f"partition_distribution_alpha_{alpha}.csv"
        dist_df.to_csv(meta_csv_path, index=False)

        client_summary = {
            int(client_id): {
                "num_samples": len(indices),
                "class_counts": {
                    c_name: int(np.sum(y[indices] == c_id))
                    for c_id, c_name in enumerate(class_names)
                },
            }
            for client_id, indices in client_indices.items()
        }

        overall_meta["partitions"][f"alpha_{alpha}"] = {
            "alpha": alpha,
            "min_samples": int(min(len(indices) for indices in client_indices.values())),
            "max_samples": int(max(len(indices) for indices in client_indices.values())),
            "clients": client_summary,
        }

        # 5. Generate verification plots
        if eda_path:
            heatmap_data = dist_df.set_index("client")[class_names]
            distributions_by_alpha[alpha] = heatmap_data

            # Heatmap of client label distributions
            plot_client_distribution(
                client_distribution=heatmap_data,
                title=f"Dirichlet Non-IID Partition (α = {alpha}, N = {num_clients} Clients)",
                save_path=eda_path / f"dirichlet_heatmap_alpha_{alpha}.png",
            )

            # Bar plot of sample counts per client
            plot_client_sample_counts(
                client_indices=client_indices,
                alpha=alpha,
                save_path=eda_path / f"client_sample_counts_alpha_{alpha}.png",
            )

        all_results[alpha] = {
            "client_indices": client_indices,
            "distribution": dist_df,
        }

        logger.info(
            f"Alpha {alpha}: Client samples range [{min(len(idx) for idx in client_indices.values())} – "
            f"{max(len(idx) for idx in client_indices.values())}]"
        )

    # Save overall partition metadata JSON
    with open(metadata_dir / "partition_metadata.json", "w") as f:
        json.dump(overall_meta, f, indent=2)

    # Cross-alpha comparison plot
    if eda_path and len(distributions_by_alpha) > 1:
        plot_multi_alpha_comparison(
            distributions_by_alpha=distributions_by_alpha,
            save_path=eda_path / "dirichlet_alphas_comparison.png",
        )
        logger.info(f"Saved Dirichlet cross-alpha comparison plot to {eda_path}")

    return all_results


# ---------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Partition ECG heartbeat dataset across federated clients via Dirichlet distribution."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (overrides config).",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=None,
        help="Number of federated clients (overrides config).",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=None,
        help="List of Dirichlet alpha parameters (overrides config).",
    )
    parser.add_argument(
        "--split_to_partition",
        type=str,
        choices=["train", "all"],
        default=None,
        help="Whether to partition 'train' split or 'all' samples.",
    )
    parser.add_argument(
        "--partitions_dir",
        type=str,
        default=None,
        help="Output directory for client partition indices.",
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default=None,
        help="Output directory for partition metadata.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()
    config = load_config(args.config)

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed)
    logger.info(f"Initialized global random seed: {seed}")

    part_cfg = config.get("partition", {})
    num_clients = args.num_clients or part_cfg.get("num_clients", 8)
    alphas = args.alphas or part_cfg.get("alphas", [0.1, 0.5, 1.0])
    split_choice = args.split_to_partition or part_cfg.get("split_to_partition", "train")
    min_samples = part_cfg.get("min_samples_per_client", 10)
    partitions_dir = args.partitions_dir or part_cfg.get("partitions_dir", "data/partitions")
    metadata_dir = args.metadata_dir or part_cfg.get("metadata_dir", "data/processed/metadata")
    eda_dir = config.get("figures", {}).get("eda_dir", "results/figures/eda")
    processed_dir = Path(config.get("dataset", {}).get("processed_dir", "data/processed"))

    # Load targets y
    target_file = processed_dir / "beats" / (f"y_{split_choice}.npy" if split_choice == "train" else "y.npy")
    if not target_file.exists():
        raise FileNotFoundError(
            f"Preprocessed targets not found at {target_file}. Please run src/preprocessing.py first!"
        )

    y = np.load(target_file)
    logger.info(f"Loaded {len(y)} labels from {target_file} for federated partitioning.")

    run_federated_partitions(
        y=y,
        num_clients=num_clients,
        alphas=alphas,
        seed=seed,
        min_samples_per_client=min_samples,
        partitions_dir=partitions_dir,
        metadata_dir=metadata_dir,
        eda_dir=eda_dir,
    )

    logger.info("Federated partitioning completed successfully.")


if __name__ == "__main__":
    main()
