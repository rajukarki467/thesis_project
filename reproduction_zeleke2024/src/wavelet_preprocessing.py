"""
Wavelet Preprocessing Pipeline for MIT-BIH ECG Arrhythmia Classification.
Replicating Zeleke & Bochicchio (IEEE BigData 2024):
"Federated Kolmogorov-Arnold Networks for Health Data Analysis: A Study Using ECG Signal"

Pipeline:
1. Load raw MIT-BIH recordings from WFDB files.
2. 9-level Discrete Wavelet Transform (DWT) decomposition using Daubechies 5 ('db5') wavelet.
3. Denoising: Zero out level-9 approximation coefficients (cA9) to eliminate baseline wander,
   and detail coefficients (cD1, cD2) to eliminate high-frequency noise and powerline interference.
4. Reconstruct denoised ECG signal using inverse DWT (pywt.waverec).
5. Segment individual beats centered around annotated R-peaks.
6. Z-score normalization: z = (x - mean) / std.
7. Resampling to target feature dimension (default n=32).
8. 80/20 train/test split.
9. Dirichlet Non-IID client partitioning across federated clients.
10. Automatic caching into reproduction_zeleke2024/data/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pywt
from scipy import interpolate
from sklearn.model_selection import train_test_split
import wfdb

logger = logging.getLogger("wavelet_preprocessing")

# Standard AAMI EC57 class mappings to 5 primary groups
AAMI_MAPPING: Dict[str, int] = {
    # 0: Normal and bundle branch block beats (N)
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0, "B": 0,
    # 1: Supraventricular ectopic beats (S)
    "A": 1, "a": 1, "J": 1, "S": 1,
    # 2: Ventricular ectopic beats (V)
    "V": 2, "E": 2, "r": 2,
    # 3: Fusion beats (F)
    "F": 3,
    # 4: Paced and unclassifiable beats (Q)
    "/": 4, "f": 4, "Q": 4,
}

CLASS_NAMES: List[str] = [
    "Normal (N)",
    "Supraventricular (S)",
    "Ventricular (V)",
    "Fusion (F)",
    "Unclassified (Q)",
]


def apply_wavelet_denoising_db5(
    signal: np.ndarray,
    level: int = 9,
    wavelet: str = "db5",
    suppress_baseline: bool = True,
    suppress_hf_noise: bool = True,
) -> np.ndarray:
    """
    Decompose ECG signal using 9 levels of 'db5' discrete wavelet transform
    to suppress baseline wander and high-frequency noise, matching Zeleke & Bochicchio (2024).

    Parameters
    ----------
    signal : np.ndarray
        1-D raw ECG signal array.
    level : int
        Decomposition level (default: 9).
    wavelet : str
        Mother wavelet family (default: 'db5').
    suppress_baseline : bool
        Whether to zero approximation coefficients cA9 (0 - 0.35 Hz) to eliminate baseline drift.
    suppress_hf_noise : bool
        Whether to zero high-frequency detail coefficients cD1 (90-180 Hz) and cD2 (45-90 Hz).

    Returns
    -------
    clean_signal : np.ndarray
        Reconstructed denoised ECG signal of same length.
    """
    signal = np.asarray(signal, dtype=np.float64)
    orig_len = len(signal)

    # Multi-level DWT decomposition: [cA9, cD9, cD8, ..., cD1]
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # 1. Suppress baseline wander (< 0.35 Hz) by clearing approximation coefficients
    if suppress_baseline:
        coeffs[0] = np.zeros_like(coeffs[0])

    # 2. Suppress high-frequency noise and powerline interference (> 45 Hz)
    if suppress_hf_noise:
        if len(coeffs) > 1:
            coeffs[-1] = np.zeros_like(coeffs[-1])  # cD1 (90 - 180 Hz)
        if len(coeffs) > 2:
            coeffs[-2] = np.zeros_like(coeffs[-2])  # cD2 (45 - 90 Hz)

    # 3. Inverse DWT reconstruction
    reconstructed = pywt.waverec(coeffs, wavelet)
    return reconstructed[:orig_len]


def zscore_normalize(segment: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Standardize a single beat segment: z = (x - mean) / std."""
    mean = np.mean(segment)
    std = np.std(segment)
    if std < epsilon:
        return segment - mean
    return (segment - mean) / std


def resample_beat_linear(beat: np.ndarray, target_length: int = 32) -> np.ndarray:
    """Downsample beat segment to target length (n=32) using linear interpolation."""
    orig_length = len(beat)
    if orig_length == target_length:
        return beat
    x_old = np.linspace(0.0, 1.0, orig_length)
    x_new = np.linspace(0.0, 1.0, target_length)
    interp_fn = interpolate.interp1d(x_old, beat, kind="linear")
    return interp_fn(x_new)


def segment_and_extract_beats(
    raw_dir: Union[str, Path] = "data/raw/mit-bih-arrhythmia-database-1.0.0",
    window_size: int = 360,
    pre_r_samples: int = 180,
    target_features: int = 32,
    wavelet: str = "db5",
    level: int = 9,
    max_records: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract, wavelet-denoise, and segment ECG beats from MIT-BIH recordings.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (N_beats, target_features).
    y : np.ndarray
        Class labels array of shape (N_beats,).
    record_ids : np.ndarray
        Array containing recording ID for each beat.
    """
    raw_path = Path(raw_dir)
    records_file = raw_path / "RECORDS"
    if not records_file.exists():
        raise FileNotFoundError(f"RECORDS file not found at: {records_file}")

    records = [line.strip() for line in records_file.read_text().splitlines() if line.strip()]
    if max_records is not None:
        records = records[:max_records]

    logger.info(f"Processing {len(records)} MIT-BIH records with 9-level {wavelet} wavelet denoising...")

    all_beats: List[np.ndarray] = []
    all_labels: List[int] = []
    all_record_ids: List[str] = []

    post_r_samples = window_size - pre_r_samples

    for rec_id in records:
        rec_path = str(raw_path / rec_id)
        try:
            record = wfdb.rdrecord(rec_path)
            ann = wfdb.rdann(rec_path, "atr")
        except Exception as e:
            logger.warning(f"Skipping record {rec_id} due to read error: {e}")
            continue

        # Extract primary lead (channel 0: MLII or modified lead)
        raw_signal = record.p_signal[:, 0]

        # 9-level db5 Wavelet Denoising
        denoised_signal = apply_wavelet_denoising_db5(
            raw_signal, level=level, wavelet=wavelet, suppress_baseline=True, suppress_hf_noise=True
        )

        sig_len = len(denoised_signal)
        for sample_idx, symbol in zip(ann.sample, ann.symbol):
            if symbol not in AAMI_MAPPING:
                continue
            class_label = AAMI_MAPPING[symbol]

            start = sample_idx - pre_r_samples
            end = sample_idx + post_r_samples
            if start < 0 or end > sig_len:
                continue

            beat = denoised_signal[start:end]
            if len(beat) != window_size:
                continue

            # Z-score normalization
            norm_beat = zscore_normalize(beat)

            # Resample to target features (e.g. 32)
            feat_beat = resample_beat_linear(norm_beat, target_length=target_features)

            all_beats.append(feat_beat)
            all_labels.append(class_label)
            all_record_ids.append(rec_id)

    X = np.asarray(all_beats, dtype=np.float32)
    y = np.asarray(all_labels, dtype=np.int64)
    rec_arr = np.asarray(all_record_ids)

    logger.info(f"Extracted {len(y)} beats across 5 classes: {np.bincount(y, minlength=5)}")
    return X, y, rec_arr


def partition_clients_dirichlet(
    y: np.ndarray,
    num_clients: int = 5,
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    """
    Non-IID Dirichlet partitioning across simulated federated clients.

    Parameters
    ----------
    y : np.ndarray
        Array of integer class labels.
    num_clients : int
        Number of federated client nodes.
    alpha : float
        Dirichlet concentration parameter (lower = more heterogeneous/skewed).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    client_partitions : Dict[int, np.ndarray]
        Dictionary mapping client_id (0..num_clients-1) to indices in y.
    """
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(y))
    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
        split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, split_points)
        for i in range(num_clients):
            client_indices[i].extend(splits[i])

    # Convert to sorted numpy arrays
    partitions = {i: np.sort(np.asarray(client_indices[i], dtype=np.int64)) for i in range(num_clients)}
    return partitions


def load_and_preprocess_dataset(
    config: Dict[str, Any],
    cache_dir: Optional[Union[str, Path]] = None,
    force_recompute: bool = False,
    dry_run: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """
    High-level entry point to load or compute the preprocessed dataset.

    Returns
    -------
    X_train : np.ndarray
        Training features (80% split).
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test features (20% hold-out split).
    y_test : np.ndarray
        Test labels.
    client_partitions : Dict[int, np.ndarray]
        Client index mappings for local training datasets.
    """
    raw_dir = config.get("dataset", {}).get("raw_dir", "data/raw/mit-bih-arrhythmia-database-1.0.0")
    if cache_dir is None:
        cache_dir = config.get("dataset", {}).get("cache_dir", "reproduction_zeleke2024/data")
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    num_clients = 2 if dry_run else config.get("experiment", {}).get("num_clients", 5)
    alpha = config.get("experiment", {}).get("dirichlet_alpha", 0.5)
    seed = config.get("experiment", {}).get("seed", 42)
    target_features = config.get("dataset", {}).get("target_features", 32)
    test_ratio = config.get("dataset", {}).get("test_ratio", 0.20)

    train_feat_file = cache_path / f"X_train_f{target_features}.npy"
    train_label_file = cache_path / "y_train.npy"
    test_feat_file = cache_path / f"X_test_f{target_features}.npy"
    test_label_file = cache_path / "y_test.npy"
    partition_file = cache_path / f"partitions_N{num_clients}_a{alpha}_s{seed}.json"

    # Check cache
    if not force_recompute and not dry_run and train_feat_file.exists() and test_feat_file.exists():
        logger.info(f"Loading cached wavelet-preprocessed dataset from {cache_path}...")
        X_train = np.load(train_feat_file)
        y_train = np.load(train_label_file)
        X_test = np.load(test_feat_file)
        y_test = np.load(test_label_file)

        if partition_file.exists():
            with open(partition_file, "r") as f:
                raw_parts = json.load(f)
            client_partitions = {int(k): np.array(v, dtype=np.int64) for k, v in raw_parts.items()}
        else:
            client_partitions = partition_clients_dirichlet(y_train, num_clients=num_clients, alpha=alpha, seed=seed)
            with open(partition_file, "w") as f:
                json.dump({str(k): v.tolist() for k, v in client_partitions.items()}, f)

        return X_train, y_train, X_test, y_test, client_partitions

    # If dry-run, use a fast subset (3 records)
    if dry_run:
        logger.info("Dry-run mode enabled: extracting from 3 MIT-BIH records...")
        X_full, y_full, _ = segment_and_extract_beats(
            raw_dir=raw_dir,
            target_features=target_features,
            max_records=3,
        )
    else:
        logger.info("Extracting beats from all 48 MIT-BIH recordings with 9-level db5 wavelet denoising...")
        X_full, y_full, _ = segment_and_extract_beats(
            raw_dir=raw_dir,
            target_features=target_features,
            max_records=None,
        )

    # Stratified 80/20 train/test split matching the paper
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_ratio, random_state=seed, stratify=y_full
    )

    client_partitions = partition_clients_dirichlet(
        y_train, num_clients=num_clients, alpha=alpha, seed=seed
    )

    # Save to cache if not dry-run
    if not dry_run:
        np.save(train_feat_file, X_train)
        np.save(train_label_file, y_train)
        np.save(test_feat_file, X_test)
        np.save(test_label_file, y_test)
        with open(partition_file, "w") as f:
            json.dump({str(k): v.tolist() for k, v in client_partitions.items()}, f)
        logger.info(f"Saved preprocessed data to {cache_path}")

    return X_train, y_train, X_test, y_test, client_partitions
