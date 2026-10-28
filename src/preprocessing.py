"""
ECG Preprocessing Pipeline for the MIT-BIH Arrhythmia Database.

Standard AAMI Heartbeat Classification Preprocessing:
1. Load .dat and .atr files via wfdb.
2. Apply a 0.5–40 Hz bandpass Butterworth filter (order 3) to eliminate baseline
   wander and high-frequency noise.
3. Detect R-peaks and extract 360-sample windows (1.0 s at 360 Hz).
4. Normalize each beat: z = (x - mu) / sigma.
5. Resample each beat to n=32 features using linear interpolation
   (scipy.interpolate or torch.nn.functional.interpolate).
6. Group MIT-BIH annotations into standard AAMI classes (N, SVEB, VEB, F, Q).
7. Enforce patient-level split (split by recording IDs, e.g., 48 recordings
   separated cleanly into train/val/test).
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
from scipy import interpolate
from scipy.signal import butter, filtfilt, find_peaks, sosfiltfilt
import yaml

# Optional PyTorch import for torch-based interpolation
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False

from src.dataset import get_record_ids, load_annotations, load_record
from src.utils.reproducibility import set_seed
from src.utils.splits import split_records
from src.visualization import (
    plot_aami_beat_classes,
    plot_patient_split_distribution,
    plot_r_peaks,
    plot_raw_vs_filtered,
)

logger = logging.getLogger("preprocessing")

# ---------------------------------------------------------------------
# AAMI EC57 Heartbeat Class Groupings
# ---------------------------------------------------------------------

AAMI_MAPPING: Dict[str, str] = {
    # N: Normal and bundle branch block beats
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "B": "N",
    # SVEB: Supraventricular ectopic beats
    "A": "SVEB",
    "a": "SVEB",
    "J": "SVEB",
    "S": "SVEB",
    # VEB: Ventricular ectopic beats
    "V": "VEB",
    "E": "VEB",
    "r": "VEB",
    # F: Fusion of ventricular and normal beats
    "F": "F",
    # Q: Paced, fusion of paced, and unclassifiable beats
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}

AAMI_CLASSES: List[str] = ["N", "SVEB", "VEB", "F", "Q"]
CLASS_TO_ID: Dict[str, int] = {c: i for i, c in enumerate(AAMI_CLASSES)}
ID_TO_CLASS: Dict[int, str] = {i: c for i, c in enumerate(AAMI_CLASSES)}


# ---------------------------------------------------------------------
# Signal Validation
# ---------------------------------------------------------------------

def _validate_signal(sig: np.ndarray) -> np.ndarray:
    """Validate and convert an ECG signal into a 1-D float64 array."""
    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim != 1:
        raise ValueError(f"Expected 1-D ECG signal, got shape {sig.shape}.")
    if sig.size == 0:
        raise ValueError("ECG signal is empty.")
    if not np.all(np.isfinite(sig)):
        raise ValueError("ECG signal contains NaN or infinite values.")
    return sig


# ---------------------------------------------------------------------
# Step 1: Bandpass Butterworth Filtering (0.5–40 Hz, Order 3)
# ---------------------------------------------------------------------

def bandpass_filter(
    signal: np.ndarray,
    fs: float = 360.0,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 3,
    use_sos: bool = True,
) -> np.ndarray:
    """
    Apply a 0.5–40 Hz bandpass Butterworth filter (order 3) to eliminate
    baseline wander (low frequency) and high-frequency noise/powerline interference.

    Uses zero-phase bidirectional filtering (sosfiltfilt) to prevent phase
    distortion and preserve the exact timing of R-peaks and morphological waves.

    Parameters
    ----------
    signal : np.ndarray
        Raw 1-D ECG signal.
    fs : float
        Sampling frequency in Hz (default 360.0 Hz for MIT-BIH).
    lowcut : float
        Lower cutoff frequency in Hz (default 0.5 Hz).
    highcut : float
        Upper cutoff frequency in Hz (default 40.0 Hz).
    order : int
        Filter order (default 3).
    use_sos : bool
        Whether to use second-order sections (SOS) for numerical stability.

    Returns
    -------
    np.ndarray
        Zero-phase bandpass-filtered ECG signal.
    """
    sig = _validate_signal(signal)
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if not 0 < lowcut < highcut < fs / 2.0:
        raise ValueError(
            f"Cutoffs must satisfy 0 < lowcut ({lowcut}) < highcut ({highcut}) < Nyquist ({fs / 2.0})."
        )
    if order < 1:
        raise ValueError("Filter order must be >= 1.")

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    if use_sos:
        sos = butter(order, [low, high], btype="bandpass", output="sos")
        return sosfiltfilt(sos, sig)
    else:
        b, a = butter(order, [low, high], btype="bandpass")
        return filtfilt(b, a, sig)


# ---------------------------------------------------------------------
# Step 2: R-Peak Detection
# ---------------------------------------------------------------------

def detect_r_peaks(
    signal: np.ndarray,
    fs: float = 360.0,
    distance_seconds: float = 0.25,
    prominence: Optional[float] = None,
    method: str = "pan_tompkins",
) -> np.ndarray:
    """
    Detect R-peaks in an ECG signal using energy envelope analysis.

    Parameters
    ----------
    signal : np.ndarray
        ECG signal (preferably filtered).
    fs : float
        Sampling frequency in Hz.
    distance_seconds : float
        Minimum physiological refractory period between consecutive beats (default 0.25s).
    prominence : Optional[float]
        Peak prominence threshold. If None, derived adaptively from the signal envelope.
    method : str
        Peak detection strategy ('pan_tompkins' or 'find_peaks').

    Returns
    -------
    np.ndarray
        1-D array of detected R-peak sample indices.
    """
    sig = _validate_signal(signal)
    min_distance = max(1, int(round(distance_seconds * fs)))

    if method == "pan_tompkins":
        # 1. First-order difference approximation of derivative
        diff = np.diff(sig, prepend=sig[0])
        # 2. Squaring to accentuate high-frequency QRS slope
        squared = diff ** 2
        # 3. Moving-window integration (~150 ms window)
        window_len = max(1, int(round(0.15 * fs)))
        kernel = np.ones(window_len) / window_len
        integrated = np.convolve(squared, kernel, mode="same")

        # 4. Adaptive prominence threshold
        if prominence is None:
            prominence = float(np.mean(integrated) + 0.5 * np.std(integrated))

        # 5. Candidate peaks in energy envelope
        candidate_peaks, _ = find_peaks(
            integrated, distance=min_distance, prominence=prominence
        )

        # 6. Local search back in the ECG signal for exact R-peak maximum
        refined_peaks = []
        search_radius = max(1, int(round(0.08 * fs)))  # ~80 ms search window
        sig_len = len(sig)

        for p in candidate_peaks:
            w_start = max(0, p - search_radius)
            w_end = min(sig_len, p + search_radius + 1)
            # Find the index of maximum absolute deflection
            local_segment = sig[w_start:w_end]
            if len(local_segment) > 0:
                local_peak = w_start + int(np.argmax(local_segment))
                refined_peaks.append(local_peak)

        if not refined_peaks:
            return np.empty((0,), dtype=np.int64)

        # Remove duplicate peak detections that fall within min_distance
        refined_peaks = np.sort(np.unique(refined_peaks))
        clean_peaks = [refined_peaks[0]]
        for p in refined_peaks[1:]:
            if p - clean_peaks[-1] >= min_distance:
                clean_peaks.append(p)

        return np.asarray(clean_peaks, dtype=np.int64)

    else:
        # Standard find_peaks fallback
        if prominence is None:
            prominence = float(0.5 * np.std(sig))
        peaks, _ = find_peaks(sig, distance=min_distance, prominence=prominence)
        return peaks.astype(np.int64)


# ---------------------------------------------------------------------
# Step 3: Beat Window Extraction (360 samples = 1.0 s)
# ---------------------------------------------------------------------

def segment_beats(
    signal: np.ndarray,
    r_peaks: Sequence[int],
    fs: float = 360.0,
    window_size: int = 360,
    pre_r_samples: int = 180,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract fixed 360-sample (1.0 s) windows centered around R-peaks.

    Parameters
    ----------
    signal : np.ndarray
        ECG signal.
    r_peaks : Sequence[int]
        Sample indices of R-peaks.
    fs : float
        Sampling frequency (default 360 Hz).
    window_size : int
        Total length of extracted beat segment (default 360 samples = 1.0 s).
    pre_r_samples : int
        Number of samples preceding the R-peak (default 180 samples = 0.5 s).

    Returns
    -------
    beats : np.ndarray
        Array of shape (num_valid_beats, window_size).
    valid_peaks : np.ndarray
        Indices of R-peaks corresponding to valid segments.
    """
    sig = _validate_signal(signal)
    post_r_samples = window_size - pre_r_samples

    if pre_r_samples < 0 or post_r_samples < 0:
        raise ValueError("pre_r_samples must satisfy 0 <= pre_r_samples <= window_size.")

    beats = []
    valid_peaks = []
    sig_len = len(sig)

    for peak in r_peaks:
        start = int(peak - pre_r_samples)
        end = int(peak + post_r_samples)

        # Discard beats that truncate at signal boundaries
        if start < 0 or end > sig_len:
            continue

        beat = sig[start:end]
        if len(beat) != window_size:
            continue

        beats.append(beat)
        valid_peaks.append(peak)

    if not beats:
        return np.empty((0, window_size), dtype=np.float64), np.empty((0,), dtype=np.int64)

    return np.asarray(beats, dtype=np.float64), np.asarray(valid_peaks, dtype=np.int64)


# ---------------------------------------------------------------------
# Step 4: Beat Normalization z = (x - mu) / sigma
# ---------------------------------------------------------------------

def normalize_beat(
    beat: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Normalize an individual ECG beat using z-score normalization:
        z = (x - mu) / sigma

    Parameters
    ----------
    beat : np.ndarray
        ECG beat segment (1-D or 2-D array of beats).
    epsilon : float
        Small positive constant to prevent division by zero for near-constant segments.

    Returns
    -------
    np.ndarray
        Normalized beat with mean ~ 0 and std ~ 1.
    """
    beat = np.asarray(beat, dtype=np.float64)
    if beat.ndim == 1:
        mu = np.mean(beat)
        sigma = np.std(beat)
        if sigma < epsilon:
            return beat - mu
        return (beat - mu) / sigma
    elif beat.ndim == 2:
        mu = np.mean(beat, axis=-1, keepdims=True)
        sigma = np.std(beat, axis=-1, keepdims=True)
        safe_sigma = np.where(sigma < epsilon, 1.0, sigma)
        norm = (beat - mu) / safe_sigma
        # For segments where sigma < epsilon, just center
        constant_mask = (sigma < epsilon).squeeze(-1)
        if np.any(constant_mask):
            norm[constant_mask] = beat[constant_mask] - mu[constant_mask]
        return norm
    else:
        raise ValueError(f"Expected 1-D or 2-D beat array, got ndim={beat.ndim}.")


# Backward compatibility alias
zscore_normalize = normalize_beat


# ---------------------------------------------------------------------
# Step 5: Linear Interpolation Resampling to n=32 Features
# ---------------------------------------------------------------------

def resample_beat(
    beat: np.ndarray,
    target_length: int = 32,
    method: str = "scipy",
) -> np.ndarray:
    """
    Resample a single beat to n=32 features using linear interpolation.

    Parameters
    ----------
    beat : np.ndarray
        1-D beat signal of original length (e.g. 360 samples).
    target_length : int
        Target number of features (n=32).
    method : str
        'scipy' for scipy.interpolate.interp1d or 'torch' for torch.nn.functional.interpolate.

    Returns
    -------
    np.ndarray
        1-D array of shape (target_length,).
    """
    orig_len = len(beat)
    if orig_len == target_length:
        return beat.copy()

    if method == "torch" and TORCH_AVAILABLE:
        t = torch.from_numpy(beat).unsqueeze(0).unsqueeze(0).float()
        res = F.interpolate(t, size=target_length, mode="linear", align_corners=True)
        return res.squeeze().numpy().astype(np.float64)
    else:
        # scipy linear interpolation
        x_old = np.linspace(0.0, 1.0, orig_len)
        x_new = np.linspace(0.0, 1.0, target_length)
        f = interpolate.interp1d(x_old, beat, kind="linear")
        return f(x_new).astype(np.float64)


def resample_beats_batch(
    beats: np.ndarray,
    target_length: int = 32,
    method: str = "scipy",
) -> np.ndarray:
    """
    Resample a batch of beats (shape: num_beats x orig_length) to (num_beats x target_length).

    Parameters
    ----------
    beats : np.ndarray
        2-D array of shape (num_beats, orig_length).
    target_length : int
        Target feature length (32).
    method : str
        'scipy' or 'torch'.

    Returns
    -------
    np.ndarray
        2-D array of shape (num_beats, target_length).
    """
    beats = np.asarray(beats, dtype=np.float64)
    if beats.size == 0 or beats.shape[0] == 0:
        return np.empty((0, target_length), dtype=np.float64)

    num_beats, orig_len = beats.shape
    if orig_len == target_length:
        return beats.copy()

    if method == "torch" and TORCH_AVAILABLE:
        t = torch.from_numpy(beats).unsqueeze(1).float()  # (N, 1, L)
        res = F.interpolate(t, size=target_length, mode="linear", align_corners=True)
        return res.squeeze(1).numpy().astype(np.float64)
    else:
        x_old = np.linspace(0.0, 1.0, orig_len)
        x_new = np.linspace(0.0, 1.0, target_length)
        f = interpolate.interp1d(x_old, beats, axis=1, kind="linear")
        return f(x_new).astype(np.float64)


# Backward compatibility alias
extract_feature = resample_beats_batch


# ---------------------------------------------------------------------
# Step 6: AAMI Annotation Mapping and Beat Extraction
# ---------------------------------------------------------------------

def map_annotation_symbol(symbol: str) -> Optional[str]:
    """
    Map an individual MIT-BIH annotation symbol to standard AAMI classes
    (N, SVEB, VEB, F, Q). Returns None for non-beat annotations.
    """
    return AAMI_MAPPING.get(symbol, None)


def extract_beat_segments(
    signal: np.ndarray,
    annotation_samples: Sequence[int],
    annotation_symbols: Sequence[str],
    fs: float = 360.0,
    window_size: int = 360,
    pre_r_samples: int = 180,
    before_seconds: Optional[float] = None,
    after_seconds: Optional[float] = None,
    normalize: bool = True,
    target_features: int = 32,
    resample_method: str = "scipy",
    return_all: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, List[str]],
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray],
]:
    """
    Extract fixed 360-sample (1.0 s) windows around annotated R-peaks,
    normalize (z-score), and resample to n=32 features via linear interpolation.

    Supports both legacy 3-element return (X, y, symbols) and full 5-element return
    (resampled_beats, raw_beats, labels, symbols, valid_peaks).
    """
    sig = _validate_signal(signal)

    # Handle before_seconds and after_seconds if provided
    if before_seconds is not None:
        pre_r_samples = int(round(before_seconds * fs))
    if after_seconds is not None:
        post_r_samples = int(round(after_seconds * fs))
        window_size = pre_r_samples + post_r_samples
    else:
        post_r_samples = window_size - pre_r_samples

    sig_len = len(sig)

    extracted_raw = []
    labels = []
    orig_symbols = []
    valid_peaks = []

    for sample, sym in zip(annotation_samples, annotation_symbols):
        aami_class = map_annotation_symbol(sym)
        if aami_class is None:
            # Skip non-beat annotations (+, ~, !, ", |, x, [, ])
            continue

        start = int(sample - pre_r_samples)
        end = int(sample + post_r_samples)

        if start < 0 or end > sig_len:
            continue

        beat = sig[start:end]
        if len(beat) != window_size:
            continue

        extracted_raw.append(beat)
        labels.append(CLASS_TO_ID[aami_class])
        orig_symbols.append(sym)
        valid_peaks.append(sample)

    if not extracted_raw:
        empty_res = np.empty((0, target_features), dtype=np.float64)
        empty_raw = np.empty((0, window_size), dtype=np.float64)
        empty_labels = np.empty((0,), dtype=np.int64)
        empty_peaks = np.empty((0,), dtype=np.int64)
        if return_all:
            return empty_res, empty_raw, empty_labels, [], empty_peaks
        return empty_res, empty_labels, []

    raw_beats_arr = np.asarray(extracted_raw, dtype=np.float64)

    if normalize:
        raw_beats_arr = normalize_beat(raw_beats_arr)

    # Resample to n=32 features
    resampled_beats_arr = resample_beats_batch(
        raw_beats_arr, target_length=target_features, method=resample_method
    )

    if normalize:
        # Standardize the resampled 32-feature representation
        resampled_beats_arr = normalize_beat(resampled_beats_arr)

    labels_arr = np.asarray(labels, dtype=np.int64)
    peaks_arr = np.asarray(valid_peaks, dtype=np.int64)

    if return_all:
        return (
            resampled_beats_arr,
            raw_beats_arr,
            labels_arr,
            orig_symbols,
            peaks_arr,
        )

    return resampled_beats_arr, labels_arr, orig_symbols


# ---------------------------------------------------------------------
# Record Processing Function
# ---------------------------------------------------------------------

def process_record(
    record_id: str,
    raw_dir: Union[str, Path] = "data/raw/mit-bih-arrhythmia-database-1.0.0",
    channel_index: int = 0,
    fs: float = 360.0,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 3,
    window_size: int = 360,
    pre_r_samples: int = 180,
    target_features: int = 32,
    resample_method: str = "scipy",
) -> Dict[str, Any]:
    """
    Process a single MIT-BIH record through the complete preprocessing pipeline.
    """
    raw_dir = Path(raw_dir)
    rec = load_record(raw_dir, record_id)
    ann = load_annotations(raw_dir, record_id)

    raw_signal = rec.p_signal[:, channel_index]
    actual_fs = float(rec.fs) if rec.fs else fs

    # 1. Bandpass filter
    filtered_signal = bandpass_filter(
        raw_signal, fs=actual_fs, lowcut=lowcut, highcut=highcut, order=order
    )

    # 2. Extract beats
    resampled_beats, raw_beats, labels, symbols, valid_peaks = extract_beat_segments(
        signal=filtered_signal,
        annotation_samples=ann.sample,
        annotation_symbols=ann.symbol,
        fs=actual_fs,
        window_size=window_size,
        pre_r_samples=pre_r_samples,
        normalize=True,
        target_features=target_features,
        resample_method=resample_method,
        return_all=True,
    )

    return {
        "record_id": record_id,
        "raw_signal": raw_signal,
        "filtered_signal": filtered_signal,
        "fs": actual_fs,
        "resampled_beats": resampled_beats,
        "raw_beats": raw_beats,
        "labels": labels,
        "symbols": symbols,
        "valid_peaks": valid_peaks,
        "annotation_samples": ann.sample,
        "annotation_symbols": ann.symbol,
    }


# ---------------------------------------------------------------------
# Step 7: Patient-Level Split Pipeline Across All Records
# ---------------------------------------------------------------------

def process_all_records(
    record_ids: Optional[Sequence[str]] = None,
    raw_dir: Union[str, Path] = "data/raw/mit-bih-arrhythmia-database-1.0.0",
    channel_index: int = 0,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 3,
    window_size: int = 360,
    pre_r_samples: int = 180,
    target_features: int = 32,
    resample_method: str = "scipy",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    processed_dir: Union[str, Path] = "data/processed",
    eda_dir: Optional[Union[str, Path]] = "results/figures/eda",
) -> Dict[str, Any]:
    """
    Process all MIT-BIH records, enforce strict patient-level split (by recording IDs),
    save processed arrays and metadata, and produce EDA verification plots.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    beats_dir = processed_dir / "beats"
    metadata_dir = processed_dir / "metadata"
    beats_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if eda_dir:
        eda_path = Path(eda_dir)
        eda_path.mkdir(parents=True, exist_ok=True)
    else:
        eda_path = None

    if record_ids is None:
        record_ids = get_record_ids(raw_dir)

    logger.info(f"Loaded {len(record_ids)} recording IDs.")

    # 1. Enforce strict patient-level split (by recording IDs)
    train_recs, val_recs, test_recs = split_records(
        record_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    logger.info(
        f"Patient-level split: {len(train_recs)} train, {len(val_recs)} val, {len(test_recs)} test."
    )

    # Validate zero patient overlap
    train_set, val_set, test_set = set(train_recs), set(val_recs), set(test_recs)
    assert train_set.isdisjoint(val_set), "Train and Val records overlap!"
    assert train_set.isdisjoint(test_set), "Train and Test records overlap!"
    assert val_set.isdisjoint(test_set), "Val and Test records overlap!"

    split_map = {}
    for r in train_recs:
        split_map[r] = "train"
    for r in val_recs:
        split_map[r] = "val"
    for r in test_recs:
        split_map[r] = "test"

    split_data = {
        "train": {"X": [], "X_raw": [], "y": [], "record_ids": [], "symbols": []},
        "val": {"X": [], "X_raw": [], "y": [], "record_ids": [], "symbols": []},
        "test": {"X": [], "X_raw": [], "y": [], "record_ids": [], "symbols": []},
    }

    # Store exemplar beat for each AAMI class for plotting
    class_raw_examples: Dict[str, np.ndarray] = {}
    class_res_examples: Dict[str, np.ndarray] = {}
    sample_record_result: Optional[Dict[str, Any]] = None

    # 2. Process all records
    for i, rec_id in enumerate(record_ids):
        res = process_record(
            record_id=rec_id,
            raw_dir=raw_dir,
            channel_index=channel_index,
            lowcut=lowcut,
            highcut=highcut,
            order=order,
            window_size=window_size,
            pre_r_samples=pre_r_samples,
            target_features=target_features,
            resample_method=resample_method,
        )

        # Cache first record for filter and peak detection plots
        if sample_record_result is None:
            sample_record_result = res

        split_name = split_map[rec_id]
        n_beats = len(res["labels"])

        if n_beats > 0:
            split_data[split_name]["X"].append(res["resampled_beats"])
            split_data[split_name]["X_raw"].append(res["raw_beats"])
            split_data[split_name]["y"].append(res["labels"])
            split_data[split_name]["record_ids"].extend([rec_id] * n_beats)
            split_data[split_name]["symbols"].extend(res["symbols"])

            # Check for class exemplars
            for cls_name in AAMI_CLASSES:
                if cls_name not in class_raw_examples:
                    cls_id = CLASS_TO_ID[cls_name]
                    match_idx = np.where(res["labels"] == cls_id)[0]
                    if len(match_idx) > 0:
                        class_raw_examples[cls_name] = res["raw_beats"][match_idx[0]]
                        class_res_examples[cls_name] = res["resampled_beats"][match_idx[0]]

        logger.info(
            f"[{i + 1:02d}/{len(record_ids):02d}] Record {rec_id} ({split_name:5s}): {n_beats:5d} valid beats."
        )

    # 3. Concatenate and save splits
    results_summary = {"total_beats": 0, "splits": {}}
    all_X, all_X_raw, all_y, all_records, all_symbols = [], [], [], [], []

    for s_name in ["train", "val", "test"]:
        s_X = np.concatenate(split_data[s_name]["X"], axis=0) if split_data[s_name]["X"] else np.empty((0, target_features))
        s_X_raw = np.concatenate(split_data[s_name]["X_raw"], axis=0) if split_data[s_name]["X_raw"] else np.empty((0, window_size))
        s_y = np.concatenate(split_data[s_name]["y"], axis=0) if split_data[s_name]["y"] else np.empty((0,), dtype=np.int64)
        s_recs = np.asarray(split_data[s_name]["record_ids"])
        s_syms = np.asarray(split_data[s_name]["symbols"])

        # Save split arrays
        np.save(beats_dir / f"X_{s_name}.npy", s_X)
        np.save(beats_dir / f"X_raw_{s_name}.npy", s_X_raw)
        np.save(beats_dir / f"y_{s_name}.npy", s_y)
        np.save(beats_dir / f"record_ids_{s_name}.npy", s_recs)
        np.save(beats_dir / f"symbols_{s_name}.npy", s_syms)

        # Collect global arrays
        if len(s_X) > 0:
            all_X.append(s_X)
            all_X_raw.append(s_X_raw)
            all_y.append(s_y)
            all_records.append(s_recs)
            all_symbols.append(s_syms)

        # Class breakdown
        class_counts = {c: int(np.sum(s_y == CLASS_TO_ID[c])) for c in AAMI_CLASSES}
        results_summary["splits"][s_name] = {
            "num_patients": len(train_recs if s_name == "train" else (val_recs if s_name == "val" else test_recs)),
            "patient_ids": train_recs if s_name == "train" else (val_recs if s_name == "val" else test_recs),
            "num_beats": int(len(s_y)),
            "class_counts": class_counts,
        }
        results_summary["total_beats"] += int(len(s_y))

    # Save combined dataset arrays
    full_X = np.concatenate(all_X, axis=0)
    full_X_raw = np.concatenate(all_X_raw, axis=0)
    full_y = np.concatenate(all_y, axis=0)
    full_records = np.concatenate(all_records, axis=0)
    full_symbols = np.concatenate(all_symbols, axis=0)

    np.save(beats_dir / "X.npy", full_X)
    np.save(beats_dir / "X_raw.npy", full_X_raw)
    np.save(beats_dir / "y.npy", full_y)
    np.save(beats_dir / "record_ids.npy", full_records)
    np.save(beats_dir / "symbols.npy", full_symbols)

    logger.info(
        f"Total heartbeats extracted: {full_X.shape[0]} | Feature dimension: {full_X.shape[1]}"
    )

    # 4. Save metadata files
    patient_splits_meta = {
        "seed": seed,
        "train_patients": train_recs,
        "val_patients": val_recs,
        "test_patients": test_recs,
        "summary": results_summary,
    }
    with open(metadata_dir / "patient_splits.json", "w") as f:
        json.dump(patient_splits_meta, f, indent=2)

    # Class distribution CSV
    class_dist_rows = []
    for s_name in ["train", "val", "test"]:
        row = {"split": s_name, "total_beats": results_summary["splits"][s_name]["num_beats"]}
        row.update(results_summary["splits"][s_name]["class_counts"])
        class_dist_rows.append(row)
    # Full dataset row
    total_row = {"split": "all", "total_beats": results_summary["total_beats"]}
    for c in AAMI_CLASSES:
        total_row[c] = int(np.sum(full_y == CLASS_TO_ID[c]))
    class_dist_rows.append(total_row)
    pd.DataFrame(class_dist_rows).to_csv(metadata_dir / "class_distribution.csv", index=False)

    # 5. Save intermediate EDA verification plots
    if eda_path and sample_record_result:
        # Plot 1: Raw vs Filtered ECG
        plot_raw_vs_filtered(
            raw_signal=sample_record_result["raw_signal"],
            filtered_signal=sample_record_result["filtered_signal"],
            fs=sample_record_result["fs"],
            start_seconds=0.0,
            duration_seconds=10.0,
            title=f"Record {sample_record_result['record_id']} Raw vs Filtered (0.5–40 Hz Butterworth Order 3)",
            save_path=eda_path / "raw_vs_filtered_ecg.png",
        )

        # Plot 2: R-peak Detection
        detected_peaks = detect_r_peaks(sample_record_result["filtered_signal"], fs=sample_record_result["fs"])
        plot_r_peaks(
            signal=sample_record_result["filtered_signal"],
            fs=sample_record_result["fs"],
            r_peaks=detected_peaks,
            annotated_peaks=sample_record_result["annotation_samples"],
            annotated_symbols=sample_record_result["annotation_symbols"],
            start_seconds=0.0,
            duration_seconds=10.0,
            title=f"Record {sample_record_result['record_id']} Filtered Signal with R-peaks and AAMI Annotations",
            save_path=eda_path / "r_peak_detection.png",
        )

        # Plot 3: AAMI Class Examples (360 samples vs 32 features)
        plot_aami_beat_classes(
            class_raw_beats=class_raw_examples,
            class_resampled_beats=class_res_examples,
            save_path=eda_path / "aami_class_examples.png",
        )

        # Plot 4: Patient Split Distribution
        split_counts = {
            s_name: results_summary["splits"][s_name]["class_counts"]
            for s_name in ["train", "val", "test"]
        }
        plot_patient_split_distribution(
            split_counts=split_counts,
            save_path=eda_path / "patient_split_distribution.png",
        )

        logger.info(f"Intermediate verification plots saved to {eda_path}")

    return {
        "X": full_X,
        "y": full_y,
        "record_ids": full_records,
        "symbols": full_symbols,
        "summary": results_summary,
    }


# ---------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess MIT-BIH Arrhythmia ECG dataset into normalized, resampled AAMI beats."
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
        "--raw_dir",
        type=str,
        default=None,
        help="Path to raw MIT-BIH dataset directory.",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=None,
        help="Output directory for processed arrays.",
    )
    parser.add_argument(
        "--resample_method",
        type=str,
        choices=["scipy", "torch"],
        default=None,
        help="Linear interpolation method ('scipy' or 'torch').",
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

    raw_dir = args.raw_dir or config.get("dataset", {}).get("raw_dir", "data/raw/mit-bih-arrhythmia-database-1.0.0")
    processed_dir = args.processed_dir or config.get("dataset", {}).get("processed_dir", "data/processed")
    eda_dir = config.get("figures", {}).get("eda_dir", "results/figures/eda")

    channel_index = config.get("dataset", {}).get("channel", 0)
    filter_cfg = config.get("preprocessing", {}).get("filter", {})
    order = filter_cfg.get("order", 3)
    lowcut = filter_cfg.get("lowcut", 0.5)
    highcut = filter_cfg.get("highcut", 40.0)

    window_cfg = config.get("preprocessing", {}).get("window", {})
    window_size = window_cfg.get("window_size", 360)
    pre_r_samples = window_cfg.get("pre_r_samples", 180)

    resample_cfg = config.get("preprocessing", {}).get("resampling", {})
    target_features = resample_cfg.get("target_features", 32)
    resample_method = args.resample_method or resample_cfg.get("method", "scipy")

    split_cfg = config.get("split", {})
    train_ratio = split_cfg.get("train_ratio", 0.70)
    val_ratio = split_cfg.get("val_ratio", 0.15)
    test_ratio = split_cfg.get("test_ratio", 0.15)

    logger.info("Executing ECG preprocessing pipeline...")
    logger.info(
        f"Filter: Order {order} Butterworth ({lowcut}–{highcut} Hz) | Window: {window_size} samples (1.0 s) | Features: n={target_features}"
    )

    process_all_records(
        raw_dir=raw_dir,
        channel_index=channel_index,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
        window_size=window_size,
        pre_r_samples=pre_r_samples,
        target_features=target_features,
        resample_method=resample_method,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        processed_dir=processed_dir,
        eda_dir=eda_dir,
    )

    logger.info("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
