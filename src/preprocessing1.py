from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.signal import butter, filtfilt


# ---------------------------------------------------------------------
# AAMI-style annotation mapping
# ---------------------------------------------------------------------

AAMI_MAPPING = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",

    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",

    "V": "V",
    "E": "V",

    "F": "F",

    "/": "Q",
    "f": "Q",
    "Q": "Q",
}


def map_annotation_symbol(
    symbol: str
) -> Optional[str]:
    """
    Map a MIT-BIH annotation symbol
    to an AAMI-style class.

    Returns None for symbols that
    are not part of the selected
    heartbeat classes.
    """

    return AAMI_MAPPING.get(
        symbol,
        None
    )


# ---------------------------------------------------------------------
# ECG filtering
# ---------------------------------------------------------------------

def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional ECG signal.

    fs : float
        Sampling frequency.

    lowcut : float
        Lower cutoff frequency.

    highcut : float
        Upper cutoff frequency.

    order : int
        Butterworth filter order.
    """

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype="band"
    )

    filtered = filtfilt(
        b,
        a,
        signal
    )

    return filtered


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def zscore_normalize(
    signal: np.ndarray
) -> np.ndarray:
    """
    Normalize one ECG segment.
    """

    mean = np.mean(signal)
    std = np.std(signal)

    if std < 1e-8:
        return signal - mean

    return (
        signal - mean
    ) / std


# ---------------------------------------------------------------------
# Beat extraction
# ---------------------------------------------------------------------

def extract_beat_segments(
    signal: np.ndarray,
    annotation_samples: np.ndarray,
    annotation_symbols: List[str],
    fs: float,
    before_seconds: float = 0.2,
    after_seconds: float = 0.4,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract fixed-length ECG beat segments
    around annotation locations.

    Returns
    -------
    X : np.ndarray
        Shape: (number_of_beats, segment_length)

    y : np.ndarray
        Integer class labels.

    symbols : list[str]
        Original annotation symbols.
    """

    before_samples = int(
        before_seconds * fs
    )

    after_samples = int(
        after_seconds * fs
    )

    segment_length = (
        before_samples
        + after_samples
    )

    X = []
    y = []
    original_symbols = []

    class_to_id = {
        "N": 0,
        "S": 1,
        "V": 2,
        "F": 3,
        "Q": 4,
    }

    for sample, symbol in zip(
        annotation_samples,
        annotation_symbols
    ):

        class_name = map_annotation_symbol(
            symbol
        )

        if class_name is None:
            continue

        start = (
            sample
            - before_samples
        )

        end = (
            sample
            + after_samples
        )

        # Ignore annotations too close
        # to the beginning or end.
        if start < 0:
            continue

        if end > len(signal):
            continue

        segment = signal[
            start:end
        ]

        if len(segment) != segment_length:
            continue

        if normalize:
            segment = zscore_normalize(
                segment
            )

        X.append(segment)

        y.append(
            class_to_id[class_name]
        )

        original_symbols.append(
            symbol
        )

    if not X:
        raise ValueError(
            "No valid heartbeat segments extracted."
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        original_symbols
    )


def process_all_records(
    records,
    data_dir,
    channel_index=0,
    before_seconds=0.2,
    after_seconds=0.4,
):
    """
    Process all records and return
    concatenated ECG beat dataset.
    """

    from src.dataset import (
        load_record,
        load_annotations
    )

    all_X = []
    all_y = []
    all_record_ids = []
    all_symbols = []

    for record_name in records:

        print(
            f"Processing record {record_name}..."
        )

        record = load_record(
            record_name,
            data_dir
        )

        annotation = load_annotations(
            record_name,
            data_dir
        )

        signal = record.p_signal[
            :, channel_index
        ]

        signal = bandpass_filter(
            signal,
            record.fs
        )

        X, y, symbols = extract_beat_segments(
            signal,
            annotation.sample,
            annotation.symbol,
            record.fs,
            before_seconds,
            after_seconds,
            normalize=True
        )

        all_X.append(X)
        all_y.append(y)

        all_record_ids.extend(
            [record_name] * len(y)
        )

        all_symbols.extend(
            symbols
        )

    X = np.concatenate(
        all_X,
        axis=0
    )

    y = np.concatenate(
        all_y,
        axis=0
    )

    record_ids = np.asarray(
        all_record_ids
    )

    symbols = np.asarray(
        all_symbols
    )

    return (
        X,
        y,
        record_ids,
        symbols
    )