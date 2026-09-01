"""
ECG preprocessing utilities for the MIT-BIH Arrhythmia Database


Pipeline:
    Raw ECG
        ↓
    Band-pass filtering
        ↓
    R-peak detection
        ↓
    Beat segmentation
        ↓
    Beat normalization
        ↓
    Fixed-length representation
"""

from __future__ import annotations
from typing import Optional, Sequence

import numpy as np 
from scipy import signal
from scipy.signal import butter,filtfilt,find_peaks,resample

# default preprocessing parameters

DEFAULT_LOW_CUT = 0.5
DEFAULT_HIGH_CUT = 40.0
DEFAULT_FILTER_ORDER = 4

DEFAULT_BEAT_BEFORE = 0.2
DEFAULT_BEAT_AFTER = 0.4

DEFAULT_FEATURE_LENGTH = 32

# Validation helpers
def _validate_signal(signal:np.ndarray) -> np.ndarray:
    """validate and convert  an ECG signal to a 1-D float array"""
    signal = np.asarray(signal,dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError(f"Expecteda 1-D ECG signal ,got shape{signal.shape}.")

    if signal.size == 0:
        raise ValueError("ECG signal is empty.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("ECG signal contains NaN or infinite values.")

    return signal 


# step 1 : Band-pass filtering

def bandpass_filter(
        signal : np.ndarray,
        fs: float,
        low_cut: float = DEFAULT_LOW_CUT,
        high_cut: float =DEFAULT_HIGH_CUT,
        order: int = DEFAULT_FILTER_ORDER,
) -> np.ndarray:
    """
    Apply a Butterwoth band-pass filter to an ECG signal .

    parametr:
    --------------
    signal : 1-D ECG signal
    fs: Sampling frequency in Hz
    low_cut : Lower cutoff frequency in Hz
    high_cut : upper cutoff frequency in Hz
    order: Butterwoth filter order.

    returns:
    ---------------
    np.ndarray: Filter ECG signal.

    """

    signal = _validate_signal(signal)

    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")

    if not 0 < low_cut < high_cut < fs /2 :
        raise ValueError(
            "Cutoff Frequency must satisfy "
            "0 < low_cut < high_cut < Nyquist frequency."
        )

    if order < 1: 
        raise ValueError("Filter order Must be > =1 ")

    nyquist = fs /2.0
    low = low_cut /nyquist
    high = high_cut/nyquist

    b,a = butter(order,[low,high],btype='bandpass')
    return filtfilt(b,a,signal)


# Step 2 : R-peak detection

def detect_r_peaks(
    signal: np.ndarray,
    fs: float,
    distance_seconds: float = 0.25,
    prominance: Optional[float] = None,   
) -> np.ndarray:
    """"
    Detect candidate R-peaks in an ECG signal.

    This funcion provides the initial peak-detection interface.
    Validate against MIT_BIH  annotations will be perfoms.

    """

    signal = _validate_signal(signal)

    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")

    distance_samples = max(
        1,int(distance_seconds*fs),
    )

    peaks, _ = find_peaks(
        signal,
        distance=distance_samples,
        prominence=prominance,
    )

    return peaks.astype(np.int64)


# Step 3 : Beat Segmentation

def segment_beats(
    signal: np.ndarray,
    r_peaks : Sequence[int],
    fs: float,
    before : float = DEFAULT_BEAT_BEFORE,
    after : float = DEFAULT_BEAT_AFTER,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract fixed-window beats around  detected p-peaks.

    parameters:
    ---------------
    signal: 1-D ECG signal.
    r_peaks: Sample indices of detected R-peaks.
    fs: Sampling Frequency.
    before : seconds before each R-peak.
    after : Second after each R-peaks.

    return 
    -------
    beats: Array with Shape (number_of_valid_beats, Beats_length).
    valid-peaks:
    R-peaks positions corresponding to returned beats
    """

    signal = _validate_signal(signal)
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")

    if before <= 0 or after <= 0 :
        raise ValueError("Before and after must be positive.")

    r_peaks = np.array(r_peaks,dtype=np.int64)

    before_samples = int (round(before * fs))
    after_samples = int (round(after * fs))

    beats = []
    valid_peaks = []

    for peak in r_peaks:
        start = peak - before_samples
        end = peak + after_samples

        if start < 0 or  end > len(signal):
            continue

        beat = signal[start:end]
        if len(beat) != before_samples + after_samples:
            continue

        beats.append(beat)
        valid_peaks.append(peak)

    if not beats:
        return (
            np.empty((0, before_samples + after_samples)),
            np.empty((0,), dtype=np.int64),
        )    

    return (
        np.asarray(beats, dtype=np.float64),
        np.asarray(valid_peaks, dtype=np.int64),
    )


# Step 4 : Beat Normalization 

def normalize_signal(
        signal : np.ndarray,
        epsilon: float = 1e-8
 ) -> np.ndarra:
    """
    Normalize an ECG signal using z-score normalization.

    Parameters
    ----------
    signal : np.ndarray
        Input ECG signal.
    epsilon : float, optional
        Small value to avoid division by zero (default is 1e-8).

    Returns
    -------
    np.ndarray
        Normalized ECG signal.
        Normally have :
        mean = 0
        standard deviation = 1
    """
    signal  = _validate_signal(signal)
    mean = np.mean(signal)
    std = np.std(signal)

    if std < epsilon:
        return signal - mean

    return (signal - mean) / std


# Step 5 : Fixed-length representation

def extract_feature(
        beats: np.ndarray,
        feature_length: int = DEFAULT_FEATURE_LENGTH
) -> np.ndarray:
    """
    convert the variable -length beats into fixed -length representation .

    Each beat is resampled to 'feature_length' samples.
    """
    beats = np.asarray(beats, dtype=np.float64)

    if beats.ndim != 2:
        raise ValueError(f"Expected 2-D array of beats, got shape {beats.shape}.")

    if feature_length <= 0:
        raise ValueError("Feature length must be positive.")

    if beats.shape[0] == 0:
        return np.empty((0, feature_length), dtype=np.float64)

    features = np.asarray(
        [resample(beat, feature_length) for beat in beats],
        dtype=np.float64,
    )

    return features