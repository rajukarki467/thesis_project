import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib
# Use Agg backend if non-interactive
if os.environ.get("DISPLAY") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_ecg(
    signal: np.ndarray,
    fs: float,
    channel_name: str = "ECG",
    start_seconds: float = 0,
    duration_seconds: float = 10,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Plot a section of an ECG signal.
    """
    start_sample = int(start_seconds * fs)
    end_sample = min(int((start_seconds + duration_seconds) * fs), len(signal))

    segment = signal[start_sample:end_sample]
    time = np.arange(len(segment)) / fs + start_seconds

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time, segment, linewidth=0.8, color="#1f77b4")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(title or f"{channel_name} ECG")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_raw_vs_filtered(
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    fs: float,
    start_seconds: float = 0.0,
    duration_seconds: float = 10.0,
    title: str = "Raw vs Butterworth Bandpass (0.5–40 Hz, Order 3) Filtered ECG",
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Compare raw vs Butterworth bandpass filtered ECG signal over a time window.
    """
    start_sample = int(start_seconds * fs)
    end_sample = min(int((start_seconds + duration_seconds) * fs), len(raw_signal))

    raw_seg = raw_signal[start_sample:end_sample]
    filt_seg = filtered_signal[start_sample:end_sample]
    time = np.arange(len(raw_seg)) / fs + start_seconds

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(time, raw_seg, color="#d62728", linewidth=0.9, label="Raw Signal")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].set_title(f"{title} - Raw Signal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(time, filt_seg, color="#2ca02c", linewidth=0.9, label="Filtered Signal (0.5-40 Hz, order 3)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude (mV)")
    axes[1].set_title("Filtered Signal (Baseline Wander & High-Freq Noise Removed)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_r_peaks(
    signal: np.ndarray,
    fs: float,
    r_peaks: np.ndarray,
    annotated_peaks: Optional[np.ndarray] = None,
    annotated_symbols: Optional[Sequence[str]] = None,
    start_seconds: float = 0.0,
    duration_seconds: float = 10.0,
    title: str = "ECG Signal with R-peak Detection",
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Plot filtered ECG signal with detected and annotated R-peaks.
    """
    start_sample = int(start_seconds * fs)
    end_sample = min(int((start_seconds + duration_seconds) * fs), len(signal))

    segment = signal[start_sample:end_sample]
    time = np.arange(len(segment)) / fs + start_seconds

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(time, segment, color="#1f77b4", linewidth=0.9, label="Filtered ECG")

    # Filter peaks in window
    peaks_in_window = [p for p in r_peaks if start_sample <= p < end_sample]
    if peaks_in_window:
        peak_times = np.array(peaks_in_window) / fs
        peak_vals = signal[peaks_in_window]
        ax.scatter(peak_times, peak_vals, color="#d62728", marker="o", s=60, label="Detected R-peaks", zorder=4)

    if annotated_peaks is not None and annotated_symbols is not None:
        for p, s in zip(annotated_peaks, annotated_symbols):
            if start_sample <= p < end_sample:
                ax.annotate(
                    s,
                    xy=(p / fs, signal[p]),
                    xytext=(0, 15),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    color="#1a237e",
                    arrowprops=dict(arrowstyle="->", color="#1a237e", lw=0.8),
                )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_aami_beat_classes(
    class_raw_beats: Dict[str, np.ndarray],
    class_resampled_beats: Dict[str, np.ndarray],
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Plot representative beats for each AAMI class (N, SVEB, VEB, F, Q),
    displaying both the 360-sample (1.0 s) normalized beat and the n=32 resampled representation.
    """
    classes = [c for c in ["N", "SVEB", "VEB", "F", "Q"] if c in class_raw_beats]
    n_classes = len(classes)
    if n_classes == 0:
        return

    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4), sharey=True)
    if n_classes == 1:
        axes = [axes]

    colors = {
        "N": "#2ca02c",
        "SVEB": "#ff7f0e",
        "VEB": "#d62728",
        "F": "#9467bd",
        "Q": "#8c564b",
    }

    for ax, cls in zip(axes, classes):
        raw = class_raw_beats[cls]
        res = class_resampled_beats[cls]
        t_raw = np.linspace(-0.5, 0.5, len(raw))
        t_res = np.linspace(-0.5, 0.5, len(res))
        c = colors.get(cls, "#333333")

        ax.plot(t_raw, raw, color=c, alpha=0.5, linewidth=1.2, label=f"360-pt ({cls})")
        ax.plot(t_res, res, color="black", linestyle="--", marker=".", markersize=4, linewidth=1.0, label="32-pt resampled")
        ax.set_title(f"AAMI Class: {cls}", fontweight="bold")
        ax.set_xlabel("Time (s) rel. to R-peak")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_ylabel("Normalized Amplitude (z-score)")
    fig.suptitle("AAMI Heartbeat Classes: 360-sample Windows vs 32-feature Linear Interpolation", y=1.02, fontsize=13)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_patient_split_distribution(
    split_counts: Dict[str, Dict[str, int]],
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Plot beat counts across AAMI classes for Train, Val, and Test patient-level splits.
    """
    df = pd.DataFrame(split_counts).fillna(0).astype(int)
    classes = [c for c in ["N", "SVEB", "VEB", "F", "Q"] if c in df.index]
    df = df.loc[classes]

    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind="bar", ax=ax, width=0.75, colormap="viridis")

    ax.set_title("Patient-Level Split: Beat Class Distribution (Train / Val / Test)", fontsize=13, fontweight="bold")
    ax.set_xlabel("AAMI Heartbeat Class", fontsize=11)
    ax.set_ylabel("Number of Heartbeats (log scale)", fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(title="Split")
    plt.xticks(rotation=0)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_client_distribution(
    client_distribution: pd.DataFrame,
    title: str = "Client Label Distribution",
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Plot client/class distribution as a heatmap.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        client_distribution,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": "Proportion of Client Data"},
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("AAMI Beat Class", fontsize=11)
    ax.set_ylabel("Client ID", fontsize=11)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_client_sample_counts(
    client_indices: Dict[int, Sequence[int]],
    alpha: float,
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Plot number of samples assigned to each client.
    """
    clients = sorted(client_indices.keys())
    counts = [len(client_indices[c]) for c in clients]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar([f"Client {c}" for c in clients], counts, color="#3f51b5", edgecolor="black", alpha=0.8)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(f"Client Sample Count Distribution (Dirichlet α = {alpha})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Total Beats Allocated")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_multi_alpha_comparison(
    distributions_by_alpha: Dict[float, pd.DataFrame],
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Side-by-side heatmaps illustrating non-IID class skew across Dirichlet alphas.
    """
    alphas = sorted(distributions_by_alpha.keys())
    n = len(alphas)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        df = distributions_by_alpha[alpha]
        sns.heatmap(df, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(f"Dirichlet α = {alpha}", fontweight="bold", fontsize=12)
        ax.set_xlabel("Class")

    axes[0].set_ylabel("Client ID")
    fig.suptitle("Federated Non-IID Class Heterogeneity Comparison Across Dirichlet Alphas", y=1.02, fontsize=14)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
