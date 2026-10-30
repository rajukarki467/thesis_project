import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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


# ==============================================================================
# Thesis Deliverable Figures (Chapter 4 of the Master Proposal)
# ==============================================================================

def plot_pareto_frontier(
    results_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Figure 4.1: Pareto Frontier (Accuracy vs. Communication Cost)",
):
    """
    Figure 4.1: Plot Global Test Accuracy (%) vs Communication Cost (Active Parameters / Round)
    across all baseline models and the adaptive KAN controller across Dirichlet skews.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color & marker palettes
    model_styles = {
        "FedAvg-MLP": {"color": "#e74c3c", "marker": "s", "label": "FedAvg-MLP (Fixed Budget)"},
        "FedAvg-KAN": {"color": "#3498db", "marker": "o", "label": "FedAvg-KAN (Fixed G=5, k=3)"},
        "FedProx-KAN": {"color": "#2ecc71", "marker": "^", "label": "FedProx-KAN (μ=0.01)"},
        "Heuristic-KAN": {"color": "#f39c12", "marker": "D", "label": "Heuristic-Adaptive KAN"},
        "PPO-Adaptive-KAN": {"color": "#9b59b6", "marker": "*", "label": "PPO-Adaptive KAN (Ours)"},
        "PPO-KAN": {"color": "#9b59b6", "marker": "*", "label": "PPO-Adaptive KAN (Ours)"},
    }

    alpha_alphas = {0.1: 1.0, 0.5: 0.75, 1.0: 0.5}

    df = results_df.copy()
    if "Accuracy (%)" not in df.columns and "Final Accuracy" in df.columns:
        df["Accuracy (%)"] = df["Final Accuracy"].apply(lambda v: v * 100.0 if v <= 1.0 else v)
    elif "Accuracy (%)" in df.columns:
        if df["Accuracy (%)"].max() <= 1.0:
            df["Accuracy (%)"] = df["Accuracy (%)"] * 100.0

    if "Params/Round" not in df.columns:
        if "Params/round" in df.columns:
            df["Params/Round"] = df["Params/round"]
        elif "Active Parameters" in df.columns:
            df["Params/Round"] = df["Active Parameters"]
        elif "mean_parameters_per_round" in df.columns:
            df["Params/Round"] = df["mean_parameters_per_round"]

    method_col = "Method" if "Method" in df.columns else "Algorithm"
    alpha_col = "Alpha" if "Alpha" in df.columns else "Dirichlet Alpha"

    for method, group in df.groupby(method_col):
        style = model_styles.get(method, {"color": "#7f8c8d", "marker": "o", "label": method})

        # Sort by parameters
        sorted_g = group.sort_values(by="Params/Round")
        x_vals = sorted_g["Params/Round"].values
        y_vals = sorted_g["Accuracy (%)"].values

        ax.plot(
            x_vals,
            y_vals,
            color=style["color"],
            linestyle="--",
            alpha=0.6,
            linewidth=1.5,
        )

        for _, row in group.iterrows():
            a_val = float(row[alpha_col])
            pt_alpha = alpha_alphas.get(a_val, 0.8)
            ax.scatter(
                row["Params/Round"],
                row["Accuracy (%)"],
                color=style["color"],
                marker=style["marker"],
                s=120,
                alpha=pt_alpha,
                edgecolors="black",
                linewidth=1.2,
                zorder=5,
            )
            # Label alpha
            ax.annotate(
                f"α={a_val}",
                xy=(row["Params/Round"], row["Accuracy (%)"]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color=style["color"],
            )

    # Custom legend for models
    handles = []
    for m, s in model_styles.items():
        if m in df[method_col].values:
            handles.append(
                plt.Line2D([0], [0], marker=s["marker"], color=s["color"], label=s["label"], markersize=9)
            )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Communication Cost: Transmitted Parameters per Round", fontsize=11, fontweight="bold")
    ax.set_ylabel("Held-Out Patient Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(handles=handles, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_spline_curves(
    client_models: Dict[str, Any],
    feature_indices: Sequence[int] = (0, 7, 15, 23),
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Figure 4.2: Spline Interpretability Across Heterogeneous Edge Profiles",
):
    """
    Figure 4.2: Extract and plot 1D B-spline activation curves phi(x) from client FastKAN layers.
    Compares low-complexity clients (G=3, G=5) against fine clients (G=10) on key ECG features.
    """
    import torch
    import torch.nn.functional as F

    profiles = list(client_models.keys())
    n_features = len(feature_indices)
    fig, axes = plt.subplots(len(profiles), n_features, figsize=(4 * n_features, 3 * len(profiles)), sharey=True)

    if len(profiles) == 1:
        axes = np.expand_dims(axes, 0)
    if n_features == 1:
        axes = np.expand_dims(axes, 1)

    x_domain = torch.linspace(-1.0, 1.0, steps=200)

    colors = {
        "Low (G=3, k=2)": "#3498db",
        "Medium (G=5, k=3)": "#2ecc71",
        "High (G=10, k=4)": "#e74c3c",
    }

    for row_idx, profile_name in enumerate(profiles):
        model = client_models[profile_name]
        c = colors.get(profile_name, "#2c3e50")
        first_layer = model.layers[0]

        with torch.no_grad():
            silu_x = F.silu(x_domain).numpy()

            for col_idx, feat_idx in enumerate(feature_indices):
                ax = axes[row_idx, col_idx]

                x_input = torch.zeros(len(x_domain), first_layer.in_features)
                x_input[:, feat_idx] = x_domain
                bases_feat = first_layer.compute_b_splines(x_input)[:, feat_idx, :].cpu().numpy()

                if first_layer.base_weight is not None:
                    wb = float(first_layer.base_weight[0, feat_idx].cpu())
                else:
                    wb = 1.0

                w_spline = first_layer.spline_weight[0, feat_idx].cpu().numpy()
                phi = wb * silu_x + np.dot(bases_feat, w_spline)

                ax.plot(x_domain.numpy(), phi, color=c, linewidth=2.0, label=r"$\phi_{0, i}(x)$")
                ax.plot(x_domain.numpy(), wb * silu_x, color="gray", linestyle=":", linewidth=1.0, label="Base SiLU")

                ax.set_title(f"Feat {feat_idx} | {profile_name}", fontsize=10, fontweight="bold")
                ax.set_xlabel("Normalized Input $x$", fontsize=9)
                ax.grid(True, alpha=0.3)
                if row_idx == 0 and col_idx == 0:
                    ax.legend(loc="upper left", fontsize=7)

        axes[row_idx, 0].set_ylabel(f"{profile_name}\nActivation $\\phi(x)$", fontsize=9, fontweight="bold")

    fig.suptitle(title, y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_classwise_f1_comparison(
    metrics_by_model: Dict[str, Dict[str, float]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Figure 4.3: Per-Class AAMI Detection F1-Scores Across Allocation Strategies",
):
    """
    Figure 4.3: Grouped bar chart comparing per-class detection (N, SVEB, VEB, F, Q)
    to verify minority classes (SVEB, VEB, F) are protected under adaptive scheduling.
    """
    classes = ["N", "SVEB", "VEB", "F", "Q"]
    models = list(metrics_by_model.keys())

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(classes))
    width = 0.8 / len(models)

    palette = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    for idx, (model_name, c_metrics) in enumerate(metrics_by_model.items()):
        f1_scores = [c_metrics.get(cls, 0.0) * 100.0 for cls in classes]
        offset = (idx - len(models) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            f1_scores,
            width,
            label=model_name,
            color=palette[idx % len(palette)],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
        )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("ANSI/AAMI EC57 Heartbeat Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("F1-Score (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_controller_overhead(
    overhead_data: Dict[str, float],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Figure 4.4: Computational Overhead & Execution Latency Breakdown",
):
    """
    Figure 4.4: Bar plot breakdown of communication round wall-clock time
    (Local training vs. Communication vs. RL inference overhead) and peak memory.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [2, 1]})

    components = [k for k in ["Local Client Training", "Server Aggregation", "RL Decision Overhead"] if k in overhead_data]
    if not components:
        components = list(overhead_data.keys())
    latencies = [overhead_data[k] for k in components]
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    bars = ax1.bar(components, latencies, color=colors[:len(components)], edgecolor="black", width=0.55)
    ax1.set_title("Round Latency Breakdown (ms)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Duration (milliseconds, log scale)", fontsize=10)
    ax1.set_yscale("log")
    ax1.grid(True, linestyle=":", alpha=0.5, which="both", axis="y")

    for bar, val in zip(bars, latencies):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.15,
            f"{val:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    mem_labels = ["Controller RAM", "Edge Model Peak"]
    mem_vals = [overhead_data.get("ram_mb", 1.2), overhead_data.get("edge_peak_mb", 4.5)]
    ax2.bar(mem_labels, mem_vals, color=["#9b59b6", "#f39c12"], edgecolor="black", width=0.45)
    ax2.set_title("Peak Memory Footprint (MB)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Memory (MB)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")

    for idx, (label, val) in enumerate(zip(mem_labels, mem_vals)):
        ax2.text(idx, val + 0.1, f"{val:.1f} MB", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_reward_sensitivity(
    sensitivity_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Reward Sensitivity Sweep: λ1 (Parameter Penalty) vs. λ2 (Latency Penalty)",
):
    """
    Heatmap of test accuracy and parameter volume across (lambda1, lambda2) grid.
    """
    pivot_acc = sensitivity_df.pivot(index="lambda1", columns="lambda2", values="test_accuracy")
    pivot_params = sensitivity_df.pivot(index="lambda1", columns="lambda2", values="total_parameters")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(pivot_acc * 100.0, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax1, cbar_kws={"label": "Accuracy (%)"})
    ax1.set_title("Test Accuracy (%)", fontweight="bold", fontsize=11)
    ax1.set_xlabel("λ2 (Latency Weight)", fontsize=10)
    ax1.set_ylabel("λ1 (Parameter Weight)", fontsize=10)

    sns.heatmap(pivot_params, annot=True, fmt=".0f", cmap="OrRd_r", ax=ax2, cbar_kws={"label": "Parameters / Round"})
    ax2.set_title("Total Parameters Transmitted", fontweight="bold", fontsize=11)
    ax2.set_xlabel("λ2 (Latency Weight)", fontsize=10)
    ax2.set_ylabel("λ1 (Parameter Weight)", fontsize=10)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Communication & Efficiency Visualizations (Step 2 of FedKAN Evaluation)
# =============================================================================

METHOD_STYLE: Dict[str, Dict[str, Any]] = {
    "FedAvg-MLP": {
        "color": "#7f7f7f",
        "marker": "v",
        "linestyle": "--",
        "label": "FedAvg-MLP",
    },
    "FedAvg-KAN (fixed 5,3)": {
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
        "label": "FedAvg-KAN (fixed 5,3)",
    },
    "FedProx-KAN": {
        "color": "#9467bd",
        "marker": "D",
        "linestyle": "--",
        "label": "FedProx-KAN",
    },
    "Heuristic-KAN": {
        "color": "#ff7f0e",
        "marker": "s",
        "linestyle": "-.",
        "label": "Heuristic-KAN",
    },
    "PPO-Adaptive-KAN": {
        "color": "#2ca02c",
        "marker": "*",
        "linestyle": "-",
        "label": "PPO-Adaptive-KAN (Ours)",
    },
    "Uniform-Fleet-(3,2)": {
        "color": "#17becf",
        "marker": "^",
        "linestyle": ":",
        "label": "Fixed (3,2) KAN",
    },
    "Uniform-Fleet-(10,4)": {
        "color": "#8c564b",
        "marker": "p",
        "linestyle": ":",
        "label": "Fixed (10,4) KAN",
    },
    "Random-config KAN": {
        "color": "#e377c2",
        "marker": "x",
        "linestyle": ":",
        "label": "Random-config KAN",
    },
}

LABEL_OFFSETS: Dict[str, Tuple[int, int]] = {
    "FedAvg-MLP": (-10, -18),
    "FedAvg-KAN (fixed 5,3)": (10, 8),
    "FedProx-KAN": (-15, 12),
    "Heuristic-KAN": (10, -16),
    "PPO-Adaptive-KAN": (-20, 10),
    "Uniform-Fleet-(3,2)": (-25, -16),
    "Uniform-Fleet-(10,4)": (10, 8),
    "Random-config KAN": (12, -8),
}


def _save_dual_format(fig: plt.Figure, save_path: Union[str, Path]) -> None:
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")


def _compute_pareto_frontier(points: Sequence[Tuple[float, float]]) -> List[int]:
    """
    Compute indices of Pareto-optimal points for minimization of x (communication)
    and maximization of y (accuracy / metric).
    """
    pareto_indices: List[int] = []
    for i, (xi, yi) in enumerate(points):
        dominated = False
        for j, (xj, yj) in enumerate(points):
            if i != j:
                if xj <= xi and yj >= yi and (xj < xi or yj > yi):
                    dominated = True
                    break
        if not dominated:
            pareto_indices.append(i)
    pareto_indices.sort(key=lambda idx: points[idx][0])
    return pareto_indices


def plot_accuracy_vs_cum_comm(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    log_x: bool = False,
    title: Optional[str] = "Fig C1: Test Accuracy vs. Cumulative Communication Across Skew Levels",
) -> plt.Figure:
    """
    Fig C1: Accuracy vs Cumulative Communication convergence curves (1x3 subplots across alpha).
    Mean curves with shaded +/- std bands across seeds.
    """
    alphas = sorted(df["alpha"].unique())
    n_alphas = len(alphas)
    fig, axes = plt.subplots(1, n_alphas, figsize=(5.5 * n_alphas, 5.0), sharey=True)
    if n_alphas == 1:
        axes = [axes]

    handles_dict = {}

    for ax_idx, alpha in enumerate(alphas):
        ax = axes[ax_idx]
        alpha_df = df[df["alpha"] == alpha]

        for method, style in METHOD_STYLE.items():
            m_df = alpha_df[alpha_df["method"] == method]
            if m_df.empty:
                continue

            grp = m_df.groupby("round").agg({
                "cum_MB": "mean",
                "test_acc": ["mean", "std"],
            })
            x = grp[("cum_MB", "mean")].values
            y_mean = grp[("test_acc", "mean")].values
            y_std = grp[("test_acc", "std")].fillna(0.0).values

            line = ax.plot(
                x,
                y_mean,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                markersize=6,
                label=style["label"],
                alpha=0.9,
            )[0]
            if method not in handles_dict:
                handles_dict[method] = line

            if m_df["seed"].nunique() > 1:
                ax.fill_between(
                    x,
                    y_mean - y_std,
                    y_mean + y_std,
                    color=style["color"],
                    alpha=0.15,
                )

        if log_x:
            ax.set_xscale("log")
        ax.set_title(f"Non-IID Skew α = {alpha}", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Cumulative Communication (MB)", fontsize=11, fontweight="bold")
        if ax_idx == 0:
            ax.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_ylim(65, 92)

    fig.legend(
        handles=list(handles_dict.values()),
        labels=[h.get_label() for h in handles_dict.values()],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=4,
        framealpha=0.9,
        fontsize=9,
    )
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.12)
    plt.tight_layout()

    if save_path:
        _save_dual_format(fig, save_path)
    return fig


def plot_f1_vs_cum_comm(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    log_x: bool = False,
    include_balanced_acc: bool = True,
    title: Optional[str] = "Fig C2: Arrhythmia Detection Metrics vs. Cumulative Communication",
) -> plt.Figure:
    """
    Fig C2: Test Macro F1 (and Balanced Accuracy) vs Cumulative Communication.
    2x3 layout (Row 1: Macro F1, Row 2: Balanced Accuracy) across alpha levels.
    """
    alphas = sorted(df["alpha"].unique())
    n_alphas = len(alphas)
    n_rows = 2 if include_balanced_acc else 1
    fig, axes = plt.subplots(n_rows, n_alphas, figsize=(5.5 * n_alphas, 4.5 * n_rows), sharey="row")

    if n_alphas == 1:
        axes = np.expand_dims(axes, 1)
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    metrics = [("test_macro_f1", "Test Macro F1 (%)")]
    if include_balanced_acc:
        metrics.append(("test_balanced_acc", "Test Balanced Acc (%)"))

    handles_dict = {}

    for row_idx, (m_col, m_label) in enumerate(metrics):
        for col_idx, alpha in enumerate(alphas):
            ax = axes[row_idx, col_idx]
            alpha_df = df[df["alpha"] == alpha]

            for method, style in METHOD_STYLE.items():
                m_df = alpha_df[alpha_df["method"] == method]
                if m_df.empty:
                    continue

                grp = m_df.groupby("round").agg({
                    "cum_MB": "mean",
                    m_col: ["mean", "std"],
                })
                x = grp[("cum_MB", "mean")].values
                y_mean = grp[(m_col, "mean")].values
                y_std = grp[(m_col, "std")].fillna(0.0).values

                line = ax.plot(
                    x,
                    y_mean,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=1.8,
                    markersize=6,
                    label=style["label"],
                    alpha=0.9,
                )[0]
                if method not in handles_dict:
                    handles_dict[method] = line

                if m_df["seed"].nunique() > 1:
                    ax.fill_between(
                        x,
                        y_mean - y_std,
                        y_mean + y_std,
                        color=style["color"],
                        alpha=0.15,
                    )

            if log_x:
                ax.set_xscale("log")
            if row_idx == 0:
                ax.set_title(f"Non-IID Skew α = {alpha}", fontsize=12, fontweight="bold", pad=8)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Cumulative Communication (MB)", fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(m_label, fontsize=11, fontweight="bold")
            ax.grid(True, linestyle=":", alpha=0.6)

    fig.legend(
        handles=list(handles_dict.values()),
        labels=[h.get_label() for h in handles_dict.values()],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=4,
        framealpha=0.9,
        fontsize=9,
    )
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.09)
    plt.tight_layout()

    if save_path:
        _save_dual_format(fig, save_path)
    return fig


def plot_final_perf_vs_total_comm(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = "Fig C3: Final Performance vs. Total Communication (Pareto Frontiers)",
) -> plt.Figure:
    """
    Fig C3: Final performance vs total communication (Pareto scatter plot with error bars).
    2x3 layout: Row 1 = Test Accuracy vs Cum Comm; Row 2 = Test Macro F1 vs Cum Comm.
    Pareto frontier lines connecting non-dominated methods.
    """
    alphas = sorted(df["alpha"].unique())
    n_alphas = len(alphas)
    fig, axes = plt.subplots(2, n_alphas, figsize=(5.8 * n_alphas, 9.5), sharey="row")
    if n_alphas == 1:
        axes = np.expand_dims(axes, 1)

    max_round = df["round"].max()
    final_df = df[df["round"] == max_round]

    metrics = [
        ("test_acc", "Final Test Accuracy (%)", (68, 90)),
        ("test_macro_f1", "Final Macro F1 (%)", (20, 52)),
    ]

    for row_idx, (m_col, m_label, y_limits) in enumerate(metrics):
        for col_idx, alpha in enumerate(alphas):
            ax = axes[row_idx, col_idx]
            alpha_df = final_df[final_df["alpha"] == alpha]

            point_data: List[Dict[str, Any]] = []
            pts_for_pareto: List[Tuple[float, float]] = []

            for method, style in METHOD_STYLE.items():
                m_df = alpha_df[alpha_df["method"] == method]
                if m_df.empty:
                    continue

                x_mean = float(m_df["cum_MB"].mean())
                x_std = float(m_df["cum_MB"].std()) if m_df["seed"].nunique() > 1 else 0.0
                y_mean = float(m_df[m_col].mean())
                y_std = float(m_df[m_col].std()) if m_df["seed"].nunique() > 1 else 0.0

                point_data.append({
                    "method": method,
                    "style": style,
                    "x": x_mean,
                    "x_std": x_std,
                    "y": y_mean,
                    "y_std": y_std,
                })
                pts_for_pareto.append((x_mean, y_mean))

            # Compute Pareto-optimal frontier
            pareto_indices = _compute_pareto_frontier(pts_for_pareto)
            pareto_pts = [pts_for_pareto[idx] for idx in pareto_indices]

            # Plot dashed Pareto line
            if len(pareto_pts) > 1:
                px = [p[0] for p in pareto_pts]
                py = [p[1] for p in pareto_pts]
                ax.plot(
                    px,
                    py,
                    linestyle="--",
                    color="#d62728",
                    linewidth=1.8,
                    alpha=0.75,
                    label="Pareto Frontier" if (row_idx == 0 and col_idx == 0) else None,
                    zorder=2,
                )

            # Plot each point with error bars
            for idx, pinfo in enumerate(point_data):
                is_pareto = idx in pareto_indices
                style = pinfo["style"]

                ax.errorbar(
                    pinfo["x"],
                    pinfo["y"],
                    xerr=pinfo["x_std"],
                    yerr=pinfo["y_std"],
                    fmt=style["marker"],
                    color=style["color"],
                    ecolor=style["color"],
                    elinewidth=1.2,
                    capsize=3,
                    markersize=8,
                    zorder=4,
                )

                if is_pareto:
                    ax.scatter(
                        [pinfo["x"]],
                        [pinfo["y"]],
                        s=160,
                        facecolor="none",
                        edgecolor="#d62728",
                        linewidth=2.0,
                        zorder=5,
                    )

                # Offset annotation
                dx, dy = LABEL_OFFSETS.get(pinfo["method"], (8, 8))
                short_name = (
                    "PPO (Ours)" if "PPO" in pinfo["method"] else
                    "Heuristic" if "Heuristic" in pinfo["method"] else
                    "Fixed (5,3)" if "fixed 5,3" in pinfo["method"] else
                    "Fixed (3,2)" if "3,2" in pinfo["method"] else
                    "Fixed (10,4)" if "10,4" in pinfo["method"] else
                    "Random" if "Random" in pinfo["method"] else
                    "FedProx" if "FedProx" in pinfo["method"] else
                    "FedAvg-MLP"
                )
                ax.annotate(
                    short_name,
                    xy=(pinfo["x"], pinfo["y"]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=7.5,
                    fontweight="bold" if is_pareto else "normal",
                    color="#2c3e50",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        alpha=0.8,
                        edgecolor="#d62728" if is_pareto else "#cccccc",
                        linewidth=0.8 if is_pareto else 0.5,
                    ),
                    zorder=6,
                )

            if row_idx == 0:
                ax.set_title(f"Non-IID Skew α = {alpha}", fontsize=12, fontweight="bold", pad=8)
            if row_idx == 1:
                ax.set_xlabel("Total Communication (MB)", fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(m_label, fontsize=11, fontweight="bold")
            ax.set_ylim(y_limits)
            ax.grid(True, linestyle=":", alpha=0.6)

    # Custom legend for Pareto marker
    custom_handles = [
        plt.Line2D([0], [0], linestyle="--", color="#d62728", linewidth=1.8, label="Pareto Frontier"),
        plt.Line2D([0], [0], marker="o", color="none", markeredgecolor="#d62728", markeredgewidth=2.0, markersize=8, label="Pareto-Optimal Point"),
    ]
    for method, style in METHOD_STYLE.items():
        custom_handles.append(
            plt.Line2D([0], [0], marker=style["marker"], color=style["color"], linestyle="none", markersize=7, label=style["label"])
        )

    fig.legend(
        handles=custom_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=5,
        framealpha=0.9,
        fontsize=8.5,
    )
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.07)
    plt.tight_layout()

    if save_path:
        _save_dual_format(fig, save_path)
    return fig


def plot_small_multiples_per_method(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    per_method_dir: Optional[Union[str, Path]] = None,
    title: Optional[str] = "Fig C4: Per-Method Convergence Profiles (Accuracy & Macro F1)",
) -> plt.Figure:
    """
    Fig C4: Small multiples grid (rows = methods, cols = alphas).
    Dual-Y axis per panel (Left: Accuracy %, Right: Macro F1 %).
    Also exports per-method standalone figures to per_method_dir.
    """
    alphas = sorted(df["alpha"].unique())
    methods = [m for m in METHOD_STYLE.keys() if m in df["method"].unique()]
    n_rows = len(methods)
    n_cols = len(alphas)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 2.5 * n_rows), sharex="col")
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    for r_idx, method in enumerate(methods):
        style = METHOD_STYLE[method]
        for c_idx, alpha in enumerate(alphas):
            ax1 = axes[r_idx, c_idx]
            sub = df[(df["method"] == method) & (df["alpha"] == alpha)]

            if sub.empty:
                continue

            grp = sub.groupby("round").agg({
                "cum_MB": "mean",
                "test_acc": ["mean", "std"],
                "test_macro_f1": ["mean", "std"],
            })
            x = grp[("cum_MB", "mean")].values
            acc_m = grp[("test_acc", "mean")].values
            f1_m = grp[("test_macro_f1", "mean")].values

            # Left axis: Accuracy
            line1 = ax1.plot(x, acc_m, color=style["color"], marker=style["marker"], markersize=4, linewidth=1.5, label="Test Acc (%)")
            ax1.set_ylim(60, 95)
            ax1.grid(True, linestyle=":", alpha=0.5)

            # Right axis: Macro F1
            ax2 = ax1.twinx()
            line2 = ax2.plot(x, f1_m, color="#d62728", marker="s", markersize=3, linestyle="--", linewidth=1.2, label="Macro F1 (%)")
            ax2.set_ylim(15, 60)

            if r_idx == 0:
                ax1.set_title(f"α = {alpha}", fontsize=11, fontweight="bold")
            if r_idx == n_rows - 1:
                ax1.set_xlabel("Cumulative Comm (MB)", fontsize=9, fontweight="bold")
            if c_idx == 0:
                ax1.set_ylabel(f"{method}\nAcc (%)", fontsize=8.5, fontweight="bold")
            if c_idx == n_cols - 1:
                ax2.set_ylabel("Macro F1 (%)", fontsize=8.5, color="#d62728", fontweight="bold")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        _save_dual_format(fig, save_path)

    # Save individual standalone plots
    if per_method_dir:
        pm_path = Path(per_method_dir)
        pm_path.mkdir(parents=True, exist_ok=True)
        for method in methods:
            for alpha in alphas:
                sub = df[(df["method"] == method) & (df["alpha"] == alpha)]
                if sub.empty:
                    continue
                ind_fig, ind_ax1 = plt.subplots(figsize=(6, 4))
                grp = sub.groupby("round").agg({
                    "cum_MB": "mean",
                    "test_acc": ["mean", "std"],
                    "test_macro_f1": ["mean", "std"],
                })
                x = grp[("cum_MB", "mean")].values
                acc_m = grp[("test_acc", "mean")].values
                f1_m = grp[("test_macro_f1", "mean")].values

                style = METHOD_STYLE[method]
                ind_ax1.plot(x, acc_m, color=style["color"], marker=style["marker"], linewidth=1.8, label="Test Accuracy (%)")
                ind_ax1.set_xlabel("Cumulative Communication (MB)", fontweight="bold")
                ind_ax1.set_ylabel("Test Accuracy (%)", color=style["color"], fontweight="bold")
                ind_ax1.set_ylim(60, 95)
                ind_ax1.grid(True, linestyle=":", alpha=0.6)

                ind_ax2 = ind_ax1.twinx()
                ind_ax2.plot(x, f1_m, color="#d62728", marker="s", linestyle="--", linewidth=1.5, label="Test Macro F1 (%)")
                ind_ax2.set_ylabel("Test Macro F1 (%)", color="#d62728", fontweight="bold")
                ind_ax2.set_ylim(15, 60)

                clean_name = method.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")
                ind_ax1.set_title(f"{method} Convergence (α = {alpha})", fontweight="bold")
                ind_fig.tight_layout()
                _save_dual_format(ind_fig, pm_path / f"{clean_name}_alpha_{alpha}")
                plt.close(ind_fig)

    return fig


def plot_comm_to_target_acc(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    targets: Sequence[float] = (70.0, 75.0, 80.0, 85.0),
    title: Optional[str] = "Fig C5: Cumulative Communication Required to Reach Target Accuracy Thresholds",
) -> plt.Figure:
    """
    Fig C5: Bar chart showing cumulative MB needed to first reach target accuracies.
    Methods that never reach a threshold receive hatched bars labeled 'Not Reached'.
    """
    alphas = sorted(df["alpha"].unique())
    n_alphas = len(alphas)
    fig, axes = plt.subplots(1, n_alphas, figsize=(5.8 * n_alphas, 5.0), sharey=True)
    if n_alphas == 1:
        axes = [axes]

    methods = [m for m in METHOD_STYLE.keys() if m in df["method"].unique()]
    n_methods = len(methods)
    bar_width = 0.8 / len(targets)
    target_palette = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]

    for ax_idx, alpha in enumerate(alphas):
        ax = axes[ax_idx]
        alpha_df = df[df["alpha"] == alpha]
        x_indices = np.arange(n_methods)

        for t_idx, target in enumerate(targets):
            offsets = (t_idx - len(targets) / 2 + 0.5) * bar_width
            mb_vals = []
            hatched_flags = []

            for method in methods:
                sub = alpha_df[alpha_df["method"] == method]
                # Average accuracy per round
                grp = sub.groupby("round").agg({"cum_MB": "mean", "test_acc": "mean"}).reset_index()
                reached = grp[grp["test_acc"] >= target]
                if not reached.empty:
                    first_mb = reached.iloc[0]["cum_MB"]
                    mb_vals.append(first_mb)
                    hatched_flags.append(False)
                else:
                    # Not reached: set bar height to max cum_MB with hatch
                    max_mb = grp["cum_MB"].max() if not grp.empty else 4.0
                    mb_vals.append(max_mb)
                    hatched_flags.append(True)

            for b_idx in range(n_methods):
                val = mb_vals[b_idx]
                is_hatched = hatched_flags[b_idx]
                ax.bar(
                    x_indices[b_idx] + offsets,
                    val,
                    width=bar_width,
                    color=target_palette[t_idx % len(target_palette)] if not is_hatched else "#e0e0e0",
                    edgecolor="black" if not is_hatched else "#d62728",
                    hatch="//" if is_hatched else None,
                    linewidth=0.7,
                    label=f"≥ {target:.0f}% Acc" if (ax_idx == 0 and b_idx == 0) else None,
                )
                if is_hatched:
                    ax.text(
                        x_indices[b_idx] + offsets,
                        val + 0.1,
                        "NR",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        fontweight="bold",
                        color="#d62728",
                        rotation=90,
                    )

        ax.set_title(f"Non-IID Skew α = {alpha}", fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks(x_indices)
        short_labels = [
            "MLP" if "MLP" in m else
            "Fixed(5,3)" if "fixed 5,3" in m else
            "FedProx" if "FedProx" in m else
            "Heuristic" if "Heuristic" in m else
            "PPO" if "PPO" in m else
            "Fixed(3,2)" if "3,2" in m else
            "Fixed(10,4)" if "10,4" in m else "Random"
            for m in methods
        ]
        ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=9, fontweight="bold")
        if ax_idx == 0:
            ax.set_ylabel("Cumulative MB to Reach Threshold", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5, axis="y")
        ax.set_ylim(0, 6.5)

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=len(targets),
        framealpha=0.9,
        fontsize=9.5,
    )
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.11)
    plt.tight_layout()

    if save_path:
        _save_dual_format(fig, save_path)
    return fig


def plot_comm_breakdown_and_configs(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    alpha_focus: float = 0.5,
    title: Optional[str] = "Fig C6: Per-Round Communication Volume and Active Configuration Breakdown",
) -> plt.Figure:
    """
    Fig C6: Stacked communication breakdown (Uplink/Downlink) and active (G,k) config histograms over rounds.
    Shows the underlying dynamic resource allocation of PPO vs. Heuristic vs. Fixed references.
    """
    import ast

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    sub_alpha = df[df["alpha"] == alpha_focus]
    rounds = sorted(df["round"].unique())

    # Panel (0, 0): Per-round params for PPO-Adaptive-KAN vs Fixed budgets
    ax_comm_ppo = axes[0, 0]
    ppo_df = sub_alpha[sub_alpha["method"] == "PPO-Adaptive-KAN"].groupby("round").mean(numeric_only=True)
    if not ppo_df.empty:
        up = ppo_df["params_up"].values
        down = ppo_df["params_down"].values
        ax_comm_ppo.bar(rounds, up, label="Uplink (Client -> Server)", color="#2ca02c", alpha=0.8, width=0.55)
        ax_comm_ppo.bar(rounds, down, bottom=up, label="Downlink (Server -> Client)", color="#98df8a", alpha=0.8, width=0.55)

    ax_comm_ppo.axhline(42792 * 2, color="#1f77b4", linestyle="--", linewidth=1.5, label="Fixed (5,3) Round Budget (85.6k)")
    ax_comm_ppo.axhline(71208 * 2, color="#8c564b", linestyle=":", linewidth=1.5, label="Fine (10,4) Fleet Budget (142.4k)")
    ax_comm_ppo.axhline(28576 * 2, color="#17becf", linestyle="-.", linewidth=1.5, label="Coarse (3,2) Fleet Budget (57.2k)")
    ax_comm_ppo.set_title("PPO-Adaptive-KAN Round Volume vs. Reference Budgets", fontweight="bold", fontsize=11)
    ax_comm_ppo.set_xlabel("Federated Round", fontweight="bold")
    ax_comm_ppo.set_ylabel("Total Parameters (Up + Down)", fontweight="bold")
    ax_comm_ppo.grid(True, linestyle=":", alpha=0.5)
    ax_comm_ppo.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # Panel (0, 1): Per-round params for Heuristic-KAN
    ax_comm_heu = axes[0, 1]
    heu_df = sub_alpha[sub_alpha["method"] == "Heuristic-KAN"].groupby("round").mean(numeric_only=True)
    if not heu_df.empty:
        up_h = heu_df["params_up"].values
        down_h = heu_df["params_down"].values
        ax_comm_heu.bar(rounds, up_h, label="Uplink", color="#ff7f0e", alpha=0.8, width=0.55)
        ax_comm_heu.bar(rounds, down_h, bottom=up_h, label="Downlink", color="#ffbb78", alpha=0.8, width=0.55)
    ax_comm_heu.axhline(42792 * 2, color="#1f77b4", linestyle="--", linewidth=1.5, label="Fixed (5,3) Budget")
    ax_comm_heu.axhline(71208 * 2, color="#8c564b", linestyle=":", linewidth=1.5, label="Fine (10,4) Budget")
    ax_comm_heu.set_title("Heuristic-KAN Round Volume vs. Reference Budgets", fontweight="bold", fontsize=11)
    ax_comm_heu.set_xlabel("Federated Round", fontweight="bold")
    ax_comm_heu.set_ylabel("Total Parameters (Up + Down)", fontweight="bold")
    ax_comm_heu.grid(True, linestyle=":", alpha=0.5)
    ax_comm_heu.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # Helper to parse config histograms
    def _parse_histograms(method_name: str) -> Dict[str, List[float]]:
        m_rows = sub_alpha[sub_alpha["method"] == method_name]
        time_series: Dict[str, List[float]] = {}
        for r in rounds:
            r_rows = m_rows[m_rows["round"] == r]
            dicts = []
            for h_str in r_rows["config_histogram"]:
                if pd.isna(h_str) or not str(h_str).strip() or str(h_str).strip() == "{}":
                    continue
                try:
                    parsed = ast.literal_eval(str(h_str))
                    if isinstance(parsed, dict):
                        dicts.append(parsed)
                except Exception:
                    pass
            all_keys = set().union(*[d.keys() for d in dicts]) if dicts else set()
            for k in all_keys:
                mean_val = np.mean([d.get(k, 0.0) for d in dicts])
                time_series.setdefault(k, []).append(mean_val)
        return time_series

    # Panel (1, 0): PPO Config Selection Area
    ax_hist_ppo = axes[1, 0]
    ppo_hist = _parse_histograms("PPO-Adaptive-KAN")
    if ppo_hist:
        keys = sorted(ppo_hist.keys())
        y_stack = np.array([ppo_hist[k] for k in keys])
        ax_hist_ppo.stackplot(rounds, y_stack, labels=keys, alpha=0.8)
        ax_hist_ppo.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.85, title="Config (G,k)")
    ax_hist_ppo.set_title("PPO-Adaptive Client Fleet Configuration Histogram", fontweight="bold", fontsize=11)
    ax_hist_ppo.set_xlabel("Federated Round", fontweight="bold")
    ax_hist_ppo.set_ylabel("Client Fleet Fraction", fontweight="bold")
    ax_hist_ppo.set_ylim(0, 1.0)
    ax_hist_ppo.grid(True, linestyle=":", alpha=0.5)

    # Panel (1, 1): Heuristic Config Selection Area
    ax_hist_heu = axes[1, 1]
    heu_hist = _parse_histograms("Heuristic-KAN")
    if heu_hist:
        keys_h = sorted(heu_hist.keys())
        y_stack_h = np.array([heu_hist[k] for k in keys_h])
        ax_hist_heu.stackplot(rounds, y_stack_h, labels=keys_h, alpha=0.8)
        ax_hist_heu.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.85, title="Config (G,k)")
    ax_hist_heu.set_title("Heuristic Client Fleet Configuration Histogram", fontweight="bold", fontsize=11)
    ax_hist_heu.set_xlabel("Federated Round", fontweight="bold")
    ax_hist_heu.set_ylabel("Client Fleet Fraction", fontweight="bold")
    ax_hist_heu.set_ylim(0, 1.0)
    ax_hist_heu.grid(True, linestyle=":", alpha=0.5)

    if title:
        fig.suptitle(f"{title} (α = {alpha_focus})", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    if save_path:
        _save_dual_format(fig, save_path)
    return fig


def plot_acc_vs_comm_time(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    time_col: str = "est_comm_time_s",
    title: Optional[str] = "Fig C7: Test Accuracy vs. Cumulative Simulated Communication Time",
) -> plt.Figure:
    """
    Fig C7: Test Accuracy vs. Cumulative Simulated Transmission Latency (seconds).
    Demonstrates wall-clock edge network efficiency with simulated 1-10 Mbps channel bandwidths.
    """
    alphas = sorted(df["alpha"].unique())
    n_alphas = len(alphas)
    fig, axes = plt.subplots(1, n_alphas, figsize=(5.5 * n_alphas, 5.0), sharey=True)
    if n_alphas == 1:
        axes = [axes]

    handles_dict = {}

    for ax_idx, alpha in enumerate(alphas):
        ax = axes[ax_idx]
        alpha_df = df[df["alpha"] == alpha]

        for method, style in METHOD_STYLE.items():
            m_df = alpha_df[alpha_df["method"] == method]
            if m_df.empty:
                continue

            # Compute cumulative latency per seed then aggregate
            m_df = m_df.sort_values(["seed", "round"]).copy()
            m_df["cum_time_s"] = m_df.groupby("seed")[time_col].cumsum()

            grp = m_df.groupby("round").agg({
                "cum_time_s": "mean",
                "test_acc": ["mean", "std"],
            })
            x = grp[("cum_time_s", "mean")].values
            y_mean = grp[("test_acc", "mean")].values
            y_std = grp[("test_acc", "std")].fillna(0.0).values

            line = ax.plot(
                x,
                y_mean,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                markersize=6,
                label=style["label"],
                alpha=0.9,
            )[0]
            if method not in handles_dict:
                handles_dict[method] = line

            if m_df["seed"].nunique() > 1:
                ax.fill_between(
                    x,
                    y_mean - y_std,
                    y_mean + y_std,
                    color=style["color"],
                    alpha=0.15,
                )

        ax.set_title(f"Non-IID Skew α = {alpha}", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Cumulative Simulated Latency (s)", fontsize=11, fontweight="bold")
        if ax_idx == 0:
            ax.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_ylim(65, 92)

    fig.legend(
        handles=list(handles_dict.values()),
        labels=[h.get_label() for h in handles_dict.values()],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=4,
        framealpha=0.9,
        fontsize=9,
    )
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.12)
    plt.tight_layout()

    if save_path:
        _save_dual_format(fig, save_path)
    return fig


def plot_savings_summary_bar(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = "Fig C8: Relative Parameter Communication Savings Across Reference Baselines",
) -> plt.Figure:
    """
    Fig C8: Communication savings summary grouped bar chart.
    Reports % parameter reduction relative to:
      1) Fixed (5,3) Reference fleet (42,792 params/round)
      2) Fine (10,4) Fleet (71,208 params/round)
      3) Coarse (3,2) Fleet (28,576 params/round)
    Negative bars indicate communication overhead exceeding the respective reference.
    """
    ref_fixed53 = 42792.0
    ref_fine104 = 71208.0
    ref_coarse32 = 28576.0

    alphas = sorted(df["alpha"].unique())
    n_alphas = len(alphas)
    fig, axes = plt.subplots(1, n_alphas, figsize=(6.0 * n_alphas, 5.2), sharey=True)
    if n_alphas == 1:
        axes = [axes]

    methods = [m for m in METHOD_STYLE.keys() if m in df["method"].unique()]
    x_indices = np.arange(len(methods))
    bar_width = 0.25

    ref_colors = ["#1f77b4", "#8c564b", "#17becf"]
    ref_labels = ["vs. Fixed (5,3)", "vs. Fine (10,4)", "vs. Coarse (3,2)"]

    for ax_idx, alpha in enumerate(alphas):
        ax = axes[ax_idx]
        alpha_df = df[df["alpha"] == alpha]

        savings_fixed53 = []
        savings_fine104 = []
        savings_coarse32 = []

        for method in methods:
            m_df = alpha_df[alpha_df["method"] == method]
            avg_params = float(m_df["params_up"].mean()) if not m_df.empty else ref_fixed53
            s_f53 = (1.0 - avg_params / ref_fixed53) * 100.0
            s_f104 = (1.0 - avg_params / ref_fine104) * 100.0
            s_c32 = (1.0 - avg_params / ref_coarse32) * 100.0

            savings_fixed53.append(s_f53)
            savings_fine104.append(s_f104)
            savings_coarse32.append(s_c32)

        ax.bar(
            x_indices - bar_width,
            savings_fixed53,
            width=bar_width,
            color=ref_colors[0],
            edgecolor="black",
            linewidth=0.6,
            label=ref_labels[0] if ax_idx == 0 else None,
        )
        ax.bar(
            x_indices,
            savings_fine104,
            width=bar_width,
            color=ref_colors[1],
            edgecolor="black",
            linewidth=0.6,
            label=ref_labels[1] if ax_idx == 0 else None,
        )
        ax.bar(
            x_indices + bar_width,
            savings_coarse32,
            width=bar_width,
            color=ref_colors[2],
            edgecolor="black",
            linewidth=0.6,
            label=ref_labels[2] if ax_idx == 0 else None,
        )

        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_title(f"Non-IID Skew α = {alpha}", fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks(x_indices)
        short_labels = [
            "MLP" if "MLP" in m else
            "Fixed(5,3)" if "fixed 5,3" in m else
            "FedProx" if "FedProx" in m else
            "Heuristic" if "Heuristic" in m else
            "PPO" if "PPO" in m else
            "Fixed(3,2)" if "3,2" in m else
            "Fixed(10,4)" if "10,4" in m else "Random"
            for m in methods
        ]
        ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=9, fontweight="bold")
        if ax_idx == 0:
            ax.set_ylabel("Communication Savings (%)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5, axis="y")
        ax.set_ylim(-75, 75)

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=3,
        framealpha=0.9,
        fontsize=9.5,
    )
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.11)
    plt.tight_layout()

    if save_path:
        _save_dual_format(fig, save_path)
    return fig


def generate_all_comm_figures(
    df: pd.DataFrame,
    output_dir: Union[str, Path] = "results/figures/comm",
    log_x: bool = False,
) -> Dict[str, Path]:
    """
    Master pipeline: generate and save all communication cost vs performance figures (C1-C8).
    Returns dictionary mapping figure name to output file path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_files = {}

    # C1
    p_c1 = out_dir / "fig_c1_acc_vs_cum_comm"
    plot_accuracy_vs_cum_comm(df, save_path=p_c1, log_x=log_x)
    generated_files["C1"] = p_c1

    # C2
    p_c2 = out_dir / "fig_c2_f1_vs_cum_comm"
    plot_f1_vs_cum_comm(df, save_path=p_c2, log_x=log_x)
    generated_files["C2"] = p_c2

    # C3
    p_c3 = out_dir / "fig_c3_pareto_scatter"
    plot_final_perf_vs_total_comm(df, save_path=p_c3)
    generated_files["C3"] = p_c3

    # C4
    p_c4 = out_dir / "fig_c4_small_multiples"
    plot_small_multiples_per_method(df, save_path=p_c4, per_method_dir=out_dir / "per_method")
    generated_files["C4"] = p_c4

    # C5
    p_c5 = out_dir / "fig_c5_comm_to_target_acc"
    plot_comm_to_target_acc(df, save_path=p_c5)
    generated_files["C5"] = p_c5

    # C6
    p_c6 = out_dir / "fig_c6_comm_breakdown"
    plot_comm_breakdown_and_configs(df, save_path=p_c6, alpha_focus=0.5)
    generated_files["C6"] = p_c6

    # C7
    p_c7 = out_dir / "fig_c7_acc_vs_comm_time"
    plot_acc_vs_comm_time(df, save_path=p_c7)
    generated_files["C7"] = p_c7

    # C8
    p_c8 = out_dir / "fig_c8_savings_summary"
    plot_savings_summary_bar(df, save_path=p_c8)
    generated_files["C8"] = p_c8

    return generated_files


