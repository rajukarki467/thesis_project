import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

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

