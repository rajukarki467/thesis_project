import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_ecg(
    signal,
    fs,
    channel_name="ECG",
    start_seconds=0,
    duration_seconds=10,
    title=None
):
    """
    Plot a section of an ECG signal.
    """

    start_sample = int(start_seconds * fs)
    end_sample = int(
        (start_seconds + duration_seconds) * fs
    )

    end_sample = min(
        end_sample,
        len(signal)
    )

    segment = signal[
        start_sample:end_sample
    ]

    time = np.arange(
        len(segment)
    ) / fs + start_seconds

    plt.figure(figsize=(14, 4))

    plt.plot(
        time,
        segment,
        linewidth=0.8
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (mV)")

    if title:
        plt.title(title)
    else:
        plt.title(
            f"{channel_name} ECG"
        )

    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_two_channel_ecg(
    record,
    start_seconds=0,
    duration_seconds=10
):
    """
    Plot both ECG channels.
    """

    fs = record.fs

    start_sample = int(
        start_seconds * fs
    )

    end_sample = int(
        (start_seconds + duration_seconds) * fs
    )

    end_sample = min(
        end_sample,
        record.p_signal.shape[0]
    )

    segment = record.p_signal[
        start_sample:end_sample
    ]

    time = np.arange(
        len(segment)
    ) / fs + start_seconds

    fig, axes = plt.subplots(
        record.p_signal.shape[1],
        1,
        figsize=(14, 7),
        sharex=True
    )

    if record.p_signal.shape[1] == 1:
        axes = [axes]

    for i, ax in enumerate(axes):

        ax.plot(
            time,
            segment[:, i],
            linewidth=0.8
        )

        ax.set_ylabel(
            f"{record.sig_name[i]}\n(mV)"
        )

        ax.grid(
            True,
            alpha=0.3
        )

    axes[-1].set_xlabel(
        "Time (seconds)"
    )

    fig.suptitle(
        "MIT-BIH ECG Record"
    )

    plt.tight_layout()
    plt.show()


def plot_annotation_distribution(
    distribution,
    top_n=20
):
    """
    Plot annotation symbol distribution.
    """

    data = distribution.head(top_n)

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=data,
        x="symbol",
        y="count"
    )

    plt.xlabel("Annotation Symbol")
    plt.ylabel("Count")
    plt.title(
        "MIT-BIH Annotation Distribution"
    )

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def plot_class_distribution(
    labels,
    class_names=None
):
    """
    Plot class distribution.
    """

    counts = (
        np.asarray(labels)
    )

    unique, frequencies = np.unique(
        counts,
        return_counts=True
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        unique,
        frequencies
    )

    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.title(
        "Class Distribution"
    )

    plt.tight_layout()
    plt.show()


def plot_client_distribution(
    client_distribution,
    title="Client Label Distribution"
):
    """
    Plot client/class distribution as heatmap.

    client_distribution:
        DataFrame
        rows = clients
        columns = classes
    """

    plt.figure(
        figsize=(12, 8)
    )

    sns.heatmap(
        client_distribution,
        annot=True,
        fmt=".2f",
        cmap="Blues"
    )

    plt.title(title)

    plt.xlabel("Class")
    plt.ylabel("Client")

    plt.tight_layout()
    plt.show()