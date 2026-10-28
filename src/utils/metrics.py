"""
Evaluation metrics for ECG heartbeat classification.
"""

from typing import Any, Dict, Optional, Sequence, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

AAMI_CLASSES = ["N", "SVEB", "VEB", "F", "Q"]


def compute_metrics(
    y_true: Union[np.ndarray, Sequence[int]],
    y_pred: Union[np.ndarray, Sequence[int]],
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for AAMI ECG classes.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.
    class_names : Optional[Sequence[str]]
        Names of the classes (defaults to standard AAMI classes).

    Returns
    -------
    Dict[str, Any]
        Dictionary with accuracy, macro/micro/weighted F1, precision, recall,
        and per-class breakdown.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if class_names is None:
        class_names = AAMI_CLASSES

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)

    class_metrics = {}
    for idx, c_name in enumerate(class_names):
        f1_val = float(per_class_f1[idx]) if idx < len(per_class_f1) else 0.0
        rec_val = float(per_class_recall[idx]) if idx < len(per_class_recall) else 0.0
        prec_val = float(per_class_precision[idx]) if idx < len(per_class_precision) else 0.0
        class_metrics[c_name] = {
            "f1": f1_val,
            "recall": rec_val,
            "precision": prec_val,
        }

    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "class_metrics": class_metrics,
        "confusion_matrix": conf_mat,
    }


def compute_communication_savings(
    baseline_params: float,
    adaptive_params: float,
) -> float:
    """
    Compute percentage communication parameter savings relative to fixed baseline:
        Savings (%) = (Params_base - Params_adapt) / Params_base * 100%
    """
    if baseline_params <= 0:
        return 0.0
    savings = ((baseline_params - adaptive_params) / baseline_params) * 100.0
    return float(savings)


def compute_non_iid_degradation(
    acc_uniform: float,
    acc_skewed: float,
) -> float:
    """
    Compute relative performance degradation rate under non-IID label skew:
        Degradation (%) = (Acc_uniform - Acc_skewed) / Acc_uniform * 100%
    """
    if acc_uniform <= 0:
        return 0.0
    deg = ((acc_uniform - acc_skewed) / acc_uniform) * 100.0
    return float(deg)


def generate_latex_table(
    df: Any,
    caption: str = "Master Performance Comparison Across Non-IID Skews (ANSI/AAMI EC57)",
    label: str = "tab:master_benchmark",
) -> str:
    """
    Generate LaTeX table matching Table 4.1 in the master thesis proposal.
    Expected columns: [Method, Alpha, Accuracy (%), Macro F1 (%), Params/round, Comm Savings (%)].
    """
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Dirichlet $\alpha$} & \textbf{Accuracy (\%)} & \textbf{Macro F1 (\%)} & \textbf{Params/Round} & \textbf{Comm. Savings (\%)} \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        method = str(row.get("Method", row.get("Algorithm", "")))
        alpha = f"{row.get('Alpha', ''):.1f}" if isinstance(row.get('Alpha'), (int, float)) else str(row.get('Alpha', ''))
        acc = f"{float(row.get('Accuracy (%)', row.get('Final Accuracy', 0)) * (100 if row.get('Final Accuracy', 0) <= 1.0 else 1)):.2f}\\%"
        f1 = f"{float(row.get('Macro F1 (%)', row.get('Final Macro F1', 0)) * (100 if row.get('Final Macro F1', 0) <= 1.0 else 1)):.2f}\\%"
        params = f"{int(row.get('Params/round', row.get('Active Parameters', 0))):,}"
        savings = f"{float(row.get('Comm Savings (%)', row.get('Savings (%)', 0.0))):+.1f}\\%"

        lines.append(f"{method} & {alpha} & {acc} & {f1} & {params} & {savings} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def generate_markdown_table(df: Any) -> str:
    """
    Generate clean GitHub-flavored Markdown table matching Table 4.1.
    """
    headers = ["Method", "Dirichlet $\\alpha$", "Accuracy (%)", "Macro F1 (%)", "Params/Round", "Comm. Savings (%)"]
    sep = ["---", ":---:", ":---:", ":---:", ":---:", ":---:"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]

    for _, row in df.iterrows():
        method = str(row.get("Method", row.get("Algorithm", "")))
        alpha = f"{row.get('Alpha', ''):.1f}" if isinstance(row.get('Alpha'), (int, float)) else str(row.get('Alpha', ''))
        acc_raw = float(row.get("Accuracy (%)", row.get("Final Accuracy", 0)))
        acc_val = acc_raw * 100.0 if acc_raw <= 1.0 else acc_raw
        f1_raw = float(row.get("Macro F1 (%)", row.get("Final Macro F1", 0)))
        f1_val = f1_raw * 100.0 if f1_raw <= 1.0 else f1_raw
        params_val = int(row.get("Params/round", row.get("Active Parameters", 0)))
        sav_val = float(row.get("Comm Savings (%)", row.get("Savings (%)", 0.0)))

        lines.append(
            f"| **{method}** | ${alpha}$ | {acc_val:.2f}% | {f1_val:.2f}% | {params_val:,} | {sav_val:+.1f}% |"
        )
    return "\n".join(lines)

