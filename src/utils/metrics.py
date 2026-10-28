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
