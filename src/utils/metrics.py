"""
DermaSense AI — Metrics Utility
Tính toán các metrics chuyên dụng cho bài toán da liễu.
"""
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Tính toàn bộ metrics cho bài toán phân loại da liễu.

    Args:
        y_true: Nhãn thật (shape: [N,]).
        y_pred: Nhãn dự đoán (shape: [N,]).
        y_prob: Xác suất softmax (shape: [N, C]) — cần cho AUC-ROC.
        class_names: Tên các lớp bệnh.

    Returns:
        Dictionary chứa tất cả metrics.
    """
    metrics = {}

    # Accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    # Per-class & weighted metrics
    metrics["precision_weighted"] = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["recall_weighted"] = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["f1_weighted"] = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Cohen's Kappa
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)

    # AUC-ROC (cần xác suất)
    if y_prob is not None:
        try:
            num_classes = y_prob.shape[1]
            if num_classes == 2:
                metrics["auc_roc"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics["auc_roc_macro"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )
        except ValueError:
            # Nếu có lớp chỉ xuất hiện 1 lần trong y_true
            metrics["auc_roc_macro"] = float("nan")

    return metrics


def compute_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: int = 1,
) -> Dict[str, float]:
    """Tính Sensitivity, Specificity, NPV cho bài toán nhị phân (Stage A).

    Args:
        y_true: Nhãn thật nhị phân.
        y_pred: Nhãn dự đoán nhị phân.
        positive_label: Nhãn "cần chú ý" (mặc định 1).

    Returns:
        Dictionary chứa sensitivity, specificity, npv, ppv.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # = Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> str:
    """In báo cáo phân loại chi tiết (wrapper sklearn).

    Returns:
        Report string.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    print(report)
    return report
