"""Evaluation: AUC / F1 / precision / recall / confusion for each probe and the ensemble.

Pure metrics over probability arrays — fully CPU, no model needed. Used by train.py and tests.
"""

from __future__ import annotations

import numpy as np


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(prob) >= threshold).astype(int)
    auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(auc),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def print_report(name: str, m: dict) -> None:
    print(
        f"{name:<16} AUC={m['auc']:.3f}  F1={m['f1']:.3f}  "
        f"P={m['precision']:.3f}  R={m['recall']:.3f}  "
        f"(TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']})"
    )
