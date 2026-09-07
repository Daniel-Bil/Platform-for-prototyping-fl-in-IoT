"""Metric aggregation helpers for distributed evaluation."""
from __future__ import annotations

from typing import Any, Iterable


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_confusion(tp: int, tn: int, fp: int, fn: int) -> dict[str, float | int]:
    total = tp + tn + fp + fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    # Class-0 F1: treating normal/negative as the positive class.
    normal_precision = _safe_div(tn, tn + fn)
    normal_recall = _safe_div(tn, tn + fp)
    normal_f1 = _safe_div(2.0 * normal_precision * normal_recall, normal_precision + normal_recall)

    return {
        "test_samples": int(total),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": _safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "normal_f1": normal_f1,
        "macro_f1": (f1 + normal_f1) / 2.0,
    }


def aggregate_eval_records(records: Iterable[dict[str, Any]]) -> dict[str, float | int | None]:
    records = list(records)
    if not records:
        return {
            "test_samples": 0,
            "test_loss": None,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "specificity": None,
            "f1": None,
            "normal_f1": None,
            "macro_f1": None,
        }

    tp = sum(int(r["tp"]) for r in records)
    tn = sum(int(r["tn"]) for r in records)
    fp = sum(int(r["fp"]) for r in records)
    fn = sum(int(r["fn"]) for r in records)
    counts = summarize_confusion(tp, tn, fp, fn)
    total_samples = int(sum(int(r["test_samples"]) for r in records))
    weighted_loss_num = sum(float(r["test_loss"]) * int(r["test_samples"]) for r in records)
    counts["test_samples"] = total_samples
    counts["test_loss"] = float(weighted_loss_num / total_samples) if total_samples else None
    return counts
