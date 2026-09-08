"""Metric aggregation helpers for distributed evaluation.

Besides ordinary confusion-matrix aggregation, this module contains the
validation-threshold selection used by the deployment platform.  Clients keep
their validation predictions local and send only confusion counts calculated
for a common threshold grid.  The cloud sums those counts and selects one
global threshold by macro-F1, so no raw sensor rows or prediction vectors have
to leave the clients.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence


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


def make_threshold_grid(
    minimum: float = 0.05,
    maximum: float = 0.95,
    step: float = 0.01,
) -> list[float]:
    """Return a deterministic inclusive probability-threshold grid."""
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("threshold minimum must be in [0, 1]")
    if not 0.0 <= maximum <= 1.0:
        raise ValueError("threshold maximum must be in [0, 1]")
    if maximum < minimum:
        raise ValueError("threshold maximum must be >= minimum")
    if step <= 0.0:
        raise ValueError("threshold step must be > 0")

    # Integer indexing avoids the usual floating-point arange surprises.
    count = int(round((maximum - minimum) / step))
    values = [minimum + idx * step for idx in range(count + 1)]
    if values[-1] < maximum - 1e-12:
        values.append(maximum)
    values[-1] = maximum
    return [round(float(value), 10) for value in values]


def aggregate_threshold_records(
    records: Iterable[dict[str, Any]],
    thresholds: Sequence[float],
) -> list[dict[str, float | int]]:
    """Sum distributed validation confusion counts for every threshold.

    Each record must contain ``threshold_counts`` in the same order as the
    supplied threshold grid.  The result includes ordinary metrics for every
    candidate threshold and can therefore be used without centralizing
    predictions.
    """
    thresholds = [float(value) for value in thresholds]
    records = list(records)
    if not thresholds:
        raise ValueError("threshold grid is empty")
    if not records:
        return []

    totals = [{"tp": 0, "tn": 0, "fp": 0, "fn": 0} for _ in thresholds]
    for record in records:
        candidates = record.get("threshold_counts")
        if not isinstance(candidates, list) or len(candidates) != len(thresholds):
            raise ValueError("validation record threshold grid length mismatch")
        for idx, (expected_threshold, candidate) in enumerate(zip(thresholds, candidates)):
            if not isinstance(candidate, dict):
                raise ValueError("invalid validation threshold record")
            candidate_threshold = float(candidate.get("threshold"))
            if abs(candidate_threshold - expected_threshold) > 1e-8:
                raise ValueError("validation threshold grid mismatch")
            for key in ("tp", "tn", "fp", "fn"):
                value = int(candidate.get(key, -1))
                if value < 0:
                    raise ValueError("negative validation confusion count")
                totals[idx][key] += value

    result: list[dict[str, float | int]] = []
    for threshold, counts in zip(thresholds, totals):
        metrics = summarize_confusion(
            tp=int(counts["tp"]),
            tn=int(counts["tn"]),
            fp=int(counts["fp"]),
            fn=int(counts["fn"]),
        )
        result.append({"threshold": float(threshold), **metrics})
    return result


def select_threshold_by_macro_f1(
    records: Iterable[dict[str, Any]],
    thresholds: Sequence[float],
    preferred_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Select one global validation threshold by maximum macro-F1.

    Ties are resolved deterministically by choosing the threshold closest to
    ``preferred_threshold`` (0.5 by default), then the lower threshold.  The
    returned object is the full aggregated validation metric row for the
    selected candidate.
    """
    aggregated = aggregate_threshold_records(records, thresholds)
    if not aggregated:
        raise ValueError("cannot select a threshold without validation records")

    return max(
        aggregated,
        key=lambda row: (
            float(row["macro_f1"]),
            -abs(float(row["threshold"]) - float(preferred_threshold)),
            -float(row["threshold"]),
        ),
    )
