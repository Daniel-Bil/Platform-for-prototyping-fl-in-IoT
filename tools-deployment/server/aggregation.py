"""Server-side aggregation strategies."""
from __future__ import annotations

from typing import Sequence
import numpy as np


def weighted_fedavg(
    updates: Sequence[Sequence[np.ndarray]],
    sample_counts: Sequence[int],
) -> list[np.ndarray]:
    """Sample-weighted Federated Averaging.

    w_global = sum(n_k * w_k) / sum(n_k)
    """
    if not updates:
        raise ValueError("cannot aggregate zero client updates")
    if len(updates) != len(sample_counts):
        raise ValueError("updates and sample_counts must have equal length")
    if any(count <= 0 for count in sample_counts):
        raise ValueError("every client sample count must be positive")

    expected_layers = len(updates[0])
    if any(len(update) != expected_layers for update in updates):
        raise ValueError("client updates have different numbers of tensors")

    total = float(sum(sample_counts))
    result: list[np.ndarray] = []
    for layer_index in range(expected_layers):
        reference = np.asarray(updates[0][layer_index])
        accumulator = np.zeros(reference.shape, dtype=np.float64)
        for client_weights, count in zip(updates, sample_counts):
            layer = np.asarray(client_weights[layer_index])
            if layer.shape != reference.shape:
                raise ValueError(f"tensor shape mismatch at index {layer_index}")
            accumulator += layer.astype(np.float64, copy=False) * float(count)
        result.append((accumulator / total).astype(reference.dtype, copy=False))
    return result
