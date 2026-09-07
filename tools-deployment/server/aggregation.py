"""Server-side aggregation strategies for the deployment platform."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _validate_updates(updates: Sequence[Sequence[np.ndarray]]) -> int:
    if not updates:
        raise ValueError("cannot aggregate zero client updates")
    expected_layers = len(updates[0])
    if expected_layers == 0:
        raise ValueError("client update contains zero tensors")
    if any(len(update) != expected_layers for update in updates):
        raise ValueError("client updates have different numbers of tensors")
    return expected_layers


def weighted_fedavg(
    updates: Sequence[Sequence[np.ndarray]],
    sample_counts: Sequence[int],
) -> list[np.ndarray]:
    """Sample-weighted Federated Averaging.

    w_global = sum(n_k * w_k) / sum(n_k)
    """
    expected_layers = _validate_updates(updates)
    if len(updates) != len(sample_counts):
        raise ValueError("updates and sample_counts must have equal length")
    if any(count <= 0 for count in sample_counts):
        raise ValueError("every client sample count must be positive")

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


def mean_fedavg(updates: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
    """Unweighted model average, useful at an explicit hierarchy level."""
    expected_layers = _validate_updates(updates)
    result: list[np.ndarray] = []
    for layer_index in range(expected_layers):
        reference = np.asarray(updates[0][layer_index])
        stacked = []
        for update in updates:
            layer = np.asarray(update[layer_index])
            if layer.shape != reference.shape:
                raise ValueError(f"tensor shape mismatch at index {layer_index}")
            stacked.append(layer.astype(np.float64, copy=False))
        result.append(np.mean(stacked, axis=0).astype(reference.dtype, copy=False))
    return result


def fedma_aggregate(
    global_weights: Sequence[np.ndarray],
    local_weights_list: Sequence[Sequence[np.ndarray]],
) -> list[np.ndarray]:
    """Federated Matched Averaging compatible with tools2/method_fedma.py.

    Hidden filters/neurons are aligned to the current global model using the
    Hungarian assignment algorithm before averaging.  The implementation
    supports Conv1D, Conv2D-shaped tensors, Dense layers and Conv->Flatten->Dense
    transitions.  This deliberately mirrors the thesis simulation semantics so
    distributed and single-machine experiments remain comparable.
    """
    if not local_weights_list:
        raise ValueError("cannot aggregate zero client updates")
    if any(len(weights) != len(global_weights) for weights in local_weights_list):
        raise ValueError("FedMA client tensor count differs from global model")

    # Import lazily so FedAvg/FedProx/FedPAQ deployments do not need scipy at
    # module-import time.
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    num_clients = len(local_weights_list)
    new_global_weights: list[np.ndarray] = []
    prev_input_perm: list[np.ndarray | None] = [None] * num_clients
    current_output_perm: list[np.ndarray | None] = [None] * num_clients

    for layer_idx, global_layer_raw in enumerate(global_weights):
        global_layer = np.asarray(global_layer_raw)
        layer_shape = global_layer.shape
        is_bias = len(layer_shape) == 1
        is_dense = len(layer_shape) == 2
        is_conv1d = len(layer_shape) == 3
        is_conv2d = len(layer_shape) == 4
        is_last_layer_weights = layer_idx >= len(global_weights) - 2
        matched_local_layers: list[np.ndarray] = []

        for client_idx, local_w in enumerate(local_weights_list):
            client_layer = np.asarray(local_w[layer_idx]).copy()
            if client_layer.shape != global_layer.shape:
                raise ValueError(f"FedMA tensor shape mismatch at index {layer_idx}")

            if is_bias:
                perm = current_output_perm[client_idx]
                if perm is not None and len(perm) == len(client_layer):
                    client_layer = client_layer[perm]
                matched_local_layers.append(client_layer)
                continue

            in_perm = prev_input_perm[client_idx]
            if in_perm is not None:
                if is_conv1d and len(in_perm) == client_layer.shape[1]:
                    client_layer = client_layer[:, in_perm, :]
                elif is_conv2d and len(in_perm) == client_layer.shape[2]:
                    client_layer = client_layer[:, :, in_perm, :]
                elif is_dense:
                    dim_in = client_layer.shape[0]
                    if len(in_perm) == dim_in:
                        client_layer = client_layer[in_perm, :]
                    elif len(in_perm) > 0 and dim_in % len(in_perm) == 0:
                        channels = len(in_perm)
                        spatial_or_time = dim_in // channels
                        out_units = client_layer.shape[1]
                        reshaped = client_layer.reshape(spatial_or_time, channels, out_units)
                        client_layer = reshaped[:, in_perm, :].reshape(-1, out_units)

            if is_last_layer_weights:
                current_output_perm[client_idx] = None
            else:
                if is_conv1d or is_conv2d:
                    num_filters = layer_shape[-1]
                    g_flat = global_layer.reshape(-1, num_filters).T
                    c_flat = client_layer.reshape(-1, num_filters).T
                elif is_dense:
                    g_flat = global_layer.T
                    c_flat = client_layer.T
                else:
                    if not layer_shape:
                        raise ValueError(f"FedMA cannot match scalar tensor {layer_idx}")
                    units = layer_shape[-1]
                    g_flat = global_layer.reshape(-1, units).T
                    c_flat = client_layer.reshape(-1, units).T

                cost_matrix = cdist(g_flat, c_flat, metric="euclidean")
                _, col_ind = linear_sum_assignment(cost_matrix)
                current_output_perm[client_idx] = col_ind

                if is_conv1d:
                    client_layer = client_layer[:, :, col_ind]
                elif is_conv2d:
                    client_layer = client_layer[:, :, :, col_ind]
                elif is_dense:
                    client_layer = client_layer[:, col_ind]

            matched_local_layers.append(client_layer)

        if not is_bias:
            prev_input_perm = list(current_output_perm)

        new_global_weights.append(
            np.mean(np.stack(matched_local_layers, axis=0), axis=0).astype(global_layer.dtype, copy=False)
        )

    return new_global_weights
