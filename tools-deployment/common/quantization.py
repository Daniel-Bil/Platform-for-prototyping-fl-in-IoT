"""Quantization helpers used by the FedPAQ deployment path.

The single-machine tools2 implementation performs per-tensor uniform min/max
quantization.  In the deployment implementation we keep the same numerical
idea, but actually transmit the quantized integers over the wire instead of
immediately dequantizing them on the client.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def quantize_weights(
    weights: Sequence[np.ndarray],
    bits: int = 8,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Uniformly quantize each tensor to unsigned integers.

    Returns the integer tensors plus JSON-serializable per-tensor metadata that
    is sufficient for deterministic dequantization on the server.
    """
    if bits < 1 or bits > 8:
        raise ValueError("FedPAQ bits must be between 1 and 8")

    levels = (1 << bits) - 1
    quantized: list[np.ndarray] = []
    params: list[dict[str, Any]] = []

    for value in weights:
        tensor = np.asarray(value)
        if tensor.dtype.kind not in "fiu":
            raise ValueError(f"cannot quantize tensor with dtype {tensor.dtype}")

        float_tensor = tensor.astype(np.float32, copy=False)
        min_value = float(np.min(float_tensor))
        max_value = float(np.max(float_tensor))

        if max_value == min_value:
            scale = 0.0
            q = np.zeros(float_tensor.shape, dtype=np.uint8)
        else:
            scale = (max_value - min_value) / float(levels)
            q = np.rint((float_tensor - min_value) / scale)
            q = np.clip(q, 0, levels).astype(np.uint8)

        quantized.append(q)
        params.append(
            {
                "bits": bits,
                "min": min_value,
                "scale": scale,
                "shape": list(float_tensor.shape),
            }
        )

    return quantized, params


def dequantize_weights(
    quantized: Sequence[np.ndarray],
    params: Sequence[dict[str, Any]],
    reference_weights: Sequence[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Restore quantized FedPAQ tensors to floating-point tensors."""
    if len(quantized) != len(params):
        raise ValueError("quantized tensors and quantization parameters differ in length")
    if reference_weights is not None and len(quantized) != len(reference_weights):
        raise ValueError("quantized tensors do not match reference model tensor count")

    restored: list[np.ndarray] = []
    for index, (q_value, meta) in enumerate(zip(quantized, params)):
        q = np.asarray(q_value)
        bits = int(meta.get("bits", 0))
        if bits < 1 or bits > 8:
            raise ValueError(f"invalid FedPAQ bit width for tensor {index}: {bits}")
        if q.dtype.kind not in "ui":
            raise ValueError(f"FedPAQ tensor {index} is not integer encoded")

        expected_shape = tuple(int(dim) for dim in meta.get("shape", []))
        if expected_shape and q.shape != expected_shape:
            raise ValueError(
                f"FedPAQ tensor {index} shape mismatch: {q.shape} != {expected_shape}"
            )

        min_value = float(meta["min"])
        scale = float(meta["scale"])
        if scale == 0.0:
            value = np.full(q.shape, min_value, dtype=np.float32)
        else:
            value = min_value + q.astype(np.float32) * scale

        if reference_weights is not None:
            reference = np.asarray(reference_weights[index])
            if value.shape != reference.shape:
                raise ValueError(
                    f"FedPAQ tensor {index} does not match global model shape "
                    f"{value.shape} != {reference.shape}"
                )
            dtype = reference.dtype if reference.dtype.kind == "f" else np.float32
            value = value.astype(dtype, copy=False)

        restored.append(value)

    return restored
