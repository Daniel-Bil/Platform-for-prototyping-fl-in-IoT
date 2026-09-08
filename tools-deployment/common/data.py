"""Local sensor dataset loading compatible with the tools2 prepared FL dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = ("value_temp", "value_hum", "value_acid", "value_PV")


@dataclass(frozen=True)
class ClientData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    @property
    def train_samples(self) -> int:
        return int(len(self.X_train))


def _create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Build causal windows and classify the LAST sample inside each window.

    For an anomaly-detection / cleaning task, a window ``X[t-seq_len+1:t+1]``
    must be paired with ``y[t]``.  The previous implementation paired it with
    ``y[t+1]``, which silently changed the task into next-step anomaly
    prediction and was especially harmful around fault boundaries.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if len(X) < seq_len:
        return (
            np.empty((0, seq_len, X.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=y.dtype),
        )

    ends = range(seq_len - 1, len(X))
    Xs = np.stack([X[end - seq_len + 1 : end + 1] for end in ends])
    ys = np.asarray([y[end] for end in range(seq_len - 1, len(y))])
    return Xs.astype(np.float32, copy=False), ys


def _read_split(path: Path, features: tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing dataset split: {path}")
    frame = pd.read_csv(path)
    required = set(features) | {"label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def load_client_data(
    client_dir: str | Path,
    seq_len: int = 6,
    features: tuple[str, ...] = DEFAULT_FEATURES,
) -> ClientData:
    client_dir = Path(client_dir)
    train_df = _read_split(client_dir / "train.csv", features)
    val_df = _read_split(client_dir / "val.csv", features)
    test_df = _read_split(client_dir / "test.csv", features)

    scaler = StandardScaler()
    X_train_raw = scaler.fit_transform(train_df[list(features)]).astype(np.float32)
    X_val_raw = scaler.transform(val_df[list(features)]).astype(np.float32)
    X_test_raw = scaler.transform(test_df[list(features)]).astype(np.float32)

    X_train, y_train = _create_sequences(X_train_raw, train_df["label"].to_numpy(), seq_len)
    X_val, y_val = _create_sequences(X_val_raw, val_df["label"].to_numpy(), seq_len)
    X_test, y_test = _create_sequences(X_test_raw, test_df["label"].to_numpy(), seq_len)

    if not len(X_train):
        raise ValueError("training split contains too few rows for the configured sequence length")

    return ClientData(X_train, y_train, X_val, y_val, X_test, y_test)
