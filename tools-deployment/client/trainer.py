"""Client-side local training."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import numpy as np

from common.data import ClientData
from common.model import build_model_from_config


@dataclass(frozen=True)
class TrainResult:
    weights: list[np.ndarray]
    train_seconds: float
    final_loss: float | None
    final_accuracy: float | None


class LocalTrainer:
    """Runs local training while deliberately resetting optimizer state each FL round."""

    def __init__(self, model_config: dict[str, Any], data: ClientData, seq_len: int, num_features: int):
        self.model_config = model_config
        self.data = data
        self.seq_len = seq_len
        self.num_features = num_features

    def train_fedavg(
        self,
        global_weights: Sequence[np.ndarray],
        local_epochs: int,
        batch_size: int,
    ) -> TrainResult:
        # A fresh model also means a fresh Adam optimizer, matching the tools2
        # single-machine simulation and standard FedAvg local-round semantics.
        model = build_model_from_config(
            self.model_config,
            seq_len=self.seq_len,
            num_features=self.num_features,
        )
        model.set_weights(list(global_weights))
        started = time.perf_counter()
        history = model.fit(
            self.data.X_train,
            self.data.y_train,
            epochs=local_epochs,
            batch_size=batch_size,
            verbose=0,
            shuffle=True,
        )
        elapsed = time.perf_counter() - started
        loss_history = history.history.get("loss", [])
        accuracy_history = history.history.get("accuracy", [])
        return TrainResult(
            weights=[np.asarray(w) for w in model.get_weights()],
            train_seconds=elapsed,
            final_loss=float(loss_history[-1]) if loss_history else None,
            final_accuracy=float(accuracy_history[-1]) if accuracy_history else None,
        )
