"""Client-side local training and evaluation strategies."""
from __future__ import annotations

from dataclasses import dataclass
import random
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
    final_objective: float | None = None


@dataclass(frozen=True)
class EvalResult:
    test_samples: int
    test_loss: float
    accuracy: float
    tp: int
    tn: int
    fp: int
    fn: int
    eval_seconds: float


class LocalTrainer:
    """Runs local FL work with a fresh optimizer/model for every operation."""

    def __init__(self, model_config: dict[str, Any], data: ClientData, seq_len: int, num_features: int):
        self.model_config = model_config
        self.data = data
        self.seq_len = seq_len
        self.num_features = num_features

    @staticmethod
    def _set_seed(seed: int | None) -> None:
        if seed is None:
            return
        import tensorflow as tf

        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)

    def _fresh_model(self, global_weights: Sequence[np.ndarray], seed: int | None = None):
        # A fresh model resets optimizer state each FL round, matching tools2 and
        # standard synchronous FL local-round semantics.
        self._set_seed(seed)
        model = build_model_from_config(
            self.model_config,
            seq_len=self.seq_len,
            num_features=self.num_features,
        )
        model.set_weights(list(global_weights))
        return model

    def train_standard(
        self,
        global_weights: Sequence[np.ndarray],
        local_epochs: int,
        batch_size: int,
        seed: int | None = None,
    ) -> TrainResult:
        """Ordinary local optimization used by FedAvg/FedMA/FedPAQ/HierFedAvg."""
        model = self._fresh_model(global_weights, seed=seed)
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

    # Backward-compatible name used by the first deployment release/tests.
    train_fedavg = train_standard

    def train_fedprox(
        self,
        global_weights: Sequence[np.ndarray],
        local_epochs: int,
        batch_size: int,
        mu: float,
        seed: int | None = None,
    ) -> TrainResult:
        """FedProx local training with proximal regularization.

        Objective:
            F_k(w) + (mu / 2) * ||w - w_global||^2

        The server still uses FedAvg-style sample-weighted aggregation; FedProx
        changes the local objective, not the aggregation rule.
        """
        if mu < 0:
            raise ValueError("FedProx mu must be non-negative")

        import tensorflow as tf

        model = self._fresh_model(global_weights, seed=seed)
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.BinaryCrossentropy()

        # After set_weights(), these constants are exactly the trainable global
        # parameters from the beginning of this federated round.
        global_trainable = [tf.constant(v.numpy()) for v in model.trainable_variables]

        @tf.function
        def train_step(x_batch, y_batch):
            with tf.GradientTape() as tape:
                predictions = model(x_batch, training=True)
                base_loss = loss_fn(y_batch, predictions)
                proximal_term = tf.add_n(
                    [
                        tf.reduce_sum(tf.square(local_var - global_var))
                        for local_var, global_var in zip(model.trainable_variables, global_trainable)
                    ]
                )
                objective = base_loss + (tf.cast(mu, base_loss.dtype) / 2.0) * proximal_term
            gradients = tape.gradient(objective, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return base_loss, objective, predictions

        dataset = (
            tf.data.Dataset.from_tensor_slices((self.data.X_train, self.data.y_train))
            .shuffle(
                max(1, min(len(self.data.X_train), 10000)),
                seed=seed,
                reshuffle_each_iteration=True,
            )
            .batch(batch_size)
        )

        started = time.perf_counter()
        final_base_loss: float | None = None
        final_objective: float | None = None
        final_accuracy: float | None = None

        for _epoch in range(local_epochs):
            mean_base = tf.keras.metrics.Mean()
            mean_objective = tf.keras.metrics.Mean()
            accuracy = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
            for x_batch, y_batch in dataset:
                base_loss, objective, predictions = train_step(x_batch, y_batch)
                mean_base.update_state(base_loss)
                mean_objective.update_state(objective)
                accuracy.update_state(tf.reshape(y_batch, tf.shape(predictions)), predictions)
            final_base_loss = float(mean_base.result().numpy())
            final_objective = float(mean_objective.result().numpy())
            final_accuracy = float(accuracy.result().numpy())

        elapsed = time.perf_counter() - started
        return TrainResult(
            weights=[np.asarray(w) for w in model.get_weights()],
            train_seconds=elapsed,
            final_loss=final_base_loss,
            final_accuracy=final_accuracy,
            final_objective=final_objective,
        )

    def evaluate(
        self,
        global_weights: Sequence[np.ndarray],
        batch_size: int = 256,
        seed: int | None = None,
    ) -> EvalResult:
        """Evaluate the global model locally and return only sufficient statistics.

        No test samples or predictions leave the client.  Confusion counts are
        enough for the cloud to reconstruct exact global accuracy/precision/
        recall/F1 across all participating clients.
        """
        if len(self.data.X_test) == 0:
            raise ValueError("test split contains no sequences")

        model = self._fresh_model(global_weights, seed=seed)
        started = time.perf_counter()
        probabilities = np.asarray(
            model.predict(self.data.X_test, batch_size=batch_size, verbose=0)
        ).reshape(-1)
        elapsed = time.perf_counter() - started

        y_true = np.asarray(self.data.y_test).reshape(-1).astype(np.int64)
        if probabilities.shape[0] != y_true.shape[0]:
            raise RuntimeError("prediction count does not match test labels")
        y_pred = (probabilities >= 0.5).astype(np.int64)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        # Binary cross entropy from probabilities.  This avoids shipping raw
        # predictions and is directly sample-weightable at the cloud.
        eps = np.finfo(np.float32).eps
        clipped = np.clip(probabilities.astype(np.float64), eps, 1.0 - eps)
        truth = y_true.astype(np.float64)
        test_loss = float(-np.mean(truth * np.log(clipped) + (1.0 - truth) * np.log(1.0 - clipped)))
        accuracy = float((tp + tn) / len(y_true))

        return EvalResult(
            test_samples=int(len(y_true)),
            test_loss=test_loss,
            accuracy=accuracy,
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
            eval_seconds=elapsed,
        )
