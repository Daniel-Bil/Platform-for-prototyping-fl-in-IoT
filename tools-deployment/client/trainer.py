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
    final_proximal_term: float | None = None


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

        FedProx changes only the local objective. The cloud still performs the
        same sample-weighted averaging as FedAvg.

        This implementation intentionally uses ``model.fit`` with a custom
        Keras ``train_step`` rather than a Python loop over every batch.  The
        previous deployment implementation was mathematically correct, but
        paid a Python->TensorFlow call for each batch and made FedProx appear
        roughly three times slower than FedAvg on the small thesis model.
        Keeping the loop inside Keras also makes the mu=0 path directly
        comparable with ordinary FedAvg training (same optimizer, loss, batch
        size and shuffle semantics).
        """
        if mu < 0:
            raise ValueError("FedProx mu must be non-negative")

        import tensorflow as tf

        base_model = self._fresh_model(global_weights, seed=seed)
        global_trainable = [tf.constant(v.numpy()) for v in base_model.trainable_variables]

        class FedProxModel(tf.keras.Model):
            def __init__(self, *, inputs, outputs, global_reference, prox_mu):
                super().__init__(inputs=inputs, outputs=outputs, name="FedProxDeploymentModel")
                self._global_reference = list(global_reference)
                self._prox_mu = tf.constant(float(prox_mu), dtype=tf.float32)
                self.base_loss_tracker = tf.keras.metrics.Mean(name="base_loss")
                self.objective_tracker = tf.keras.metrics.Mean(name="objective")
                self.proximal_tracker = tf.keras.metrics.Mean(name="proximal_term")

            @property
            def metrics(self):
                # Keras resets every object returned here at each epoch.
                return [
                    self.base_loss_tracker,
                    self.objective_tracker,
                    self.proximal_tracker,
                    *self.compiled_metrics.metrics,
                ]

            def train_step(self, data):
                x_batch, y_batch = data
                with tf.GradientTape() as tape:
                    predictions = self(x_batch, training=True)
                    base_loss = self.compiled_loss(
                        y_batch, predictions, regularization_losses=self.losses
                    )
                    proximal_term = tf.add_n([
                        tf.reduce_sum(tf.square(local_var - global_var))
                        for local_var, global_var in zip(
                            self.trainable_variables, self._global_reference
                        )
                    ])
                    objective = base_loss + (
                        tf.cast(self._prox_mu, base_loss.dtype) / 2.0
                    ) * proximal_term

                gradients = tape.gradient(objective, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

                self.base_loss_tracker.update_state(base_loss)
                self.objective_tracker.update_state(objective)
                self.proximal_tracker.update_state(proximal_term)
                self.compiled_metrics.update_state(y_batch, predictions)

                return {metric.name: metric.result() for metric in self.metrics}

        # Functional wrapper shares the exact variables initialized in
        # base_model, therefore the global reference above corresponds to the
        # beginning of this federated round.
        model = FedProxModel(
            inputs=base_model.inputs,
            outputs=base_model.outputs,
            global_reference=global_trainable,
            prox_mu=mu,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5)],
        )

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

        def last(name: str) -> float | None:
            values = history.history.get(name, [])
            return float(values[-1]) if values else None

        # IMPORTANT: return only the underlying neural-network tensors.
        #
        # ``FedProxModel`` owns metric trackers (base_loss/objective/proximal
        # term and compiled metrics). Keras tracks those metric variables as
        # part of the wrapper object, so ``model.get_weights()`` can include
        # metric state in addition to the actual layer weights.  Those extra
        # tensors are local bookkeeping and must never be sent to the FL
        # server.  ``base_model`` shares the exact layer variables used by the
        # wrapper, so after training it contains the updated network weights
        # and nothing else.
        trained_weights = [np.asarray(w) for w in base_model.get_weights()]

        if len(trained_weights) != len(global_weights) or any(
            candidate.shape != np.asarray(reference).shape
            for candidate, reference in zip(trained_weights, global_weights)
        ):
            raise RuntimeError(
                "FedProx produced neural-network weights incompatible with the round global model"
            )

        return TrainResult(
            weights=trained_weights,
            train_seconds=elapsed,
            final_loss=last("base_loss"),
            final_accuracy=last("accuracy"),
            final_objective=last("objective"),
            final_proximal_term=last("proximal_term"),
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
