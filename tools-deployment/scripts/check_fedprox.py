#!/usr/bin/env python3
"""Small on-device FedProx diagnostic.

Purpose:
  * compare ordinary local training with FedProx(mu=0), which should be a
    near-equivalent optimization problem;
  * measure the implementation overhead of the FedProx path;
  * report the real proximal term/objective for the configured mu.

Run this on a provisioned device using the deployment venv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
if str(DEPLOYMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_ROOT))

from common.data import ClientData, DEFAULT_FEATURES, load_client_data
from common.model import build_model_from_config, load_model_config
from client.trainer import LocalTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FedAvg and FedProx local training")
    parser.add_argument("--data", required=True, help="client dataset directory")
    parser.add_argument(
        "--model",
        default=str(DEPLOYMENT_ROOT / "config" / "default_model.json"),
        help="tools2-compatible model JSON",
    )
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit-train-samples",
        type=int,
        default=0,
        help="0 = full training split; otherwise use first N sequences for a quick diagnostic",
    )
    return parser.parse_args()


def limited(data: ClientData, count: int) -> ClientData:
    if count <= 0 or count >= data.train_samples:
        return data
    return ClientData(
        X_train=data.X_train[:count],
        y_train=data.y_train[:count],
        X_val=data.X_val,
        y_val=data.y_val,
        X_test=data.X_test,
        y_test=data.y_test,
    )


def weight_delta(a: list[np.ndarray], b: list[np.ndarray]) -> dict[str, float]:
    flat_a = np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in a])
    flat_b = np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in b])
    diff = flat_a - flat_b
    return {
        "l2": float(np.linalg.norm(diff)),
        "max_abs": float(np.max(np.abs(diff))) if len(diff) else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if len(diff) else 0.0,
    }


def main() -> None:
    args = parse_args()
    if args.mu < 0:
        raise SystemExit("--mu must be >= 0")

    data = limited(
        load_client_data(args.data, seq_len=args.seq_len, features=DEFAULT_FEATURES),
        args.limit_train_samples,
    )
    model_config = load_model_config(args.model)

    # Build one deterministic initial global state and reuse it for all three
    # local-training paths.
    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    initial_model = build_model_from_config(
        model_config, seq_len=args.seq_len, num_features=len(DEFAULT_FEATURES)
    )
    initial_weights = [np.asarray(w) for w in initial_model.get_weights()]

    trainer = LocalTrainer(model_config, data, args.seq_len, len(DEFAULT_FEATURES))
    fedavg = trainer.train_standard(
        initial_weights, args.epochs, args.batch_size, seed=args.seed
    )
    prox_zero = trainer.train_fedprox(
        initial_weights, args.epochs, args.batch_size, mu=0.0, seed=args.seed
    )
    prox = trainer.train_fedprox(
        initial_weights, args.epochs, args.batch_size, mu=args.mu, seed=args.seed
    )

    result = {
        "train_samples": data.train_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "fedavg": {
            "seconds": fedavg.train_seconds,
            "loss": fedavg.final_loss,
            "accuracy": fedavg.final_accuracy,
        },
        "fedprox_mu0": {
            "seconds": prox_zero.train_seconds,
            "loss": prox_zero.final_loss,
            "objective": prox_zero.final_objective,
            "proximal_term": prox_zero.final_proximal_term,
            "accuracy": prox_zero.final_accuracy,
            "weight_delta_vs_fedavg": weight_delta(prox_zero.weights, fedavg.weights),
        },
        "fedprox": {
            "mu": args.mu,
            "seconds": prox.train_seconds,
            "loss": prox.final_loss,
            "objective": prox.final_objective,
            "proximal_term": prox.final_proximal_term,
            "proximal_contribution": (
                None
                if prox.final_loss is None or prox.final_objective is None
                else prox.final_objective - prox.final_loss
            ),
            "accuracy": prox.final_accuracy,
            "weight_delta_vs_fedavg": weight_delta(prox.weights, fedavg.weights),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
