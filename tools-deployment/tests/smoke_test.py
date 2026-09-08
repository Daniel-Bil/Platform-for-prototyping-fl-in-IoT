#!/usr/bin/env python3
"""TensorFlow-free smoke tests for deployment transport, aggregation and evaluation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.metrics import (
    aggregate_eval_records,
    make_threshold_grid,
    select_threshold_by_macro_f1,
    summarize_confusion,
)
from common.protocol import recv_message, send_message
from common.quantization import dequantize_weights, quantize_weights
from server.aggregation import fedma_aggregate, weighted_fedavg
import server.server as server_module
from edge.edge import HierFedAvgEdge


class FakeModel:
    def get_weights(self):
        return [np.array([0.0], dtype=np.float32)]


def patch_fake_server_model() -> None:
    server_module.load_model_config = lambda _path: {
        "architecture": {"nodes": [{"id": "input", "type": "Input"}], "edges": []}
    }
    server_module.build_model_from_config = lambda _config, _seq, _features: FakeModel()


def server_args(port: int, tmp: str, algorithm: str, rounds: int = 1, evaluate_every: int = 0):
    return argparse.Namespace(
        host="127.0.0.1",
        port=port,
        model="unused.json",
        algorithm=algorithm,
        rounds=rounds,
        local_epochs=1,
        batch_size=4,
        seq_len=6,
        features=["a", "b", "c", "d"],
        seed=42,
        run_id=None,
        campaign_id=None,
        repetition=None,
        requested_clients=0,
        fedprox_mu=0.01,
        fedpaq_bits=8,
        join_window=0.05,
        initial_join_window=None,
        round_timeout=5.0,
        evaluation_timeout=5.0,
        evaluate_every=evaluate_every,
        eval_batch_size=64,
        threshold_min=0.05,
        threshold_max=0.95,
        threshold_step=0.05,
        threshold_preferred=0.5,
        handshake_timeout=2.0,
        backlog=16,
        results_dir=tmp,
        auth_token=None,
        log_level="INFO",
    )


def free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def test_protocol_and_aggregation() -> None:
    left, right = socket.socketpair()
    arrays = [np.arange(6, dtype=np.float32).reshape(2, 3)]
    sender = threading.Thread(target=lambda: send_message(left, {"type": "PING"}, arrays))
    sender.start()
    message = recv_message(right)
    sender.join()
    left.close()
    right.close()
    assert message.metadata["type"] == "PING"
    np.testing.assert_array_equal(message.arrays[0], arrays[0])

    result = weighted_fedavg(
        [[np.array([1.0], dtype=np.float32)], [np.array([5.0], dtype=np.float32)]],
        [1, 3],
    )
    np.testing.assert_allclose(result[0], np.array([4.0], dtype=np.float32))


def test_metrics() -> None:
    result = summarize_confusion(tp=30, tn=50, fp=10, fn=10)
    assert result["test_samples"] == 100
    assert abs(result["accuracy"] - 0.8) < 1e-9
    assert abs(result["precision"] - 0.75) < 1e-9
    assert abs(result["recall"] - 0.75) < 1e-9
    assert abs(result["f1"] - 0.75) < 1e-9

    combined = aggregate_eval_records([
        {"test_samples": 10, "test_loss": 0.2, "tp": 4, "tn": 4, "fp": 1, "fn": 1},
        {"test_samples": 30, "test_loss": 0.6, "tp": 12, "tn": 12, "fp": 3, "fn": 3},
    ])
    assert combined["test_samples"] == 40
    assert abs(combined["test_loss"] - 0.5) < 1e-9
    assert abs(combined["accuracy"] - 0.8) < 1e-9

    thresholds = [0.3, 0.5, 0.7]
    selected = select_threshold_by_macro_f1([
        {
            "threshold_counts": [
                {"threshold": 0.3, "tp": 8, "tn": 2, "fp": 8, "fn": 2},
                {"threshold": 0.5, "tp": 7, "tn": 8, "fp": 2, "fn": 3},
                {"threshold": 0.7, "tp": 2, "tn": 10, "fp": 0, "fn": 8},
            ]
        }
    ], thresholds)
    assert abs(float(selected["threshold"]) - 0.5) < 1e-9
    assert make_threshold_grid(0.1, 0.3, 0.1) == [0.1, 0.2, 0.3]

    # Regression for the real-data sanity run: a fixed 0.5 threshold can
    # collapse to the majority class even when lower probability scores still
    # separate anomalies well. Validation macro-F1 must pick the useful lower
    # threshold instead of silently accepting the all-normal classifier.
    rescued = select_threshold_by_macro_f1([
        {
            "threshold_counts": [
                {"threshold": 0.1, "tp": 4, "tn": 14, "fp": 1, "fn": 1},
                {"threshold": 0.5, "tp": 0, "tn": 15, "fp": 0, "fn": 5},
            ]
        }
    ], [0.1, 0.5])
    assert abs(float(rescued["threshold"]) - 0.1) < 1e-9


def test_fedpaq_quantization_and_wire_reduction() -> None:
    rng = np.random.default_rng(42)
    original = [rng.normal(size=(128, 64)).astype(np.float32), rng.normal(size=(64,)).astype(np.float32)]
    quantized, params = quantize_weights(original, bits=8)
    restored = dequantize_weights(quantized, params, reference_weights=original)

    for source, candidate in zip(original, restored):
        dynamic_range = float(source.max() - source.min())
        tolerance = dynamic_range / 255.0 + 1e-6
        np.testing.assert_allclose(candidate, source, atol=tolerance, rtol=0)
        assert candidate.dtype == source.dtype

    left, right = socket.socketpair()
    float_wire = send_message(left, {"type": "FLOAT"}, original)
    _ = recv_message(right)
    quant_wire = send_message(right, {"type": "Q", "quantization": params}, quantized)
    _ = recv_message(left)
    left.close()
    right.close()
    assert quant_wire < float_wire


def test_fedma_permutation_matching() -> None:
    global_weights = [
        np.array([[[1.0, 10.0]]], dtype=np.float32),
        np.array([0.1, 0.9], dtype=np.float32),
        np.array([[2.0], [20.0]], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
    ]
    client_a = [value.copy() for value in global_weights]
    client_b = [
        np.array([[[10.0, 1.0]]], dtype=np.float32),
        np.array([0.9, 0.1], dtype=np.float32),
        np.array([[20.0], [2.0]], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
    ]
    matched = fedma_aggregate(global_weights, [client_a, client_b])
    for expected, actual in zip(global_weights, matched):
        np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_distributed_evaluation_and_result_files() -> None:
    patch_fake_server_model()
    port = free_port()

    def fake_client(client_id: str, train_samples: int, delta: float, counts: tuple[int, int, int, int], loss: float):
        sock = None
        deadline = time.time() + 3.0
        while time.time() < deadline:
            try:
                sock = socket.create_connection(("127.0.0.1", port), timeout=0.2)
                sock.settimeout(5.0)
                break
            except OSError:
                time.sleep(0.02)
        assert sock is not None
        send_message(sock, {"type": "HELLO", "client_id": client_id})
        welcome = recv_message(sock)
        assert welcome.metadata["type"] == "WELCOME"
        send_message(sock, {"type": "READY", "client_id": client_id, "train_samples": train_samples})
        while True:
            message = recv_message(sock)
            msg_type = message.metadata["type"]
            if msg_type == "STOP":
                break
            round_id = int(message.metadata["round"])
            if msg_type == "TRAIN":
                send_message(sock, {
                    "type": "UPDATE", "round": round_id, "client_id": client_id,
                    "algorithm": "FedAvg", "train_seconds": 0.01,
                    "final_loss": 0.1, "final_accuracy": 0.9,
                }, [message.arrays[0] + delta])
            elif msg_type == "VALIDATE_THRESHOLDS":
                tp, tn, fp, fn = counts
                n = tp + tn + fp + fn
                threshold_counts = [
                    {"threshold": float(t), "tp": tp, "tn": tn, "fp": fp, "fn": fn}
                    for t in message.metadata["thresholds"]
                ]
                send_message(sock, {
                    "type": "VALIDATION_RESULT", "round": round_id, "client_id": client_id,
                    "algorithm": "FedAvg", "val_samples": n, "val_loss": loss,
                    "threshold_counts": threshold_counts, "eval_seconds": 0.01,
                })
            elif msg_type == "EVALUATE":
                tp, tn, fp, fn = counts
                n = tp + tn + fp + fn
                send_message(sock, {
                    "type": "EVAL_RESULT", "round": round_id, "client_id": client_id,
                    "algorithm": "FedAvg", "test_samples": n, "test_loss": loss,
                    "accuracy": (tp + tn) / n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                    "eval_seconds": 0.02,
                })
        sock.close()

    with tempfile.TemporaryDirectory() as tmp:
        server = server_module.FederatedServer(server_args(port, tmp, "FedAvg", evaluate_every=1))
        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        a = threading.Thread(target=fake_client, args=("A", 10, 1.0, (4, 4, 1, 1), 0.2))
        b = threading.Thread(target=fake_client, args=("B", 30, 3.0, (12, 12, 3, 3), 0.6))
        a.start(); b.start()
        a.join(8.0); b.join(8.0); server_thread.join(8.0)
        assert not server_thread.is_alive()

        # Weighted FedAvg: (10*1 + 30*3) / 40 = 2.5
        np.testing.assert_allclose(server.global_weights[0], np.array([2.5], dtype=np.float32))
        assert len(server.round_rows) == 1
        row = server.round_rows[0]
        assert row["test_samples"] == 40
        assert abs(row["test_loss"] - 0.5) < 1e-9
        assert abs(row["accuracy"] - 0.8) < 1e-9
        assert abs(float(row["selected_threshold"]) - 0.5) < 1e-9
        assert abs(float(row["validation_macro_f1"]) - 0.8) < 1e-9
        assert row["bytes_total_train"] > 0
        assert row["bytes_total_eval"] > 0

        assert (server.run_dir / "config.json").exists()
        assert (server.run_dir / "rounds.csv").exists()
        assert (server.run_dir / "participants.csv").exists()
        assert (server.run_dir / "summary.json").exists()
        assert (server.run_dir / "global_weights.npz").exists()
        summary = json.loads((server.run_dir / "summary.json").read_text())
        assert summary["completed_rounds"] == 1
        assert abs(summary["final_metrics"]["f1"] - 0.8) < 1e-9
        with (server.run_dir / "participants.csv").open(newline="", encoding="utf-8") as handle:
            participant_rows = list(csv.DictReader(handle))
        assert {r["participant_id"] for r in participant_rows} == {"A", "B"}


def test_hierfedavg_sample_weighted_edge_path() -> None:
    patch_fake_server_model()
    cloud_port = free_port()
    edge_port = free_port()

    with tempfile.TemporaryDirectory() as tmp:
        cloud = server_module.FederatedServer(server_args(cloud_port, tmp, "HierFedAvg", evaluate_every=1))
        cloud_thread = threading.Thread(target=cloud.serve, daemon=True)
        cloud_thread.start()

        edge_args = argparse.Namespace(
            edge_id="edge-A", cloud="127.0.0.1", cloud_port=cloud_port,
            listen_host="127.0.0.1", listen_port=edge_port, child_join_window=0.05,
            child_wait_timeout=3.0, round_timeout=4.0, handshake_timeout=2.0,
            backlog=16, reconnect_delay=0.02, connect_timeout=3.0,
            auth_token=None, log_level="INFO",
        )
        edge = HierFedAvgEdge(edge_args)
        edge_thread = threading.Thread(target=lambda: (edge.run(), edge.stop()), daemon=True)
        edge_thread.start()

        def fake_child(client_id: str, delta: float, samples: int):
            sock = None
            deadline = time.time() + 3.0
            while time.time() < deadline:
                try:
                    sock = socket.create_connection(("127.0.0.1", edge_port), timeout=0.2)
                    sock.settimeout(5.0)
                    break
                except OSError:
                    time.sleep(0.02)
            assert sock is not None
            send_message(sock, {"type": "HELLO", "client_id": client_id})
            welcome = recv_message(sock)
            assert welcome.metadata["algorithm"] == "HierFedAvg"
            send_message(sock, {"type": "READY", "client_id": client_id, "train_samples": samples})
            train = recv_message(sock)
            assert train.metadata["type"] == "TRAIN"
            round_id = int(train.metadata["round"])
            send_message(sock, {
                "type": "UPDATE", "round": round_id,
                "client_id": client_id, "algorithm": "HierFedAvg", "train_seconds": 0.01,
            }, [train.arrays[0] + delta])
            validate = recv_message(sock)
            assert validate.metadata["type"] == "VALIDATE_THRESHOLDS"
            thresholds = [float(value) for value in validate.metadata["thresholds"]]
            # Same balanced 80%-accurate confusion matrix for every candidate;
            # the cloud tie-break therefore selects 0.5.
            val_n = samples
            val_tp = int(val_n * 0.4)
            val_tn = int(val_n * 0.4)
            val_fp = int(val_n * 0.1)
            val_fn = val_n - val_tp - val_tn - val_fp
            send_message(sock, {
                "type": "VALIDATION_RESULT", "round": round_id, "client_id": client_id,
                "algorithm": "HierFedAvg", "val_samples": val_n, "val_loss": 0.5,
                "threshold_counts": [
                    {"threshold": t, "tp": val_tp, "tn": val_tn, "fp": val_fp, "fn": val_fn}
                    for t in thresholds
                ],
                "eval_seconds": 0.01,
            })
            evaluate = recv_message(sock)
            assert evaluate.metadata["type"] == "EVALUATE"
            assert abs(float(evaluate.metadata["threshold"]) - 0.5) < 1e-9
            # 80% accuracy on each child, with test size mirroring its train weight.
            test_n = samples
            tp = int(test_n * 0.4)
            tn = int(test_n * 0.4)
            fp = int(test_n * 0.1)
            fn = test_n - tp - tn - fp
            send_message(sock, {
                "type": "EVAL_RESULT", "round": round_id, "client_id": client_id,
                "algorithm": "HierFedAvg", "test_samples": test_n, "test_loss": 0.5,
                "accuracy": (tp + tn) / test_n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "eval_seconds": 0.01,
            })
            stop = recv_message(sock)
            assert stop.metadata["type"] == "STOP"
            sock.close()

        # Unequal samples prove local edge weighting: (10*1 + 30*3) / 40 = 2.5.
        c1 = threading.Thread(target=fake_child, args=("C1", 1.0, 10))
        c2 = threading.Thread(target=fake_child, args=("C2", 3.0, 30))
        c1.start(); c2.start()
        c1.join(8.0); c2.join(8.0); cloud_thread.join(8.0); edge_thread.join(8.0)
        assert not cloud_thread.is_alive()
        np.testing.assert_allclose(cloud.global_weights[0], np.array([2.5], dtype=np.float32))
        assert cloud.round_rows[0]["train_samples"] == 40
        assert cloud.round_rows[0]["bytes_edges_to_children_train"] > 0
        assert cloud.round_rows[0]["bytes_children_to_edges_train"] > 0
        assert cloud.round_rows[0]["bytes_edges_to_children_eval"] > 0
        assert cloud.round_rows[0]["bytes_children_to_edges_eval"] > 0
        assert cloud.round_rows[0]["test_samples"] == 40
        assert abs(cloud.round_rows[0]["accuracy"] - 0.8) < 1e-9


def test_dynamic_late_join() -> None:
    patch_fake_server_model()
    port = free_port()
    seen = {"A": [], "B": []}
    first_round_seen = threading.Event()

    def fake_client(client_id: str, delta: float, start_after: threading.Event | None = None) -> None:
        if start_after is not None:
            assert start_after.wait(3.0)
        sock = None
        deadline = time.time() + 3.0
        while time.time() < deadline:
            try:
                sock = socket.create_connection(("127.0.0.1", port), timeout=0.2)
                sock.settimeout(4.0)
                break
            except OSError:
                time.sleep(0.02)
        assert sock is not None
        send_message(sock, {"type": "HELLO", "client_id": client_id})
        welcome = recv_message(sock)
        samples = 10 if client_id == "A" else 30
        send_message(sock, {"type": "READY", "client_id": client_id, "train_samples": samples})
        while True:
            message = recv_message(sock)
            if message.metadata["type"] == "STOP":
                break
            assert message.metadata["type"] == "TRAIN"
            round_id = int(message.metadata["round"])
            seen[client_id].append(round_id)
            if client_id == "A" and round_id == 1:
                first_round_seen.set()
                time.sleep(0.15)
            send_message(sock, {
                "type": "UPDATE", "round": round_id, "client_id": client_id,
                "algorithm": "FedAvg", "train_seconds": 0.01,
            }, [message.arrays[0] + delta])
        sock.close()

    with tempfile.TemporaryDirectory() as tmp:
        server = server_module.FederatedServer(server_args(port, tmp, "FedAvg", rounds=2, evaluate_every=0))
        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        a = threading.Thread(target=fake_client, args=("A", 1.0))
        b = threading.Thread(target=fake_client, args=("B", 3.0, first_round_seen))
        a.start(); b.start()
        a.join(8.0); b.join(8.0); server_thread.join(8.0)
        assert seen["A"] == [1, 2]
        assert seen["B"] == [2]
        np.testing.assert_allclose(server.global_weights[0], np.array([3.5], dtype=np.float32))


if __name__ == "__main__":
    test_protocol_and_aggregation()
    test_metrics()
    test_fedpaq_quantization_and_wire_reduction()
    test_fedma_permutation_matching()
    test_distributed_evaluation_and_result_files()
    test_hierfedavg_sample_weighted_edge_path()
    test_dynamic_late_join()
    print("All deployment smoke tests passed.")
