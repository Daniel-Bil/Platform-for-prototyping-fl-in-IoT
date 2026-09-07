#!/usr/bin/env python3
"""TensorFlow-free smoke tests for deployment transport and aggregators."""
from __future__ import annotations

import argparse
from io import BytesIO
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

from common.protocol import recv_message, send_message
from common.quantization import dequantize_weights, quantize_weights
from server.aggregation import fedma_aggregate, weighted_fedavg
import server.server as server_module
from edge.edge import HierFedAvgEdge


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
    # Tiny Conv1D -> Dense-shaped model. Client B represents the same hidden
    # filters in swapped order; FedMA should align it back to the global order.
    global_weights = [
        np.array([[[1.0, 10.0]]], dtype=np.float32),  # conv kernel [k,in,out]
        np.array([0.1, 0.9], dtype=np.float32),       # conv bias
        np.array([[2.0], [20.0]], dtype=np.float32),  # dense input by conv channel
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



def test_hierfedavg_real_edge_path() -> None:
    class FakeModel:
        def get_weights(self):
            return [np.array([0.0], dtype=np.float32)]

    server_module.load_model_config = lambda _path: {
        "architecture": {"nodes": [{"id": "input", "type": "Input"}], "edges": []}
    }
    server_module.build_model_from_config = lambda _config, _seq, _features: FakeModel()

    def free_port() -> int:
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        return port

    cloud_port = free_port()
    edge_port = free_port()

    with tempfile.TemporaryDirectory() as tmp:
        cloud_args = argparse.Namespace(
            host="127.0.0.1",
            port=cloud_port,
            model="unused.json",
            algorithm="HierFedAvg",
            rounds=1,
            local_epochs=1,
            batch_size=4,
            seq_len=6,
            features=["a", "b", "c", "d"],
            fedprox_mu=0.01,
            fedpaq_bits=8,
            join_window=0.05,
            round_timeout=5.0,
            handshake_timeout=2.0,
            backlog=16,
            results_dir=tmp,
            auth_token=None,
            log_level="INFO",
        )
        cloud = server_module.FederatedServer(cloud_args)
        cloud_thread = threading.Thread(target=cloud.serve, daemon=True)
        cloud_thread.start()

        edge_args = argparse.Namespace(
            edge_id="edge-A",
            cloud="127.0.0.1",
            cloud_port=cloud_port,
            listen_host="127.0.0.1",
            listen_port=edge_port,
            child_join_window=0.05,
            round_timeout=4.0,
            handshake_timeout=2.0,
            backlog=16,
            reconnect_delay=0.02,
            connect_timeout=3.0,
            auth_token=None,
            log_level="INFO",
        )
        edge = HierFedAvgEdge(edge_args)

        def edge_runner():
            try:
                edge.run()
            finally:
                edge.stop()

        edge_thread = threading.Thread(target=edge_runner, daemon=True)
        edge_thread.start()

        def fake_child(client_id: str, delta: float):
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
            send_message(sock, {"type": "READY", "client_id": client_id, "train_samples": 10})
            message = recv_message(sock)
            assert message.metadata["type"] == "TRAIN"
            assert message.metadata["algorithm"] == "HierFedAvg"
            send_message(
                sock,
                {
                    "type": "UPDATE",
                    "round": int(message.metadata["round"]),
                    "client_id": client_id,
                    "algorithm": "HierFedAvg",
                    "train_seconds": 0.01,
                },
                [message.arrays[0] + delta],
            )
            stop = recv_message(sock)
            assert stop.metadata["type"] == "STOP"
            sock.close()

        c1 = threading.Thread(target=fake_child, args=("C1", 1.0))
        c2 = threading.Thread(target=fake_child, args=("C2", 3.0))
        c1.start()
        c2.start()
        c1.join(8.0)
        c2.join(8.0)
        cloud_thread.join(8.0)
        edge_thread.join(8.0)

        assert not c1.is_alive()
        assert not c2.is_alive()
        assert not cloud_thread.is_alive()
        np.testing.assert_allclose(cloud.global_weights[0], np.array([2.0], dtype=np.float32))

def test_dynamic_late_join() -> None:
    class FakeModel:
        def get_weights(self):
            return [np.array([0.0], dtype=np.float32)]

    server_module.load_model_config = lambda _path: {
        "architecture": {"nodes": [{"id": "input", "type": "Input"}], "edges": []}
    }
    server_module.build_model_from_config = lambda _config, _seq, _features: FakeModel()

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

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
        assert welcome.metadata["type"] == "WELCOME"
        assert welcome.metadata["algorithm"] == "FedAvg"
        samples = 10 if client_id == "A" else 30
        send_message(sock, {"type": "READY", "client_id": client_id, "train_samples": samples})

        while True:
            message = recv_message(sock)
            if message.metadata["type"] == "STOP":
                break
            round_id = int(message.metadata["round"])
            seen[client_id].append(round_id)
            if client_id == "A" and round_id == 1:
                first_round_seen.set()
                time.sleep(0.15)
            send_message(
                sock,
                {
                    "type": "UPDATE",
                    "round": round_id,
                    "client_id": client_id,
                    "algorithm": "FedAvg",
                    "train_seconds": 0.01,
                },
                [message.arrays[0] + delta],
            )
        sock.close()

    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(
            host="127.0.0.1",
            port=port,
            model="unused.json",
            algorithm="FedAvg",
            rounds=2,
            local_epochs=1,
            batch_size=4,
            seq_len=6,
            features=["a", "b", "c", "d"],
            fedprox_mu=0.01,
            fedpaq_bits=8,
            join_window=0.1,
            round_timeout=3.0,
            handshake_timeout=2.0,
            backlog=16,
            results_dir=tmp,
            auth_token=None,
            log_level="INFO",
        )
        server = server_module.FederatedServer(args)
        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        a = threading.Thread(target=fake_client, args=("A", 1.0))
        b = threading.Thread(target=fake_client, args=("B", 3.0, first_round_seen))
        a.start()
        b.start()
        a.join(8.0)
        b.join(8.0)
        server_thread.join(8.0)

        assert seen["A"] == [1, 2]
        assert seen["B"] == [2]
        np.testing.assert_allclose(server.global_weights[0], np.array([3.5], dtype=np.float32))


if __name__ == "__main__":
    test_protocol_and_aggregation()
    test_fedpaq_quantization_and_wire_reduction()
    test_fedma_permutation_matching()
    test_hierfedavg_real_edge_path()
    test_dynamic_late_join()
    print("All deployment smoke tests passed.")
