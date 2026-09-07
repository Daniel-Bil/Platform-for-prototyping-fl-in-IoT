#!/usr/bin/env python3
"""TensorFlow-free smoke tests for transport, aggregation and dynamic round cohorts."""
from __future__ import annotations

import argparse
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
from server.aggregation import weighted_fedavg
import server.server as server_module


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
                {"type": "UPDATE", "round": round_id, "client_id": client_id, "train_seconds": 0.01},
                [message.arrays[0] + delta],
            )
        sock.close()

    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(
            host="127.0.0.1",
            port=port,
            model="unused.json",
            rounds=2,
            local_epochs=1,
            batch_size=4,
            seq_len=6,
            features=["a", "b", "c", "d"],
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
        assert seen["B"] == [2]  # late client must not enter a round already in progress
        np.testing.assert_allclose(server.global_weights[0], np.array([3.5], dtype=np.float32))


if __name__ == "__main__":
    test_protocol_and_aggregation()
    test_dynamic_late_join()
    print("All deployment smoke tests passed.")
