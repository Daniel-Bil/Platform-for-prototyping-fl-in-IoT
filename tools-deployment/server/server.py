#!/usr/bin/env python3
"""Dynamic-client federated learning server.

No client count is configured. At the beginning of every round the server takes
a snapshot of all clients that are currently registered. Clients that connect
while a round is running remain connected and automatically join the next round.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import queue
import signal
import socket
import sys
import threading
import time
from typing import Any

import numpy as np

DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
if str(DEPLOYMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_ROOT))

from common.data import DEFAULT_FEATURES
from common.model import build_model_from_config, load_model_config
from common.protocol import ConnectionClosed, Message, ProtocolError, recv_message, send_message
from server.aggregation import weighted_fedavg

LOG = logging.getLogger("fl-server")


@dataclass
class ClientSession:
    client_id: str
    sock: socket.socket
    address: tuple[str, int]
    train_samples: int
    inbox: queue.Queue[Message] = field(default_factory=queue.Queue)
    send_lock: threading.Lock = field(default_factory=threading.Lock)
    alive: bool = True
    bytes_sent: int = 0
    bytes_received: int = 0

    def send(self, metadata: dict[str, Any], arrays=None) -> None:
        with self.send_lock:
            self.bytes_sent += send_message(self.sock, metadata, arrays)

    def close(self) -> None:
        self.alive = False
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


class ClientRegistry:
    def __init__(self) -> None:
        self._clients: dict[str, ClientSession] = {}
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)

    def register(self, session: ClientSession) -> None:
        with self._changed:
            old = self._clients.get(session.client_id)
            if old is not None and old is not session:
                LOG.warning("Replacing previous connection for client %s", session.client_id)
                old.close()
            self._clients[session.client_id] = session
            self._changed.notify_all()

    def remove(self, client_id: str, session: ClientSession) -> None:
        with self._changed:
            if self._clients.get(client_id) is session:
                self._clients.pop(client_id, None)
                self._changed.notify_all()

    def snapshot(self) -> list[ClientSession]:
        with self._lock:
            return [client for client in self._clients.values() if client.alive]

    def wait_for_any(self, shutdown: threading.Event) -> bool:
        with self._changed:
            while not shutdown.is_set() and not any(c.alive for c in self._clients.values()):
                self._changed.wait(timeout=1.0)
            return not shutdown.is_set()

    def close_all(self) -> None:
        for client in self.snapshot():
            client.close()


class FederatedServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.shutdown = threading.Event()
        self.registry = ClientRegistry()
        self.model_config = load_model_config(args.model)
        self.model = build_model_from_config(self.model_config, args.seq_len, len(args.features))
        self.global_weights = [np.asarray(w) for w in self.model.get_weights()]
        self.listener: socket.socket | None = None
        self.run_dir = self._create_run_dir(Path(args.results_dir))
        self.rounds_csv = self.run_dir / "rounds.csv"
        self._write_run_config()

    @staticmethod
    def _create_run_dir(base: Path) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = base / stamp
        suffix = 1
        while run_dir.exists():
            run_dir = base / f"{stamp}-{suffix}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _write_run_config(self) -> None:
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "algorithm": "FedAvg",
            "host": self.args.host,
            "port": self.args.port,
            "rounds": self.args.rounds,
            "local_epochs": self.args.local_epochs,
            "batch_size": self.args.batch_size,
            "seq_len": self.args.seq_len,
            "features": list(self.args.features),
            "round_timeout": self.args.round_timeout,
            "join_window": self.args.join_window,
            "model_path": str(Path(self.args.model).resolve()),
        }
        (self.run_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _client_reader(self, session: ClientSession) -> None:
        try:
            while not self.shutdown.is_set() and session.alive:
                message = recv_message(session.sock)
                session.bytes_received += message.wire_bytes
                session.inbox.put(message)
        except (ConnectionClosed, OSError, ProtocolError) as exc:
            if not self.shutdown.is_set():
                LOG.info("Client %s disconnected: %s", session.client_id, exc)
        finally:
            session.alive = False
            self.registry.remove(session.client_id, session)
            session.close()

    def _bootstrap_connection(self, client_sock: socket.socket, address: tuple[str, int]) -> None:
        try:
            client_sock.settimeout(self.args.handshake_timeout)
            hello = recv_message(client_sock)
            meta = hello.metadata
            if meta.get("type") != "HELLO":
                raise ProtocolError("first message must be HELLO")
            client_id = str(meta.get("client_id", "")).strip()
            if not client_id or len(client_id) > 128:
                raise ProtocolError("invalid client_id")
            if self.args.auth_token and meta.get("auth_token") != self.args.auth_token:
                raise ProtocolError("authentication failed")

            # The server is the source of truth for preprocessing and model config.
            welcome_bytes = send_message(
                client_sock,
                {
                    "type": "WELCOME",
                    "client_id": client_id,
                    "algorithm": "FedAvg",
                    "seq_len": self.args.seq_len,
                    "features": list(self.args.features),
                    "model_config": self.model_config,
                },
            )
            ready = recv_message(client_sock)
            if ready.metadata.get("type") != "READY":
                raise ProtocolError("client must send READY after WELCOME")
            if ready.metadata.get("client_id") != client_id:
                raise ProtocolError("READY client_id does not match HELLO")
            train_samples = int(ready.metadata.get("train_samples", 0))
            if train_samples <= 0:
                raise ProtocolError("client must report a positive train_samples value")

            client_sock.settimeout(None)
            session = ClientSession(client_id, client_sock, address, train_samples)
            session.bytes_received += hello.wire_bytes + ready.wire_bytes
            session.bytes_sent += welcome_bytes
            self.registry.register(session)
            LOG.info("Client %-20s ready from %s:%s (%d train samples)", client_id, *address, train_samples)
            threading.Thread(target=self._client_reader, args=(session,), daemon=True).start()
        except Exception as exc:
            LOG.warning("Rejected connection from %s:%s: %s", *address, exc)
            try:
                send_message(client_sock, {"type": "ERROR", "error": str(exc)})
            except Exception:
                pass
            try:
                client_sock.close()
            except OSError:
                pass

    def _accept_loop(self) -> None:
        assert self.listener is not None
        self.listener.settimeout(1.0)
        while not self.shutdown.is_set():
            try:
                client_sock, address = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if not self.shutdown.is_set():
                    LOG.exception("Listener failed")
                break
            threading.Thread(
                target=self._bootstrap_connection,
                args=(client_sock, address),
                daemon=True,
            ).start()

    def _weights_are_compatible(self, arrays: list[np.ndarray]) -> bool:
        if len(arrays) != len(self.global_weights):
            return False
        return all(
            np.asarray(candidate).shape == np.asarray(reference).shape
            and np.asarray(candidate).dtype.kind in "fiu"
            for candidate, reference in zip(arrays, self.global_weights)
        )

    def _collect_round_updates(
        self,
        cohort: list[ClientSession],
        round_id: int,
    ) -> tuple[list[list[np.ndarray]], list[int], list[str]]:
        pending = {client.client_id: client for client in cohort if client.alive}
        updates: list[list[np.ndarray]] = []
        sample_counts: list[int] = []
        successful_ids: list[str] = []
        deadline = time.monotonic() + self.args.round_timeout

        while pending and time.monotonic() < deadline and not self.shutdown.is_set():
            progressed = False
            for client_id, session in list(pending.items()):
                if not session.alive:
                    pending.pop(client_id, None)
                    continue
                try:
                    message = session.inbox.get_nowait()
                except queue.Empty:
                    continue
                progressed = True
                meta = message.metadata
                if meta.get("type") == "UPDATE" and int(meta.get("round", -1)) == round_id:
                    if not message.arrays:
                        LOG.warning("Round %d: %s returned an empty update", round_id, client_id)
                    elif not self._weights_are_compatible(message.arrays):
                        LOG.warning("Round %d: %s returned incompatible model tensors", round_id, client_id)
                    else:
                        updates.append(message.arrays)
                        sample_counts.append(session.train_samples)
                        successful_ids.append(client_id)
                        LOG.info(
                            "Round %d: update <- %s | train %.2fs | loss %s",
                            round_id,
                            client_id,
                            float(meta.get("train_seconds", 0.0)),
                            meta.get("final_loss"),
                        )
                    pending.pop(client_id, None)
                elif meta.get("type") == "CLIENT_ERROR" and int(meta.get("round", -1)) == round_id:
                    LOG.error("Round %d: client %s failed: %s", round_id, client_id, meta.get("error"))
                    pending.pop(client_id, None)
                else:
                    LOG.warning("Ignoring unexpected message from %s: %s", client_id, meta.get("type"))
            if not progressed:
                time.sleep(0.05)

        for client_id in pending:
            LOG.warning("Round %d: timed out waiting for %s", round_id, client_id)
        return updates, sample_counts, successful_ids

    def _append_round_result(self, row: dict[str, Any]) -> None:
        exists = self.rounds_csv.exists()
        with self.rounds_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _save_weights(self) -> None:
        np.savez_compressed(
            self.run_dir / "global_weights.npz",
            **{f"arr_{i:05d}": w for i, w in enumerate(self.global_weights)},
        )

    def run_rounds(self) -> None:
        round_id = 1
        while not self.shutdown.is_set() and (self.args.rounds == 0 or round_id <= self.args.rounds):
            LOG.info("Waiting for at least one connected client...")
            if not self.registry.wait_for_any(self.shutdown):
                break

            if self.args.join_window > 0:
                LOG.info("Round %d join window: %.1fs", round_id, self.args.join_window)
                self.shutdown.wait(self.args.join_window)
                if self.shutdown.is_set():
                    break

            cohort = self.registry.snapshot()
            if not cohort:
                continue

            cohort_ids = [client.client_id for client in cohort]
            LOG.info("=== ROUND %d | cohort=%s ===", round_id, ", ".join(cohort_ids))
            before_sent = sum(client.bytes_sent for client in cohort)
            before_recv = sum(client.bytes_received for client in cohort)

            for client in cohort:
                if not client.alive:
                    continue
                try:
                    client.send(
                        {
                            "type": "TRAIN",
                            "round": round_id,
                            "algorithm": "FedAvg",
                            "local_epochs": self.args.local_epochs,
                            "batch_size": self.args.batch_size,
                        },
                        self.global_weights,
                    )
                except OSError as exc:
                    LOG.warning("Round %d: failed to send to %s: %s", round_id, client.client_id, exc)
                    client.close()

            started = time.perf_counter()
            updates, sample_counts, successful_ids = self._collect_round_updates(cohort, round_id)
            aggregation_seconds = 0.0
            if updates:
                agg_started = time.perf_counter()
                self.global_weights = weighted_fedavg(updates, sample_counts)
                aggregation_seconds = time.perf_counter() - agg_started
                self._save_weights()
                LOG.info("Round %d aggregated %d update(s) in %.4fs", round_id, len(updates), aggregation_seconds)
            else:
                LOG.error("Round %d produced no valid updates; global model unchanged", round_id)

            elapsed = time.perf_counter() - started
            after_sent = sum(client.bytes_sent for client in cohort)
            after_recv = sum(client.bytes_received for client in cohort)
            self._append_round_result(
                {
                    "round": round_id,
                    "cohort_size": len(cohort),
                    "successful_updates": len(updates),
                    "client_ids": ";".join(cohort_ids),
                    "successful_client_ids": ";".join(successful_ids),
                    "train_samples": sum(sample_counts),
                    "round_seconds": round(elapsed, 6),
                    "aggregation_seconds": round(aggregation_seconds, 6),
                    "bytes_server_to_clients": max(0, after_sent - before_sent),
                    "bytes_clients_to_server": max(0, after_recv - before_recv),
                }
            )
            round_id += 1

    def serve(self) -> None:
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((self.args.host, self.args.port))
        self.listener.listen(self.args.backlog)
        LOG.info("Server listening on %s:%d", self.args.host, self.args.port)
        LOG.info("Results: %s", self.run_dir)
        threading.Thread(target=self._accept_loop, daemon=True).start()
        try:
            self.run_rounds()
        finally:
            self.stop()

    def stop(self) -> None:
        if self.shutdown.is_set():
            return
        self.shutdown.set()
        for client in self.registry.snapshot():
            try:
                client.send({"type": "STOP", "reason": "experiment_finished"})
            except Exception:
                pass
        self.registry.close_all()
        if self.listener is not None:
            try:
                self.listener.close()
            except OSError:
                pass
        self._save_weights()
        LOG.info("Server stopped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic-client federated learning server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--model", required=True, help="Path to a tools2-compatible model JSON")
    parser.add_argument("--rounds", type=int, default=3, help="0 = run until Ctrl+C")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES))
    parser.add_argument("--join-window", type=float, default=2.0, help="Seconds to admit newly connected clients before each round snapshot")
    parser.add_argument("--round-timeout", type=float, default=300.0)
    parser.add_argument("--handshake-timeout", type=float, default=10.0)
    parser.add_argument("--backlog", type=int, default=128)
    parser.add_argument("--results-dir", default=str(DEPLOYMENT_ROOT / "results"))
    parser.add_argument("--auth-token", default=os.environ.get("FL_AUTH_TOKEN"), help="Optional shared token; prefer FL_AUTH_TOKEN env var")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    server = FederatedServer(args)

    def request_stop(_signum=None, _frame=None):
        LOG.info("Shutdown requested")
        server.stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    server.serve()


if __name__ == "__main__":
    main()
