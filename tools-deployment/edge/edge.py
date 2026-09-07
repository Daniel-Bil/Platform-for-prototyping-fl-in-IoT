#!/usr/bin/env python3
"""HierFedAvg edge aggregator.

The edge is a client of the cloud server and simultaneously a server for normal
training clients.  It receives the global model from the cloud, broadcasts it
to its current child cohort, averages child updates locally, and returns one
edge-level model to the cloud.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import queue
import socket
import sys
import threading
import time
from typing import Any

import numpy as np

DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
if str(DEPLOYMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_ROOT))

from common.protocol import ConnectionClosed, Message, ProtocolError, recv_message, send_message
from common.sessions import ClientRegistry, ClientSession
from server.aggregation import mean_fedavg

LOG = logging.getLogger("fl-edge")


class HierFedAvgEdge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.shutdown = threading.Event()
        self.registry = ClientRegistry()
        self.listener: socket.socket | None = None
        self.cloud_sock: socket.socket | None = None
        self.model_config: dict[str, Any] | None = None
        self.seq_len: int | None = None
        self.features: list[str] | None = None

    def _child_reader(self, session: ClientSession) -> None:
        try:
            while not self.shutdown.is_set() and session.alive:
                message = recv_message(session.sock)
                session.bytes_received += message.wire_bytes
                session.inbox.put(message)
        except (ConnectionClosed, OSError, ProtocolError) as exc:
            if not self.shutdown.is_set():
                LOG.info("Child %s disconnected: %s", session.client_id, exc)
        finally:
            session.alive = False
            self.registry.remove(session.client_id, session)
            session.close()

    def _bootstrap_child(self, child_sock: socket.socket, address: tuple[str, int]) -> None:
        try:
            if self.model_config is None or self.seq_len is None or self.features is None:
                raise ProtocolError("edge is not configured by cloud yet")
            child_sock.settimeout(self.args.handshake_timeout)
            hello = recv_message(child_sock)
            meta = hello.metadata
            if meta.get("type") != "HELLO":
                raise ProtocolError("first message must be HELLO")
            client_id = str(meta.get("client_id", "")).strip()
            if not client_id or len(client_id) > 128:
                raise ProtocolError("invalid client_id")
            role = str(meta.get("role", "client"))
            if role != "client":
                raise ProtocolError("edge accepts role='client' only")
            if self.args.auth_token and meta.get("auth_token") != self.args.auth_token:
                raise ProtocolError("authentication failed")

            welcome_bytes = send_message(
                child_sock,
                {
                    "type": "WELCOME",
                    "client_id": client_id,
                    "role": "client",
                    "algorithm": "HierFedAvg",
                    "seq_len": self.seq_len,
                    "features": self.features,
                    "model_config": self.model_config,
                },
            )
            ready = recv_message(child_sock)
            if ready.metadata.get("type") != "READY":
                raise ProtocolError("client must send READY after WELCOME")
            if ready.metadata.get("client_id") != client_id:
                raise ProtocolError("READY client_id does not match HELLO")
            train_samples = int(ready.metadata.get("train_samples", 0))
            if train_samples <= 0:
                raise ProtocolError("client must report positive train_samples")

            child_sock.settimeout(None)
            session = ClientSession(client_id, child_sock, address, train_samples, role="client")
            session.bytes_received += hello.wire_bytes + ready.wire_bytes
            session.bytes_sent += welcome_bytes
            self.registry.register(session)
            LOG.info("Child %-20s ready from %s:%s (%d samples)", client_id, *address, train_samples)
            threading.Thread(target=self._child_reader, args=(session,), daemon=True).start()
        except Exception as exc:
            LOG.warning("Rejected child from %s:%s: %s", *address, exc)
            try:
                send_message(child_sock, {"type": "ERROR", "error": str(exc)})
            except Exception:
                pass
            try:
                child_sock.close()
            except OSError:
                pass

    def _accept_loop(self) -> None:
        assert self.listener is not None
        self.listener.settimeout(1.0)
        while not self.shutdown.is_set():
            try:
                child_sock, address = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if not self.shutdown.is_set():
                    LOG.exception("Edge listener failed")
                return
            threading.Thread(target=self._bootstrap_child, args=(child_sock, address), daemon=True).start()

    def _start_child_listener(self) -> None:
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((self.args.listen_host, self.args.listen_port))
        self.listener.listen(self.args.backlog)
        LOG.info("Edge %s listening for children on %s:%d", self.args.edge_id, self.args.listen_host, self.args.listen_port)
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _connect_cloud(self) -> socket.socket:
        deadline = None if self.args.connect_timeout <= 0 else time.monotonic() + self.args.connect_timeout
        while not self.shutdown.is_set():
            try:
                sock = socket.create_connection((self.args.cloud, self.args.cloud_port), timeout=10.0)
                sock.settimeout(None)
                return sock
            except OSError as exc:
                if deadline is not None and time.monotonic() >= deadline:
                    raise ConnectionError("unable to connect to cloud") from exc
                LOG.warning("Cloud unavailable (%s); retrying in %.1fs", exc, self.args.reconnect_delay)
                self.shutdown.wait(self.args.reconnect_delay)
        raise ConnectionError("edge stopped before cloud connection")

    @staticmethod
    def _compatible(arrays: list[np.ndarray], global_weights: list[np.ndarray]) -> bool:
        return len(arrays) == len(global_weights) and all(
            np.asarray(candidate).shape == np.asarray(reference).shape
            and np.asarray(candidate).dtype.kind in "fiu"
            for candidate, reference in zip(arrays, global_weights)
        )

    def _collect_child_updates(
        self,
        cohort: list[ClientSession],
        round_id: int,
        global_weights: list[np.ndarray],
    ) -> tuple[list[list[np.ndarray]], list[str], int]:
        pending = {client.client_id: client for client in cohort if client.alive}
        updates: list[list[np.ndarray]] = []
        successful_ids: list[str] = []
        successful_samples = 0
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
                    if str(meta.get("algorithm", "")) != "HierFedAvg":
                        LOG.warning("Round %d: child %s returned wrong algorithm", round_id, client_id)
                    elif not message.arrays or not self._compatible(message.arrays, global_weights):
                        LOG.warning("Round %d: child %s returned incompatible tensors", round_id, client_id)
                    else:
                        updates.append([np.asarray(v) for v in message.arrays])
                        successful_ids.append(client_id)
                        successful_samples += session.train_samples
                        LOG.info(
                            "Round %d: child update <- %s | train %.2fs | loss %s",
                            round_id,
                            client_id,
                            float(meta.get("train_seconds", 0.0)),
                            meta.get("final_loss"),
                        )
                    pending.pop(client_id, None)
                elif meta.get("type") == "CLIENT_ERROR" and int(meta.get("round", -1)) == round_id:
                    LOG.error("Round %d: child %s failed: %s", round_id, client_id, meta.get("error"))
                    pending.pop(client_id, None)
                else:
                    LOG.warning("Ignoring unexpected child message from %s: %s", client_id, meta.get("type"))
            if not progressed:
                time.sleep(0.05)

        for client_id in pending:
            LOG.warning("Round %d: timed out waiting for child %s", round_id, client_id)
        return updates, successful_ids, successful_samples

    def _run_cloud_round(self, message: Message) -> None:
        assert self.cloud_sock is not None
        meta = message.metadata
        round_id = int(meta["round"])
        if str(meta.get("algorithm", "")) != "HierFedAvg":
            raise ProtocolError("edge received non-HierFedAvg TRAIN message")
        global_weights = [np.asarray(value) for value in message.arrays]
        if not global_weights:
            raise ProtocolError("cloud TRAIN contains no global weights")

        if not self.registry.snapshot():
            LOG.info(
                "Round %d: waiting up to %.1fs for at least one child client",
                round_id,
                self.args.child_wait_timeout,
            )
            deadline = time.monotonic() + self.args.child_wait_timeout
            while (
                not self.shutdown.is_set()
                and not self.registry.snapshot()
                and time.monotonic() < deadline
            ):
                self.shutdown.wait(0.1)
            if not self.registry.snapshot():
                send_message(
                    self.cloud_sock,
                    {
                        "type": "CLIENT_ERROR",
                        "round": round_id,
                        "client_id": self.args.edge_id,
                        "error": "edge timed out waiting for a child client",
                    },
                )
                return

        if self.args.child_join_window > 0:
            self.shutdown.wait(self.args.child_join_window)
        cohort = self.registry.snapshot()
        if not cohort:
            send_message(
                self.cloud_sock,
                {
                    "type": "CLIENT_ERROR",
                    "round": round_id,
                    "client_id": self.args.edge_id,
                    "error": "edge has no connected child clients",
                },
            )
            return

        child_ids = [child.client_id for child in cohort]
        LOG.info("=== EDGE ROUND %d | children=%s ===", round_id, ", ".join(child_ids))
        child_train = {
            "type": "TRAIN",
            "round": round_id,
            "algorithm": "HierFedAvg",
            "local_epochs": int(meta["local_epochs"]),
            "batch_size": int(meta["batch_size"]),
        }
        for child in cohort:
            try:
                child.send(child_train, global_weights)
            except OSError as exc:
                LOG.warning("Round %d: failed to send to child %s: %s", round_id, child.client_id, exc)
                child.close()

        started = time.perf_counter()
        updates, successful_ids, successful_samples = self._collect_child_updates(
            cohort, round_id, global_weights
        )
        if not updates:
            send_message(
                self.cloud_sock,
                {
                    "type": "CLIENT_ERROR",
                    "round": round_id,
                    "client_id": self.args.edge_id,
                    "error": "edge received no valid child updates",
                },
            )
            return

        agg_started = time.perf_counter()
        edge_weights = mean_fedavg(updates)
        aggregation_seconds = time.perf_counter() - agg_started
        elapsed = time.perf_counter() - started
        send_message(
            self.cloud_sock,
            {
                "type": "UPDATE",
                "round": round_id,
                "client_id": self.args.edge_id,
                "algorithm": "HierFedAvg",
                "train_seconds": round(elapsed, 6),
                "final_loss": None,
                "child_count": len(updates),
                "child_ids": successful_ids,
                "child_train_samples": successful_samples,
                "edge_aggregation_seconds": round(aggregation_seconds, 6),
            },
            edge_weights,
        )
        LOG.info(
            "Round %d: edge update -> cloud (%d child updates, aggregate %.4fs)",
            round_id,
            len(updates),
            aggregation_seconds,
        )

    def run(self) -> int:
        self.cloud_sock = self._connect_cloud()
        LOG.info("Connected edge %s to cloud %s:%d", self.args.edge_id, self.args.cloud, self.args.cloud_port)
        send_message(
            self.cloud_sock,
            {
                "type": "HELLO",
                "client_id": self.args.edge_id,
                "role": "edge",
                "auth_token": self.args.auth_token,
            },
        )
        welcome = recv_message(self.cloud_sock)
        if welcome.metadata.get("type") == "ERROR":
            raise RuntimeError(welcome.metadata.get("error", "cloud rejected edge"))
        if welcome.metadata.get("type") != "WELCOME":
            raise ProtocolError("cloud did not send WELCOME")
        if welcome.metadata.get("algorithm") != "HierFedAvg":
            raise ProtocolError("edge may only attach to a HierFedAvg cloud experiment")

        self.model_config = welcome.metadata["model_config"]
        self.seq_len = int(welcome.metadata["seq_len"])
        self.features = [str(v) for v in welcome.metadata["features"]]
        self._start_child_listener()

        # Cloud-level tools2 HierFedAvg averages edges equally, so this value is
        # a positive registration placeholder rather than an aggregation weight.
        send_message(
            self.cloud_sock,
            {
                "type": "READY",
                "client_id": self.args.edge_id,
                "train_samples": 1,
            },
        )
        LOG.info("Edge %s registered with cloud; waiting for rounds", self.args.edge_id)

        while not self.shutdown.is_set():
            message = recv_message(self.cloud_sock)
            msg_type = message.metadata.get("type")
            if msg_type == "STOP":
                LOG.info("Cloud stopped experiment: %s", message.metadata.get("reason", ""))
                break
            if msg_type != "TRAIN":
                LOG.warning("Ignoring cloud message: %s", msg_type)
                continue
            try:
                self._run_cloud_round(message)
            except Exception as exc:
                LOG.exception("Cloud round failed")
                send_message(
                    self.cloud_sock,
                    {
                        "type": "CLIENT_ERROR",
                        "round": int(message.metadata.get("round", -1)),
                        "client_id": self.args.edge_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
        return 0

    def stop(self) -> None:
        if self.shutdown.is_set():
            return
        self.shutdown.set()
        for child in self.registry.snapshot():
            try:
                child.send({"type": "STOP", "reason": "edge_stopped"})
            except Exception:
                pass
        self.registry.close_all()
        if self.listener is not None:
            try:
                self.listener.close()
            except OSError:
                pass
        if self.cloud_sock is not None:
            try:
                self.cloud_sock.close()
            except OSError:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HierFedAvg edge aggregator")
    parser.add_argument("--edge-id", required=True)
    parser.add_argument("--cloud", required=True)
    parser.add_argument("--cloud-port", type=int, default=8090)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8091)
    parser.add_argument("--child-join-window", type=float, default=1.0)
    parser.add_argument("--child-wait-timeout", type=float, default=60.0)
    parser.add_argument("--round-timeout", type=float, default=300.0)
    parser.add_argument("--handshake-timeout", type=float, default=30.0)
    parser.add_argument("--backlog", type=int, default=128)
    parser.add_argument("--reconnect-delay", type=float, default=3.0)
    parser.add_argument("--connect-timeout", type=float, default=0.0)
    parser.add_argument("--auth-token", default=os.environ.get("FL_AUTH_TOKEN"))
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    edge = HierFedAvgEdge(args)
    try:
        raise SystemExit(edge.run())
    except KeyboardInterrupt:
        raise SystemExit(130)
    finally:
        edge.stop()


if __name__ == "__main__":
    main()
