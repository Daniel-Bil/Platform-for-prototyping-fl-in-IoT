#!/usr/bin/env python3
"""HierFedAvg edge aggregator.

The edge is a client of the cloud and a server for training clients. Child
updates are sample-weighted locally; the edge reports the number of represented
samples so the cloud can sample-weight edge models as well.
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

from common.metrics import aggregate_eval_records
from common.protocol import ConnectionClosed, Message, ProtocolError, recv_message, send_message
from common.sessions import ClientRegistry, ClientSession
from server.aggregation import weighted_fedavg

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
        self.seed: int = 42
        self.round_successful_children: dict[int, list[str]] = {}

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
                    "seed": self.seed,
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
        except ConnectionClosed as exc:
            LOG.debug("Probe/closed child connection from %s:%s: %s", *address, exc)
            try:
                child_sock.close()
            except OSError:
                pass
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
    ) -> tuple[list[list[np.ndarray]], list[int], list[str], list[dict[str, Any]]]:
        pending = {client.client_id: client for client in cohort if client.alive}
        updates: list[list[np.ndarray]] = []
        sample_counts: list[int] = []
        successful_ids: list[str] = []
        records: list[dict[str, Any]] = []
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
                        sample_counts.append(session.train_samples)
                        successful_ids.append(client_id)
                        records.append({
                            "client_id": client_id,
                            "train_samples": session.train_samples,
                            "train_seconds": float(meta.get("train_seconds", 0.0)),
                            "final_loss": meta.get("final_loss"),
                            "final_accuracy": meta.get("final_accuracy"),
                            "wire_bytes": int(message.wire_bytes),
                        })
                        LOG.info(
                            "Round %d: child update <- %s | n=%d | train %.2fs | loss %s",
                            round_id, client_id, session.train_samples,
                            float(meta.get("train_seconds", 0.0)), meta.get("final_loss"),
                        )
                    pending.pop(client_id, None)
                elif meta.get("type") == "CLIENT_ERROR" and int(meta.get("round", -1)) == round_id:
                    LOG.error("Round %d: child %s failed: %s", round_id, client_id, meta.get("error"))
                    pending.pop(client_id, None)
                else:
                    LOG.warning("Ignoring unexpected child message from %s during training: %s", client_id, meta.get("type"))
            if not progressed:
                time.sleep(0.05)

        for client_id in pending:
            LOG.warning("Round %d: timed out waiting for child %s", round_id, client_id)
        return updates, sample_counts, successful_ids, records

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
            LOG.info("Round %d: waiting up to %.1fs for at least one child client", round_id, self.args.child_wait_timeout)
            deadline = time.monotonic() + self.args.child_wait_timeout
            while not self.shutdown.is_set() and not self.registry.snapshot() and time.monotonic() < deadline:
                self.shutdown.wait(0.1)
            if not self.registry.snapshot():
                send_message(self.cloud_sock, {
                    "type": "CLIENT_ERROR", "round": round_id, "client_id": self.args.edge_id,
                    "error": "edge timed out waiting for a child client",
                })
                return

        if self.args.child_join_window > 0:
            self.shutdown.wait(self.args.child_join_window)
        cohort = self.registry.snapshot()
        if not cohort:
            send_message(self.cloud_sock, {
                "type": "CLIENT_ERROR", "round": round_id, "client_id": self.args.edge_id,
                "error": "edge has no connected child clients",
            })
            return

        child_ids = [child.client_id for child in cohort]
        LOG.info("=== EDGE ROUND %d | children=%s ===", round_id, ", ".join(child_ids))
        before_sent = sum(child.bytes_sent for child in cohort)
        before_recv = sum(child.bytes_received for child in cohort)
        child_train = {
            "type": "TRAIN",
            "round": round_id,
            "algorithm": "HierFedAvg",
            "local_epochs": int(meta["local_epochs"]),
            "batch_size": int(meta["batch_size"]),
            "seed": int(meta.get("seed", self.seed + round_id * 1000)),
        }
        for child in cohort:
            try:
                child.send(child_train, global_weights)
            except OSError as exc:
                LOG.warning("Round %d: failed to send to child %s: %s", round_id, child.client_id, exc)
                child.close()

        started = time.perf_counter()
        updates, sample_counts, successful_ids, records = self._collect_child_updates(cohort, round_id, global_weights)
        if not updates:
            send_message(self.cloud_sock, {
                "type": "CLIENT_ERROR", "round": round_id, "client_id": self.args.edge_id,
                "error": "edge received no valid child updates",
            })
            return

        agg_started = time.perf_counter()
        edge_weights = weighted_fedavg(updates, sample_counts)
        aggregation_seconds = time.perf_counter() - agg_started
        elapsed = time.perf_counter() - started
        after_sent = sum(child.bytes_sent for child in cohort)
        after_recv = sum(child.bytes_received for child in cohort)
        successful_samples = int(sum(sample_counts))
        self.round_successful_children[round_id] = list(successful_ids)

        def weighted_optional(key: str):
            pairs = [(r.get(key), int(r["train_samples"])) for r in records if r.get(key) is not None]
            if not pairs:
                return None
            total = sum(n for _, n in pairs)
            return sum(float(v) * n for v, n in pairs) / total

        send_message(
            self.cloud_sock,
            {
                "type": "UPDATE",
                "round": round_id,
                "client_id": self.args.edge_id,
                "algorithm": "HierFedAvg",
                "train_seconds": round(elapsed, 6),
                "final_loss": weighted_optional("final_loss"),
                "final_accuracy": weighted_optional("final_accuracy"),
                "child_count": len(updates),
                "child_ids": successful_ids,
                "child_train_samples": successful_samples,
                "edge_aggregation_seconds": round(aggregation_seconds, 6),
                "edge_child_train_bytes_down": max(0, after_sent - before_sent),
                "edge_child_train_bytes_up": max(0, after_recv - before_recv),
            },
            edge_weights,
        )
        LOG.info(
            "Round %d: edge update -> cloud (%d children, %d samples, aggregate %.4fs)",
            round_id, len(updates), successful_samples, aggregation_seconds,
        )

    def _collect_child_evaluations(
        self,
        cohort: list[ClientSession],
        round_id: int,
    ) -> list[dict[str, Any]]:
        pending = {child.client_id: child for child in cohort if child.alive}
        records: list[dict[str, Any]] = []
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
                if meta.get("type") == "EVAL_RESULT" and int(meta.get("round", -1)) == round_id:
                    try:
                        n = int(meta.get("test_samples", 0))
                        tp, tn, fp, fn = (int(meta.get(k, -1)) for k in ("tp", "tn", "fp", "fn"))
                        if n <= 0 or min(tp, tn, fp, fn) < 0 or tp + tn + fp + fn != n:
                            raise ProtocolError("invalid child evaluation counts")
                        records.append({
                            "client_id": client_id,
                            "test_samples": n,
                            "test_loss": float(meta["test_loss"]),
                            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                            "eval_seconds": float(meta.get("eval_seconds", 0.0)),
                            "wire_bytes": int(message.wire_bytes),
                        })
                    except (KeyError, TypeError, ValueError, ProtocolError) as exc:
                        LOG.warning("Round %d: rejected child evaluation from %s: %s", round_id, client_id, exc)
                    pending.pop(client_id, None)
                elif meta.get("type") == "CLIENT_ERROR" and int(meta.get("round", -1)) == round_id:
                    LOG.error("Round %d: child %s evaluation failed: %s", round_id, client_id, meta.get("error"))
                    pending.pop(client_id, None)
                else:
                    LOG.warning("Ignoring unexpected child message from %s during evaluation: %s", client_id, meta.get("type"))
            if not progressed:
                time.sleep(0.05)
        for client_id in pending:
            LOG.warning("Round %d: timed out waiting for child evaluation %s", round_id, client_id)
        return records

    def _run_cloud_evaluation(self, message: Message) -> None:
        assert self.cloud_sock is not None
        meta = message.metadata
        round_id = int(meta["round"])
        global_weights = [np.asarray(value) for value in message.arrays]
        if not global_weights:
            raise ProtocolError("cloud EVALUATE contains no global weights")

        successful_ids = set(self.round_successful_children.get(round_id, []))
        cohort = [c for c in self.registry.snapshot() if c.client_id in successful_ids]
        if not cohort:
            send_message(self.cloud_sock, {
                "type": "CLIENT_ERROR", "phase": "evaluate", "round": round_id,
                "client_id": self.args.edge_id, "error": "no successful child clients available for evaluation",
            })
            return

        before_sent = sum(child.bytes_sent for child in cohort)
        before_recv = sum(child.bytes_received for child in cohort)
        eval_meta = {
            "type": "EVALUATE",
            "round": round_id,
            "algorithm": "HierFedAvg",
            "batch_size": int(meta.get("batch_size", 256)),
            "seed": int(meta.get("seed", self.seed + round_id * 1000 + 500)),
        }
        for child in cohort:
            try:
                child.send(eval_meta, global_weights)
            except OSError as exc:
                LOG.warning("Round %d: failed to send evaluation to child %s: %s", round_id, child.client_id, exc)
                child.close()

        started = time.perf_counter()
        records = self._collect_child_evaluations(cohort, round_id)
        after_sent = sum(child.bytes_sent for child in cohort)
        after_recv = sum(child.bytes_received for child in cohort)
        if not records:
            send_message(self.cloud_sock, {
                "type": "CLIENT_ERROR", "phase": "evaluate", "round": round_id,
                "client_id": self.args.edge_id, "error": "edge received no valid child evaluations",
            })
            return

        metrics = aggregate_eval_records(records)
        send_message(
            self.cloud_sock,
            {
                "type": "EVAL_RESULT",
                "round": round_id,
                "client_id": self.args.edge_id,
                "algorithm": "HierFedAvg",
                "test_samples": metrics["test_samples"],
                "test_loss": metrics["test_loss"],
                "accuracy": metrics["accuracy"],
                "tp": metrics["tp"], "tn": metrics["tn"], "fp": metrics["fp"], "fn": metrics["fn"],
                "eval_seconds": round(time.perf_counter() - started, 6),
                "child_count": len(records),
                "edge_child_eval_bytes_down": max(0, after_sent - before_sent),
                "edge_child_eval_bytes_up": max(0, after_recv - before_recv),
            },
        )
        LOG.info(
            "Round %d: edge evaluation -> cloud (n=%d, acc=%.4f)",
            round_id, metrics["test_samples"], metrics["accuracy"],
        )

    def run(self) -> int:
        self.cloud_sock = self._connect_cloud()
        LOG.info("Connected edge %s to cloud %s:%d", self.args.edge_id, self.args.cloud, self.args.cloud_port)
        send_message(self.cloud_sock, {
            "type": "HELLO", "client_id": self.args.edge_id, "role": "edge", "auth_token": self.args.auth_token,
        })
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
        self.seed = int(welcome.metadata.get("seed", 42))
        self._start_child_listener()

        # Registration requires a positive value. Real aggregation weight is sent
        # per round as child_train_samples after the edge knows its successful cohort.
        send_message(self.cloud_sock, {"type": "READY", "client_id": self.args.edge_id, "train_samples": 1})
        LOG.info("Edge %s registered with cloud; waiting for rounds", self.args.edge_id)

        while not self.shutdown.is_set():
            message = recv_message(self.cloud_sock)
            msg_type = message.metadata.get("type")
            if msg_type == "STOP":
                LOG.info("Cloud stopped experiment: %s", message.metadata.get("reason", ""))
                break
            try:
                if msg_type == "TRAIN":
                    self._run_cloud_round(message)
                elif msg_type == "EVALUATE":
                    self._run_cloud_evaluation(message)
                else:
                    LOG.warning("Ignoring cloud message: %s", msg_type)
            except Exception as exc:
                LOG.exception("Cloud %s failed", msg_type)
                send_message(self.cloud_sock, {
                    "type": "CLIENT_ERROR",
                    "phase": str(msg_type).lower(),
                    "round": int(message.metadata.get("round", -1)),
                    "client_id": self.args.edge_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
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
