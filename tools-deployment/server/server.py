#!/usr/bin/env python3
"""Dynamic-client federated learning cloud server.

The networking layer has no configured client count.  Every round snapshots the
participants that are currently registered.  Direct algorithms use cloud->client
communication; HierFedAvg uses cloud->edge->client.

After every successful aggregation the global model is evaluated *on the clients*.
Clients return only confusion counts and aggregate loss, so raw test data never
leaves the edge devices.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import queue
import random
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np

DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
if str(DEPLOYMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_ROOT))

from common.data import DEFAULT_FEATURES
from common.metrics import aggregate_eval_records
from common.model import build_model_from_config, load_model_config
from common.protocol import ConnectionClosed, Message, ProtocolError, recv_message, send_message
from common.quantization import dequantize_weights
from common.sessions import ClientRegistry, ClientSession
from server.aggregation import fedma_aggregate, weighted_fedavg

LOG = logging.getLogger("fl-server")
SUPPORTED_ALGORITHMS = ("FedAvg", "FedProx", "FedPAQ", "FedMA", "HierFedAvg")

ROUND_FIELDS = [
    "round", "algorithm", "cohort_size", "successful_updates", "evaluation_results",
    "client_ids", "successful_client_ids", "train_samples", "test_samples",
    "round_seconds", "training_phase_seconds", "aggregation_seconds", "evaluation_phase_seconds",
    "test_loss", "accuracy", "precision", "recall", "specificity", "f1", "normal_f1", "macro_f1",
    "tp", "tn", "fp", "fn",
    "bytes_cloud_to_participants_train", "bytes_participants_to_cloud_train",
    "bytes_cloud_to_participants_eval", "bytes_participants_to_cloud_eval",
    "bytes_edges_to_children_train", "bytes_children_to_edges_train",
    "bytes_edges_to_children_eval", "bytes_children_to_edges_eval",
    "bytes_total_train", "bytes_total_eval", "bytes_total",
]

PARTICIPANT_FIELDS = [
    "round", "algorithm", "participant_id", "role", "train_samples", "train_seconds",
    "final_loss", "final_accuracy", "final_objective", "final_proximal_term", "update_wire_bytes",
    "test_samples", "test_loss", "test_accuracy", "tp", "tn", "fp", "fn",
    "eval_seconds", "evaluation_wire_bytes", "child_count", "child_train_samples",
    "edge_child_train_bytes_down", "edge_child_train_bytes_up",
    "edge_child_eval_bytes_down", "edge_child_eval_bytes_up",
]


class FederatedServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.algorithm = str(args.algorithm)
        self.shutdown = threading.Event()
        self.registry = ClientRegistry()
        self._set_initial_seed(int(args.seed))
        self.model_config = load_model_config(args.model)
        self.model = build_model_from_config(self.model_config, args.seq_len, len(args.features))
        self.global_weights = [np.asarray(w) for w in self.model.get_weights()]
        self.listener: socket.socket | None = None
        self.run_dir = self._create_run_dir(Path(args.results_dir), self.algorithm)
        self.rounds_csv = self.run_dir / "rounds.csv"
        self.participants_csv = self.run_dir / "participants.csv"
        self.round_rows: list[dict[str, Any]] = []
        self._write_run_config()

    @staticmethod
    def _set_initial_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import tensorflow as tf
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            # TensorFlow-free smoke tests replace the model builder.
            pass

    @staticmethod
    def _create_run_dir(base: Path, algorithm: str) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_algorithm = "".join(ch for ch in algorithm if ch.isalnum() or ch in "-_")
        run_dir = base / f"{stamp}_{safe_algorithm}"
        suffix = 1
        while run_dir.exists():
            run_dir = base / f"{stamp}_{safe_algorithm}-{suffix}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    @staticmethod
    def _git_revision() -> str | None:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=DEPLOYMENT_ROOT.parent,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

    def _algorithm_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.algorithm == "FedProx":
            params["fedprox_mu"] = float(self.args.fedprox_mu)
        if self.algorithm == "FedPAQ":
            params["fedpaq_bits"] = int(self.args.fedpaq_bits)
        return params

    def _write_run_config(self) -> None:
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": self._git_revision(),
            "algorithm": self.algorithm,
            "algorithm_params": self._algorithm_params(),
            "seed": int(self.args.seed),
            "host": self.args.host,
            "port": self.args.port,
            "rounds": self.args.rounds,
            "local_epochs": self.args.local_epochs,
            "batch_size": self.args.batch_size,
            "seq_len": self.args.seq_len,
            "features": list(self.args.features),
            "round_timeout": self.args.round_timeout,
            "evaluation_timeout": self.args.evaluation_timeout,
            "evaluate_every": self.args.evaluate_every,
            "eval_batch_size": self.args.eval_batch_size,
            "join_window": self.args.join_window,
            "model_path": str(Path(self.args.model).resolve()),
            "hierarchy": "cloud->edge->client" if self.algorithm == "HierFedAvg" else "cloud->client",
        }
        (self.run_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _expected_role(self) -> str:
        return "edge" if self.algorithm == "HierFedAvg" else "client"

    def _client_reader(self, session: ClientSession) -> None:
        try:
            while not self.shutdown.is_set() and session.alive:
                message = recv_message(session.sock)
                session.bytes_received += message.wire_bytes
                session.inbox.put(message)
        except (ConnectionClosed, OSError, ProtocolError) as exc:
            if not self.shutdown.is_set():
                LOG.info("%s %s disconnected: %s", session.role.capitalize(), session.client_id, exc)
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
            role = str(meta.get("role", "client"))
            expected_role = self._expected_role()
            if role != expected_role:
                raise ProtocolError(f"{self.algorithm} cloud expects role={expected_role!r}, got {role!r}")
            if self.args.auth_token and meta.get("auth_token") != self.args.auth_token:
                raise ProtocolError("authentication failed")

            welcome_metadata = {
                "type": "WELCOME",
                "client_id": client_id,
                "role": role,
                "algorithm": self.algorithm,
                "seq_len": self.args.seq_len,
                "features": list(self.args.features),
                "model_config": self.model_config,
                "seed": int(self.args.seed),
                **self._algorithm_params(),
            }
            welcome_bytes = send_message(client_sock, welcome_metadata)

            ready = recv_message(client_sock)
            if ready.metadata.get("type") != "READY":
                raise ProtocolError("participant must send READY after WELCOME")
            if ready.metadata.get("client_id") != client_id:
                raise ProtocolError("READY client_id does not match HELLO")
            train_samples = int(ready.metadata.get("train_samples", 0))
            if train_samples <= 0:
                raise ProtocolError("participant must report positive train_samples")

            client_sock.settimeout(None)
            session = ClientSession(client_id, client_sock, address, train_samples, role=role)
            session.bytes_received += hello.wire_bytes + ready.wire_bytes
            session.bytes_sent += welcome_bytes
            self.registry.register(session)
            if role == "edge":
                LOG.info("Edge %-20s ready from %s:%s", client_id, *address)
            else:
                LOG.info("Client %-20s ready from %s:%s (%d train samples)", client_id, *address, train_samples)
            threading.Thread(target=self._client_reader, args=(session,), daemon=True).start()
        except ConnectionClosed as exc:
            # Ansible wait_for and ordinary TCP health checks connect and close
            # without speaking FLP1.  That is not an experiment warning.
            LOG.debug("Probe/closed connection from %s:%s: %s", *address, exc)
            try:
                client_sock.close()
            except OSError:
                pass
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
            threading.Thread(target=self._bootstrap_connection, args=(client_sock, address), daemon=True).start()

    def _weights_are_compatible(self, arrays: list[np.ndarray]) -> bool:
        if len(arrays) != len(self.global_weights):
            return False
        return all(
            np.asarray(candidate).shape == np.asarray(reference).shape
            and np.asarray(candidate).dtype.kind in "fiu"
            for candidate, reference in zip(arrays, self.global_weights)
        )

    def _decode_update(self, message: Message) -> list[np.ndarray]:
        meta = message.metadata
        update_algorithm = str(meta.get("algorithm", self.algorithm))
        if update_algorithm != self.algorithm:
            raise ProtocolError(f"update algorithm {update_algorithm!r} differs from experiment {self.algorithm!r}")
        if not message.arrays:
            raise ProtocolError("empty model update")

        if self.algorithm == "FedPAQ":
            quantization = meta.get("quantization")
            if not isinstance(quantization, list):
                raise ProtocolError("FedPAQ update is missing quantization metadata")
            bits = int(meta.get("fedpaq_bits", 0))
            if bits != int(self.args.fedpaq_bits):
                raise ProtocolError(f"FedPAQ update uses {bits} bits; server configured for {self.args.fedpaq_bits}")
            arrays = dequantize_weights(message.arrays, quantization, reference_weights=self.global_weights)
        else:
            arrays = [np.asarray(value) for value in message.arrays]

        if not self._weights_are_compatible(arrays):
            raise ProtocolError("update model tensors are incompatible with the global model")
        return arrays

    def _collect_round_updates(
        self,
        cohort: list[ClientSession],
        round_id: int,
    ) -> tuple[list[list[np.ndarray]], list[int], list[str], list[dict[str, Any]]]:
        pending = {participant.client_id: participant for participant in cohort if participant.alive}
        updates: list[list[np.ndarray]] = []
        sample_counts: list[int] = []
        successful_ids: list[str] = []
        records: list[dict[str, Any]] = []
        deadline = time.monotonic() + self.args.round_timeout

        while pending and time.monotonic() < deadline and not self.shutdown.is_set():
            progressed = False
            for participant_id, session in list(pending.items()):
                if not session.alive:
                    pending.pop(participant_id, None)
                    continue
                try:
                    message = session.inbox.get_nowait()
                except queue.Empty:
                    continue

                progressed = True
                meta = message.metadata
                msg_type = meta.get("type")
                message_round = int(meta.get("round", -1))

                if msg_type == "UPDATE" and message_round == round_id:
                    try:
                        arrays = self._decode_update(message)
                        if self.algorithm == "HierFedAvg":
                            aggregation_samples = int(meta.get("child_train_samples", 0))
                            if aggregation_samples <= 0:
                                raise ProtocolError("HierFedAvg edge update must report child_train_samples > 0")
                        else:
                            aggregation_samples = session.train_samples
                    except (ProtocolError, ValueError) as exc:
                        LOG.warning("Round %d: rejected update from %s: %s", round_id, participant_id, exc)
                    else:
                        updates.append(arrays)
                        sample_counts.append(aggregation_samples)
                        successful_ids.append(participant_id)
                        records.append({
                            "participant_id": participant_id,
                            "role": session.role,
                            "train_samples": aggregation_samples,
                            "train_seconds": float(meta.get("train_seconds", 0.0)),
                            "final_loss": meta.get("final_loss"),
                            "final_accuracy": meta.get("final_accuracy"),
                            "final_objective": meta.get("final_objective"),
                            "final_proximal_term": meta.get("final_proximal_term"),
                            "update_wire_bytes": int(message.wire_bytes),
                            "child_count": int(meta.get("child_count", 0) or 0),
                            "child_train_samples": int(meta.get("child_train_samples", 0) or 0),
                            "edge_child_train_bytes_down": int(meta.get("edge_child_train_bytes_down", 0) or 0),
                            "edge_child_train_bytes_up": int(meta.get("edge_child_train_bytes_up", 0) or 0),
                        })
                        extra = ""
                        if self.algorithm == "HierFedAvg":
                            extra = f" | children {meta.get('child_count', '?')} | samples {aggregation_samples}"
                        elif self.algorithm == "FedProx":
                            extra = (
                                f" | objective {meta.get('final_objective')}"
                                f" | prox {meta.get('final_proximal_term')}"
                            )
                        LOG.info(
                            "Round %d: update <- %s | train %.2fs | loss %s%s",
                            round_id,
                            participant_id,
                            float(meta.get("train_seconds", 0.0)),
                            meta.get("final_loss"),
                            extra,
                        )
                    pending.pop(participant_id, None)
                elif msg_type == "CLIENT_ERROR" and message_round == round_id:
                    LOG.error("Round %d: participant %s failed: %s", round_id, participant_id, meta.get("error"))
                    pending.pop(participant_id, None)
                else:
                    LOG.warning("Ignoring unexpected message from %s during training: %s", participant_id, msg_type)

            if not progressed:
                time.sleep(0.05)

        for participant_id in pending:
            LOG.warning("Round %d: timed out waiting for %s", round_id, participant_id)
        return updates, sample_counts, successful_ids, records

    def _aggregate(self, updates: list[list[np.ndarray]], sample_counts: list[int]) -> list[np.ndarray]:
        if self.algorithm in {"FedAvg", "FedProx", "FedPAQ", "HierFedAvg"}:
            # HierFedAvg updates are already edge-level weighted averages; the
            # cloud weights those edges by the number of child samples represented.
            return weighted_fedavg(updates, sample_counts)
        if self.algorithm == "FedMA":
            return fedma_aggregate(self.global_weights, updates)
        raise RuntimeError(f"unsupported algorithm {self.algorithm}")

    def _collect_evaluations(
        self,
        cohort: list[ClientSession],
        successful_ids: list[str],
        round_id: int,
    ) -> list[dict[str, Any]]:
        targets = {
            p.client_id: p
            for p in cohort
            if p.alive and p.client_id in set(successful_ids)
        }
        if not targets:
            return []

        eval_metadata = {
            "type": "EVALUATE",
            "round": round_id,
            "algorithm": self.algorithm,
            "batch_size": int(self.args.eval_batch_size),
            "seed": int(self.args.seed + round_id * 1000 + 500),
        }
        for participant_id, session in list(targets.items()):
            try:
                session.send(eval_metadata, self.global_weights)
            except OSError as exc:
                LOG.warning("Round %d: failed to send evaluation model to %s: %s", round_id, participant_id, exc)
                session.close()
                targets.pop(participant_id, None)

        records: list[dict[str, Any]] = []
        pending = dict(targets)
        deadline = time.monotonic() + self.args.evaluation_timeout
        while pending and time.monotonic() < deadline and not self.shutdown.is_set():
            progressed = False
            for participant_id, session in list(pending.items()):
                if not session.alive:
                    pending.pop(participant_id, None)
                    continue
                try:
                    message = session.inbox.get_nowait()
                except queue.Empty:
                    continue
                progressed = True
                meta = message.metadata
                msg_type = meta.get("type")
                message_round = int(meta.get("round", -1))
                if msg_type == "EVAL_RESULT" and message_round == round_id:
                    try:
                        test_samples = int(meta.get("test_samples", 0))
                        tp, tn, fp, fn = (int(meta.get(k, -1)) for k in ("tp", "tn", "fp", "fn"))
                        if test_samples <= 0 or min(tp, tn, fp, fn) < 0:
                            raise ProtocolError("invalid evaluation counts")
                        if tp + tn + fp + fn != test_samples:
                            raise ProtocolError("evaluation confusion counts do not sum to test_samples")
                        record = {
                            "participant_id": participant_id,
                            "role": session.role,
                            "test_samples": test_samples,
                            "test_loss": float(meta["test_loss"]),
                            "test_accuracy": float(meta.get("accuracy", (tp + tn) / test_samples)),
                            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                            "eval_seconds": float(meta.get("eval_seconds", 0.0)),
                            "evaluation_wire_bytes": int(message.wire_bytes),
                            "edge_child_eval_bytes_down": int(meta.get("edge_child_eval_bytes_down", 0) or 0),
                            "edge_child_eval_bytes_up": int(meta.get("edge_child_eval_bytes_up", 0) or 0),
                        }
                    except (KeyError, TypeError, ValueError, ProtocolError) as exc:
                        LOG.warning("Round %d: rejected evaluation from %s: %s", round_id, participant_id, exc)
                    else:
                        records.append(record)
                        LOG.info(
                            "Round %d: evaluation <- %s | n=%d | acc=%.4f",
                            round_id, participant_id, test_samples, record["test_accuracy"],
                        )
                    pending.pop(participant_id, None)
                elif msg_type == "CLIENT_ERROR" and message_round == round_id:
                    LOG.error("Round %d: evaluation failed on %s: %s", round_id, participant_id, meta.get("error"))
                    pending.pop(participant_id, None)
                else:
                    LOG.warning("Ignoring unexpected message from %s during evaluation: %s", participant_id, msg_type)
            if not progressed:
                time.sleep(0.05)

        for participant_id in pending:
            LOG.warning("Round %d: timed out waiting for evaluation from %s", round_id, participant_id)
        return records

    @staticmethod
    def _append_csv(path: Path, fields: list[str], row: dict[str, Any]) -> None:
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow({field: row.get(field) for field in fields})

    def _save_participant_rows(
        self,
        round_id: int,
        update_records: list[dict[str, Any]],
        eval_records: list[dict[str, Any]],
    ) -> None:
        eval_by_id = {record["participant_id"]: record for record in eval_records}
        for update in update_records:
            row = {
                "round": round_id,
                "algorithm": self.algorithm,
                **update,
                **eval_by_id.get(update["participant_id"], {}),
            }
            self._append_csv(self.participants_csv, PARTICIPANT_FIELDS, row)

    def _save_weights(self) -> None:
        np.savez_compressed(
            self.run_dir / "global_weights.npz",
            **{f"arr_{i:05d}": w for i, w in enumerate(self.global_weights)},
        )

    def _write_summary(self) -> None:
        if not self.round_rows:
            summary = {
                "algorithm": self.algorithm,
                "completed_rounds": 0,
                "run_dir": str(self.run_dir),
            }
        else:
            final = self.round_rows[-1]
            summary = {
                "algorithm": self.algorithm,
                "algorithm_params": self._algorithm_params(),
                "seed": int(self.args.seed),
                "completed_rounds": len(self.round_rows),
                "final_round": int(final["round"]),
                "final_metrics": {
                    key: final.get(key)
                    for key in (
                        "test_samples", "test_loss", "accuracy", "precision", "recall",
                        "specificity", "f1", "normal_f1", "macro_f1", "tp", "tn", "fp", "fn"
                    )
                },
                "total_network_bytes": int(sum(int(r["bytes_total"]) for r in self.round_rows)),
                "total_training_network_bytes": int(sum(int(r["bytes_total_train"]) for r in self.round_rows)),
                "total_evaluation_network_bytes": int(sum(int(r["bytes_total_eval"]) for r in self.round_rows)),
                "total_round_seconds": float(sum(float(r["round_seconds"]) for r in self.round_rows)),
                "mean_aggregation_seconds": float(
                    sum(float(r["aggregation_seconds"]) for r in self.round_rows) / len(self.round_rows)
                ),
                "run_dir": str(self.run_dir),
            }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def run_rounds(self) -> None:
        round_id = 1
        while not self.shutdown.is_set() and (self.args.rounds == 0 or round_id <= self.args.rounds):
            role_name = "edge" if self.algorithm == "HierFedAvg" else "client"
            LOG.info("Waiting for at least one connected %s...", role_name)
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

            cohort_ids = [participant.client_id for participant in cohort]
            LOG.info("=== ROUND %d | algorithm=%s | cohort=%s ===", round_id, self.algorithm, ", ".join(cohort_ids))
            round_started = time.perf_counter()
            before_train_sent = sum(p.bytes_sent for p in cohort)
            before_train_recv = sum(p.bytes_received for p in cohort)

            train_metadata = {
                "type": "TRAIN",
                "round": round_id,
                "algorithm": self.algorithm,
                "local_epochs": self.args.local_epochs,
                "batch_size": self.args.batch_size,
                "seed": int(self.args.seed + round_id * 1000),
                **self._algorithm_params(),
            }
            for participant in cohort:
                if not participant.alive:
                    continue
                try:
                    participant.send(train_metadata, self.global_weights)
                except OSError as exc:
                    LOG.warning("Round %d: failed to send to %s: %s", round_id, participant.client_id, exc)
                    participant.close()

            training_started = time.perf_counter()
            updates, sample_counts, successful_ids, update_records = self._collect_round_updates(cohort, round_id)
            training_phase_seconds = time.perf_counter() - training_started
            after_train_sent = sum(p.bytes_sent for p in cohort)
            after_train_recv = sum(p.bytes_received for p in cohort)

            aggregation_seconds = 0.0
            if updates:
                agg_started = time.perf_counter()
                self.global_weights = self._aggregate(updates, sample_counts)
                aggregation_seconds = time.perf_counter() - agg_started
                self._save_weights()
                LOG.info("Round %d %s aggregated %d update(s) in %.4fs", round_id, self.algorithm, len(updates), aggregation_seconds)
            else:
                LOG.error("Round %d produced no valid updates; global model unchanged", round_id)

            eval_records: list[dict[str, Any]] = []
            evaluation_phase_seconds = 0.0
            before_eval_sent = after_train_sent
            before_eval_recv = after_train_recv
            if updates and self.args.evaluate_every > 0 and round_id % self.args.evaluate_every == 0:
                eval_started = time.perf_counter()
                eval_records = self._collect_evaluations(cohort, successful_ids, round_id)
                evaluation_phase_seconds = time.perf_counter() - eval_started
            after_eval_sent = sum(p.bytes_sent for p in cohort)
            after_eval_recv = sum(p.bytes_received for p in cohort)

            metrics = aggregate_eval_records(eval_records)
            if eval_records:
                LOG.info(
                    "Round %d GLOBAL | n=%d | loss=%.5f | acc=%.4f | F1=%.4f | macro-F1=%.4f",
                    round_id,
                    metrics["test_samples"], metrics["test_loss"], metrics["accuracy"],
                    metrics["f1"], metrics["macro_f1"],
                )

            cloud_train_down = max(0, after_train_sent - before_train_sent)
            cloud_train_up = max(0, after_train_recv - before_train_recv)
            cloud_eval_down = max(0, after_eval_sent - before_eval_sent)
            cloud_eval_up = max(0, after_eval_recv - before_eval_recv)
            edge_train_down = sum(int(r.get("edge_child_train_bytes_down", 0)) for r in update_records)
            edge_train_up = sum(int(r.get("edge_child_train_bytes_up", 0)) for r in update_records)
            edge_eval_down = sum(int(r.get("edge_child_eval_bytes_down", 0)) for r in eval_records)
            edge_eval_up = sum(int(r.get("edge_child_eval_bytes_up", 0)) for r in eval_records)
            total_train = cloud_train_down + cloud_train_up + edge_train_down + edge_train_up
            total_eval = cloud_eval_down + cloud_eval_up + edge_eval_down + edge_eval_up

            row = {
                "round": round_id,
                "algorithm": self.algorithm,
                "cohort_size": len(cohort),
                "successful_updates": len(updates),
                "evaluation_results": len(eval_records),
                "client_ids": ";".join(cohort_ids),
                "successful_client_ids": ";".join(successful_ids),
                "train_samples": sum(sample_counts),
                **metrics,
                "round_seconds": round(time.perf_counter() - round_started, 6),
                "training_phase_seconds": round(training_phase_seconds, 6),
                "aggregation_seconds": round(aggregation_seconds, 6),
                "evaluation_phase_seconds": round(evaluation_phase_seconds, 6),
                "bytes_cloud_to_participants_train": cloud_train_down,
                "bytes_participants_to_cloud_train": cloud_train_up,
                "bytes_cloud_to_participants_eval": cloud_eval_down,
                "bytes_participants_to_cloud_eval": cloud_eval_up,
                "bytes_edges_to_children_train": edge_train_down,
                "bytes_children_to_edges_train": edge_train_up,
                "bytes_edges_to_children_eval": edge_eval_down,
                "bytes_children_to_edges_eval": edge_eval_up,
                "bytes_total_train": total_train,
                "bytes_total_eval": total_eval,
                "bytes_total": total_train + total_eval,
            }
            self.round_rows.append(row)
            self._append_csv(self.rounds_csv, ROUND_FIELDS, row)
            self._save_participant_rows(round_id, update_records, eval_records)
            self._write_summary()
            round_id += 1

    def serve(self) -> None:
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((self.args.host, self.args.port))
        self.listener.listen(self.args.backlog)
        LOG.info("Server listening on %s:%d", self.args.host, self.args.port)
        LOG.info("Algorithm: %s", self.algorithm)
        LOG.info("Results: %s", self.run_dir)
        threading.Thread(target=self._accept_loop, daemon=True).start()
        try:
            self.run_rounds()
        finally:
            self.stop()

    def stop(self) -> None:
        already_stopping = self.shutdown.is_set()
        self.shutdown.set()
        if not already_stopping:
            for participant in self.registry.snapshot():
                try:
                    participant.send({"type": "STOP", "reason": "experiment_finished"})
                except Exception:
                    pass
        self.registry.close_all()
        if self.listener is not None:
            try:
                self.listener.close()
            except OSError:
                pass
        self._save_weights()
        self._write_summary()
        if not already_stopping:
            LOG.info("Server stopped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic-client federated learning server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--model", required=True, help="Path to a tools2-compatible model JSON")
    parser.add_argument("--algorithm", choices=SUPPORTED_ALGORITHMS, default="FedAvg")
    parser.add_argument("--rounds", type=int, default=3, help="0 = run until Ctrl+C")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--fedpaq-bits", type=int, default=8)
    parser.add_argument("--join-window", type=float, default=2.0)
    parser.add_argument("--round-timeout", type=float, default=300.0)
    parser.add_argument("--evaluation-timeout", type=float, default=300.0)
    parser.add_argument("--evaluate-every", type=int, default=1, help="0 disables distributed evaluation")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--handshake-timeout", type=float, default=10.0)
    parser.add_argument("--backlog", type=int, default=128)
    parser.add_argument("--results-dir", default=str(DEPLOYMENT_ROOT / "results"))
    parser.add_argument("--auth-token", default=os.environ.get("FL_AUTH_TOKEN"), help="Optional shared token; prefer FL_AUTH_TOKEN env var")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    if args.fedprox_mu < 0:
        parser.error("--fedprox-mu must be >= 0")
    if not 1 <= args.fedpaq_bits <= 8:
        parser.error("--fedpaq-bits must be between 1 and 8")
    if args.evaluate_every < 0:
        parser.error("--evaluate-every must be >= 0")
    if args.eval_batch_size <= 0:
        parser.error("--eval-batch-size must be > 0")
    return args


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
