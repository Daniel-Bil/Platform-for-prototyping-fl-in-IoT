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
import platform
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
from common.metrics import (
    aggregate_eval_records,
    make_threshold_grid,
    select_threshold_by_macro_f1,
    summarize_confusion,
)
from common.model import build_model_from_config, load_model_config
from common.protocol import ConnectionClosed, Message, ProtocolError, recv_message, send_message
from common.quantization import dequantize_weights
from common.sessions import ClientRegistry, ClientSession
from server.aggregation import fedma_aggregate, weighted_fedavg

LOG = logging.getLogger("fl-server")
SUPPORTED_ALGORITHMS = ("FedAvg", "FedProx", "FedPAQ", "FedMA", "HierFedAvg")

ROUND_FIELDS = [
    "round", "algorithm", "requested_clients", "cohort_size", "successful_updates", "evaluation_results",
    "client_ids", "successful_client_ids", "logical_client_count", "logical_client_ids",
    "train_samples", "test_samples",
    "selected_threshold", "validation_samples", "validation_loss", "validation_accuracy",
    "validation_precision", "validation_recall", "validation_specificity", "validation_f1",
    "validation_normal_f1", "validation_macro_f1", "validation_tp", "validation_tn",
    "validation_fp", "validation_fn",
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
    "round", "algorithm", "participant_id", "role", "parent_id", "dataset_name",
    "train_samples", "val_samples", "profile_test_samples",
    "train_positive", "val_positive", "test_positive",
    "train_positive_rate", "val_positive_rate", "test_positive_rate",
    "train_seconds", "final_loss", "final_accuracy", "final_objective", "final_proximal_term",
    "update_wire_bytes", "selected_threshold", "validation_samples", "validation_loss",
    "validation_accuracy", "validation_macro_f1", "validation_wire_bytes",
    "test_samples", "test_loss", "test_accuracy", "tp", "tn", "fp", "fn",
    "eval_seconds", "evaluation_wire_bytes", "child_count", "child_train_samples",
    "edge_aggregation_seconds", "child_ids",
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
        self.started_utc = datetime.now(timezone.utc)
        self.started_monotonic = time.perf_counter()
        self.run_dir = self._create_run_dir(Path(args.results_dir), self.algorithm, args.run_id)
        self.rounds_csv = self.run_dir / "rounds.csv"
        self.participants_csv = self.run_dir / "participants.csv"
        self.round_rows: list[dict[str, Any]] = []
        self.failure_reason: str | None = None
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
    def _create_run_dir(base: Path, algorithm: str, run_id: str | None = None) -> Path:
        base.mkdir(parents=True, exist_ok=True)
        if run_id:
            safe_run_id = "".join(ch for ch in str(run_id) if ch.isalnum() or ch in "-_.")
            if not safe_run_id or safe_run_id != str(run_id):
                raise ValueError("--run-id may contain only letters, digits, '-', '_' and '.'")
            run_dir = base / safe_run_id
            if run_dir.exists():
                raise FileExistsError(f"result run directory already exists: {run_dir}")
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir

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
        try:
            import tensorflow as tf
            tensorflow_version = tf.__version__
        except Exception:
            tensorflow_version = None

        model_params = int(self.model.count_params()) if hasattr(self.model, "count_params") else None
        model_weight_bytes = int(sum(np.asarray(w).nbytes for w in self.global_weights))
        payload = {
            "created_utc": self.started_utc.isoformat(),
            "run_id": self.run_dir.name,
            "campaign_id": self.args.campaign_id,
            "repetition": self.args.repetition,
            "requested_clients": self.args.requested_clients,
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
            "threshold_selection": {
                "mode": "validation_macro_f1",
                "minimum": self.args.threshold_min,
                "maximum": self.args.threshold_max,
                "step": self.args.threshold_step,
                "preferred": self.args.threshold_preferred,
            },
            "join_window": self.args.join_window,
            "initial_join_window": self.args.initial_join_window,
            "model_path": str(Path(self.args.model).resolve()),
            "hierarchy": "cloud->edge->client" if self.algorithm == "HierFedAvg" else "cloud->client",
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "tensorflow": tensorflow_version,
            },
            "model": {
                "parameter_count": model_params,
                "weight_bytes_float": model_weight_bytes,
            },
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
            profile = {
                "dataset_name": ready.metadata.get("dataset_name"),
                "train_samples": train_samples,
                "val_samples": int(ready.metadata.get("val_samples", 0) or 0),
                "test_samples": int(ready.metadata.get("test_samples", 0) or 0),
                "train_positive": int(ready.metadata.get("train_positive", 0) or 0),
                "val_positive": int(ready.metadata.get("val_positive", 0) or 0),
                "test_positive": int(ready.metadata.get("test_positive", 0) or 0),
            }
            session = ClientSession(
                client_id, client_sock, address, train_samples, role=role, profile=profile
            )
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

    def _weights_compatibility_error(self, arrays: list[np.ndarray]) -> str | None:
        """Return a human-readable incompatibility reason, or ``None``.

        Keeping this diagnostic explicit is useful on real machines: a generic
        "incompatible tensors" message hides whether a client returned an
        extra Keras bookkeeping tensor, a wrong architecture, or a bad dtype.
        """
        if len(arrays) != len(self.global_weights):
            return f"tensor count mismatch: got {len(arrays)}, expected {len(self.global_weights)}"

        for index, (candidate, reference) in enumerate(zip(arrays, self.global_weights)):
            candidate_array = np.asarray(candidate)
            reference_array = np.asarray(reference)
            if candidate_array.shape != reference_array.shape:
                return (
                    f"tensor {index} shape mismatch: got {candidate_array.shape}, "
                    f"expected {reference_array.shape}"
                )
            if candidate_array.dtype.kind not in "fiu":
                return f"tensor {index} has unsupported dtype {candidate_array.dtype}"
        return None

    def _weights_are_compatible(self, arrays: list[np.ndarray]) -> bool:
        # Backward-compatible helper used by tests/older code.
        return self._weights_compatibility_error(arrays) is None

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

        incompatibility = self._weights_compatibility_error(arrays)
        if incompatibility is not None:
            raise ProtocolError(
                f"update model tensors are incompatible with the global model: {incompatibility}"
            )
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
                            "parent_id": None,
                            "dataset_name": session.profile.get("dataset_name"),
                            "train_samples": aggregation_samples,
                            "val_samples": int(session.profile.get("val_samples", 0) or 0),
                            "profile_test_samples": int(session.profile.get("test_samples", 0) or 0),
                            "train_positive": int(session.profile.get("train_positive", 0) or 0),
                            "val_positive": int(session.profile.get("val_positive", 0) or 0),
                            "test_positive": int(session.profile.get("test_positive", 0) or 0),
                            "train_seconds": float(meta.get("train_seconds", 0.0)),
                            "final_loss": meta.get("final_loss"),
                            "final_accuracy": meta.get("final_accuracy"),
                            "final_objective": meta.get("final_objective"),
                            "final_proximal_term": meta.get("final_proximal_term"),
                            "update_wire_bytes": int(message.wire_bytes),
                            "child_count": int(meta.get("child_count", 0) or 0),
                            "child_ids": list(meta.get("child_ids", []) or []),
                            "child_train_samples": int(meta.get("child_train_samples", 0) or 0),
                            "edge_aggregation_seconds": float(meta.get("edge_aggregation_seconds", 0.0) or 0.0),
                            "edge_child_train_bytes_down": int(meta.get("edge_child_train_bytes_down", 0) or 0),
                            "edge_child_train_bytes_up": int(meta.get("edge_child_train_bytes_up", 0) or 0),
                            "child_update_records": list(meta.get("child_update_records", []) or []),
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

    def _collect_threshold_validations(
        self,
        cohort: list[ClientSession],
        successful_ids: list[str],
        round_id: int,
        thresholds: list[float],
    ) -> list[dict[str, Any]]:
        targets = {
            p.client_id: p
            for p in cohort
            if p.alive and p.client_id in set(successful_ids)
        }
        if not targets:
            return []

        validation_metadata = {
            "type": "VALIDATE_THRESHOLDS",
            "round": round_id,
            "algorithm": self.algorithm,
            "batch_size": int(self.args.eval_batch_size),
            "thresholds": [float(value) for value in thresholds],
            "seed": int(self.args.seed + round_id * 1000 + 400),
        }
        for participant_id, session in list(targets.items()):
            try:
                session.send(validation_metadata, self.global_weights)
            except OSError as exc:
                LOG.warning(
                    "Round %d: failed to send threshold validation model to %s: %s",
                    round_id,
                    participant_id,
                    exc,
                )
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
                if msg_type == "VALIDATION_RESULT" and message_round == round_id:
                    try:
                        val_samples = int(meta.get("val_samples", 0))
                        threshold_counts = meta.get("threshold_counts")
                        if val_samples <= 0:
                            raise ProtocolError("validation result must report val_samples > 0")
                        if not isinstance(threshold_counts, list) or len(threshold_counts) != len(thresholds):
                            raise ProtocolError("validation threshold grid length mismatch")
                        for expected, candidate in zip(thresholds, threshold_counts):
                            if not isinstance(candidate, dict):
                                raise ProtocolError("invalid validation threshold candidate")
                            if abs(float(candidate.get("threshold")) - float(expected)) > 1e-8:
                                raise ProtocolError("validation threshold grid mismatch")
                            counts = [int(candidate.get(key, -1)) for key in ("tp", "tn", "fp", "fn")]
                            if min(counts) < 0 or sum(counts) != val_samples:
                                raise ProtocolError("invalid validation confusion counts")
                        record = {
                            "participant_id": participant_id,
                            "role": session.role,
                            "val_samples": val_samples,
                            "val_loss": float(meta["val_loss"]),
                            "threshold_counts": threshold_counts,
                            "eval_seconds": float(meta.get("eval_seconds", 0.0)),
                            "validation_wire_bytes": int(message.wire_bytes),
                            "edge_child_eval_bytes_down": int(meta.get("edge_child_eval_bytes_down", 0) or 0),
                            "edge_child_eval_bytes_up": int(meta.get("edge_child_eval_bytes_up", 0) or 0),
                            "child_validation_records": list(meta.get("child_validation_records", []) or []),
                        }
                    except (KeyError, TypeError, ValueError, ProtocolError) as exc:
                        LOG.warning("Round %d: rejected validation result from %s: %s", round_id, participant_id, exc)
                    else:
                        records.append(record)
                        LOG.info(
                            "Round %d: validation <- %s | n=%d | candidates=%d",
                            round_id,
                            participant_id,
                            val_samples,
                            len(threshold_counts),
                        )
                    pending.pop(participant_id, None)
                elif msg_type == "CLIENT_ERROR" and message_round == round_id:
                    LOG.error("Round %d: validation failed on %s: %s", round_id, participant_id, meta.get("error"))
                    pending.pop(participant_id, None)
                else:
                    LOG.warning("Ignoring unexpected message from %s during validation: %s", participant_id, msg_type)
            if not progressed:
                time.sleep(0.05)

        for participant_id in pending:
            LOG.warning("Round %d: timed out waiting for threshold validation from %s", round_id, participant_id)
        return records

    def _collect_evaluations(
        self,
        cohort: list[ClientSession],
        successful_ids: list[str],
        round_id: int,
        threshold: float,
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
            "threshold": float(threshold),
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
                            "selected_threshold": float(meta.get("threshold", threshold)),
                            "test_samples": test_samples,
                            "test_loss": float(meta["test_loss"]),
                            "test_accuracy": float(meta.get("accuracy", (tp + tn) / test_samples)),
                            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                            "eval_seconds": float(meta.get("eval_seconds", 0.0)),
                            "evaluation_wire_bytes": int(message.wire_bytes),
                            "edge_child_eval_bytes_down": int(meta.get("edge_child_eval_bytes_down", 0) or 0),
                            "edge_child_eval_bytes_up": int(meta.get("edge_child_eval_bytes_up", 0) or 0),
                            "child_eval_records": list(meta.get("child_eval_records", []) or []),
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

    @staticmethod
    def _positive_rate(positive: Any, samples: Any) -> float | None:
        try:
            positive_i = int(positive or 0)
            samples_i = int(samples or 0)
        except (TypeError, ValueError):
            return None
        return float(positive_i / samples_i) if samples_i > 0 else None

    @staticmethod
    def _validation_at_threshold(
        record: dict[str, Any],
        threshold: float | None,
    ) -> dict[str, Any]:
        if threshold is None:
            return {}
        candidates = record.get("threshold_counts") or []
        for candidate in candidates:
            try:
                if abs(float(candidate.get("threshold")) - float(threshold)) <= 1e-8:
                    counts = summarize_confusion(
                        tp=int(candidate.get("tp", 0)),
                        tn=int(candidate.get("tn", 0)),
                        fp=int(candidate.get("fp", 0)),
                        fn=int(candidate.get("fn", 0)),
                    )
                    return counts
            except (TypeError, ValueError):
                continue
        return {}

    def _save_participant_rows(
        self,
        round_id: int,
        update_records: list[dict[str, Any]],
        validation_records: list[dict[str, Any]],
        eval_records: list[dict[str, Any]],
        selected_threshold: float | None,
    ) -> None:
        validation_by_id = {record["participant_id"]: record for record in validation_records}
        eval_by_id = {record["participant_id"]: record for record in eval_records}
        for update in update_records:
            validation_record = validation_by_id.get(update["participant_id"], {})
            validation_selected = self._validation_at_threshold(validation_record, selected_threshold)
            eval_record = eval_by_id.get(update["participant_id"], {})
            row = {
                "round": round_id,
                "algorithm": self.algorithm,
                **update,
                "selected_threshold": selected_threshold,
                "validation_samples": validation_record.get("val_samples"),
                "validation_loss": validation_record.get("val_loss"),
                "validation_accuracy": validation_selected.get("accuracy"),
                "validation_macro_f1": validation_selected.get("macro_f1"),
                "validation_wire_bytes": validation_record.get("validation_wire_bytes"),
                **eval_record,
            }
            row["child_ids"] = ";".join(str(x) for x in (update.get("child_ids") or []))
            row["train_positive_rate"] = self._positive_rate(row.get("train_positive"), row.get("train_samples"))
            row["val_positive_rate"] = self._positive_rate(row.get("val_positive"), row.get("val_samples"))
            row["test_positive_rate"] = self._positive_rate(row.get("test_positive"), row.get("profile_test_samples"))
            self._append_csv(self.participants_csv, PARTICIPANT_FIELDS, row)

            # For HierFedAvg, preserve edge-level rows *and* materialize the
            # actual child measurements so analysis can compare local training
            # time and non-IID class balance at the same logical-client level as
            # direct algorithms.  No raw samples are included.
            if self.algorithm != "HierFedAvg":
                continue

            child_eval_by_id = {
                str(item.get("client_id")): item
                for item in (eval_record.get("child_eval_records") or [])
                if item.get("client_id") is not None
            }
            child_validation_by_id = {
                str(item.get("client_id")): item
                for item in (validation_record.get("child_validation_records") or [])
                if item.get("client_id") is not None
            }
            for child_update in update.get("child_update_records") or []:
                child_id = str(child_update.get("client_id", ""))
                if not child_id:
                    continue
                child_eval = child_eval_by_id.get(child_id, {})
                child_validation = child_validation_by_id.get(child_id, {})
                child_validation_selected = self._validation_at_threshold(
                    child_validation,
                    selected_threshold,
                )
                child_row = {
                    "round": round_id,
                    "algorithm": self.algorithm,
                    "participant_id": child_id,
                    "role": "client",
                    "parent_id": update["participant_id"],
                    "dataset_name": child_update.get("dataset_name"),
                    "train_samples": child_update.get("train_samples"),
                    "val_samples": child_update.get("val_samples"),
                    "profile_test_samples": child_update.get("test_samples_profile"),
                    "train_positive": child_update.get("train_positive"),
                    "val_positive": child_update.get("val_positive"),
                    "test_positive": child_update.get("test_positive"),
                    "train_seconds": child_update.get("train_seconds"),
                    "final_loss": child_update.get("final_loss"),
                    "final_accuracy": child_update.get("final_accuracy"),
                    "update_wire_bytes": child_update.get("wire_bytes"),
                    "selected_threshold": selected_threshold,
                    "validation_samples": child_validation.get("val_samples"),
                    "validation_loss": child_validation.get("val_loss"),
                    "validation_accuracy": child_validation_selected.get("accuracy"),
                    "validation_macro_f1": child_validation_selected.get("macro_f1"),
                    "validation_wire_bytes": child_validation.get("wire_bytes"),
                    "test_samples": child_eval.get("test_samples"),
                    "test_loss": child_eval.get("test_loss"),
                    "test_accuracy": (
                        None
                        if not child_eval.get("test_samples")
                        else (int(child_eval.get("tp", 0)) + int(child_eval.get("tn", 0)))
                             / int(child_eval["test_samples"])
                    ),
                    "tp": child_eval.get("tp"),
                    "tn": child_eval.get("tn"),
                    "fp": child_eval.get("fp"),
                    "fn": child_eval.get("fn"),
                    "eval_seconds": child_eval.get("eval_seconds"),
                    "evaluation_wire_bytes": child_eval.get("wire_bytes"),
                }
                child_row["train_positive_rate"] = self._positive_rate(
                    child_row.get("train_positive"), child_row.get("train_samples")
                )
                child_row["val_positive_rate"] = self._positive_rate(
                    child_row.get("val_positive"), child_row.get("val_samples")
                )
                child_row["test_positive_rate"] = self._positive_rate(
                    child_row.get("test_positive"), child_row.get("profile_test_samples")
                )
                self._append_csv(self.participants_csv, PARTICIPANT_FIELDS, child_row)

    def _save_weights(self) -> None:
        np.savez_compressed(
            self.run_dir / "global_weights.npz",
            **{f"arr_{i:05d}": w for i, w in enumerate(self.global_weights)},
        )

    def _write_summary(self) -> None:
        common = {
            "status": "failed" if self.failure_reason else "completed",
            "failure_reason": self.failure_reason,
            "run_id": self.run_dir.name,
            "campaign_id": self.args.campaign_id,
            "repetition": self.args.repetition,
            "requested_clients": int(self.args.requested_clients or 0),
            "algorithm": self.algorithm,
            "algorithm_params": self._algorithm_params(),
            "seed": int(self.args.seed),
            "completed_rounds": len(self.round_rows),
            "started_utc": self.started_utc.isoformat(),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "server_wall_seconds": float(time.perf_counter() - self.started_monotonic),
            "run_dir": str(self.run_dir),
        }
        if not self.round_rows:
            summary = common
        else:
            final = self.round_rows[-1]
            summary = {
                **common,
                "final_round": int(final["round"]),
                "final_metrics": {
                    key: final.get(key)
                    for key in (
                        "test_samples", "test_loss", "accuracy", "precision", "recall",
                        "specificity", "f1", "normal_f1", "macro_f1", "tp", "tn", "fp", "fn",
                        "selected_threshold", "validation_samples", "validation_loss",
                        "validation_accuracy", "validation_precision", "validation_recall",
                        "validation_specificity", "validation_f1", "validation_normal_f1",
                        "validation_macro_f1", "validation_tp", "validation_tn",
                        "validation_fp", "validation_fn"
                    )
                },
                "total_network_bytes": int(sum(int(r["bytes_total"]) for r in self.round_rows)),
                "total_training_network_bytes": int(sum(int(r["bytes_total_train"]) for r in self.round_rows)),
                "total_evaluation_network_bytes": int(sum(int(r["bytes_total_eval"]) for r in self.round_rows)),
                "total_round_seconds": float(sum(float(r["round_seconds"]) for r in self.round_rows)),
                "mean_round_seconds": float(
                    sum(float(r["round_seconds"]) for r in self.round_rows) / len(self.round_rows)
                ),
                "mean_training_phase_seconds": float(
                    sum(float(r["training_phase_seconds"]) for r in self.round_rows) / len(self.round_rows)
                ),
                "mean_evaluation_phase_seconds": float(
                    sum(float(r["evaluation_phase_seconds"]) for r in self.round_rows) / len(self.round_rows)
                ),
                "mean_aggregation_seconds": float(
                    sum(float(r["aggregation_seconds"]) for r in self.round_rows) / len(self.round_rows)
                ),
                "final_logical_client_count": int(final.get("logical_client_count", 0) or 0),
                "min_logical_client_count": int(min(int(r.get("logical_client_count", 0) or 0) for r in self.round_rows)),
                "max_logical_client_count": int(max(int(r.get("logical_client_count", 0) or 0) for r in self.round_rows)),
            }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def run_rounds(self) -> None:
        round_id = 1
        while not self.shutdown.is_set() and (self.args.rounds == 0 or round_id <= self.args.rounds):
            role_name = "edge" if self.algorithm == "HierFedAvg" else "client"
            LOG.info("Waiting for at least one connected %s...", role_name)
            if not self.registry.wait_for_any(self.shutdown):
                break

            join_window = (
                self.args.initial_join_window
                if round_id == 1 and self.args.initial_join_window is not None
                else self.args.join_window
            )
            if join_window > 0:
                LOG.info("Round %d join window: %.1fs", round_id, join_window)
                self.shutdown.wait(join_window)
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
                self.failure_reason = f"round_{round_id}_produced_no_valid_updates"
                self._write_summary()
                raise RuntimeError(
                    f"Round {round_id} produced no valid updates; aborting experiment"
                )

            validation_records: list[dict[str, Any]] = []
            eval_records: list[dict[str, Any]] = []
            selected_threshold: float | None = None
            validation_metrics: dict[str, Any] = {
                "test_samples": 0,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "specificity": None,
                "f1": None,
                "normal_f1": None,
                "macro_f1": None,
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
            }
            validation_loss: float | None = None
            evaluation_phase_seconds = 0.0
            before_eval_sent = after_train_sent
            before_eval_recv = after_train_recv
            if updates and self.args.evaluate_every > 0 and round_id % self.args.evaluate_every == 0:
                eval_started = time.perf_counter()
                threshold_grid = make_threshold_grid(
                    self.args.threshold_min,
                    self.args.threshold_max,
                    self.args.threshold_step,
                )
                validation_records = self._collect_threshold_validations(
                    cohort,
                    successful_ids,
                    round_id,
                    threshold_grid,
                )
                if not validation_records:
                    self.failure_reason = f"round_{round_id}_produced_no_validation_results"
                    self._write_summary()
                    raise RuntimeError(
                        f"Round {round_id} produced no valid threshold-validation results"
                    )

                selected = select_threshold_by_macro_f1(
                    validation_records,
                    threshold_grid,
                    preferred_threshold=float(self.args.threshold_preferred),
                )
                selected_threshold = float(selected["threshold"])
                validation_metrics = dict(selected)
                validation_total = sum(int(record["val_samples"]) for record in validation_records)
                validation_loss = (
                    sum(float(record["val_loss"]) * int(record["val_samples"]) for record in validation_records)
                    / validation_total
                    if validation_total
                    else None
                )
                LOG.info(
                    "Round %d: selected threshold %.3f on validation | n=%d | macro-F1=%.4f | F1=%.4f",
                    round_id,
                    selected_threshold,
                    validation_metrics["test_samples"],
                    validation_metrics["macro_f1"],
                    validation_metrics["f1"],
                )
                eval_records = self._collect_evaluations(
                    cohort,
                    successful_ids,
                    round_id,
                    threshold=selected_threshold,
                )
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
            edge_eval_down = sum(
                int(r.get("edge_child_eval_bytes_down", 0))
                for r in [*validation_records, *eval_records]
            )
            edge_eval_up = sum(
                int(r.get("edge_child_eval_bytes_up", 0))
                for r in [*validation_records, *eval_records]
            )
            total_train = cloud_train_down + cloud_train_up + edge_train_down + edge_train_up
            total_eval = cloud_eval_down + cloud_eval_up + edge_eval_down + edge_eval_up

            if self.algorithm == "HierFedAvg":
                logical_client_ids = []
                for record in update_records:
                    logical_client_ids.extend(str(value) for value in (record.get("child_ids") or []))
            else:
                logical_client_ids = list(successful_ids)
            # Preserve order while de-duplicating, useful if a malformed edge ever
            # reports a child twice.
            logical_client_ids = list(dict.fromkeys(logical_client_ids))

            row = {
                "round": round_id,
                "algorithm": self.algorithm,
                "requested_clients": int(self.args.requested_clients or 0),
                "cohort_size": len(cohort),
                "successful_updates": len(updates),
                "evaluation_results": len(eval_records),
                "client_ids": ";".join(cohort_ids),
                "successful_client_ids": ";".join(successful_ids),
                "logical_client_count": len(logical_client_ids),
                "logical_client_ids": ";".join(logical_client_ids),
                "train_samples": sum(sample_counts),
                **metrics,
                "selected_threshold": selected_threshold,
                "validation_samples": int(validation_metrics.get("test_samples", 0) or 0),
                "validation_loss": validation_loss,
                "validation_accuracy": validation_metrics.get("accuracy"),
                "validation_precision": validation_metrics.get("precision"),
                "validation_recall": validation_metrics.get("recall"),
                "validation_specificity": validation_metrics.get("specificity"),
                "validation_f1": validation_metrics.get("f1"),
                "validation_normal_f1": validation_metrics.get("normal_f1"),
                "validation_macro_f1": validation_metrics.get("macro_f1"),
                "validation_tp": validation_metrics.get("tp"),
                "validation_tn": validation_metrics.get("tn"),
                "validation_fp": validation_metrics.get("fp"),
                "validation_fn": validation_metrics.get("fn"),
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
            self._save_participant_rows(
                round_id,
                update_records,
                validation_records,
                eval_records,
                selected_threshold,
            )
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
                    participant.send({
                        "type": "STOP",
                        "reason": self.failure_reason or "experiment_finished",
                    })
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
    parser.add_argument("--run-id", default=None, help="Optional exact result-directory name for benchmark orchestration")
    parser.add_argument("--campaign-id", default=None, help="Optional benchmark campaign identifier stored as metadata")
    parser.add_argument("--repetition", type=int, default=None, help="Optional benchmark repetition number stored as metadata")
    parser.add_argument("--requested-clients", type=int, default=0, help="Benchmark metadata only; never controls client admission")
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--fedpaq-bits", type=int, default=8)
    parser.add_argument("--join-window", type=float, default=2.0)
    parser.add_argument("--initial-join-window", type=float, default=None, help="Optional longer first-round registration grace period")
    parser.add_argument("--round-timeout", type=float, default=300.0)
    parser.add_argument("--evaluation-timeout", type=float, default=300.0)
    parser.add_argument("--evaluate-every", type=int, default=1, help="0 disables distributed evaluation")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--threshold-min", type=float, default=0.0, help="Minimum validation threshold candidate")
    parser.add_argument("--threshold-max", type=float, default=1.0, help="Maximum validation threshold candidate")
    parser.add_argument("--threshold-step", type=float, default=0.01, help="Validation threshold grid step")
    parser.add_argument("--threshold-preferred", type=float, default=0.5, help="Tie-break threshold when macro-F1 is equal")
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
    if not 0.0 <= args.threshold_min <= 1.0:
        parser.error("--threshold-min must be in [0, 1]")
    if not 0.0 <= args.threshold_max <= 1.0:
        parser.error("--threshold-max must be in [0, 1]")
    if args.threshold_max < args.threshold_min:
        parser.error("--threshold-max must be >= --threshold-min")
    if args.threshold_step <= 0:
        parser.error("--threshold-step must be > 0")
    if not 0.0 <= args.threshold_preferred <= 1.0:
        parser.error("--threshold-preferred must be in [0, 1]")
    if args.requested_clients < 0:
        parser.error("--requested-clients must be >= 0")
    if args.repetition is not None and args.repetition < 1:
        parser.error("--repetition must be >= 1")
    if args.initial_join_window is not None and args.initial_join_window < 0:
        parser.error("--initial-join-window must be >= 0")
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
    try:
        server.serve()
    except RuntimeError as exc:
        LOG.error("Experiment failed: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
