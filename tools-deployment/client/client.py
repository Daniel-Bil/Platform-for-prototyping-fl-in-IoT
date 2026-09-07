#!/usr/bin/env python3
"""Federated learning deployment client."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import socket
import sys
import time
import traceback

DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
if str(DEPLOYMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_ROOT))

from common.data import load_client_data
from common.protocol import ConnectionClosed, ProtocolError, recv_message, send_message
from client.trainer import LocalTrainer

LOG = logging.getLogger("fl-client")


def connect_with_retry(host: str, port: int, retry_seconds: float, stop_after: float) -> socket.socket:
    started = time.monotonic()
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=10.0)
            sock.settimeout(None)
            return sock
        except OSError as exc:
            if stop_after > 0 and time.monotonic() - started >= stop_after:
                raise ConnectionError(f"unable to connect to {host}:{port}") from exc
            LOG.warning("Server unavailable (%s); retrying in %.1fs", exc, retry_seconds)
            time.sleep(retry_seconds)


def run(args: argparse.Namespace) -> int:
    sock = connect_with_retry(args.server, args.port, args.reconnect_delay, args.connect_timeout)
    try:
        LOG.info("Connected to %s:%d", args.server, args.port)
        send_message(
            sock,
            {
                "type": "HELLO",
                "client_id": args.client_id,
                "auth_token": args.auth_token,
            },
        )
        welcome = recv_message(sock)
        if welcome.metadata.get("type") == "ERROR":
            raise RuntimeError(welcome.metadata.get("error", "server rejected client"))
        if welcome.metadata.get("type") != "WELCOME":
            raise ProtocolError("server did not send WELCOME")

        server_seq_len = int(welcome.metadata["seq_len"])
        server_features = tuple(str(value) for value in welcome.metadata["features"])
        if not server_features:
            raise ProtocolError("server supplied an empty feature list")

        LOG.info("Loading local data from %s using server experiment config", args.data)
        data = load_client_data(args.data, seq_len=server_seq_len, features=server_features)
        LOG.info(
            "Prepared local dataset: train=%d val=%d test=%d sequences",
            len(data.X_train), len(data.X_val), len(data.X_test),
        )
        trainer = LocalTrainer(
            model_config=welcome.metadata["model_config"],
            data=data,
            seq_len=server_seq_len,
            num_features=len(server_features),
        )
        send_message(
            sock,
            {
                "type": "READY",
                "client_id": args.client_id,
                "train_samples": data.train_samples,
            },
        )
        LOG.info("Registered as %s; waiting for training rounds", args.client_id)

        while True:
            message = recv_message(sock)
            meta = message.metadata
            msg_type = meta.get("type")

            if msg_type == "STOP":
                LOG.info("Server stopped experiment: %s", meta.get("reason", ""))
                return 0
            if msg_type != "TRAIN":
                LOG.warning("Ignoring server message: %s", msg_type)
                continue

            round_id = int(meta["round"])
            try:
                if meta.get("algorithm") != "FedAvg":
                    raise ValueError(f"unsupported algorithm {meta.get('algorithm')!r}")
                if not message.arrays:
                    raise ProtocolError("TRAIN message has no global weights")
                LOG.info("Round %d: training started", round_id)
                result = trainer.train_fedavg(
                    global_weights=message.arrays,
                    local_epochs=int(meta["local_epochs"]),
                    batch_size=int(meta["batch_size"]),
                )
                send_message(
                    sock,
                    {
                        "type": "UPDATE",
                        "round": round_id,
                        "client_id": args.client_id,
                        "train_samples": data.train_samples,
                        "train_seconds": round(result.train_seconds, 6),
                        "final_loss": result.final_loss,
                        "final_accuracy": result.final_accuracy,
                    },
                    result.weights,
                )
                LOG.info("Round %d: update sent (training %.2fs)", round_id, result.train_seconds)
            except Exception as exc:
                LOG.exception("Round %d failed", round_id)
                send_message(
                    sock,
                    {
                        "type": "CLIENT_ERROR",
                        "round": round_id,
                        "client_id": args.client_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
    finally:
        try:
            sock.close()
        except OSError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated learning deployment client")
    parser.add_argument("--server", required=True, help="Server LAN IP or DNS name")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--data", required=True, help="Directory containing train.csv/val.csv/test.csv")
    parser.add_argument("--auth-token", default=os.environ.get("FL_AUTH_TOKEN"), help="Optional shared token; prefer FL_AUTH_TOKEN env var")
    parser.add_argument("--reconnect-delay", type=float, default=3.0)
    parser.add_argument("--connect-timeout", type=float, default=0.0, help="0 = retry forever")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    try:
        raise SystemExit(run(args))
    except KeyboardInterrupt:
        LOG.info("Interrupted")
        raise SystemExit(130)
    except (ConnectionClosed, ConnectionError, ProtocolError, OSError) as exc:
        LOG.error("Connection ended: %s", exc)
        raise SystemExit(2)
    except Exception as exc:
        LOG.error("Fatal error: %s", exc)
        LOG.debug("%s", traceback.format_exc())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
