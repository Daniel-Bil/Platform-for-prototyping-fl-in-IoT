"""Wire protocol for the deployment FL platform.

A frame consists of:
    4 bytes  magic (FLP1)
    1 byte   protocol version
    8 bytes  JSON metadata length (big endian)
    8 bytes  binary NumPy payload length (big endian)
    N bytes  UTF-8 JSON metadata
    M bytes  np.savez_compressed payload containing model tensors

The protocol intentionally keeps control data human-readable while transporting
model weights as binary arrays instead of expanding floats into JSON strings.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import socket
import struct
from typing import Any, Mapping, Sequence

import numpy as np

MAGIC = b"FLP1"
PROTOCOL_VERSION = 1
_HEADER = struct.Struct("!4sBQQ")
MAX_METADATA_BYTES = 1 * 1024 * 1024
MAX_PAYLOAD_BYTES = 512 * 1024 * 1024


class ProtocolError(RuntimeError):
    pass


class ConnectionClosed(ConnectionError):
    pass


@dataclass(frozen=True)
class Message:
    metadata: dict[str, Any]
    arrays: list[np.ndarray]
    wire_bytes: int


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionClosed("peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _encode_arrays(arrays: Sequence[np.ndarray] | None) -> bytes:
    if not arrays:
        return b""
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        **{f"arr_{idx:05d}": np.asarray(value) for idx, value in enumerate(arrays)},
    )
    return buffer.getvalue()


def _decode_arrays(payload: bytes) -> list[np.ndarray]:
    if not payload:
        return []
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            keys = sorted(archive.files)
            return [np.array(archive[key], copy=True) for key in keys]
    except Exception as exc:  # corrupted/incompatible payload
        raise ProtocolError(f"invalid NumPy payload: {exc}") from exc


def send_message(
    sock: socket.socket,
    metadata: Mapping[str, Any],
    arrays: Sequence[np.ndarray] | None = None,
) -> int:
    meta = dict(metadata)
    meta.setdefault("protocol_version", PROTOCOL_VERSION)
    metadata_bytes = json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload = _encode_arrays(arrays)

    if len(metadata_bytes) > MAX_METADATA_BYTES:
        raise ProtocolError("metadata frame is too large")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ProtocolError("model payload is too large")

    header = _HEADER.pack(MAGIC, PROTOCOL_VERSION, len(metadata_bytes), len(payload))
    sock.sendall(header)
    sock.sendall(metadata_bytes)
    if payload:
        sock.sendall(payload)
    return len(header) + len(metadata_bytes) + len(payload)


def recv_message(sock: socket.socket) -> Message:
    header = _recv_exact(sock, _HEADER.size)
    magic, version, metadata_len, payload_len = _HEADER.unpack(header)

    if magic != MAGIC:
        raise ProtocolError("invalid frame magic")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")
    if metadata_len > MAX_METADATA_BYTES:
        raise ProtocolError("metadata length exceeds safety limit")
    if payload_len > MAX_PAYLOAD_BYTES:
        raise ProtocolError("payload length exceeds safety limit")

    metadata_raw = _recv_exact(sock, metadata_len)
    payload = _recv_exact(sock, payload_len) if payload_len else b""
    try:
        metadata = json.loads(metadata_raw.decode("utf-8"))
    except Exception as exc:
        raise ProtocolError(f"invalid JSON metadata: {exc}") from exc

    if not isinstance(metadata, dict):
        raise ProtocolError("metadata must be a JSON object")

    return Message(
        metadata=metadata,
        arrays=_decode_arrays(payload),
        wire_bytes=_HEADER.size + metadata_len + payload_len,
    )
