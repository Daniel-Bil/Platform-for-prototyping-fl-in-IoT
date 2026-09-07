"""Thread-safe connection registry shared by cloud and edge servers."""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import queue
import socket
import threading
from typing import Any

from common.protocol import Message, send_message

LOG = logging.getLogger("fl-sessions")


@dataclass
class ClientSession:
    client_id: str
    sock: socket.socket
    address: tuple[str, int]
    train_samples: int
    role: str = "client"
    inbox: queue.Queue[Message] = field(default_factory=queue.Queue)
    send_lock: threading.Lock = field(default_factory=threading.Lock)
    alive: bool = True
    bytes_sent: int = 0
    bytes_received: int = 0

    def send(self, metadata: dict[str, Any], arrays=None) -> int:
        with self.send_lock:
            wire_bytes = send_message(self.sock, metadata, arrays)
            self.bytes_sent += wire_bytes
            return wire_bytes

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
                LOG.warning("Replacing previous connection for %s", session.client_id)
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
