# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Cross-platform exclusive file lock via portalocker.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:
    import portalocker
    _HAS_PORTALOCKER = True
except ImportError:  # pragma: no cover
    portalocker = None  # type: ignore[assignment]
    _HAS_PORTALOCKER = False


class LockHeldError(RuntimeError):
    """Raised when the lock cannot be acquired (held by another holder)."""


class _FallbackLock:
    """Thread-local mutex used when portalocker is unavailable."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def acquire(self, timeout: float) -> bool:
        return self._lock.acquire(timeout=timeout)

    def release(self) -> None:
        self._lock.release()


_fallback_registry: dict[str, _FallbackLock] = {}
_fallback_registry_lock = threading.Lock()

# Track fds held by this process so a double-acquire in the same process
# is rejected immediately (portalocker on POSIX is permissive on same fd).
_held_in_process: set[str] = set()
_held_lock = threading.Lock()


@contextmanager
def exclusive_lock(path: Path, timeout_s: float = 0.0) -> Iterator[int]:
    """Acquire an exclusive file lock. Raises LockHeldError on contention."""
    path_str = str(path)
    with _held_lock:
        if path_str in _held_in_process:
            raise LockHeldError(f"{path} already locked by this process")
        _held_in_process.add(path_str)

    try:
        if _HAS_PORTALOCKER:
            fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                deadline = time.monotonic() + max(0.0, timeout_s)
                while True:
                    try:
                        portalocker.lock(
                            fd, portalocker.LOCK_EX | portalocker.LOCK_NB,
                        )
                        break
                    except portalocker.LockException as exc:
                        if timeout_s <= 0.0 or time.monotonic() >= deadline:
                            os.close(fd)
                            raise LockHeldError(f"{path} is locked") from exc
                        remaining = max(0.0, deadline - time.monotonic())
                        time.sleep(min(0.05, remaining))
                try:
                    yield fd
                finally:
                    try:
                        portalocker.unlock(fd)
                    finally:
                        os.close(fd)
            except LockHeldError:
                raise
        else:  # pragma: no cover — fallback path
            with _fallback_registry_lock:
                lk = _fallback_registry.setdefault(path_str, _FallbackLock())
            if not lk.acquire(timeout=timeout_s):
                raise LockHeldError(f"{path} is locked (fallback)")
            try:
                yield -1
            finally:
                lk.release()
    finally:
        with _held_lock:
            _held_in_process.discard(path_str)
