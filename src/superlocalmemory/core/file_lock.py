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

# path -> thread that reserved it. A second descriptor in the owning thread is
# not excluded by POSIX flock, so same-thread re-entry is rejected here.
# Other threads wait on this condition instead of failing immediately.
_held_owners: dict[str, int] = {}
_held_cond = threading.Condition()
_LOCK_POLL_S = 0.05


def _wait_seconds(timeout_s: float | None) -> float:
    """Seconds to wait. None, NaN, and non-positive values do not wait."""
    if timeout_s is None:
        return 0.0
    value = float(timeout_s)
    if value <= 0.0 or value != value:
        return 0.0
    return value


def _remaining(deadline: float, wait_s: float) -> float:
    if wait_s <= 0.0:
        return 0.0
    return max(0.0, deadline - time.monotonic())


def _reserve(path: Path, deadline: float, wait_s: float) -> None:
    path_str = str(path)
    me = threading.get_ident()
    with _held_cond:
        while True:
            owner = _held_owners.get(path_str)
            if owner is None:
                _held_owners[path_str] = me
                return
            if owner == me:
                raise LockHeldError(f"{path} already locked by this process")
            remaining = _remaining(deadline, wait_s)
            if remaining <= 0.0:
                raise LockHeldError(f"{path} already locked by this process")
            _held_cond.wait(timeout=remaining)


def _release_reservation(path_str: str) -> None:
    # The context may be closed by a different thread than the one that
    # entered it (the write coordinator's lease reaper does this). The
    # reservation belongs to the context, so whichever thread exits it
    # must drop the path.
    with _held_cond:
        if path_str in _held_owners:
            _held_owners.pop(path_str, None)
            _held_cond.notify_all()


def _lock_portalocker(fd: int, path: Path, deadline: float, wait_s: float) -> None:
    """Take a non-blocking exclusive lock, retrying until ``deadline``."""
    flags = portalocker.LOCK_EX | portalocker.LOCK_NB
    while True:
        try:
            portalocker.lock(fd, flags)
            return
        except portalocker.LockException as exc:
            remaining = _remaining(deadline, wait_s)
            if remaining <= 0.0:
                raise LockHeldError(f"{path} is locked") from exc
            time.sleep(min(_LOCK_POLL_S, remaining))


@contextmanager
def exclusive_lock(path: Path, timeout_s: float | None = 0.0) -> Iterator[int]:
    """Acquire an exclusive file lock.

    A positive ``timeout_s`` waits until the lock is free or the deadline
    passes, then raises ``LockHeldError``. ``0``, ``None``, and any other
    non-positive value try once. Same-thread re-entry always fails
    immediately so the caller cannot deadlock on its own lock.
    """
    path_str = str(path)
    wait_s = _wait_seconds(timeout_s)
    deadline = time.monotonic() + wait_s
    _reserve(path, deadline, wait_s)
    try:
        if _HAS_PORTALOCKER:
            fd = os.open(path_str, os.O_RDWR | os.O_CREAT, 0o600)
            locked = False
            try:
                _lock_portalocker(fd, path, deadline, wait_s)
                locked = True
                yield fd
            finally:
                if locked:
                    try:
                        portalocker.unlock(fd)
                    finally:
                        os.close(fd)
                else:
                    os.close(fd)
        else:
            with _fallback_registry_lock:
                lk = _fallback_registry.setdefault(path_str, _FallbackLock())
            if not lk.acquire(timeout=_remaining(deadline, wait_s)):
                raise LockHeldError(f"{path} is locked (fallback)")
            try:
                yield -1
            finally:
                lk.release()
    finally:
        _release_reservation(path_str)
