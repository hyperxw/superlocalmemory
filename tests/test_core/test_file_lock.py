# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for core.file_lock — cross-platform single-holder file lock."""

from __future__ import annotations

import multiprocessing
import threading
import time
from pathlib import Path

import pytest


def _imports():
    from superlocalmemory.core import file_lock as fl
    return fl


def test_lock_acquire_release(tmp_path: Path) -> None:
    fl = _imports()
    lock_file = tmp_path / "d.lock"
    with fl.exclusive_lock(lock_file) as h:
        assert h is not None
    # After release, file exists and another lock can be taken
    with fl.exclusive_lock(lock_file):
        pass


def test_double_acquire_in_same_process_rejected(tmp_path: Path) -> None:
    fl = _imports()
    lock_file = tmp_path / "d.lock"
    with fl.exclusive_lock(lock_file):
        with pytest.raises(fl.LockHeldError):
            with fl.exclusive_lock(lock_file):
                pass


def _child_tries_lock(lock_path: str, result_queue: multiprocessing.Queue) -> None:
    from superlocalmemory.core import file_lock as fl
    try:
        with fl.exclusive_lock(Path(lock_path), timeout_s=0.1):
            result_queue.put("acquired")
    except fl.LockHeldError:
        result_queue.put("held")
    except Exception as exc:
        result_queue.put(f"error:{exc}")


def test_second_process_cannot_acquire(tmp_path: Path) -> None:
    fl = _imports()
    lock_file = tmp_path / "d.lock"
    ctx = multiprocessing.get_context("spawn")
    # 30s timeout: CI runners cold-import the full dep tree (torch, onnxruntime)
    # which takes ~5-15s before the child logic even starts.
    _JOIN_TIMEOUT = 30.0
    with fl.exclusive_lock(lock_file):
        q: multiprocessing.Queue = ctx.Queue()
        p = ctx.Process(target=_child_tries_lock, args=(str(lock_file), q))
        p.start()
        p.join(timeout=_JOIN_TIMEOUT)
        assert p.exitcode == 0, (
            f"Child process did not finish within {_JOIN_TIMEOUT}s "
            f"(exitcode={p.exitcode}). CI cold-start too slow?"
        )
        result = q.get(timeout=2.0)
    assert result == "held", f"Expected 'held', got {result}"


def test_released_lock_allows_second_process(tmp_path: Path) -> None:
    fl = _imports()
    lock_file = tmp_path / "d.lock"
    with fl.exclusive_lock(lock_file):
        pass  # release
    ctx = multiprocessing.get_context("spawn")
    _JOIN_TIMEOUT = 30.0
    q: multiprocessing.Queue = ctx.Queue()
    p = ctx.Process(target=_child_tries_lock, args=(str(lock_file), q))
    p.start()
    p.join(timeout=_JOIN_TIMEOUT)
    assert p.exitcode == 0, (
        f"Child process did not finish within {_JOIN_TIMEOUT}s "
        f"(exitcode={p.exitcode}). CI cold-start too slow?"
    )
    assert q.get(timeout=2.0) == "acquired"


def _child_timed_lock(
    lock_path: str,
    timeout_s: float | None,
    result_queue: multiprocessing.Queue,
) -> None:
    from superlocalmemory.core import file_lock as fl

    started = time.monotonic()
    try:
        with fl.exclusive_lock(Path(lock_path), timeout_s=timeout_s):
            result_queue.put(("acquired", time.monotonic() - started))
    except fl.LockHeldError:
        result_queue.put(("held", time.monotonic() - started))
    except Exception as exc:
        result_queue.put((f"error:{exc}", time.monotonic() - started))


def _run_child_timed(
    lock_file: Path, timeout_s: float | None,
) -> tuple[str, float]:
    ctx = multiprocessing.get_context("spawn")
    q: multiprocessing.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_child_timed_lock, args=(str(lock_file), timeout_s, q),
    )
    proc.start()
    proc.join(timeout=30.0)
    assert proc.exitcode == 0, (
        f"Child process did not finish within 30s (exitcode={proc.exitcode})"
    )
    status, elapsed = q.get(timeout=2.0)
    return status, elapsed


def test_second_process_waits_until_lock_is_released(tmp_path: Path) -> None:
    fl = _imports()
    lock_file = tmp_path / "d.lock"
    ctx = multiprocessing.get_context("spawn")
    with fl.exclusive_lock(lock_file):
        q: multiprocessing.Queue = ctx.Queue()
        ready = ctx.Event()
        proc = ctx.Process(
            target=_child_waits_for_release,
            args=(str(lock_file), ready, q),
        )
        proc.start()
        assert ready.wait(timeout=30.0), "Child did not start the lock attempt"
        time.sleep(0.25)
    proc.join(timeout=30.0)
    assert proc.exitcode == 0, (
        f"Child process did not finish within 30s (exitcode={proc.exitcode})"
    )
    status, elapsed = q.get(timeout=2.0)
    assert status == "acquired"
    assert elapsed >= 0.2


def _child_waits_for_release(
    lock_path: str,
    ready_event: multiprocessing.Event,
    result_queue: multiprocessing.Queue,
) -> None:
    from superlocalmemory.core import file_lock as fl

    ready_event.set()
    started = time.monotonic()
    try:
        with fl.exclusive_lock(Path(lock_path), timeout_s=2.0):
            result_queue.put(("acquired", time.monotonic() - started))
    except fl.LockHeldError:
        result_queue.put(("held", time.monotonic() - started))
    except Exception as exc:
        result_queue.put((f"error:{exc}", time.monotonic() - started))


def test_contended_lock_waits_until_timeout(tmp_path: Path) -> None:
    fl = _imports()
    lock_file = tmp_path / "d.lock"
    with fl.exclusive_lock(lock_file):
        status, elapsed = _run_child_timed(lock_file, 0.35)
    assert status == "held"
    assert elapsed >= 0.25


@pytest.mark.parametrize("timeout_s", [0, 0.0, None])
def test_nonpositive_timeout_fails_without_waiting(
    tmp_path: Path, timeout_s: float | None,
) -> None:
    fl = _imports()
    lock_file = tmp_path / "d.lock"
    with fl.exclusive_lock(lock_file):
        status, elapsed = _run_child_timed(lock_file, timeout_s)
    assert status == "held"
    assert elapsed < 0.2


def test_context_can_be_released_from_another_thread(tmp_path: Path) -> None:
    fl = _imports()
    lock_file = tmp_path / "cross-thread.lock"
    context = fl.exclusive_lock(lock_file)
    context.__enter__()

    def release() -> None:
        context.__exit__(None, None, None)

    thread = threading.Thread(target=release)
    thread.start()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    with fl.exclusive_lock(lock_file, timeout_s=0):
        pass


def test_same_thread_reentry_does_not_wait(tmp_path: Path) -> None:
    fl = _imports()
    lock_file = tmp_path / "d.lock"
    with fl.exclusive_lock(lock_file):
        started = time.monotonic()
        with pytest.raises(fl.LockHeldError):
            with fl.exclusive_lock(lock_file, timeout_s=1.0):
                pass
        assert time.monotonic() - started < 0.15


def test_other_thread_waits_until_release(tmp_path: Path) -> None:
    fl = _imports()
    lock_file = tmp_path / "d.lock"
    held = threading.Event()
    release = threading.Event()
    outcome: dict[str, tuple[str, float]] = {}

    def holder() -> None:
        with fl.exclusive_lock(lock_file):
            held.set()
            assert release.wait(timeout=2.0)

    def waiter() -> None:
        started = time.monotonic()
        try:
            with fl.exclusive_lock(lock_file, timeout_s=2.0):
                outcome["result"] = ("acquired", time.monotonic() - started)
        except fl.LockHeldError:
            outcome["result"] = ("held", time.monotonic() - started)
        except Exception as exc:
            outcome["result"] = (f"error:{exc}", time.monotonic() - started)

    first = threading.Thread(target=holder)
    second = threading.Thread(target=waiter)
    first.start()
    assert held.wait(timeout=2.0)
    second.start()
    time.sleep(0.25)
    release.set()
    second.join(timeout=2.0)
    first.join(timeout=2.0)
    assert outcome["result"][0] == "acquired"
    assert outcome["result"][1] >= 0.2


def test_fallback_other_thread_waits_until_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fl = _imports()
    monkeypatch.setattr(fl, "_HAS_PORTALOCKER", False)
    lock_file = tmp_path / "fallback.lock"
    held = threading.Event()
    release = threading.Event()
    outcome: dict[str, tuple[str, float]] = {}

    def holder() -> None:
        with fl.exclusive_lock(lock_file):
            held.set()
            assert release.wait(timeout=2.0)

    def waiter() -> None:
        started = time.monotonic()
        try:
            with fl.exclusive_lock(lock_file, timeout_s=2.0):
                outcome["result"] = ("acquired", time.monotonic() - started)
        except fl.LockHeldError:
            outcome["result"] = ("held", time.monotonic() - started)
        except Exception as exc:
            outcome["result"] = (f"error:{exc}", time.monotonic() - started)

    first = threading.Thread(target=holder)
    second = threading.Thread(target=waiter)
    first.start()
    assert held.wait(timeout=2.0)
    second.start()
    time.sleep(0.25)
    release.set()
    second.join(timeout=2.0)
    first.join(timeout=2.0)
    assert outcome["result"][0] == "acquired"
    assert outcome["result"][1] >= 0.2


def test_fallback_timeout_expires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fl = _imports()
    monkeypatch.setattr(fl, "_HAS_PORTALOCKER", False)
    lock_file = tmp_path / "fallback.lock"
    held = threading.Event()
    release = threading.Event()
    outcome: dict[str, tuple[str, float]] = {}

    def holder() -> None:
        with fl.exclusive_lock(lock_file):
            held.set()
            assert release.wait(timeout=2.0)

    def waiter() -> None:
        started = time.monotonic()
        try:
            with fl.exclusive_lock(lock_file, timeout_s=0.3):
                outcome["result"] = ("acquired", time.monotonic() - started)
        except fl.LockHeldError:
            outcome["result"] = ("held", time.monotonic() - started)
        except Exception as exc:
            outcome["result"] = (f"error:{exc}", time.monotonic() - started)

    first = threading.Thread(target=holder)
    second = threading.Thread(target=waiter)
    first.start()
    assert held.wait(timeout=2.0)
    second.start()
    second.join(timeout=2.0)
    release.set()
    first.join(timeout=2.0)
    assert outcome["result"] == ("held", outcome["result"][1])
    assert outcome["result"][1] >= 0.2


def test_fallback_none_and_negative_do_not_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """None and negative timeouts reach the threading lock as one attempt.

    The in-process reservation is left unset so ``Lock.acquire`` itself sees
    the normalized timeout. ``None`` must not raise TypeError, and a negative
    value must not wait forever (``threading.Lock`` treats -1 as unbounded).
    """
    fl = _imports()
    monkeypatch.setattr(fl, "_HAS_PORTALOCKER", False)
    lock_file = tmp_path / "fallback.lock"
    with fl._fallback_registry_lock:
        fallback = fl._fallback_registry.setdefault(
            str(lock_file), fl._FallbackLock(),
        )
    assert fallback.acquire(timeout=0)

    def attempt(timeout_s: float | None, slot: str) -> None:
        started = time.monotonic()
        try:
            with fl.exclusive_lock(lock_file, timeout_s=timeout_s):
                outcome[slot] = ("acquired", time.monotonic() - started)
        except fl.LockHeldError:
            outcome[slot] = ("held", time.monotonic() - started)
        except Exception as exc:
            outcome[slot] = (f"error:{type(exc).__name__}:{exc}", time.monotonic() - started)

    outcome: dict[str, tuple[str, float]] = {}
    threads = [
        threading.Thread(target=attempt, args=(None, "none")),
        threading.Thread(target=attempt, args=(-1.0, "negative")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=0.4)
    still_running = any(thread.is_alive() for thread in threads)
    fallback.release()
    for thread in threads:
        thread.join(timeout=1.0)
    assert not still_running
    assert outcome["none"][0] == "held"
    assert outcome["none"][1] < 0.2
    assert outcome["negative"][0] == "held"
    assert outcome["negative"][1] < 0.2
