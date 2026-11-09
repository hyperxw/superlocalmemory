"""MCP coverage for manual fact classification updates."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch


class _MockServer:
    def __init__(self) -> None:
        self._tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self._tools[fn.__name__] = fn
            return fn
        return decorator


def _tool():
    from superlocalmemory.mcp.tools_core import register_core_tools

    server = _MockServer()
    register_core_tools(server, MagicMock())
    return server._tools["update_fact_type"]


def test_update_fact_type_uses_capability_owned_worker_when_daemon_offline():
    tool = _tool()
    pool = MagicMock()
    pool.update_fact_type.return_value = {
        "ok": True,
        "fact_id": "fact-1",
        "old_fact_type": "semantic",
        "fact_type": "opinion",
    }

    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=False),
        patch("superlocalmemory.core.worker_pool.WorkerPool.shared", return_value=pool),
    ):
        result = asyncio.run(tool("fact-1", "OPINION", "test-agent"))

    assert result["success"] is True
    assert result["old_fact_type"] == "semantic"
    assert result["fact_type"] == "opinion"
    pool.update_fact_type.assert_called_once_with(
        "fact-1", "opinion", "test-agent",
    )


def test_update_fact_type_rejects_invalid_value_before_mutation():
    tool = _tool()

    result = asyncio.run(tool("fact-1", "invalid", "test-agent"))

    assert result["success"] is False
    assert "invalid fact_type" in result["error"]
