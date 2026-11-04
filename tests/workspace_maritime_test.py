# -*- coding: utf-8 -*-
"""Test cases for MaritimeWorkspace.

The lifecycle class is skipped when the ``MARITIME_API_KEY`` environment
variable is not set, because it requires a live Maritime sandbox.
``TestMaritimeWorkspaceAttach`` runs everywhere against a fake client.
"""

import os
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.async_case import IsolatedAsyncioTestCase

from sandbox_test_support import CheckoutWheelTestCase

from agentscope.mcp import MCPClient, StdioMCPConfig
from agentscope.workspace import MaritimeBackend, MaritimeWorkspace
from agentscope.workspace._maritime._constants import (
    METADATA_WORKSPACE_ID_KEY,
)

# ── Maritime availability check ─────────────────────────────────────────

_MARITIME_API_KEY = os.getenv("MARITIME_API_KEY", "")
_SKIP_REASON = "MARITIME_API_KEY environment variable is not set"


# ── lifecycle tests ────────────────────────────────────────────────


@unittest.skipUnless(_MARITIME_API_KEY, _SKIP_REASON)
class TestMaritimeWorkspaceLifecycle(CheckoutWheelTestCase):
    """Test cases for MaritimeWorkspace lifecycle and MCP integration.

    Each test creates a real Maritime sandbox and tears it down
    (``pause``) afterward.  The suite is skipped entirely when
    ``MARITIME_API_KEY`` is absent so that CI runs without Maritime
    credentials
    are unaffected.
    """

    workspace_type = MaritimeWorkspace

    async def asyncSetUp(self) -> None:
        """No shared setup; each test manages its own workspace."""

    async def asyncTearDown(self) -> None:
        """No shared teardown; each test closes its own workspace."""

    async def test_initialize_and_list_mcps(self) -> None:
        """``initialize`` starts the sandbox and ``list_mcps`` enumerates MCPs.

        Verifies:
        1. The workspace initializes without raising.
        2. ``list_mcps`` returns at least the seeded MCP (playwright).
        3. Each MCP exposes at least one tool via ``list_raw_tools``.
        4. ``close`` (sandbox pause) completes without raising.
        """
        workspace = MaritimeWorkspace(
            api_key=_MARITIME_API_KEY,
            default_mcps=[
                MCPClient(
                    name="playwright",
                    mcp_config=StdioMCPConfig(
                        command="npx",
                        args=["@playwright/mcp@latest"],
                    ),
                    is_stateful=True,
                ),
            ],
        )

        await workspace.initialize()

        mcps = await workspace.list_mcps(
            agent_id="test-agent",
            session_id="test-session",
        )
        self.assertGreater(len(mcps), 0)

        for mcp in mcps:
            tools = await mcp.list_raw_tools()
            self.assertGreater(len(tools), 0)

        await workspace.close()


# ── attach / create (no network) ───────────────────────────────────


class _FakeSandboxes:
    """Stand-in for ``Maritime().sandboxes`` with scripted list pages."""

    def __init__(self, pages: list[list[Any]]) -> None:
        """Serve *pages* in order, linked by cursors."""
        self.pages = pages
        self.list_calls: list[dict[str, Any]] = []
        self.created: list[dict[str, Any]] = []
        self.connected: list[Any] = []

    def list(self, **kwargs: Any) -> Any:
        """Return the page the cursor points at."""
        self.list_calls.append(kwargs)
        index = int(kwargs.get("cursor") or 0)
        more = index + 1 < len(self.pages)
        return SimpleNamespace(
            data=self.pages[index] if self.pages else [],
            next_cursor=str(index + 1) if more else None,
        )

    def create(self, **kwargs: Any) -> Any:
        """Record the create body and return a new sandbox."""
        self.created.append(kwargs)
        return SimpleNamespace(id="sbx_new")

    def connect(self, data: Any) -> Any:
        """Wrap listed data as a client, as the SDK does without a call."""
        self.connected.append(data)
        return SimpleNamespace(id=data.id)


def _workspace(pages: list[list[Any]], **kwargs: Any) -> Any:
    """A workspace whose client is the fake above."""
    ws = MaritimeWorkspace(workspace_id="ws-1", **kwargs)
    ws._client = SimpleNamespace(sandboxes=_FakeSandboxes(pages))
    return ws


class TestMaritimeWorkspaceAttach(IsolatedAsyncioTestCase):
    """Reattachment by metadata and the create body."""

    async def test_creates_with_metadata_when_nothing_matches(self) -> None:
        """A new sandbox is tagged with the workspace id plus extras."""
        ws = _workspace(
            [[]],
            sandbox_metadata={"team": "a"},
            env={"K": "V"},
            idle_timeout_seconds=600,
        )

        await ws._provision_backend()

        fake = ws._client.sandboxes
        self.assertEqual(ws.sandbox_id, "sbx_new")
        self.assertIsInstance(ws._backend, MaritimeBackend)
        self.assertDictEqual(
            fake.list_calls[0],
            {
                "state": ["running", "paused"],
                "metadata": {METADATA_WORKSPACE_ID_KEY: "ws-1"},
                "cursor": None,
            },
        )
        self.assertDictEqual(
            fake.created[0],
            {
                "from_": "base",
                "metadata": {METADATA_WORKSPACE_ID_KEY: "ws-1", "team": "a"},
                "idle_timeout_seconds": 600,
                "env": {"K": "V"},
            },
        )

    async def test_reattaches_to_newest_match_across_pages(self) -> None:
        """Every page is read and the newest sandbox wins; none is
        created."""
        old = SimpleNamespace(id="sbx_old", created_at="2026-10-01T00:00:00Z")
        new = SimpleNamespace(id="sbx_new", created_at="2026-10-02T00:00:00Z")
        ws = _workspace([[old], [new]])

        await ws._provision_backend()

        fake = ws._client.sandboxes
        self.assertEqual(ws.sandbox_id, "sbx_new")
        self.assertEqual(len(fake.list_calls), 2)
        self.assertListEqual(fake.created, [])

    async def test_close_pauses_instead_of_killing(self) -> None:
        """Teardown keeps the sandbox for the next ``initialize``."""
        calls: list[str] = []
        ws = _workspace([[]])
        ws._sandbox = SimpleNamespace(
            pause=lambda: calls.append("pause"),
            kill=lambda: calls.append("kill"),
        )

        await ws._teardown_backend()

        self.assertListEqual(calls, ["pause"])
        self.assertIsNone(ws.sandbox_id)
