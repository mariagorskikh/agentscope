# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Test cases for :class:`MaritimeBackend`.

Validates that the three backend primitives (``exec_shell``,
``read_file``, ``write_file``) and the inherited shell-based filesystem
helpers behave correctly inside a real Maritime sandbox.

The live class is skipped unless the ``MARITIME_API_KEY`` environment
variable is set. CI runs without Maritime credentials are therefore
unaffected; when a key *is* present the tests exercise the real
``Sandbox.run`` / ``files.*`` APIs. ``TestMaritimeBackendMapping`` runs
everywhere against a fake sandbox.

A live sandbox is obtained by initializing a :class:`MaritimeWorkspace` and
reusing its already-wired :class:`MaritimeBackend` (``ws._backend``), which
avoids duplicating the sandbox bring-up logic here.
"""

import importlib.util
import os
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.async_case import IsolatedAsyncioTestCase

from sandbox_test_support import CheckoutWheelTestCase

from agentscope.tool import ExecResult
from agentscope.workspace import MaritimeWorkspace
from agentscope.workspace import MaritimeBackend
from agentscope.workspace._maritime._constants import SANDBOX_WORKDIR


# ── Maritime availability check ─────────────────────────────────────────

_MARITIME_API_KEY = os.getenv("MARITIME_API_KEY", "")
_SKIP_REASON = "MARITIME_API_KEY environment variable is not set"


@unittest.skipUnless(_MARITIME_API_KEY, _SKIP_REASON)
class TestMaritimeBackend(CheckoutWheelTestCase):
    """Test cases for ``MaritimeBackend`` against a live sandbox.

    Each test creates a real Maritime sandbox via ``MaritimeWorkspace`` and
    tears it down (``close`` → sandbox pause) afterwards.
    """

    workspace_type = MaritimeWorkspace

    async def asyncSetUp(self) -> None:
        """Start a workspace and reuse its wired backend."""
        self.workspace = MaritimeWorkspace(api_key=_MARITIME_API_KEY)
        await self.workspace.initialize()
        self.backend = self.workspace._backend
        self.assertIsInstance(self.backend, MaritimeBackend)

    async def asyncTearDown(self) -> None:
        """Pause / close the sandbox."""
        await self.workspace.close()

    # ── exec ───────────────────────────────────────────────────────

    async def test_exec_returns_stdout(self) -> None:
        """A program's stdout/exit code are captured into ``ExecResult``."""
        result = await self.backend.exec_shell(["echo", "hello world"])
        self.assertIsInstance(result, ExecResult)
        self.assertTrue(result.ok())
        self.assertEqual(result.stdout.decode().strip(), "hello world")

    async def test_exec_nonzero_exit(self) -> None:
        """A non-zero command exit is reported as a normal result."""
        result = await self.backend.exec_shell(
            ["sh", "-c", "echo oops >&2; exit 4"],
        )
        self.assertEqual(result.exit_code, 4)
        self.assertIn("oops", result.stderr.decode())

    async def test_exec_argv_quoting_preserved(self) -> None:
        """An argv element with spaces / metacharacters survives the
        argv array the backend sends to ``Sandbox.run``."""
        tricky = "a b c | ;"
        result = await self.backend.exec_shell(["echo", tricky])
        self.assertTrue(result.ok())
        self.assertEqual(result.stdout.decode().rstrip("\n"), tricky)

    async def test_exec_cwd_default_is_workdir(self) -> None:
        """With no explicit ``cwd`` the sandbox workdir is used."""
        result = await self.backend.exec_shell(["pwd"])
        self.assertTrue(result.ok())
        self.assertEqual(result.stdout.decode().strip(), SANDBOX_WORKDIR)

    # ── file I/O ───────────────────────────────────────────────────

    async def test_write_then_read_roundtrip(self) -> None:
        """Bytes written into the sandbox are read back verbatim."""
        path = f"{SANDBOX_WORKDIR}/roundtrip.txt"
        payload = b"hello\nworld\n"
        await self.backend.write_file(path, payload)
        self.assertEqual(await self.backend.read_file(path), payload)

    async def test_write_creates_parent_dirs(self) -> None:
        """``write_file`` creates missing parent directories."""
        path = f"{SANDBOX_WORKDIR}/a/b/c/file.txt"
        await self.backend.write_file(path, b"x")
        self.assertEqual(await self.backend.read_file(path), b"x")

    async def test_read_missing_file_raises(self) -> None:
        """Reading a non-existent file raises ``FileNotFoundError``."""
        with self.assertRaises(FileNotFoundError):
            await self.backend.read_file(f"{SANDBOX_WORKDIR}/nope.txt")

    # ── derived filesystem helpers (shell-based) ───────────────────

    async def test_file_exists_and_is_dir(self) -> None:
        """``file_exists`` / ``is_dir`` reflect the sandbox filesystem."""
        path = f"{SANDBOX_WORKDIR}/f.txt"
        await self.backend.write_file(path, b"x")
        self.assertTrue(await self.backend.file_exists(path))
        self.assertTrue(await self.backend.is_dir(SANDBOX_WORKDIR))
        self.assertFalse(await self.backend.is_dir(path))
        self.assertFalse(
            await self.backend.file_exists(f"{SANDBOX_WORKDIR}/missing"),
        )

    async def test_list_dir(self) -> None:
        """Non-recursive ``list_dir`` returns immediate child base names."""
        base = f"{SANDBOX_WORKDIR}/listing"
        await self.backend.write_file(f"{base}/a.txt", b"x")
        await self.backend.write_file(f"{base}/b.txt", b"x")
        entries = await self.backend.list_dir(base)
        self.assertEqual(sorted(entries), ["a.txt", "b.txt"])

    async def test_stat_mtime(self) -> None:
        """``stat_mtime`` returns a float for an existing path, None else."""
        path = f"{SANDBOX_WORKDIR}/stat.txt"
        await self.backend.write_file(path, b"x")
        mtime = await self.backend.stat_mtime(path)
        self.assertIsInstance(mtime, float)
        self.assertIsNone(
            await self.backend.stat_mtime(f"{SANDBOX_WORKDIR}/missing"),
        )

    async def test_delete_path(self) -> None:
        """``delete_path`` removes files and trees; missing is a no-op."""
        path = f"{SANDBOX_WORKDIR}/to_delete.txt"
        await self.backend.write_file(path, b"x")
        await self.backend.delete_path(path)
        self.assertFalse(await self.backend.file_exists(path))

        tree = f"{SANDBOX_WORKDIR}/tree"
        await self.backend.write_file(f"{tree}/deep/f.txt", b"x")
        await self.backend.delete_path(tree)
        self.assertFalse(await self.backend.file_exists(tree))

        # Deleting a non-existent path must not raise.
        await self.backend.delete_path(f"{SANDBOX_WORKDIR}/missing")


# ── SDK mapping (no network) ───────────────────────────────────────

_HAS_SDK = importlib.util.find_spec("maritime") is not None


class _FakeFiles:
    """In-memory stand-in for ``Sandbox.files``."""

    def __init__(self) -> None:
        """Start with an empty filesystem."""
        self.store: dict[str, bytes] = {}

    def read(self, path: str) -> bytes:
        """Return stored bytes or raise Maritime's ``file_not_found``."""
        from maritime import MaritimeSandboxError

        if path not in self.store:
            raise MaritimeSandboxError(
                404,
                code="file_not_found",
                message=f"No file at {path}",
            )
        return self.store[path]

    def write(self, path: str, data: bytes) -> None:
        """Store *data* at *path*."""
        self.store[path] = data


class _FakeSandbox:
    """Records ``run`` calls and replays a scripted result."""

    def __init__(self, result: Any = None, error: Any = None) -> None:
        """Script the next ``run`` outcome."""
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self.result = result
        self.error = error
        self.files = _FakeFiles()

    def run(self, cmd: Any, **kwargs: Any) -> Any:
        """Record the call, then return or raise the scripted outcome."""
        self.calls.append((cmd, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


def _command(exit_code: int | None, stdout: str, stderr: str) -> Any:
    """A Command-shaped object as ``Sandbox.run`` returns it."""
    return SimpleNamespace(exit_code=exit_code, stdout=stdout, stderr=stderr)


@unittest.skipUnless(_HAS_SDK, "maritime SDK is not installed")
class TestMaritimeBackendMapping(IsolatedAsyncioTestCase):
    """``MaritimeBackend`` maps the SDK onto ``BackendBase`` correctly."""

    async def test_argv_cwd_and_timeout_are_forwarded(self) -> None:
        """argv goes through untouched, cwd defaults to the workdir, and
        a fractional timeout is rounded up to whole seconds."""
        sandbox = _FakeSandbox(result=_command(0, "hi\n", ""))
        backend = MaritimeBackend(sandbox, workdir=SANDBOX_WORKDIR)

        result = await backend.exec_shell(["echo", "a b | ;"], timeout=2.5)

        self.assertTrue(result.ok())
        self.assertEqual(result.stdout, b"hi\n")
        self.assertListEqual(
            sandbox.calls,
            [
                (
                    ["echo", "a b | ;"],
                    {"cwd": SANDBOX_WORKDIR, "timeout_seconds": 3},
                ),
            ],
        )

    async def test_timeout_is_clamped_to_maritime_bounds(self) -> None:
        """Maritime accepts 1 to 3600 seconds per foreground command."""
        sandbox = _FakeSandbox(result=_command(0, "", ""))
        backend = MaritimeBackend(sandbox, workdir=SANDBOX_WORKDIR)

        await backend.exec_shell(["true"], cwd="/tmp", timeout=0.01)
        await backend.exec_shell(["true"], timeout=7200)

        self.assertEqual(sandbox.calls[0][1]["cwd"], "/tmp")
        self.assertEqual(sandbox.calls[0][1]["timeout_seconds"], 1)
        self.assertEqual(sandbox.calls[1][1]["timeout_seconds"], 3600)

    async def test_no_timeout_leaves_the_sdk_default(self) -> None:
        """``timeout=None`` sends no ``timeout_seconds`` at all."""
        sandbox = _FakeSandbox(result=_command(0, "", ""))
        backend = MaritimeBackend(sandbox, workdir=SANDBOX_WORKDIR)

        await backend.exec_shell(["true"])

        self.assertNotIn("timeout_seconds", sandbox.calls[0][1])

    async def test_nonzero_exit_is_a_normal_result(self) -> None:
        """A failing command is reported, not raised."""
        sandbox = _FakeSandbox(result=_command(4, "", "oops\n"))
        backend = MaritimeBackend(sandbox, workdir=SANDBOX_WORKDIR)

        result = await backend.exec_shell(["sh", "-c", "exit 4"])

        self.assertEqual(result.exit_code, 4)
        self.assertEqual(result.stderr, b"oops\n")

    async def test_signal_killed_process_reports_minus_one(self) -> None:
        """Maritime leaves ``exit_code`` null when a signal ended it."""
        sandbox = _FakeSandbox(result=_command(None, "partial", ""))
        backend = MaritimeBackend(sandbox, workdir=SANDBOX_WORKDIR)

        result = await backend.exec_shell(["sleep", "9"])

        self.assertEqual(result.exit_code, -1)
        self.assertEqual(result.stdout, b"partial")

    async def test_command_timeout_keeps_partial_output(self) -> None:
        """``command_timeout`` carries the output produced so far."""
        from maritime import MaritimeSandboxError

        error = MaritimeSandboxError(
            408,
            code="command_timeout",
            message="The command ran past timeoutSeconds.",
        )
        error.command = _command(None, "half", "warn\n")
        sandbox = _FakeSandbox(error=error)
        backend = MaritimeBackend(sandbox, workdir=SANDBOX_WORKDIR)

        result = await backend.exec_shell(["sleep", "99"], timeout=1)

        self.assertFalse(result.ok())
        self.assertEqual(result.exit_code, -1)
        self.assertEqual(result.stdout, b"half")
        self.assertIn(b"warn", result.stderr)
        self.assertIn(b"timeoutSeconds", result.stderr)

    async def test_transport_error_reports_minus_one(self) -> None:
        """Any other failure becomes ``exit_code == -1`` with the text."""
        sandbox = _FakeSandbox(error=ConnectionError("network down"))
        backend = MaritimeBackend(sandbox, workdir=SANDBOX_WORKDIR)

        result = await backend.exec_shell(["true"])

        self.assertEqual(result.exit_code, -1)
        self.assertEqual(result.stderr, b"network down")

    async def test_files_roundtrip_and_missing_file(self) -> None:
        """Writes land verbatim; a missing path is ``FileNotFoundError``."""
        sandbox = _FakeSandbox()
        backend = MaritimeBackend(sandbox, workdir=SANDBOX_WORKDIR)
        path = f"{SANDBOX_WORKDIR}/a/b.txt"

        await backend.write_file(path, b"\x00bytes\n")

        self.assertEqual(await backend.read_file(path), b"\x00bytes\n")
        # write_file relies on Maritime creating parents: no exec call.
        self.assertListEqual(sandbox.calls, [])
        with self.assertRaises(FileNotFoundError):
            await backend.read_file(f"{SANDBOX_WORKDIR}/missing")
