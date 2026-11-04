# -*- coding: utf-8 -*-
"""Maritime sandbox :class:`BackendBase` implementation.

Wraps the Maritime SDK's ``Sandbox.run`` and ``Sandbox.files`` APIs into
the three backend primitives (``exec_shell``, ``read_file``,
``write_file``) so that builtin tools (Bash, Read, Write, Edit, Grep,
Glob) can operate inside a Maritime sandbox transparently. All derived
filesystem helpers (``file_exists``, ``is_dir``, ``list_dir``,
``stat_mtime``, ``delete_path``) are inherited from :class:`BackendBase`,
which implements them via ``exec_shell``.

The Maritime SDK is synchronous, so every call runs in a worker thread
via :func:`asyncio.to_thread` to keep the event loop free.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from ...tool import BackendBase, ExecResult
from ._constants import MAX_COMMAND_TIMEOUT, MIN_COMMAND_TIMEOUT


class MaritimeBackend(BackendBase):
    """Backend that delegates to a running Maritime sandbox.

    Only the three abstract primitives (``exec_shell``, ``read_file``,
    ``write_file``) are implemented here; the derived filesystem helpers
    are inherited from :class:`BackendBase`.

    Args:
        sandbox (`Any`):
            A ``maritime.Sandbox`` client. A paused sandbox resumes on
            the first call that reaches it.
        workdir (`str`):
            Default working directory for ``exec_shell`` calls inside
            the sandbox.
    """

    def __init__(self, sandbox: Any, workdir: str) -> None:
        """Initialize the Maritime backend.

        Args:
            sandbox (`Any`):
                A ``maritime.Sandbox`` client.
            workdir (`str`):
                Default working directory for ``exec_shell`` calls
                inside the sandbox.
        """
        self._sandbox = sandbox
        self._workdir = workdir

    # ── exec ───────────────────────────────────────────────────────

    async def getcwd(self) -> str:
        """Return the sandbox's default working directory.

        Overrides the base class default (which would shell out to
        ``pwd``) with the cached ``workdir`` supplied at construction,
        avoiding a per-call sandbox round-trip.

        Returns:
            `str`:
                The sandbox's default working directory.
        """
        return self._workdir

    async def exec_shell(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Run a program inside the sandbox via ``Sandbox.run``.

        *command* is an argv list and is sent to Maritime as an argv
        array, so it runs without a shell and needs no quoting. Callers
        needing shell features pass ``["sh", "-c", line]``.

        Args:
            command (`list[str]`):
                Executable path/name followed by its arguments.
            cwd (`str | None`, optional):
                Working directory inside the sandbox. When ``None`` the
                backend's default ``workdir`` is used.
            timeout (`float | None`, optional):
                Maximum number of seconds to wait, rounded up to whole
                seconds and clamped to the 1 to 3600 range Maritime
                accepts. When ``None`` the Maritime default (60 s)
                applies.

        Returns:
            `ExecResult`:
                The captured exit code, stdout, and stderr. A non-zero
                command exit is reported as a normal result; timeouts
                and transport errors yield an ``exit_code`` of ``-1``.
        """
        from maritime import MaritimeSandboxError

        kwargs: dict[str, Any] = {"cwd": cwd or self._workdir}
        if timeout is not None:
            kwargs["timeout_seconds"] = min(
                max(math.ceil(timeout), MIN_COMMAND_TIMEOUT),
                MAX_COMMAND_TIMEOUT,
            )
        try:
            res = await asyncio.to_thread(
                self._sandbox.run,
                list(command),
                **kwargs,
            )
        except MaritimeSandboxError as e:
            partial = getattr(e, "command", None)
            stdout = getattr(partial, "stdout", "") or ""
            stderr = getattr(partial, "stderr", "") or ""
            return ExecResult(
                exit_code=-1,
                stdout=stdout.encode("utf-8"),
                stderr=(stderr + str(e)).encode("utf-8"),
            )
        except Exception as e:  # noqa: BLE001
            return ExecResult(
                exit_code=-1,
                stdout=b"",
                stderr=str(e).encode("utf-8"),
            )
        return ExecResult(
            # ``exit_code`` is None when a signal ended the process.
            exit_code=-1 if res.exit_code is None else int(res.exit_code),
            stdout=(res.stdout or "").encode("utf-8"),
            stderr=(res.stderr or "").encode("utf-8"),
        )

    # ── file I/O ───────────────────────────────────────────────────

    async def read_file(self, path: str) -> bytes:
        """Read a file from the sandbox via ``files.read``.

        Args:
            path (`str`):
                Absolute path to the file inside the sandbox.

        Returns:
            `bytes`:
                The raw file contents.

        Raises:
            `FileNotFoundError`:
                If the path does not exist inside the sandbox.
        """
        from maritime import MaritimeSandboxError

        try:
            data = await asyncio.to_thread(self._sandbox.files.read, path)
        except MaritimeSandboxError as exc:
            if exc.code == "file_not_found":
                raise FileNotFoundError(
                    f"not found in sandbox: {path}",
                ) from exc
            raise
        return bytes(data)

    async def write_file(self, path: str, data: bytes) -> None:
        """Write *data* to a file inside the sandbox.

        Maritime's ``files.write`` creates missing parent directories
        itself, so no ``mkdir`` round-trip is needed.

        Args:
            path (`str`):
                Absolute destination path inside the sandbox.
            data (`bytes`):
                The raw bytes to write.
        """
        await asyncio.to_thread(self._sandbox.files.write, path, data)
