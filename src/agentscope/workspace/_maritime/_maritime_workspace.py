# -*- coding: utf-8 -*-
"""MaritimeWorkspace: sandboxed workspace backed by a Maritime sandbox.

Architecture
------------

Mirrors :class:`agentscope.workspace.E2BWorkspace` but swaps the E2B SDK
for the Maritime SDK (``maritime.Maritime().sandboxes``):

* **Lifecycle.** ``initialize()`` looks up an existing sandbox by
  metadata and either reattaches to it (a paused sandbox resumes on the
  first call that reaches it) or creates a fresh one and runs the
  bootstrap shell sequence. ``close()`` calls ``sandbox.pause()`` so the
  sandbox filesystem and memory (skills, ``.mcp``, sessions, data,
  the running gateway) survive for the next ``initialize()``.
* **Idle pause.** Maritime also pauses a sandbox on its own after
  ``idle_timeout_seconds`` with no calls, and resumes it on the next
  call, so a workspace the manager never closes stops billing compute
  while nobody uses it.
* **Bootstrap.** First-time provisioning installs uv + a gateway venv
  + agentscope (``--no-deps``) and uploads the gateway script. The
  probe + install loop lives on :class:`SandboxedWorkspaceBase`; this
  subclass only supplies the sandbox-specific shell commands via
  :meth:`_bootstrap_commands`.
* **MCP gateway.** Identical to E2B: a FastAPI process inside the
  sandbox, driven through :class:`GatewayClient` via
  :meth:`MaritimeBackend.exec_shell`. No host→sandbox network
  reachability is required.
* **Service-layer index.** The host stores only ``workspace_id``; the
  sandbox carries ``METADATA_WORKSPACE_ID_KEY = workspace_id`` in its
  Maritime metadata, and ``sandboxes.list(metadata=...)`` finds it on
  cache miss.

The Maritime SDK is synchronous; every call runs in a worker thread.
"""

import asyncio
import shlex
from typing import Any

from ..._logging import logger
from ...mcp import MCPClient
from .._sandboxed_base import SandboxedWorkspaceBase
from .._utils import (
    UV_VERSION,
    _GATEWAY_BASE_REQUIREMENTS,
    DEFAULT_WORKSPACE_INSTRUCTIONS,
)
from ._constants import (
    DEFAULT_GATEWAY_PORT,
    DEFAULT_TEMPLATE,
    GATEWAY_HOME,
    METADATA_WORKSPACE_ID_KEY,
    SANDBOX_WORKDIR,
)
from ._maritime_backend import MaritimeBackend


class MaritimeWorkspace(SandboxedWorkspaceBase):
    """Workspace backed by a Maritime sandbox.

    ``default_mcps`` and ``skill_paths`` are seed-time inputs and are
    not retained as instance state past :meth:`initialize`.
    """

    _gateway_home = GATEWAY_HOME
    # The ``base`` template already ships python3 + curl + sudo, so
    # every bootstrap step is expected to finish inside 10 minutes.
    _bootstrap_cmd_timeout = 600.0

    def __init__(
        self,
        *,
        workspace_id: str | None = None,
        template: str = DEFAULT_TEMPLATE,
        api_key: str = "",
        base_url: str = "",
        size: str | None = None,
        idle_timeout_seconds: int | None = None,
        gateway_port: int = DEFAULT_GATEWAY_PORT,
        env: dict[str, str] | None = None,
        sandbox_metadata: dict[str, str] | None = None,
        extra_pip: list[str] | None = None,
        instructions: str = DEFAULT_WORKSPACE_INSTRUCTIONS,
        default_mcps: list[MCPClient] | None = None,
        skill_paths: list[str] | None = None,
    ) -> None:
        """Construct a :class:`MaritimeWorkspace`.

        The sandbox is *not* started here; call :meth:`initialize`
        (or use the workspace as an ``async`` context manager).

        Args:
            workspace_id (`str | None`, optional):
                Stable identifier; also stored in sandbox metadata for
                reattachment.
            template (`str`, defaults to `DEFAULT_TEMPLATE`):
                Maritime stock template or saved snapshot name the
                sandbox starts from.
            api_key (`str`, defaults to `""`):
                Maritime API key (``""`` falls back to
                ``MARITIME_API_KEY``).
            base_url (`str`, defaults to `""`):
                Optional API base URL (``""`` falls back to
                ``MARITIME_API_URL`` or ``https://api.maritime.sh``).
            size (`str | None`, optional):
                Sandbox size preset (e.g. ``"small"``, ``"medium"``).
                ``None`` uses Maritime's default.
            idle_timeout_seconds (`int | None`, optional):
                Seconds without calls before Maritime pauses the
                sandbox. ``None`` uses Maritime's default.
            gateway_port (`int`, defaults to `DEFAULT_GATEWAY_PORT`):
                TCP port the in-sandbox gateway listens on.
            env (`dict[str, str] | None`, optional):
                Environment variables baked into the sandbox.
            sandbox_metadata (`dict[str, str] | None`, optional):
                Extra metadata merged with the workspace-id tag.
            extra_pip (`list[str] | None`, optional):
                Extra Python packages installed into the gateway venv
                during bootstrap.
            instructions (`str`, defaults to `DEFAULT_WORKSPACE_INSTRUCTIONS`):
                System-prompt fragment template (supports ``{workdir}``).
            default_mcps (`list[MCPClient] | None`, optional):
                MCPs registered on first init when no persisted
                ``.mcp`` exists.
            skill_paths (`list[str] | None`, optional):
                Local skill dirs seeded into ``skills/`` on first init.
        """
        super().__init__(
            workspace_id=workspace_id,
            default_mcps=default_mcps,
            skill_paths=skill_paths,
        )

        # ── serializable config ─────────────────────────────────
        self.workdir = SANDBOX_WORKDIR
        self.template = template
        self.api_key = api_key
        self.base_url = base_url
        self.size = size
        self.idle_timeout_seconds = idle_timeout_seconds
        self.gateway_port = gateway_port
        self.env: dict[str, str] = dict(env or {})
        self.sandbox_metadata: dict[str, str] = dict(sandbox_metadata or {})
        self.extra_pip: list[str] = list(extra_pip or [])
        self.instructions = instructions.format(
            backend="Maritime-based",
            workdir=self.workdir,
        )

        # ── runtime state (Maritime-only) ───────────────────────
        self._client: Any = None  # maritime.Maritime
        self._sandbox: Any = None  # maritime.Sandbox
        self._backend: MaritimeBackend | None = None

    # ── lifecycle hooks ─────────────────────────────────────────

    @property
    def sandbox_id(self) -> str | None:
        """Maritime sandbox id, or ``None`` if not started."""
        return self._sandbox.id if self._sandbox else None

    async def _provision_backend(self) -> None:
        """Reattach or create the sandbox and bind the backend.

        First-time bootstrap (uv → gateway venv → agentscope → gateway
        script upload) is driven by
        :meth:`SandboxedWorkspaceBase._ensure_bootstrapped` once
        ``initialize`` has bound the backend; every step is idempotent
        so an interrupted bootstrap re-runs cleanly.
        """
        await self._attach_or_create_sandbox()
        self._backend = MaritimeBackend(
            self._sandbox,
            workdir=SANDBOX_WORKDIR,
        )

    async def _teardown_backend(self) -> None:
        """Pause the sandbox (keep disk and memory) and drop the handle.

        ``pause()``, not ``kill()``, so the next :meth:`initialize`
        can reattach via metadata lookup. Errors are swallowed.
        """
        if self._sandbox is not None:
            try:
                await asyncio.to_thread(self._sandbox.pause)
            except Exception as e:
                logger.warning("MaritimeWorkspace: pause failed: %s", e)
            self._sandbox = None

    # ── instructions ────────────────────────────────────────────

    async def get_instructions(self) -> str:
        """Return the system-prompt fragment for this workspace.

        Substitutes ``{workdir}`` in the configured template with the
        sandbox-side path (``/home/user/workspace``). The agent always
        sees sandbox-internal paths.
        """
        return self.instructions.format(workdir=SANDBOX_WORKDIR)

    # ── internals: sandbox attach / create ─────────────────────

    def _get_client(self) -> Any:
        """Return the lazily built ``maritime.Maritime`` client."""
        if self._client is None:
            from maritime import Maritime

            self._client = Maritime(
                api_key=self.api_key or None,
                base_url=self.base_url or None,
            )
        return self._client

    async def _attach_or_create_sandbox(self) -> None:
        """Reattach to an existing sandbox by metadata, or create one.

        Resolution rule: a single sandbox is expected per
        ``workspace_id``. If several match (e.g. a leaked running +
        paused pair after an unclean shutdown) we attach to the newest
        by ``created_at`` and log a warning; manual cleanup is left to
        the operator.

        Maritime's ``create`` returns once the sandbox is running, and
        a paused sandbox resumes on the first command or file call, so
        no readiness poll is needed here.
        """
        client = self._get_client()
        existing = await self._find_existing_sandbox()

        if existing is not None:
            self._sandbox = client.sandboxes.connect(existing)
            return

        create_kwargs: dict[str, Any] = {
            "from_": self.template,
            "metadata": {
                METADATA_WORKSPACE_ID_KEY: self.workspace_id,
                **self.sandbox_metadata,
            },
        }
        if self.size is not None:
            create_kwargs["size"] = self.size
        if self.idle_timeout_seconds is not None:
            create_kwargs["idle_timeout_seconds"] = self.idle_timeout_seconds
        if self.env:
            create_kwargs["env"] = self.env

        self._sandbox = await asyncio.to_thread(
            client.sandboxes.create,
            **create_kwargs,
        )

    async def _find_existing_sandbox(self) -> Any:
        """List sandboxes filtered by ``workspace_id`` metadata.

        Returns the most recent ``SandboxData`` (paused or running) or
        ``None`` if no match exists.
        """
        client = self._get_client()
        candidates: list[Any] = []
        cursor: str | None = None
        while True:
            try:
                page = await asyncio.to_thread(
                    client.sandboxes.list,
                    state=["running", "paused"],
                    metadata={METADATA_WORKSPACE_ID_KEY: self.workspace_id},
                    cursor=cursor,
                )
            except Exception as e:
                logger.warning(
                    "MaritimeWorkspace: list sandboxes failed: %s",
                    e,
                )
                break
            candidates.extend(page.data)
            cursor = page.next_cursor
            if not cursor:
                break

        if not candidates:
            return None
        if len(candidates) > 1:
            logger.warning(
                "MaritimeWorkspace: %d sandboxes match workspace_id=%r; "
                "attaching to most recent",
                len(candidates),
                self.workspace_id,
            )
        candidates.sort(key=lambda s: s.created_at, reverse=True)
        return candidates[0]

    # ── internals: bootstrap ────────────────────────────────────

    def _bootstrap_commands(self) -> list[str]:
        """Shell commands that provision this Maritime sandbox once.

        Only runs when the gateway script is missing (fresh sandbox or a
        prior bootstrap that was interrupted). Every step is idempotent
        so a resumed sandbox can re-run cleanly.

        ``--no-deps`` on agentscope is mandatory: the gateway only
        imports :class:`agentscope.mcp.MCPClient` whose transitive needs
        are installed via the gateway base requirements.
        """
        pip_pkgs = list(_GATEWAY_BASE_REQUIREMENTS) + list(self.extra_pip)
        # Quote every requirement string so entries with spaces or
        # shell metacharacters cannot break the ``sh -c`` command or
        # become a command-injection vector inside the sandbox.
        pip_args = " ".join(shlex.quote(p) for p in pip_pkgs)

        return [
            # 1. System deps + uv installer, both via sudo so uv lands
            # at /usr/local/bin (on PATH).
            "sudo apt-get update -qq "
            "&& sudo apt-get install -y --no-install-recommends "
            "curl ca-certificates ripgrep "
            "&& sudo update-ca-certificates "
            "&& sudo rm -rf /var/lib/apt/lists/*",
            f"curl -LsSf https://astral.sh/uv/{UV_VERSION}/install.sh "
            "| sudo env UV_INSTALL_DIR=/usr/local/bin "
            "INSTALLER_NO_MODIFY_PATH=1 sh",
            # 2. Gateway venv + base requirements + agentscope.
            f"uv venv {self._gateway_venv}",
            f"uv pip install --python {self._gateway_python} {pip_args}",
            f"uv pip install --python {self._gateway_python} "
            f"--no-deps 'agentscope'",
        ]
