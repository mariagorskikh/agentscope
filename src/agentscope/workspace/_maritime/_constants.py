# -*- coding: utf-8 -*-
"""Maritime-specific constants for :class:`MaritimeWorkspace`.

Path layout (venv, script, log, helper) is derived on the base class
from ``_gateway_home``. This module only carries defaults that cannot
be derived: template, port, sandbox user, metadata key, command limits.
"""

#: Default Maritime stock template. Ubuntu with python3, curl, git and
#: passwordless sudo for the ``user`` account.
DEFAULT_TEMPLATE = "base"

#: Default port the in-sandbox gateway listens on.
DEFAULT_GATEWAY_PORT = 5600

#: Sandbox-side runtime user (commands run as ``user``, not root).
SANDBOX_USER_HOME = "/home/user"

#: Sandbox-side workdir and gateway home.
SANDBOX_WORKDIR = f"{SANDBOX_USER_HOME}/workspace"
GATEWAY_HOME = f"{SANDBOX_USER_HOME}/.agentscope"

#: Sandbox metadata key used to map workspace_id → sandbox_id.
METADATA_WORKSPACE_ID_KEY = "agentscope.workspace.id"

#: Bounds Maritime accepts for one foreground command's
#: ``timeoutSeconds``.
MIN_COMMAND_TIMEOUT = 1
MAX_COMMAND_TIMEOUT = 3600
