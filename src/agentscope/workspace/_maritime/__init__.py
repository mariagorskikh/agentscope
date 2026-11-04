# -*- coding: utf-8 -*-
"""Maritime-backed workspace package.

Re-exports :class:`MaritimeWorkspace` so callers can write
``from agentscope.workspace._maritime import MaritimeWorkspace`` without
having to poke at the underlying module layout.
"""

from ._maritime_workspace import MaritimeWorkspace
from ._maritime_backend import MaritimeBackend

__all__ = ["MaritimeWorkspace", "MaritimeBackend"]
