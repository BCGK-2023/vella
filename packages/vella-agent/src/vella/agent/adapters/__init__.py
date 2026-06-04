"""Optional-extra real adapters — heavy I/O kept OUTSIDE the deterministic gate.

The cognition core ships exactly five runtime dependencies (asserted by
``tests/test_deps.py``) and importing ``vella.agent`` never pulls ``httpx`` or
``mcp``. The real provider/tool transports live here, behind the ``[openrouter]``
and ``[mcp]`` extras, and each adapter imports its heavy dependency **lazily**
(inside ``__init__``/methods, never at module top): merely importing this package
or its modules must not import ``httpx``/``mcp``, so the in-gate import-boundary and
dependency-hygiene invariants hold whether or not the extras are installed.

* :mod:`vella.agent.adapters.openrouter` — :class:`OpenRouterProvider`, a
  :class:`~vella.agent.ModelProvider` over OpenRouter's OpenAI-compatible
  chat-completions SSE stream, normalized into the canonical streaming lifecycle
  and folded through the SHARED :mod:`vella.agent._assembler` (never a private
  accumulator) so its result is byte-identical to the in-gate path.
* :mod:`vella.agent.adapters.mcp_invoker` — :class:`MCPToolInvoker`, a real
  :class:`~vella.agent.ToolInvoker` that dispatches an
  :class:`~vella.agent.MCPBinding` against the referenced ``mcp_server`` node's
  transport and maps the response to a :class:`~vella.agent.ToolResult`.

These are NOT part of the gated public surface (``vella.agent.__all__``): a user who
installed the extra imports the adapter explicitly. They are deliberately not
re-exported here either — importing the package marker must stay dependency-free.
"""

from __future__ import annotations

__all__: list[str] = []
