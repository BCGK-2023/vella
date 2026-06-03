"""Vella agent (a self-hosted cognition core over the runtime + graph).

Where ``vella.runtime`` is *physics* — the append-only log, the
optimistic-concurrency store, and the write verbs that move world state forward —
and ``vella.graph`` is a *read-only projection* that answers traversal queries from
memory, ``vella.agent`` is the *cognition core*: a data-configured interpreter that
acts ONLY through the runtime's published verbs and perceives ONLY through the
graph's published projection. It owns no storage and takes no privileged path; an
agent run, its steps, tool calls, messages, and policy are all ordinary registered
core node types.

Design principles
-----------------
* **Self-hosting is the substrate, not a feature.** The agent acts only through
  ``vella.runtime``'s public verbs; cognition is nodes/edges and ``observe_only``
  telemetry. There is no new ``Node`` subclass and no edit pushed down into
  core/runtime/graph.
* **Three Protocol seams.** ``ModelProvider`` / ``ToolInvoker`` / ``ContextAssembler``
  each have a deterministic in-gate reference impl and optional out-of-gate real
  adapters (``[openrouter]`` / ``[mcp]``), so heavy I/O never enters the gate.
* **Determinism is a property, not a hope.** The interpreter is network-free under
  its reference impls; any set-derived serialized value is ``sorted()``; the gated
  determinism artifact is byte-identical across hash seeds.
* **Depend downward only.** The agent imports only the published ``vella.core``,
  ``vella.runtime``, and ``vella.graph`` surfaces — NEVER ``vella.reconciler`` (a
  sibling, not a dependency); all three lower layers are unaware of it.

The public surface grows milestone by milestone; everything in ``__all__`` is
importable, documented, and snapshotted by the surface tripwire from M0 onward. The
node type-specs, canonical-turn models, the three Protocols, and the FSM interpreter
land in later milestones; the surface is baselined now (empty) so the tripwire
guards it from the start.
"""

from __future__ import annotations

__all__: list[str] = []
