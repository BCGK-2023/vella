"""Append-only log domain types — the ordered record of every transition.

Where ``vella.core`` is pure, frozen data, the log is the spine of the runtime:
every create/edit/delete/telemetry event becomes one immutable ``LogEntry`` at a
``Cursor`` position. State tables, observers, and replay all derive from this one
ordered stream — the log is the source of truth, the state table a fold of it.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel
from vella.core import UTCDatetime


class Cursor(BaseModel, frozen=True):
    """Opaque, totally-ordered, resumable position in the global log.

    Consumers receive cursors from the store and pass them back to
    ``observe(since=cursor)`` — never compare by value, never inspect
    internals. Adapters encode their ordering key as the token string:
    in-memory uses ``str(offset)``; a SQL adapter might use a BIGSERIAL,
    LSN, or base64-encoded key. Adapters parse the token internally via
    a private helper (e.g. ``_parse_token(cursor) -> int``).

    The surface tripwire snapshots ``{"token": {"type": "string"}}`` —
    a real, stable shape that freezes the contract without freezing ``int``.

    Deliberately carries NO ``__lt__``/``__le__``: cursors are compared by the
    owning adapter (which knows how to parse its own token), never by callers.
    """

    token: str


TransitionKind = Literal[
    "create",
    "edit",
    "set_desired",
    "upsert",
    "delete",
    "link",
    "unlink",
    "observe_only",  # telemetry: emitted to observers, no version bump, no state-table update
]
"""The kind of transition a ``LogEntry`` records.

``observe_only`` is the telemetry channel: such entries reach observers and the
log, but never bump an entity's version or touch the derived state table.
"""


class LogEntry(BaseModel, frozen=True):
    """One ordered, append-only record of a transition.

    ``payload`` holds **actual model-instance fields** in the in-memory
    adapter — ``{k: getattr(entity, k) for k in type(entity).model_fields}``
    — real typed objects (UUID, datetime, nested model instances) that core's
    ``hydrate`` door can consume directly (``hydrate`` calls
    ``model_construct`` internally, which needs typed objects, not dicts).

    CRITICAL: ``model_dump(mode="python")`` CANNOT be used — it returns plain
    dicts for nested models (verified on Pydantic 2.13: ``dump['integrations']
    [0]`` is ``dict``, ``dump['data']`` is ``dict``), and ``hydrate`` on those
    produces structurally invalid nodes.

    Canonical bytes are derived ONLY at the serialization boundary via
    ``json.dumps(entity.model_dump(mode="json"), sort_keys=True,
    separators=(",",":"))``. A SQL adapter stores this JSON form and
    reconstructs via ``parse_node``/``parse_edge`` (the portable replay path).

    Core fields keep core's own ``model_dump`` order (list fields like
    ``integrations`` and ``contributes_to`` are order-semantic and are NEVER
    re-sorted by runtime).
    """

    cursor: Cursor  # opaque position in the global log
    tenant_id: str
    entity_kind: Literal["node", "edge"]
    entity_id: UUID
    version: int  # post-transition version (unchanged for observe_only)
    transition: TransitionKind
    payload: dict[str, Any]  # model-instance fields (in-memory); JSON-mode dump (SQL)
    recorded_at: UTCDatetime


__all__ = ["Cursor", "TransitionKind", "LogEntry"]
