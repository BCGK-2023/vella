"""TRAP-1 payload-independence (M2): topology follows get(), never LogEntry.payload.

The fold computes the live id-set from typed top-level ``LogEntry`` fields, then
calls ``runtime.get(tenant_id, edge_id)`` once per live edge for authoritative
endpoints and type. It must NEVER read ``entry.payload`` for those values.

This test makes that verifiable by constructing a runtime whose ``observe()``
stream poisons every edge entry's payload with WRONG ``from_node_id``,
``to_node_id``, and ``type`` — while its ``get()`` returns the true edge. A fold
that reads payload would build the poisoned topology; a fold that reads ``get()``
builds the true topology.

Non-vacuity: mut-m2-trap1 (read endpoints/type from ``entry.payload`` instead of
``got.from_node_id`` / ``got.to_node_id`` / ``got.type`` in ``_fold.py``) makes
this test RED because the asserted from/to/type would then match the poison values
instead of the true ones.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Optional, Union
from uuid import UUID, uuid4

from vella.core import Edge, EdgeTypes, Node
from vella.runtime import Cursor, LogEntry, Runtime

from vella.graph import GraphProjection

from _fixtures import drive, make_node, thing_registry


class _PayloadPoisonRuntime(Runtime):
    """A ``Runtime`` whose ``observe()`` poisons every edge entry's payload.

    The real write verbs and ``get()`` are unchanged — ``get()`` returns the
    authoritative edge. But every ``LogEntry`` whose ``entity_kind == "edge"``
    emitted by ``observe()`` has its payload replaced with bogus values: a random
    ``from_node_id``, ``to_node_id``, and an unrecognisable ``type``. A fold that
    reads payload would build those bogus endpoints into the index; a TRAP-1-clean
    fold reads ``get()`` and builds the true topology.

    The ``poison_from``, ``poison_to`` and ``poison_type`` attributes are set by
    the test so assertions can check the index does NOT contain them.
    """

    def __init__(self) -> None:
        """Initialise with fresh poison endpoint ids."""
        super().__init__()
        self.poison_from: UUID = uuid4()
        self.poison_to: UUID = uuid4()
        self.poison_type: str = "POISONED_TYPE_DO_NOT_USE"

    async def observe(
        self, since: Optional[Cursor] = None
    ) -> AsyncGenerator[LogEntry, None]:
        """Yield log entries with edge payloads replaced by poisoned values."""
        async for entry in super().observe(since=since):
            if entry.entity_kind == "edge":
                # Replace the payload with wrong endpoints and type.  The fold
                # must NOT read these; it must call get() for authority instead.
                poisoned_payload = dict(entry.payload)
                poisoned_payload["from_node_id"] = self.poison_from
                poisoned_payload["to_node_id"] = self.poison_to
                poisoned_payload["type"] = self.poison_type
                # LogEntry is frozen — reconstruct with the poisoned payload.
                entry = LogEntry(
                    cursor=entry.cursor,
                    tenant_id=entry.tenant_id,
                    entity_kind=entry.entity_kind,
                    entity_id=entry.entity_id,
                    version=entry.version,
                    transition=entry.transition,
                    payload=poisoned_payload,
                    recorded_at=entry.recorded_at,
                )
            yield entry


def test_fold_uses_get_authority_not_payload() -> None:
    """Index topology matches the true edge from get(), not the poisoned payload."""
    drive(_case())


async def _case() -> None:
    rt = _PayloadPoisonRuntime()
    thing = thing_registry()
    tenant = "t"
    a, b = uuid4(), uuid4()

    await rt.create(make_node(thing, tenant_id=tenant, node_id=a))
    await rt.create(make_node(thing, tenant_id=tenant, node_id=b))

    # The REAL edge: a -> b, KNOWS.
    await rt.link(tenant, a, b, EdgeTypes.KNOWS)

    # Confirm the poison values are distinct from the real endpoints.
    assert rt.poison_from not in {a, b}
    assert rt.poison_to not in {a, b}

    view = await GraphProjection().fold(rt, tenant, mode="full")
    idx = view._internal_index()

    # Should have exactly one live edge.
    assert len(idx.live_edges) == 1

    # The index must reflect the TRUE endpoints (from get()), not the payload poison.
    (rec,) = idx.neighbors(a, "out")
    assert rec.from_id == a,   f"from_id wrong: got {rec.from_id}, expected {a}"
    assert rec.to_id == b,     f"to_id wrong: got {rec.to_id}, expected {b}"
    assert rec.edge_type == EdgeTypes.KNOWS, (
        f"edge_type wrong: got {rec.edge_type!r}, expected {EdgeTypes.KNOWS!r}"
    )

    # The POISON values must NOT appear anywhere in the index.
    all_from = {r.from_id for nm in idx.adj["out"].values() for bkt in nm.values() for r in bkt}
    all_to   = {r.to_id   for nm in idx.adj["out"].values() for bkt in nm.values() for r in bkt}
    all_types = {r.edge_type for nm in idx.adj["out"].values() for bkt in nm.values() for r in bkt}

    assert rt.poison_from not in all_from, "poison from_id leaked into index"
    assert rt.poison_to   not in all_to,   "poison to_id leaked into index"
    assert rt.poison_type not in all_types, "poison edge_type leaked into index"

    # In-direction mirror also uses true ids.
    (rec_in,) = idx.neighbors(b, "in")
    assert rec_in.from_id == a
    assert rec_in.to_id   == b
