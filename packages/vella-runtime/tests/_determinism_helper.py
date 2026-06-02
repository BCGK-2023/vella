"""Subprocess fixture for the determinism harness (M5, full verb set).

Run as a script under a fixed ``PYTHONHASHSEED`` (set by the parent test). It
imports the runtime, drives a FIXED verb sequence through a ``Runtime`` over a
fresh ``InMemoryStore``, serializes the resulting state-table + log to canonical
JSON, and prints it to stdout.

Reproducibility design — the ONLY nondeterminism under test is hash-seed-driven
iteration order, so every other source is pinned:

* **Fixed ids / timestamps / created_by.** Core's ``Node``/``Edge`` default
  ``id`` (uuid7), ``created_at`` and ``updated_at`` (utcnow) to freshly-minted
  values. The fixture passes EXPLICIT ``id=``, ``created_at=``, ``updated_at=``,
  ``created_by=`` so the bytes never depend on wall-clock or a random uuid.
* **Wall-clock fields core mints internally are scrubbed.** Two timestamps are
  minted by core/runtime at write time, not under the fixture's control:
  ``LogEntry.recorded_at`` (stamped by every verb via ``utcnow()``) and an
  ``Actuator``'s ``last_desired_at`` (stamped by core's ``update_desired``).
  Both are per-run wall-clock, NOT hash-seed effects — leaving them in would
  mask the real invariant under spurious timestamp noise. The fixture
  recursively scrubs these known wall-clock keys before serializing, so the
  only surviving variable is iteration order.
* **Set-derived ordering is the thing under test.** The state-table is keyed on
  a dict; we serialize it under ``sort_keys=True`` and explicitly sort the table
  keys ourselves — a runtime ``sorted()`` of its OWN derived structure, never of
  core model fields. If any serialized value derived its order from set/dict
  hash iteration, the two seeds would diverge and the parent test would fail.

This is a script, not a test module — invoked via ``subprocess.run`` so each run
gets a *fresh* interpreter with the parent-supplied hash seed. An in-process
re-import would NOT reset the hash seed (it is fixed once at interpreter start),
so a subprocess is the only sound way to vary it.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from vella.core import (
    Actuator,
    IntegrationBinding,
    Node,
    Registry,
    VellaModel,
    node_type,
)

from vella.runtime import Runtime
from vella.runtime._inmemory import InMemoryStore

# --- pinned fixture constants (no utcnow / no uuid minting) ------------------
_T = "t-fixture"
_DT = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_CREATED_BY = UUID("22222222-2222-2222-2222-222222222222")
_ID_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_ID_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_ID_C = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")


def _registry() -> type:
    reg = Registry()

    @node_type("doc", registry=reg)
    class DocData(VellaModel):
        title: str

    return DocData


def _node(DocData: type, *, node_id: UUID, title: str, with_state: bool = False,
          integrations: list[IntegrationBinding] | None = None) -> "Node[Any, Any]":
    return Node[DocData, Any](  # type: ignore[valid-type]
        id=node_id,
        type="doc",
        name=title,
        created_by=_CREATED_BY,
        data=DocData(title=title),
        tenant_id=_T,
        created_at=_DT,
        updated_at=_DT,
        integrations=integrations or [],
        state=Actuator(current=DocData(title=title)) if with_state else None,
    )


async def _drive() -> InMemoryStore:
    """Run a fixed, multi-verb sequence and return the populated store."""
    rt = Runtime()
    store = rt.store
    assert isinstance(store, InMemoryStore)
    DocData = _registry()

    # create A (plain), B (with actuator state), C (with integrations).
    await rt.create(_node(DocData, node_id=_ID_A, title="a"))
    await rt.create(_node(DocData, node_id=_ID_B, title="b", with_state=True))
    await rt.create(
        _node(
            DocData,
            node_id=_ID_C,
            title="c",
            integrations=[
                IntegrationBinding(
                    plugin="wordpress",
                    external_id="post-1",
                    contributes_to=["embedding", "data"],
                ),
                IntegrationBinding(plugin="ga4", external_id="prop-2"),
            ],
        )
    )
    # edit A, set_desired on B's actuator, telemetry on A, delete C.
    await rt.edit(_T, _ID_A, expected_version=1, name="a2")
    await rt.set_desired(_T, _ID_B, expected_version=1, title="b-desired")
    await rt.emit_telemetry(_T, _ID_A, {"cpu": 0.5})
    await rt.delete(_T, _ID_C)
    return store


# Wall-clock keys core/runtime mint at write time — per-run noise, not a
# hash-seed effect. Scrubbed recursively so the determinism check measures only
# iteration order.
_VOLATILE_KEYS = frozenset({"recorded_at", "last_desired_at", "last_updated_at"})


def _scrub(obj: Any) -> Any:
    """Recursively drop per-run wall-clock keys from a JSON-mode structure."""
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items() if k not in _VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def build_fixture() -> dict[str, object]:
    """Serialize the full state-table + log to a hash-seed-stable structure.

    The state-table is rendered as a list sorted by its own stringified key
    (runtime sorting its OWN derived structure); each entity via
    ``model_dump(mode="json")``. The log is rendered in offset order (already a
    total order, never set-derived) with ``recorded_at`` stripped.
    """
    store = asyncio.run(_drive())

    table_rows = []
    for key, row in store._index.state.items():  # noqa: SLF001
        if row.deleted:
            continue
        entity: Node[Any, Any] | None = (
            Node.hydrate(**row.entry.payload)
            if row.entry.entity_kind == "node"
            else None
        )
        assert entity is not None
        table_rows.append(
            {
                "key": f"{key[0]}|{key[1]}|{key[2]}",
                "entity": _scrub(entity.model_dump(mode="json")),
            }
        )
    table_rows.sort(key=lambda r: r["key"])  # runtime sorts its OWN derived order

    log_rows = [
        _scrub(entry.model_dump(mode="json")) for entry in store._index.log  # noqa: SLF001
    ]

    return {"state_table": table_rows, "log": log_rows}


def main() -> None:
    """Print the fixture as canonical, byte-stable JSON to stdout."""
    print(
        json.dumps(build_fixture(), sort_keys=True, separators=(",", ":")),
        end="",
    )


if __name__ == "__main__":
    main()
