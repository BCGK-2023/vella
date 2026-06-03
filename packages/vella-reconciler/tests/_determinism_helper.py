"""Subprocess fixture for the reconciler determinism artifact (M6, C3).

Run as a script under a fixed ``PYTHONHASHSEED`` (set by the parent test). It
drives a FIXED scenario through a :class:`~vella.reconciler.Reconciler` that
dead-letters several entities across DISTINCT tenants/ids, then serializes the
resulting dead-letter record SET to canonical, byte-stable JSON and prints it to
stdout.

The NAMED determinism artifact (must-fix 4 / C3) is the **sorted dead-letter
record set** serialized byte-identically across hash seeds. This mirrors core's
discipline: any set-derived serialized value is ``sorted()``. The dead-letter
store holds records in a ``dict`` keyed by ``(tenant_id, entity_id)``; dict
iteration order for tuple-of-str/UUID keys is hash-seed sensitive, so the ONLY
thing that makes the serialization reproducible is the explicit ``sorted()`` here.

Reproducibility design — every source of nondeterminism EXCEPT iteration order is
pinned:

* **Fixed ids / tenants.** The scenario uses explicit ``UUID`` and tenant-id
  constants (no ``uuid4``), so the record contents never depend on a random id.
  The ids are chosen so their SORTED order differs from their dict-insertion /
  hash-iteration order across seeds — that is what makes removing ``sorted()``
  genuinely diverge (the non-vacuity mutation the verifier runs).
* **Volatile keys scrubbed.** ``DeadLetterRecord`` has no wall-clock fields today,
  but the helper scrubs the same ``_VOLATILE_KEYS`` the runtime helper uses,
  defensively and consistently, so a future volatile field cannot silently break
  determinism. (Mirrors ``packages/vella-runtime/tests/_determinism_helper.py``.)
* **Sorted set-derived output is the thing under test.** Records are ordered via
  ``sorted(records, key=lambda r: (r.tenant_id, str(r.entity_id)))`` (UUIDs sorted
  via ``str`` — UUID objects are not orderable against the tuple's string element),
  then ``json.dumps(..., sort_keys=True, separators=(",", ":"))``. If any
  serialized value derived its order from dict/set hash iteration, two seeds would
  diverge and the parent test would fail.

This is a script, not a test module — invoked via ``subprocess.run`` so each run
gets a fresh interpreter with the parent-supplied hash seed. ``PYTHONHASHSEED`` is
read once at interpreter start, so an in-process re-import would NOT reset it; a
subprocess is the only sound way to vary it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import UUID

from vella.core import Actuator, Node, Registry as CoreRegistry, VellaModel, node_type
from vella.runtime import Runtime

from vella.reconciler import (
    Context,
    InMemoryDeadLetterStore,
    ManualClock,
    ReconcileResult,
    Reconciler,
    Registry,
)
from vella.reconciler.deadletter import DeadLetterRecord

# --- pinned scenario constants (no uuid4, no random tenant) ------------------
# Entities across THREE distinct tenants with DISTINCT ids. The ids are deliberately
# NOT in a single tidy order: their (tenant_id, str(id)) SORTED order is meant to
# differ from whatever dict-insertion / hash-iteration order a given PYTHONHASHSEED
# produces. With ≥2 dead-lettered records keyed on heterogeneous (str, UUID) tuples,
# dict iteration order is genuinely hash-seed sensitive — so the explicit sorted()
# is load-bearing.
_SCENARIO: tuple[tuple[str, UUID], ...] = (
    ("t-gamma", UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")),
    ("t-alpha", UUID("99999999-9999-9999-9999-999999999999")),
    ("t-beta", UUID("11111111-1111-1111-1111-111111111111")),
    ("t-alpha", UUID("44444444-4444-4444-4444-444444444444")),
    ("t-gamma", UUID("22222222-2222-2222-2222-222222222222")),
    ("t-beta", UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")),
)

# Mirror runtime's helper: scrub per-run wall-clock keys before serializing so the
# only surviving variable is iteration order. DeadLetterRecord has none today; the
# scrub is defensive + consistent with the runtime artifact.
_VOLATILE_KEYS = frozenset({"recorded_at", "last_desired_at", "last_updated_at"})


def _scrub(obj: Any) -> Any:
    """Recursively drop per-run wall-clock keys from a JSON-mode structure."""
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items() if k not in _VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def _core_registry() -> type:
    """Isolated core registry with one ``device`` node kind (never the global)."""
    reg = CoreRegistry()

    @node_type("device", registry=reg)
    class DeviceData(VellaModel):
        power: str = "off"

    return DeviceData


def _drifting_node(
    DeviceData: type, *, tenant_id: str, node_id: UUID
) -> "Node[Any, Any]":
    """A drifting ``device`` node (current != desired) so the worker dispatches it."""
    return Node[DeviceData, Any](  # type: ignore[valid-type]
        id=node_id,
        type="device",
        name="dev",
        # created_by is a freshly-minted UUID, but it never reaches the dead-letter
        # record (which holds only tenant_id/entity_id/reason/attempts), so it does
        # not affect the artifact bytes.
        created_by=UUID("00000000-0000-0000-0000-000000000001"),
        data=DeviceData(power="off"),
        tenant_id=tenant_id,
        state=Actuator(
            current=DeviceData(power="off"),
            desired=DeviceData(power="on"),
        ),
    )


async def _drive() -> "set[DeadLetterRecord]":
    """Run the fixed give-up scenario; return the dead-letter records as a SET.

    The records are returned as a genuine ``set`` — NOT a list in dict-insertion
    order. This matters for the non-vacuity mutation: a Python ``dict`` preserves
    insertion order (seed-independent), but ``set`` iteration over elements
    containing strings (here the records' ``tenant_id``) is genuinely
    ``PYTHONHASHSEED``-sensitive. So when the verifier removes the artifact's
    ``sorted()``, the serialization falls back to hash-driven set-iteration order
    and the seeds DIVERGE — exactly the discipline the artifact proves
    (``DeadLetterRecord`` is a frozen pydantic model, hence hashable). ``sorted()``
    is the only thing that tames that order.
    """
    rt = Runtime()
    DeviceData = _core_registry()
    deadletter = InMemoryDeadLetterStore()
    registry = Registry()

    async def always_fail(_ctx: Context) -> ReconcileResult:
        raise RuntimeError("permanent failure")

    registry.register("device", always_fail)

    # Create every scenario entity (in the scenario's tuple order).
    for tenant_id, node_id in _SCENARIO:
        await rt.create(
            _drifting_node(DeviceData, tenant_id=tenant_id, node_id=node_id)
        )

    # max_attempts=1 -> each dispatch dead-letters immediately. A long resync
    # interval the ManualClock never advances keeps the run bounded to convergence
    # (every key dead-letters, the loop goes idle).
    rec = Reconciler(
        rt,
        registry,
        ManualClock(),
        deadletter_store=deadletter,
        resync_interval=10_000.0,
        max_attempts=1,
    )
    await asyncio.wait_for(rec.run(max_steps=len(_SCENARIO) * 4), timeout=5.0)

    # Collect the records into a genuine SET (not the store's insertion-ordered
    # dict): set iteration over string-bearing elements is hash-seed sensitive, so
    # this is the set-derived structure whose order the artifact's sorted() tames.
    # Accessing the dict directly is intentional — the artifact owns the ordering.
    return set(deadletter._records.values())  # noqa: SLF001 - artifact owns the sort


def build_artifact() -> str:
    """Serialize the dead-letter record set to canonical, byte-stable JSON.

    The records are ordered by ``sorted(records, key=lambda r: (r.tenant_id,
    str(r.entity_id)))`` — the load-bearing set-derived sort — each rendered via
    ``model_dump(mode="json")`` with volatile keys scrubbed, then dumped with
    ``sort_keys=True, separators=(",", ":")``.

    Returns:
        The canonical JSON string for the sorted dead-letter record set.
    """
    records = asyncio.run(_drive())
    # THE artifact: the SORTED set-derived serialization. Removing this sorted()
    # (the non-vacuity mutation) makes the bytes depend on dict/hash iteration order,
    # so the seeds diverge.
    ordered = sorted(records, key=lambda r: (r.tenant_id, str(r.entity_id)))
    rows = [_scrub(r.model_dump(mode="json")) for r in ordered]
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


def main() -> None:
    """Print the dead-letter artifact as canonical, byte-stable JSON to stdout."""
    print(build_artifact(), end="")


if __name__ == "__main__":
    main()
