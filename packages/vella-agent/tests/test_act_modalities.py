"""Three act-modalities: imperative via ToolInvoker; declarative via set_desired.

M3 demonstrates two of the three modalities (the cognition loop is M5):

* **imperative** — an action goes through :meth:`ToolInvoker.invoke`, which returns a
  single :class:`~vella.agent.ToolResult`.
* **declarative** — an action sets an actuator's desired state via
  ``runtime.set_desired`` and lets the reconciler converge it INDEPENDENTLY. The
  agent NEVER imports ``vella.reconciler`` — it is a sibling, off the agent's graph
  entirely; the declarative path is just a runtime verb.

The no-reconciler-import property is asserted AST-based (so it holds even though
``vella-reconciler`` is not installed on this branch), mirroring the package's
import-boundary discipline. ``asyncio.run`` only — no pytest-asyncio.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from typing import Any

from vella.core import Actuator, Node, Registry, ToolDeclaration, UnresolvedRef, VellaModel, node_type
from vella.runtime import Runtime

from vella.agent import (
    BuiltinBinding,
    InMemoryToolInvoker,
    ToolData,
    ToolResult,
    agent_registry,
)

_TENANT = "t-agent"
_ACTOR = UnresolvedRef(identifier="vella:test")
_SRC = Path(__file__).resolve().parent.parent / "src" / "vella" / "agent"


def test_imperative_modality_goes_through_invoker() -> None:
    async def _impl(args: dict[str, Any]) -> ToolResult:
        return ToolResult(content={"did": args.get("what")})

    invoker = InMemoryToolInvoker({"act": _impl})
    agent_registry()
    node = Node.from_data(
        ToolData(
            declaration=ToolDeclaration(name="act", description="d"),
            binding=BuiltinBinding(registry_key="act"),
        ),
        name="act",
        created_by=_ACTOR,
    )
    result = asyncio.run(invoker.invoke(node, {"what": "turn_on"}))
    assert result.content == {"did": "turn_on"}


def test_declarative_modality_uses_set_desired_no_reconciler() -> None:
    asyncio.run(asyncio.wait_for(_case_declarative(Runtime()), timeout=5.0))


async def _case_declarative(rt: Runtime) -> None:
    # A local, isolated actuator type (fresh Registry — never the global default).
    reg = Registry()

    @node_type("test.device", registry=reg)
    class DeviceState(VellaModel):
        power: str = "off"

    node = Node.from_data(
        DeviceState(power="off"),
        name="lamp",
        created_by=_ACTOR,
        tenant_id=_TENANT,
        state=Actuator(current=DeviceState(power="off")),
    )
    await rt.create(node)

    # The declarative path: set the DESIRED target via runtime.set_desired. The
    # reconciler (a separate package, never imported here) converges current->desired.
    entry = await rt.set_desired(_TENANT, node.id, expected_version=1, power="on")
    assert entry.transition == "set_desired"

    got: Any = await rt.get(_TENANT, node.id)
    assert got is not None
    assert got.state.desired.power == "on"
    assert got.state.current.power == "off"  # unchanged — convergence is the reconciler's job


def test_no_vella_reconciler_import_in_agent_source() -> None:
    # AST-based (never executes the import) so the forbid holds even though
    # vella-reconciler is not installed on this branch.
    offenders: list[str] = []
    for path in _SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "vella.reconciler" or alias.name.startswith("vella.reconciler."):
                        offenders.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "vella.reconciler" or mod.startswith("vella.reconciler."):
                    offenders.append(f"{path.name}: from {mod} import ...")
    assert offenders == [], f"forbidden vella.reconciler imports: {offenders}"
