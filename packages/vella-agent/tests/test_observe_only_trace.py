"""The reasoning trace is ``observe_only`` and never bumps the run's version.

Locked decision #1: the token-level reasoning trace reaches the log + live
observers via ``emit_telemetry`` but never touches the state-table version. This
proves the no-version-bump invariant — the entity's version after the trace equals
its version before — and that the trace appears in ``history`` / ``observe`` as an
``observe_only`` entry. Fresh ``Runtime()`` + isolated agent registry; async cases
driven by ``asyncio.run`` (no pytest-asyncio).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from vella.agent import RunData
from vella.agent._writeback import create_run, emit_reasoning_trace
from vella.runtime import Runtime

_TENANT = "t-trace"


def _run(case: Callable[[Runtime], Any]) -> None:
    asyncio.run(asyncio.wait_for(case(Runtime()), timeout=5.0))


def test_trace_is_observe_only() -> None:
    _run(_case_trace_is_observe_only)


async def _case_trace_is_observe_only(rt: Runtime) -> None:
    run = await create_run(rt, RunData(goal="trace"), name="run", tenant_id=_TENANT)
    entry = await emit_reasoning_trace(
        rt, run.id, tenant_id=_TENANT, payload={"thinking": "step 1"}
    )
    assert entry.transition == "observe_only"


def test_trace_does_not_bump_run_version() -> None:
    _run(_case_trace_does_not_bump_run_version)


async def _case_trace_does_not_bump_run_version(rt: Runtime) -> None:
    run = await create_run(rt, RunData(goal="trace"), name="run", tenant_id=_TENANT)

    before = await rt.get(_TENANT, run.id)
    assert before is not None
    version_before = before.version

    # Multiple traces — none of them is a state transition.
    for i in range(3):
        await emit_reasoning_trace(
            rt, run.id, tenant_id=_TENANT, payload={"thinking": f"step {i}"}
        )

    after = await rt.get(_TENANT, run.id)
    assert after is not None
    # The no-version-bump invariant: the state-table version is UNCHANGED.
    assert after.version == version_before
    # And the stored run body is byte-identical (model_dump(mode="json"), not ==).
    assert after.model_dump(mode="json") == before.model_dump(mode="json")

    # The traces are present in history, all observe_only, all at the run's version.
    hist = await rt.history(_TENANT, run.id)
    traces = [e for e in hist if e.transition == "observe_only"]
    assert len(traces) == 3
    assert {e.version for e in traces} == {version_before}
