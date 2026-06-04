"""Replay/resume (initial): folding from zero reconstructs the run projection.

A materialized run is replayable: folding the runtime's log into a ``GraphView``
(``vella.graph``) from zero reconstructs the run's structure — the run node plus its
``PART_OF`` step/message children, found by graph query over the durable log. This
is the M1 slice of the Replay/resume criterion; mid-loop resume lands in M5. If the
write-back drops a node from the materialization, the reconstructed child set no
longer matches what was written — the projection assertion goes red.

Fresh ``Runtime()`` + isolated agent registry; ``GraphProjection().fold`` reads the
runtime through ``observe``/``get`` only; async cases via ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from vella.agent import MessageData, RunData, StepData
from vella.agent._writeback import append_message, append_step, create_run
from vella.core import EdgeTypes
from vella.graph import GraphProjection
from vella.runtime import Runtime

_TENANT = "t-replay"


def _run(case: Callable[[Runtime], Any]) -> None:
    asyncio.run(asyncio.wait_for(case(Runtime()), timeout=5.0))


def test_fold_reconstructs_run_projection() -> None:
    _run(_case_fold_reconstructs_run_projection)


async def _case_fold_reconstructs_run_projection(rt: Runtime) -> None:
    run = await create_run(rt, RunData(goal="replay"), name="run", tenant_id=_TENANT)
    step = await append_step(
        rt, run.id, StepData(turn_index=0), name="s0", tenant_id=_TENANT
    )
    msg = await append_message(
        rt, run.id, MessageData(role="user", text="hi"), name="m0", tenant_id=_TENANT
    )

    # Fold the durable log from zero into a frozen graph view.
    view = await GraphProjection().fold(rt, _TENANT, mode="full")

    # The run node is resident and typed.
    folded_run = await view.get(run.id)
    assert folded_run is not None and folded_run.type == "agent.run"

    # The run's children are exactly {step, msg}, reached by the PART_OF edges
    # (which point child -> run, so the run's incident PART_OF neighbours are its
    # children). Deterministic sorted ids.
    neighbours = await view.neighbors(
        run.id, edge_type=EdgeTypes.PART_OF, direction="in"
    )
    child_ids = sorted(str(n.node_id) for n in neighbours)
    assert child_ids == sorted({str(step.id), str(msg.id)})

    # The reconstructed run body matches what was written (model_dump, not ==).
    assert folded_run.model_dump(mode="json") == run.model_dump(mode="json")
