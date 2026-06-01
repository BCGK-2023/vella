"""
Type-level tests for the generic envelope.

The ``assert_type`` calls are no-ops at runtime but verified by mypy/pyright:
``assert_type(expr, T)`` fails type-checking unless ``expr`` is *exactly* ``T``.
So these are the regression guard against the ``Node[TData, TState]`` generics
eroding to ``BaseModel``/``Any`` — the exact silent failure we designed
``SerializeAsAny`` + ``parse_node`` to prevent. If a refactor breaks the generic
binding, CI fails here rather than data silently vanishing at runtime.

(Negative "must-fail-to-typecheck" tests via constructor kwargs are not reliable
under the pydantic mypy plugin, which synthesizes a permissive ``**data: Any``
``__init__`` for parametrized generics; the exact positive assertions below cover
the regression instead.)
"""

from __future__ import annotations

from uuid import uuid4

from typing_extensions import assert_type

from vella.core import Node, Overlay, VellaModel


class FooData(VellaModel):
    x: int = 0


class FooState(VellaModel):
    y: int = 0


def test_data_resolves_to_concrete_type() -> None:
    n = Node[FooData](type="foo", name="n", created_by=uuid4(), data=FooData())
    assert_type(n.data, FooData)  # fails CI if data erodes to BaseModel/Any


def test_state_value_resolves_to_concrete_type() -> None:
    overlay = Overlay[FooState](value=FooState(y=3))
    assert_type(overlay.value, FooState)
    assert overlay.value.y == 3  # also a runtime check
