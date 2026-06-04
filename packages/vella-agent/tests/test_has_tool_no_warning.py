"""Positive guard: ``Edge(type="has_tool")`` raises NO warning (regression gate).

This is NOT a suppression / ``pytest.warns`` test. It asserts a PROPERTY that
currently holds: core's ``_warn_on_unknown_edge_type`` validator stays silent on
``"has_tool"`` because that string is too dissimilar from any ``EdgeTypes`` constant
to clear ``difflib``'s 0.6 cutoff. Under ``filterwarnings=["error::UserWarning"]``
(the gate-wide setting), a warning would be promoted to an error — so if a future
core ``EdgeTypes``/cutoff change ever STARTED warning on ``"has_tool"``, this test
would fail, catching the regression at the seam the agent's discovery depends on.

The whole-module gate runs under ``error::UserWarning`` already; this test also
asserts locally via ``warnings.catch_warnings(record=True)`` so the property is
explicit and the failure message is legible.
"""

from __future__ import annotations

import warnings
from typing import Any
from uuid import uuid4

from vella.core import Edge, UnknownEdgeTypeWarning

from vella.agent import HAS_TOOL_EDGE


def test_has_tool_edge_raises_no_warning() -> None:
    assert HAS_TOOL_EDGE == "has_tool"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Edge(
            type=HAS_TOOL_EDGE,
            from_node_id=uuid4(),
            to_node_id=uuid4(),
            created_by=uuid4(),
        )
    # No UnknownEdgeTypeWarning (nor any UserWarning) for the custom has_tool string.
    assert not [w for w in caught if issubclass(w.category, UnknownEdgeTypeWarning)]
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


def test_has_tool_under_error_filter_does_not_raise() -> None:
    # Mirror the gate's filterwarnings=["error::UserWarning"] exactly: promoting a
    # UserWarning to an error must NOT trip — proving the silent property holds.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        edge: Edge[Any, Any] = Edge(
            type=HAS_TOOL_EDGE,
            from_node_id=uuid4(),
            to_node_id=uuid4(),
            created_by=uuid4(),
        )
    assert edge.type == "has_tool"
