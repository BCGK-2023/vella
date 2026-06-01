"""
Property-based hardening of the defining invariant: ``parse_node`` / ``parse_edge``
NEVER throw in non-strict mode, for ANY mapping input.

Rounds 4-9 each found a *new* single input that escaped the quarantine guards
(non-string keys, stringify collisions, migration failures, ...). That is
whack-a-mole. This file locks the whole class instead: hypothesis throws
arbitrarily-shaped, deeply-nested, hostile mappings — including non-string keys
(int/None/float/bool/tuple, as binary codecs produce) — at the parsers and asserts
they always return a typed envelope and never raise, and that the result
round-trips through ``model_dump(mode="json")`` and back without raising either.

Deterministic (``derandomize=True``) so CI never flakes; a new escape hatch makes
this fail reproducibly rather than in production.
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from vella.core import Edge, FlexibleData, Node, Registry, VellaModel, node_type, parse_edge, parse_node

# JSON-expressible scalar values (finite floats only — NaN/inf aren't our concern
# here and only add serialization noise).
_scalars = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text()
)

# Keys a real wire/codec payload can carry that are NOT strings — the adversarial
# surface that broke rounds 7-8. (Python folds True==1 itself, so no extra work.)
_weird_keys = (
    st.text()
    | st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.tuples(st.integers(), st.text())
)

_values = st.recursive(
    _scalars,
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(_weird_keys, children, max_size=4),
    max_leaves=15,
)

_arbitrary_mapping = st.dictionaries(_weird_keys, _values, max_size=6)


@settings(max_examples=400, deadline=None, derandomize=True)
@given(raw=_arbitrary_mapping)
def test_parse_node_never_throws_on_arbitrary_mapping(raw: dict[Any, Any]) -> None:
    reg = Registry()
    node = parse_node(raw, registry=reg)  # invariant: must not raise
    assert isinstance(node, Node)
    # The recovered node must itself survive a JSON round-trip without raising.
    parse_node(node.model_dump(mode="json"), registry=reg)


@settings(max_examples=400, deadline=None, derandomize=True)
@given(raw=_arbitrary_mapping)
def test_parse_edge_never_throws_on_arbitrary_mapping(raw: dict[Any, Any]) -> None:
    reg = Registry()
    edge = parse_edge(raw, registry=reg)  # invariant: must not raise
    assert isinstance(edge, Edge)
    parse_edge(edge.model_dump(mode="json"), registry=reg)


@settings(max_examples=400, deadline=None, derandomize=True)
@given(data=_values, sv=st.integers(min_value=0, max_value=5))
def test_parse_node_typed_path_never_throws(data: Any, sv: int) -> None:
    # Exercise the registered-type path (strict model + migration routing) with a
    # hostile body and schema_version, where rounds 7/9 found throwing escapes.
    reg = Registry()

    @node_type("doc", version=3, migrations={1: lambda d: d, 2: lambda d: d}, registry=reg)
    class DocData(VellaModel):
        title: str

    raw = {
        "type": "doc",
        "schema_version": sv,
        "name": "n",
        "created_by": "00000000-0000-0000-0000-000000000000",
        "data": data,
    }
    node = parse_node(raw, registry=reg)  # invariant: must not raise
    assert isinstance(node, Node)
    # A body that isn't a valid DocData must degrade to a repairable FlexibleData node.
    if not (isinstance(data, dict) and isinstance(data.get("title"), str)):
        assert isinstance(node.data, FlexibleData)
