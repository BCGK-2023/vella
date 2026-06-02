"""
Contract tests for the public value objects and the imperative registry API.

The parse/model suites cover the envelope and the tolerant round-trip; this file
locks the rest of the *exported* surface that integrations build against:
references (the unresolved→resolved transition and the discriminated union),
the convenience ref constructors, the imperative ``register_tools``/``tools_for``
path, ``known_edge_types``, and the small value objects (Embedding, Provenance,
HistoryEntry, IntegrationBinding) — including mutable-default-factory safety.
These are behavioral guards, not getters: each asserts a real branch or invariant.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from vella.core import (
    Embedding,
    HistoryEntry,
    IntegrationBinding,
    Provenance,
    Reference,
    Registry,
    ResolvedRef,
    ToolDeclaration,
    UnresolvedRef,
    VellaModel,
    channel_ref,
    document_ref,
    email_ref,
    known_edge_types,
    location_ref,
    node_type,
    person_ref,
    register_tools,
    tools_for,
)
from vella.core.edge import EdgeTypes


# --- References: the unresolved -> resolved transition -----------------------
def test_resolve_carries_metadata_and_flips_discriminator() -> None:
    ref = UnresolvedRef(identifier="alice@example.com", label="Alice", kind="person", edge_type="knows")
    node_id = uuid4()
    prov = Provenance(confidence=0.9, method="agent_inference")

    resolved = ref.resolve(node_id, provenance=prov)

    assert isinstance(resolved, ResolvedRef)
    assert resolved.resolution == "resolved"
    assert resolved.node_id == node_id
    # identifier/label/kind/edge_type carry across so the link stays traceable.
    assert resolved.identifier == "alice@example.com"
    assert resolved.label == "Alice"
    assert resolved.kind == "person"
    assert resolved.edge_type == "knows"
    assert resolved.provenance is prov


def test_resolve_without_provenance_is_allowed() -> None:
    resolved = UnresolvedRef(identifier="x").resolve(uuid4())
    assert resolved.provenance is None


def test_reference_union_discriminates_on_resolution() -> None:
    adapter: TypeAdapter[object] = TypeAdapter(Reference)
    unresolved = adapter.validate_python({"resolution": "unresolved", "identifier": "x"})
    resolved = adapter.validate_python(
        {"resolution": "resolved", "identifier": "x", "node_id": str(uuid4())}
    )
    assert isinstance(unresolved, UnresolvedRef)
    assert isinstance(resolved, ResolvedRef)


def test_reference_union_rejects_unknown_discriminator() -> None:
    adapter: TypeAdapter[object] = TypeAdapter(Reference)
    with pytest.raises(ValidationError):
        adapter.validate_python({"resolution": "mystery", "identifier": "x"})


@pytest.mark.parametrize(
    ("ctor", "expected_kind"),
    [
        (person_ref, "person"),
        (channel_ref, "channel"),
        (document_ref, "document"),
        (email_ref, "email"),
        (location_ref, "location"),
    ],
)
def test_convenience_constructors_set_kind_and_stay_unresolved(ctor: object, expected_kind: str) -> None:
    ref = ctor("ident", "a label", "knows")  # type: ignore[operator]
    assert isinstance(ref, UnresolvedRef)
    assert ref.resolution == "unresolved"
    assert ref.kind == expected_kind
    assert ref.identifier == "ident"
    assert ref.label == "a label"
    assert ref.edge_type == "knows"


def test_provenance_confidence_is_bounded() -> None:
    Provenance(confidence=0.0, method="system")
    Provenance(confidence=1.0, method="system")
    with pytest.raises(ValidationError):
        Provenance(confidence=1.5, method="system")
    with pytest.raises(ValidationError):
        Provenance(confidence=-0.1, method="system")


# --- Imperative registry API -------------------------------------------------
def test_register_tools_creates_placeholder_spec_when_type_absent() -> None:
    reg = Registry()
    register_tools("late_bound", [ToolDeclaration(name="press", description="d")], registry=reg)
    # The branch that mints a TypeSpec(data_cls=object) for a not-yet-defined type.
    assert [t.name for t in tools_for("late_bound", registry=reg)] == ["press"]
    spec = reg.get("late_bound")
    assert spec is not None and spec.data_cls is object


def test_register_tools_preserves_other_spec_fields() -> None:
    reg = Registry()

    @node_type("device", version=3, compat="FULL", registry=reg)
    class DeviceData(VellaModel):
        serial: str

    register_tools("device", [ToolDeclaration(name="reboot", description="d")], registry=reg)

    spec = reg.get("device")
    assert spec is not None
    assert [t.name for t in spec.tools] == ["reboot"]
    assert spec.version == 3            # not reset by the tools update
    assert spec.compat == "FULL"
    assert spec.data_cls is DeviceData  # data class preserved


def test_tools_for_unknown_type_is_empty() -> None:
    assert tools_for("nope", registry=Registry()) == []


def test_registry_clear_empties_specs() -> None:
    reg = Registry()
    register_tools("t", [ToolDeclaration(name="x", description="d")], registry=reg)
    assert reg.names() == ["t"]
    reg.clear()
    assert reg.names() == []


# --- Edge type constants -----------------------------------------------------
def test_known_edge_types_matches_declared_constants() -> None:
    known = known_edge_types()
    assert EdgeTypes.PART_OF in known
    assert EdgeTypes.OWNED_BY in known
    # Exactly the public string constants on EdgeTypes, no dunders or helpers.
    declared = {v for k, v in vars(EdgeTypes).items() if not k.startswith("_") and isinstance(v, str)}
    assert known == declared


# --- Small value objects -----------------------------------------------------
def test_embedding_round_trips_and_keeps_provenance_metadata() -> None:
    emb = Embedding(vector=[0.1, 0.2, 0.3], model="text-embedding-3-large", dimensions=3)
    back = Embedding.model_validate(emb.model_dump(mode="json"))
    assert back.model == "text-embedding-3-large"
    assert back.dimensions == 3
    assert back.normalized is True
    assert back.generated_from == "data"
    assert back.generated_at.tzinfo is not None  # tz-aware by construction


def test_history_entry_timestamp_is_tz_aware() -> None:
    entry = HistoryEntry(source="ingest")
    assert entry.timestamp.tzinfo is not None
    assert entry.change == {}


def test_integration_binding_defaults_and_mutable_default_isolation() -> None:
    a = IntegrationBinding(plugin="philips_hue", external_id="light-1")
    b = IntegrationBinding(plugin="google_calendar", external_id="cal-2")
    assert a.role == "primary"
    assert a.contributes_to == ["data", "state"]
    # The default-factory must hand each instance its own list, never a shared one.
    assert a.contributes_to is not b.contributes_to


def test_integration_binding_rejects_unknown_surface() -> None:
    with pytest.raises(ValidationError):
        IntegrationBinding(plugin="p", external_id="e", contributes_to=["data", "bogus"])
