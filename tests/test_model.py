"""Behavioral tests for the core model: identity, frozen posture, lifecycle."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from vella.core import (
    DEFAULT_TENANT,
    Actuator,
    Edge,
    EdgeTypes,
    Node,
    Overlay,
    ToolDeclaration,
    ToolOverride,
    UnknownEdgeTypeWarning,
    VellaError,
    VellaModel,
    node_type,
)


@node_type(
    "test_email",
    compat="BACKWARD",
    tools=[
        ToolDeclaration(
            name="reply",
            description="Reply to this email.",
            parameters={"type": "object", "properties": {"to": {"type": "string"}}},
        )
    ],
)
class EmailData(VellaModel):
    subject: str


class EmailFlags(VellaModel):
    is_read: bool = False


@node_type("test_light", state=EmailFlags)
class LightData(VellaModel):
    model: str


def make_email() -> Node[EmailData]:
    return Node[EmailData](
        type="test_email", name="hi", created_by=uuid4(), data=EmailData(subject="s")
    )


def test_id_is_uuid7_and_strictly_monotonic() -> None:
    nodes = [make_email() for _ in range(1000)]
    assert all(n.id.version == 7 for n in nodes)
    ids = [n.id for n in nodes]
    assert ids == sorted(ids), "uuid7 ids must be strictly time-ordered/monotonic"
    assert len(set(ids)) == len(ids), "ids must be unique"


def test_default_tenant_is_non_null() -> None:
    assert make_email().tenant_id == DEFAULT_TENANT


def test_frozen_blocks_mutation() -> None:
    n = make_email()
    with pytest.raises(ValidationError):
        n.name = "changed"  # type: ignore[misc]


def test_model_construct_is_locked_hydrate_works() -> None:
    with pytest.raises(VellaError):
        EmailData.model_construct(subject="x")
    assert EmailData.hydrate(subject="trusted").subject == "trusted"


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        Node[EmailData](
            type="test_email",
            name="n",
            created_by=uuid4(),
            data=EmailData(subject="s"),
            created_at=datetime(2020, 1, 1),  # naive
        )
    # tz-aware is accepted
    Node[EmailData](
        type="test_email",
        name="n",
        created_by=uuid4(),
        data=EmailData(subject="s"),
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )


def test_evolve_copies_revalidates_and_preserves_version() -> None:
    n = make_email()
    n2 = n.evolve(name="renamed")
    assert n2.name == "renamed"
    assert n.name == "hi"  # original untouched (frozen, copy-on-write)
    assert n2.version == n.version == 1  # evolve never bumps version


def test_model_copy_revalidates_and_cannot_bypass() -> None:
    n = make_email()
    # A valid update produces a new validated instance.
    n2 = n.model_copy(update={"name": "renamed"})
    assert n2.name == "renamed" and n.name == "hi"
    # An invalid update (unregistered tool override) is rejected, not silently applied.
    with pytest.raises(VellaError):
        n.model_copy(update={"tool_overrides": [ToolOverride(tool_name="ghost_tool")]})


def test_update_state_overlay_is_copy_on_write() -> None:
    n = Node[LightData, EmailFlags](
        type="test_light",
        name="lamp",
        created_by=uuid4(),
        data=LightData(model="A19"),
        state=Overlay(value=EmailFlags(is_read=False)),
    )
    n2 = n.update_state(is_read=True)
    assert isinstance(n2.state, Overlay)
    assert n2.state.value.is_read is True
    assert isinstance(n.state, Overlay) and n.state.value.is_read is False


def test_update_desired_is_idempotent() -> None:
    n = Node[LightData, EmailFlags](
        type="test_light",
        name="lamp",
        created_by=uuid4(),
        data=LightData(model="A19"),
        state=Actuator(current=EmailFlags(is_read=False)),
    )
    once = n.update_desired(is_read=True)
    twice = once.update_desired(is_read=True)
    assert isinstance(once.state, Actuator) and isinstance(twice.state, Actuator)
    assert once.state.desired is not None and once.state.desired.is_read is True
    assert twice.state.desired == once.state.desired  # idempotent


def test_from_data_rejects_unregistered_type() -> None:
    class Plain(VellaModel):
        x: int = 0

    with pytest.raises(VellaError):
        Node.from_data(Plain(), name="n", created_by=uuid4())


def test_node_type_requires_frozen() -> None:
    with pytest.raises(VellaError):

        @node_type("not_frozen")
        class NotFrozen(BaseModel):
            model_config = ConfigDict(frozen=False)
            x: int = 0


def test_tool_override_unknown_tool_rejected() -> None:
    with pytest.raises(VellaError):
        Node[EmailData](
            type="test_email",
            name="n",
            created_by=uuid4(),
            data=EmailData(subject="s"),
            tool_overrides=[ToolOverride(tool_name="does_not_exist")],
        )


def test_tool_override_unknown_param_rejected() -> None:
    with pytest.raises(VellaError):
        Node[EmailData](
            type="test_email",
            name="n",
            created_by=uuid4(),
            data=EmailData(subject="s"),
            tool_overrides=[ToolOverride(tool_name="reply", parameter_overrides={"nope": {}})],
        )


def test_tool_override_valid_accepted() -> None:
    n = Node[EmailData](
        type="test_email",
        name="n",
        created_by=uuid4(),
        data=EmailData(subject="s"),
        tool_overrides=[
            ToolOverride(tool_name="reply", parameter_overrides={"to": {"type": "string"}})
        ],
    )
    assert n.tool_overrides[0].tool_name == "reply"


def test_edge_canonical_type_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Edge(
            type=EdgeTypes.PART_OF,
            from_node_id=uuid4(),
            to_node_id=uuid4(),
            created_by=uuid4(),
        )


def test_edge_typo_warns_with_suggestion() -> None:
    with pytest.warns(UnknownEdgeTypeWarning, match="Did you mean"):
        Edge(type="prt_of", from_node_id=uuid4(), to_node_id=uuid4(), created_by=uuid4())
