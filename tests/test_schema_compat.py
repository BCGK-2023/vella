"""Unit tests for the per-type compatibility checker (Confluent semantics)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Optional

from vella.core import Registry, VellaModel, node_type

# Import the checker from scripts/ without packaging it.
_spec = importlib.util.spec_from_file_location(
    "export_schema", Path(__file__).resolve().parent.parent / "scripts" / "export_schema.py"
)
assert _spec and _spec.loader
export_schema = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_schema)
check_compat = export_schema.check_compat


def schema(props: dict[str, str], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {k: {"type": v} for k, v in props.items()},
        "required": required,
    }


OLD = schema({"a": "string", "b": "integer"}, required=["a"])


def test_add_optional_is_compatible_everywhere() -> None:
    new = schema({"a": "string", "b": "integer", "c": "string"}, required=["a"])
    assert check_compat(OLD, new, "FULL") == []


def test_add_required_breaks_backward_and_full() -> None:
    new = schema({"a": "string", "b": "integer", "c": "string"}, required=["a", "c"])
    assert check_compat(OLD, new, "BACKWARD")
    assert check_compat(OLD, new, "FULL")
    assert check_compat(OLD, new, "FORWARD") == []  # forward is fine


def test_remove_required_breaks_forward_and_full() -> None:
    new = schema({"b": "integer"}, required=[])
    assert check_compat(OLD, new, "FORWARD")
    assert check_compat(OLD, new, "FULL")
    assert check_compat(OLD, new, "BACKWARD") == []  # backward is fine


def test_type_change_breaks_all_non_none() -> None:
    new = schema({"a": "integer", "b": "integer"}, required=["a"])
    assert check_compat(OLD, new, "BACKWARD")
    assert check_compat(OLD, new, "FORWARD")
    assert check_compat(OLD, new, "FULL")
    assert check_compat(OLD, new, "NONE") == []  # NONE disables checking


def test_transitive_uses_same_rules() -> None:
    new = schema({"a": "string", "b": "integer", "c": "string"}, required=["a", "c"])
    assert check_compat(OLD, new, "BACKWARD_TRANSITIVE")


def test_existing_optional_becoming_required_breaks_backward() -> None:
    old = schema({"a": "string"}, required=[])
    new = schema({"a": "string"}, required=["a"])  # a: optional -> required
    assert check_compat(old, new, "BACKWARD")
    assert check_compat(old, new, "FULL")
    assert check_compat(old, new, "FORWARD") == []


def test_required_becoming_optional_breaks_forward() -> None:
    old = schema({"a": "string"}, required=["a"])
    new = schema({"a": "string"}, required=[])  # a: required -> optional
    assert check_compat(old, new, "FORWARD")
    assert check_compat(old, new, "BACKWARD") == []


def test_enum_narrowing_breaks_backward_widening_breaks_forward() -> None:
    old = {"type": "object", "properties": {"e": {"enum": ["a", "b"]}}, "required": ["e"]}
    narrowed = {"type": "object", "properties": {"e": {"enum": ["a"]}}, "required": ["e"]}
    widened = {"type": "object", "properties": {"e": {"enum": ["a", "b", "c"]}}, "required": ["e"]}
    assert check_compat(old, narrowed, "BACKWARD")  # removed value -> backward break
    assert check_compat(old, widened, "FORWARD")     # added value -> forward break


def test_format_change_breaks_all_non_none() -> None:
    old = {"type": "object", "properties": {"t": {"type": "string", "format": "date"}}, "required": ["t"]}
    new = {"type": "object", "properties": {"t": {"type": "string", "format": "date-time"}}, "required": ["t"]}
    assert check_compat(old, new, "BACKWARD")
    assert check_compat(old, new, "FORWARD")
    assert check_compat(old, new, "NONE") == []


def test_array_item_type_change_is_detected() -> None:
    old = {"type": "object", "properties": {"xs": {"type": "array", "items": {"type": "string"}}}, "required": ["xs"]}
    new = {"type": "object", "properties": {"xs": {"type": "array", "items": {"type": "integer"}}}, "required": ["xs"]}
    assert check_compat(old, new, "FULL")


def test_nested_object_added_required_via_ref_is_detected() -> None:
    # Mirrors pydantic output: nested model referenced via $ref/$defs.
    old = {
        "type": "object",
        "properties": {"inner": {"$ref": "#/$defs/Inner"}},
        "required": ["inner"],
        "$defs": {"Inner": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}},
    }
    new = {
        "type": "object",
        "properties": {"inner": {"$ref": "#/$defs/Inner"}},
        "required": ["inner"],
        "$defs": {
            "Inner": {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                "required": ["a", "b"],  # new required field on the nested model
            }
        },
    }
    violations = check_compat(old, new, "BACKWARD")
    assert violations and any("inner.b" in v for v in violations)


def test_optional_field_type_change_via_real_pydantic_anyof() -> None:
    # pydantic emits anyOf:[{...},{"type":"null"}] for Optional; a type change on
    # a nullable field is a real break the checker must see through.
    class V1(VellaModel):
        a: Optional[str] = None

    class V2(VellaModel):
        a: Optional[int] = None

    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "FULL")


def test_making_field_non_nullable_breaks_backward() -> None:
    class V1(VellaModel):
        a: Optional[str] = None

    class V2(VellaModel):
        a: str = ""

    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "BACKWARD")


def test_union_widening_with_format_distinguished_arms_is_detected() -> None:
    # bytes -> {"type":"string","format":"binary"}; without the format in the arm
    # fingerprint this collapses to the existing str arm and the change is invisible.
    class V1(VellaModel):
        x: int | str

    class V2(VellaModel):
        x: int | str | bytes

    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "FULL")


def test_per_type_gate_end_to_end_against_registry() -> None:
    # Exercises the same path main() uses: registry_type_schemas() + check_compat,
    # with real pydantic schemas and a declared per-type policy.
    reg_v1 = Registry()

    @node_type("account", compat="FULL", registry=reg_v1)
    class AccountV1(VellaModel):
        balance: int

    reg_v2 = Registry()

    @node_type("account", compat="FULL", registry=reg_v2)
    class AccountV2(VellaModel):
        balance: str  # breaking: int -> str

    before = export_schema.registry_type_schemas(reg_v1)
    after = export_schema.registry_type_schemas(reg_v2)
    assert "account" in before and "account" in after
    violations = check_compat(before["account"]["schema"], after["account"]["schema"], after["account"]["compat"])
    assert violations  # the gate flags the breaking change


def test_core_baseline_is_committed_and_current() -> None:
    """The committed schema/core.json must match the generated schema."""
    import json

    generated = export_schema.generate()
    baseline_path = export_schema.BASELINE
    assert baseline_path.exists(), "run scripts/export_schema.py to create the baseline"
    committed = json.loads(baseline_path.read_text())
    for name, gen_schema in generated.items():
        assert json.dumps(committed.get(name), sort_keys=True) == json.dumps(
            gen_schema, sort_keys=True
        ), f"schema/core.json is stale for {name}; re-run scripts/export_schema.py"
