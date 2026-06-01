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


def test_closing_extra_fields_breaks_backward() -> None:
    # extra="allow" -> "forbid" (the FlexibleData crystallization off-ramp): a new
    # reader that forbids extras can no longer read old data that carried them.
    from pydantic import ConfigDict

    class V1(VellaModel):
        model_config = ConfigDict(frozen=True, extra="allow")
        a: str = ""

    class V2(VellaModel):
        model_config = ConfigDict(frozen=True, extra="forbid")
        a: str = ""

    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "BACKWARD")
    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "FULL")
    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "FORWARD") == []


def test_opening_extra_fields_breaks_forward() -> None:
    from pydantic import ConfigDict

    class V1(VellaModel):
        model_config = ConfigDict(frozen=True, extra="forbid")
        a: str = ""

    class V2(VellaModel):
        model_config = ConfigDict(frozen=True, extra="allow")
        a: str = ""

    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "FORWARD")
    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "BACKWARD") == []


def test_recursive_type_does_not_crash_and_self_compat_is_clean() -> None:
    # A self-referential data class (tree/linked-list) used to recurse forever in
    # the checker. It must terminate; comparing a schema to itself yields no
    # violations.
    from typing import List

    class Tree(VellaModel):
        val: str
        children: "List[Tree]" = []

    Tree.model_rebuild()
    schema = Tree.model_json_schema()
    assert check_compat(schema, schema, "FULL") == []


def test_recursive_type_breaking_change_is_still_detected() -> None:
    from typing import List

    class TreeV1(VellaModel):
        val: str
        children: "List[TreeV1]" = []

    class TreeV2(VellaModel):
        val: int  # breaking type change on a recursive type
        children: "List[TreeV2]" = []

    TreeV1.model_rebuild()
    TreeV2.model_rebuild()
    assert check_compat(TreeV1.model_json_schema(), TreeV2.model_json_schema(), "FULL")


def test_literal_narrowing_enum_to_const_breaks_backward() -> None:
    # pydantic emits `enum` for Literal[a, b] and `const` for Literal[a]; narrowing
    # to a single member is an enum->const transition that must still be flagged.
    from typing import Literal

    class V1(VellaModel):
        status: Literal["open", "closed"] = "open"

    class V2(VellaModel):
        status: Literal["open"] = "open"

    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "BACKWARD")
    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "FORWARD") == []


def test_literal_widening_const_to_enum_breaks_forward() -> None:
    from typing import Literal

    class V1(VellaModel):
        status: Literal["open"] = "open"

    class V2(VellaModel):
        status: Literal["open", "closed"] = "open"

    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "FORWARD")
    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "BACKWARD") == []


def test_tuple_to_list_array_representation_switch_is_detected() -> None:
    # pydantic emits prefixItems for a fixed Tuple and items for a List / variadic
    # Tuple; switching representation is a breaking change neither the items nor the
    # prefixItems branch alone would catch.
    from typing import List, Tuple

    class V1(VellaModel):
        x: Tuple[int, str]

    class V2(VellaModel):
        x: List[int]

    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "FULL")
    assert check_compat(V2.model_json_schema(), V1.model_json_schema(), "FULL")  # list->tuple too


def test_list_to_list_unchanged_is_not_a_false_positive() -> None:
    from typing import List

    class V1(VellaModel):
        x: List[int]

    class V2(VellaModel):
        x: List[int]

    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "FULL") == []


def test_union_widening_is_forward_breaking_only_not_backward() -> None:
    # Adding a union arm (int -> int|str) is BACKWARD-compatible (new reader still
    # reads old int data) but FORWARD-breaking (old int-only reader rejects new str
    # data). It must NOT be flagged under the default BACKWARD policy.
    class V1(VellaModel):
        x: int

    class V2(VellaModel):
        x: "int | str"

    V2.model_rebuild()
    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "BACKWARD") == []
    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "FORWARD")


def test_union_narrowing_is_backward_breaking_only_not_forward() -> None:
    class V1(VellaModel):
        x: "int | str"

    class V2(VellaModel):
        x: int

    V1.model_rebuild()
    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "BACKWARD")
    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "FORWARD") == []


def test_adding_field_to_closed_model_breaks_forward_only() -> None:
    # extra="forbid" emits additionalProperties:false; the old reader rejects any
    # unknown field, so adding even an OPTIONAL field is a FORWARD break — but
    # BACKWARD-compatible (the new reader still accepts old data without the field).
    class V1(VellaModel):
        a: str = ""

    class V2(VellaModel):
        a: str = ""
        b: str = ""

    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "FORWARD")
    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "BACKWARD") == []


def test_removing_field_from_closed_model_breaks_backward_only() -> None:
    class V1(VellaModel):
        a: str = ""
        b: str = ""

    class V2(VellaModel):
        a: str = ""

    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "BACKWARD")
    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "FORWARD") == []


def test_adding_optional_field_to_open_model_is_compatible() -> None:
    # Control: with an OPEN content model (extra="allow"), adding an optional field
    # is fully compatible in both directions — must not be a false positive.
    from pydantic import ConfigDict

    class V1(VellaModel):
        model_config = ConfigDict(frozen=True, extra="allow")
        a: str = ""

    class V2(VellaModel):
        model_config = ConfigDict(frozen=True, extra="allow")
        a: str = ""
        b: str = ""

    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "FULL") == []


def test_model_ref_union_widening_is_forward_breaking_only() -> None:
    # The R10 direction fix used scalar unions (int|str), where deref is a no-op.
    # Model-typed arms emit $ref and the old lone-ref side derefs to an object —
    # the arm fingerprint must still compare ref-vs-ref so widening A -> A|B is
    # correctly BACKWARD-compatible and FORWARD-breaking, not a blanket type_changed.
    from typing import Union

    class A(VellaModel):
        a: int = 0

    class B(VellaModel):
        b: int = 0

    class W1(VellaModel):
        x: A

    class W2(VellaModel):
        x: "Union[A, B]"

    W2.model_rebuild()
    assert check_compat(W1.model_json_schema(), W2.model_json_schema(), "BACKWARD") == []
    assert check_compat(W1.model_json_schema(), W2.model_json_schema(), "FORWARD")
    # And the reverse narrowing is BACKWARD-breaking only.
    assert check_compat(W2.model_json_schema(), W1.model_json_schema(), "BACKWARD")
    assert check_compat(W2.model_json_schema(), W1.model_json_schema(), "FORWARD") == []


def test_model_ref_union_disjoint_change_breaks_both() -> None:
    from typing import Union

    class A(VellaModel):
        a: int = 0

    class B(VellaModel):
        b: int = 0

    class C(VellaModel):
        c: int = 0

    class D1(VellaModel):
        x: "Union[A, B]"

    class D2(VellaModel):
        x: "Union[A, C]"

    D1.model_rebuild()
    D2.model_rebuild()
    # B replaced by C: a value removed (backward) and added (forward) -> breaks both.
    assert check_compat(D1.model_json_schema(), D2.model_json_schema(), "BACKWARD")
    assert check_compat(D1.model_json_schema(), D2.model_json_schema(), "FORWARD")


def test_union_arm_interior_breaking_change_with_stable_membership_is_detected() -> None:
    # R13: the union branch compared arm SETS and returned; a breaking change INSIDE
    # an arm (added required field) when membership is unchanged slipped through as
    # []. The checker must recurse into arms shared by both sides.
    from typing import Union

    class Bank(VellaModel):
        iban: str = ""

    class CardV1(VellaModel):
        num: str = ""

    class CardV2(VellaModel):
        num: str = ""
        cvv: str  # newly required field inside the arm

    class PayV1(VellaModel):
        method: Union[CardV1, Bank]

    class PayV2(VellaModel):
        method: Union[CardV2, Bank]

    # Align the arm $def name so membership fingerprints match (mirrors a real
    # in-place edit where the class name is stable across versions).
    def _rename_card(model: type[VellaModel], old_name: str) -> dict[str, Any]:
        s: dict[str, Any] = model.model_json_schema()
        s["$defs"]["Card"] = s["$defs"].pop(old_name)
        for arm in s["properties"]["method"]["anyOf"]:
            if arm.get("$ref", "").endswith(old_name):
                arm["$ref"] = "#/$defs/Card"
        return s

    s1 = _rename_card(PayV1, "CardV1")
    s2 = _rename_card(PayV2, "CardV2")
    violations = check_compat(s1, s2, "BACKWARD")
    assert violations and any("cvv" in v for v in violations)


def test_same_container_inline_union_arms_do_not_collide() -> None:
    # R14: Dict/List union arms are emitted inline (no $ref), so a (type, format)
    # arm key collapsed Union[Dict[str,int], Dict[str,str]] to one entry and an
    # interior element-type change was silently shadowed. The arm key now folds in
    # the element shape so the two arms stay distinct and the change is flagged.
    from typing import Dict, List, Union

    class V1(VellaModel):
        x: Union[Dict[str, int], Dict[str, str]]

    class V2(VellaModel):
        x: Union[Dict[str, bool], Dict[str, str]]

    assert check_compat(V1.model_json_schema(), V2.model_json_schema(), "FULL")
    assert check_compat(V1.model_json_schema(), V1.model_json_schema(), "FULL") == []  # no false positive

    class L1(VellaModel):
        y: Union[List[int], List[str]]

    class L2(VellaModel):
        y: Union[List[bool], List[str]]

    assert check_compat(L1.model_json_schema(), L2.model_json_schema(), "FULL")


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
