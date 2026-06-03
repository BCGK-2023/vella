# Vella Core — Design

This document is the design narrative and rationale for `vella.core`. The code is
the spec; this is the *why*. "V6" below refers to the design iteration the model
was distilled from, not the package version (which starts at `0.1.0`).

## What this package is

The **sdk-core** box of the Vella architecture: pure data + construction logic,
zero first-party dependencies, publishable standalone as `vella-core`. It ships
the type *definitions* (Node, Edge, references, state, tools, embedding,
integration bindings) and nothing that does I/O. Storage, history, and tool
execution live in higher layers (runtime); the reconciliation loop lives in
`vella.reconciler`; all depend on this package — never the reverse.

Distribution: `vella` is a **namespace** (PEP 420). `vella-core` ships
`vella.core`; future siblings (`vella.agent`, `vella.vectorstore`, `vella.graph`,
`vella.runtime`) ship as their own distributions under the same namespace, each
independently versioned. The monorepo is the umbrella *repo*; the namespace is
the umbrella *feel*; separate dists keep core minimal and slow-moving.

## Principles

- **Pure, frozen data.** No behavior on the types; the only logic is
  construction-time validation and copy-on-write. The same surface is used by
  internal and external integrations — there is no privileged internal API.
- **One envelope, polymorphic via generics.** `Node[TData, TState]`. System
  types declare strict models; agent types use `FlexibleData`.
- **Inert + self-validating.** Because core does no I/O and validates on
  construction, even a hand-built node is structurally sound; there is nothing in
  core to "circumvent" — the bypass surface lives one layer up and is governed by
  the dependency rule.

## Decisions (locked)

### Enforcement posture
Models are **frozen**; strict models **forbid extra fields**; `model_construct`
is **locked** (raises) in favor of an explicit `hydrate` trusted door. The four
ways to obtain a node: strict construction, `evolve`/`update_*` (copy-on-write,
re-validating), `parse_node` (tolerant hydration), `hydrate` (trusted fast path).

### Polymorphic round-trip (the Expression Problem)
Generics are erased at runtime, so `Node.model_validate` cannot know `data`'s
concrete type and would silently drop fields. `parse_node` uses the **registry**
(open tagged union) to resolve `type → (data_cls, state_cls)` and validate into
the right `Node[Data, State]`. Closed discriminated unions are rejected — the
type set is open (plugins, runtime discovery).

**Nested polymorphism rule (contract).** `SerializeAsAny` is applied to the
top-level polymorphic slots (`Node.data`, `Edge.data`, `Overlay.value`,
`Actuator.current`/`desired`) so the *actual* object is serialized, not the
erased declared type. This does **not** propagate into a data model's own fields:
if a data class declares a field typed as a base model but stores a subclass, the
subclass fields erode on dump. Discriminated unions (like `Reference`) are safe.
For any other base-typed polymorphic sub-field, the data author **must** annotate
it `SerializeAsAny[Base]` (see `test_nested_polymorphic_field_round_trips...`).
There is no global "serialize-as-any" switch in pydantic; this is a documented
authoring rule, exercised by tests.

### Schema evolution (writer/reader, à la Avro/Confluent)
Each type declares a **per-type compatibility policy** (`FULL`/`BACKWARD`/
`FORWARD`/`NONE`, + transitive). Construction is strict; hydration is **tolerant**
(Postel). Older data is migrated up a registered version chain; data that still
fails validation is **quarantined** into a `FlexibleData` node carrying a
`vella_repair` marker rather than throwing — the same "needs review" surface as
low-confidence reference resolutions. This makes parse-failure *data*, not an
exception, and is the substrate for self-healing.

`vella_repair` is a **reserved key** in a node's `data`: quarantine writes its
marker there (preserving any real clobbered value under `shadowed`, and a
non-mapping `data` under `shadowed_data`). The marker `reason` is built from
validation error *types and field-paths only*, never input values — though an
extra/forbidden field's *key name* (not its value) from the offending input may
appear in a path. `parse_*` never throws in non-strict mode: malformed
sub-surfaces are stripped and a last-resort synthetic node is produced if even
the minimal envelope is unparseable.

### Actuator semantics (level-triggered reconciliation, à la Kubernetes/Borg)
`Actuator.desired` is **declarative full state**, not a command or a patch.
`update_desired` is idempotent and level-triggered; a reconciliation loop
(`vella.reconciler`) converges `current` toward `desired` and is robust to missed
events and restarts. Telemetry vs. meaningful change follows stream-table duality: the
history store is the **log**, a node's state is the **table** (a materialized
view). High-frequency telemetry feeds the view via `observer` bindings and is
*not* stored as per-reading node versions; meaningful transitions are.

### Identity & idempotency
`id` is a time-ordered **UUIDv7** (storage locality + free temporal sort),
opaque. Idempotency is **binding-level** on `(tenant_id, plugin, external_id)` —
the runtime upserts on it — because polysource nodes carry many bindings and so
cannot derive one id from a single natural key.

### Multi-tenancy
**Partition isolation now**: `tenant_id` is non-null (`DEFAULT_TENANT` =
`__local__`), so every node belongs to exactly one tenant and there is never a
null population to migrate. Access is always `(tenant_id, id)`. Access *control*
(scopes, redaction, delegation) is deferred — see below.

### Registry
A `Registry` **class** with a module-level `default_registry` instance (not a
bare global): `@node_type` targets the default for ergonomics; tests construct
isolated registries; runtime registration is lock-guarded.

### Quality gates
`py.typed` shipped; CI runs `mypy --strict` **and** `pyright` on the min + latest
Python, plus type-assertion tests (positive `assert_type` and negative must-fail)
for the generics. A schema-export tripwire (`scripts/export_schema.py`) enforces
per-type compatibility and fails any breaking change lacking a version bump +
migration.

## Deferred (tracked, not dropped)

1. **Graph-level invariants** — cross-node constraints (e.g. "an Invoice has
   exactly one `OWNED_BY` edge to a Person/Company") via `@node_type(invariants=…)`,
   enforced in the integration API. Also covers edge cardinality/dedup.
2. **Multi-tenant access control beyond partitioning** — scope-limited tool
   exposure, per-field redaction, delegation, cross-tenant audit. Needs a real
   permission model first; premature otherwise.
3. **FlexibleData crystallization** — an off-ramp from loose dicts to strict
   schemas (`FlexibleData.crystallize(StrictSchema)`) plus instrumentation that
   surfaces stable extra-field patterns as candidate schemas.
