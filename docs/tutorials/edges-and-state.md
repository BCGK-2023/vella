# Edges and state

This how-to covers the two things the [node tutorial](authoring-a-node-type.md)
left out:

1. Connect two nodes with a typed, directed `Edge`.
2. Attach mutable **state** — `Overlay` for the common case, `Actuator` for state
   that can be *commanded* — and change it copy-on-write.

Edges are full peers of nodes: they are polymorphic over `data` *and* `state` and
share the exact same copy-on-write state helpers (`update_state` /
`update_desired`), so everything below applies to nodes too.

<!-- Every code block below is an executable doctest (run in CI via
`pytest --doctest-glob=*.md`, with `docs/tutorials` on the testpath). Nothing here
touches global state, so the example is hermetic. -->

## 1. Connect two nodes with an `Edge`

An edge is a typed, directed relationship: it records `from_node_id`,
`to_node_id`, and a `type`. Prefer the canonical `EdgeTypes` constants — a
non-canonical string is allowed but emits a did-you-mean warning to catch typos.

```python
>>> from uuid import uuid4
>>> from pydantic import BaseModel, ConfigDict
>>> from vella.core import Edge, EdgeTypes, Overlay
>>>
>>> source_id, target_id = uuid4(), uuid4()  # two nodes you already have
>>> link = Edge(
...     type=EdgeTypes.REFERENCES,
...     from_node_id=source_id,
...     to_node_id=target_id,
...     created_by=uuid4(),
... )
>>> link.type
'references'
>>> link.id.version  # UUIDv7 — time-ordered, like a node's id
7

```

## 2. Overlay state — the 80% case

`Overlay[T]` is plain mutable state: any property that changes more often than the
core data and has no command semantics (an email's `is_read`, a task's status, an
edge's confidence). The state value is itself a model, so it stays validated.

```python
>>> class LinkState(BaseModel):
...     model_config = ConfigDict(frozen=True)
...     weight: float
...     verified: bool = False
>>>
>>> link = Edge(
...     type=EdgeTypes.REFERENCES,
...     from_node_id=source_id,
...     to_node_id=target_id,
...     created_by=uuid4(),
...     state=Overlay(value=LinkState(weight=0.5)),
... )
>>> link.state.value.weight
0.5
>>> link.state.value.verified
False

```

Change it with `update_state`, which structural-merges your fields into the
overlay's value and returns a **new** edge — the original is untouched (the same
copy-on-write contract as `evolve`):

```python
>>> stronger = link.update_state(weight=0.9)
>>> stronger.state.value.weight     # the new edge reflects the change
0.9
>>> link.state.value.weight         # the original is unchanged
0.5
>>> stronger is link
False

```

## 3. Actuator state — current vs. desired

`Actuator[T]` is for state that can be *commanded*: it holds `current` (ground
truth from the world) and `desired` (the full target state Vella wants `current`
to become). `desired` is declarative, not a patch — a reconciliation loop in the
runtime converges `current` toward it.

```python
>>> from vella.core import Actuator
>>>
>>> commanded = Edge(
...     type=EdgeTypes.OWNED_BY,
...     from_node_id=source_id,
...     to_node_id=target_id,
...     created_by=uuid4(),
...     state=Actuator(current=LinkState(weight=0.0)),
... )
>>> commanded.state.current.weight
0.0
>>> commanded.state.desired is None   # no target set yet
True

```

Set a target with `update_desired`. It is idempotent and level-triggered (a
declarative target, not a command), and leaves `current` — the ground truth —
alone:

```python
>>> targeted = commanded.update_desired(weight=1.0)
>>> targeted.state.desired.weight     # the target we want
1.0
>>> targeted.state.current.weight     # ground truth is untouched
0.0

```

## 4. The helpers guard their state kind

`update_state` and `update_desired` are not interchangeable: each refuses the
wrong kind of state rather than silently doing the wrong thing.

```python
>>> commanded.update_state(weight=0.5)  # doctest: +ELLIPSIS
Traceback (most recent call last):
    ...
vella.core.errors.VellaError: update_state requires Overlay state; use update_desired for Actuator.

```

## See also

- [Authoring a node type](authoring-a-node-type.md) for `@node_type`,
  `Node.from_data`, and `evolve`.
- The [architecture diagram](../architecture.md) for how `Node`, `Edge`, and the
  state envelopes relate.
- The [API reference](../api.md) for `Edge`, `Overlay`, `Actuator`, and the full
  surface.
