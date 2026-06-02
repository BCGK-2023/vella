# Authoring a node type

This how-to walks through the three things you do most with `vella-core`:

1. Define a strict, frozen `@node_type`.
2. Construct a `Node` from a data instance with `Node.from_data`.
3. Evolve that node copy-on-write — the original is never mutated.

<!-- Every code block below is an executable doctest (run in CI via
`pytest --doctest-glob=*.md`, with `docs/tutorials` on the testpath). The example
registers into a *local* `Registry()` so it stays hermetic and never mutates the
process-global `default_registry`. -->

## 1. Define a strict `@node_type`

A node type is just a frozen pydantic data class decorated with `@node_type`. The
decorator records the type in a registry and stamps the type name onto the class.
Keep the example hermetic: construct a local `Registry()` and pass `registry=` so
the global `default_registry` stays untouched.

```python
>>> from uuid import uuid4
>>> from pydantic import ConfigDict
>>> from vella.core import FlexibleData, Node, Registry, node_type
>>>
>>> registry = Registry()  # hermetic: never touches default_registry
>>> @node_type("task", compat="BACKWARD", registry=registry)
... class TaskData(FlexibleData):
...     model_config = ConfigDict(extra="forbid", frozen=True)  # strict + frozen
...     title: str
...     done: bool = False
>>>
>>> registry.names()
['task']

```

`@node_type` requires a frozen class. `FlexibleData` is frozen by default, but a
plain mutable model is rejected up front — so you never get a node type that
silently allows mutation:

```python
>>> from pydantic import BaseModel
>>> @node_type("oops", registry=registry)  # doctest: +ELLIPSIS
... class NotFrozen(BaseModel):  # not frozen
...     title: str
Traceback (most recent call last):
    ...
vella.core.errors.VellaError: @node_type('oops') requires a frozen data class ...

```

## 2. Construct a `Node` with `from_data`

`Node.from_data` reads the type name stamped on the data class, so you do not
repeat it. The envelope carries provenance (`created_by`), a time-ordered id, and
your strict `data` payload.

```python
>>> task = Node.from_data(
...     TaskData(title="Ship the docs"),
...     name="Ship the docs",
...     created_by=uuid4(),
... )
>>> task.type
'task'
>>> task.data.title
'Ship the docs'
>>> task.data.done
False
>>> task.id.version  # UUIDv7 — time-ordered for storage locality
7

```

## 3. Evolve copy-on-write

Nodes are frozen. To change one you call `evolve`, which re-validates and returns
a *new* node — the original is left intact. This is the copy-on-write contract
that makes change auditable.

```python
>>> done = task.evolve(data=TaskData(title="Ship the docs", done=True))
>>> done.data.done       # the new node reflects the change
True
>>> task.data.done       # the original is unchanged
False
>>> done is task
False

```

Because `evolve` re-validates, the new node still satisfies every constraint of
the type — you cannot evolve your way into an invalid node.

## See also

- The [API reference](../api.md) for the full public surface.
- The [architecture diagram](../architecture.md) for how `Node`, `Edge`, and
  state envelopes relate.
- `docs/examples/quickstart.py` for the same flow as a runnable script.
