# vella-core

The pure-data core model for the Vella graph SDK — `Node`, `Edge`, references,
state, tools, and the type registry. Zero first-party dependencies (pydantic v2
and typing_extensions only), publishable standalone so anyone can build
integrations without the rest of the stack.

```bash
pip install vella-core
```

## What it is

`vella-core` is **data, not behavior**. Nodes and edges are frozen pydantic
models that carry a payload, mutable state, provenance, and tool declarations —
nothing else. Every change goes through copy-on-write validation, so an invalid
node can never exist and every mutation is auditable.

- **Pure, frozen data.** `model_construct` is locked; you cannot bypass
  validation.
- **One envelope, polymorphic via generics.** `Node[TData, TState]` and
  `Edge[TData, TState]`. Strict models for system-managed types; `FlexibleData`
  for agent-managed ones.
- **Structural multi-tenancy.** Every node and edge belongs to exactly one
  `tenant_id` (default `__local__`); never null.
- **Time-ordered ids.** UUIDv7 for storage locality.

## The four doors to a node

There is exactly one way in for each situation — pick the door, not a back door:

| Door | Use it when | Validates? |
|------|-------------|-----------|
| **construct** (`Node.from_data`) | You have a typed, trusted data instance | Yes |
| **`evolve`** | You want to change a node copy-on-write | Yes (re-validates) |
| **`parse_node`** | You are hydrating untyped/external data | Yes, tolerant — quarantines bad data instead of throwing |
| **`hydrate`** | You are loading already-trusted storage rows | No (trusted fast path) |

`parse_node` is the tolerant reader: it resolves types via the registry, migrates
across schema versions, and quarantines unparseable data as a repairable
`FlexibleData` node rather than raising — the substrate for self-healing.

## Start here

- **[Authoring a node type](tutorials/authoring-a-node-type.md)** — define a
  `@node_type`, build a `Node` with `from_data`, and `evolve` it copy-on-write.
- **[Edges and state](tutorials/edges-and-state.md)** — connect nodes with a typed
  `Edge` and manage `Overlay` / `Actuator` state.
- **[Conventions](conventions.md)** — the rules every model follows.
- **[Architecture](architecture.md)** — how `Node`, `Edge`, and the state
  envelopes relate (with a diagram).
- **[API reference](api.md)** — the full public surface, generated directly from
  the source docstrings and type hints.
- **[Catalogs](catalogs/index.md)** — generated tables of edge types, the public
  API, compatibility policies, and the type registry.

!!! note "Examples register locally"

    Examples that touch the type registry construct their own `Registry()` and
    pass `registry=` so they stay hermetic — they never mutate the process-global
    `default_registry`. Copy that pattern in your own tests.
