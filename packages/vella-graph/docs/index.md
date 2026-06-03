# vella-graph

A read-only graph/traversal projection for the Vella SDK — a traversal view over
the `vella-runtime` log. Where `vella-runtime` is *physics* (the append-only log,
the optimistic-concurrency store, and the write verbs that move state forward),
`vella-graph` is a *read-only projection* that folds the runtime's `observe()`
stream into an in-memory adjacency index and answers graph queries from it.

```bash
pip install vella-graph
```

## What it is

`vella-graph` builds on `vella-runtime` and adds a **read-only graph projection**
over it:

- **The index is forced; read-through is impossible.** The runtime exposes only
  `get`/`history`/`observe` — no list/scan. Every query is answered from an
  in-memory, type-partitioned, bidirectional adjacency index folded from
  `observe()`.
- **Maximal expression, fast via structure.** Working-set memory buys latency:
  anchoring, type-pruning, and baked weights make expressive queries fast without
  amputating capability.
- **Determinism is a property, not a hope.** Every query returns `sorted()` ids;
  the gated determinism artifact is topology-derived and byte-identical across
  hash seeds. Any set-derived serialized value is `sorted()`.
- **Depends downward only.** The graph depends on `vella-runtime` and
  `vella-core`; neither depends on it.

The public surface grows milestone by milestone and is snapshotted by a surface
tripwire so accidental breaking changes fail the gate.

## Start here

- **[API reference](api.md)** — the full public surface, generated directly from
  the source docstrings and type hints.
