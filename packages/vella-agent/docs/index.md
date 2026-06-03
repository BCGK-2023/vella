# vella-agent

A self-hosted cognition core for the Vella SDK — a data-configured agent
interpreter over the `vella-runtime` log and the `vella-graph` projection. Where
`vella-runtime` is *physics* (the append-only log, the optimistic-concurrency store,
and the write verbs that move state forward) and `vella-graph` is a *read-only
projection* answering traversal queries from memory, `vella-agent` is the
*cognition core*: it acts only through the runtime's verbs and perceives only
through the graph.

```bash
pip install vella-agent
```

## What it is

`vella-agent` builds on `vella-runtime` and `vella-graph` and adds a **self-hosted
agent interpreter** over them:

- **Self-hosting is the substrate, not a feature.** The agent acts only through the
  runtime's public verbs; an agent run, its steps, tool calls, messages, and policy
  are ordinary registered core node types. There is no new `Node` subclass and no
  privileged path.
- **Three Protocol seams.** `ModelProvider` / `ToolInvoker` / `ContextAssembler`
  each have a deterministic in-gate reference impl and optional out-of-gate real
  adapters (`[openrouter]` / `[mcp]`), so heavy I/O never enters the gate.
- **Determinism is a property, not a hope.** The interpreter is network-free under
  its reference impls; any set-derived serialized value is `sorted()`; the gated
  determinism artifact is byte-identical across hash seeds.
- **Depends downward only.** The agent depends on `vella-runtime`, `vella-graph`,
  and `vella-core` — never `vella-reconciler` (a sibling, not a dependency); all
  three lower layers are unaware of it.

The public surface grows milestone by milestone and is snapshotted by a surface
tripwire so accidental breaking changes fail the gate.

## Start here

- **[API reference](api.md)** — the full public surface, generated directly from
  the source docstrings and type hints.
