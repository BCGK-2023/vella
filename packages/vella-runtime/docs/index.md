# vella-runtime

The runtime physics/state substrate for the Vella graph SDK — the append-only
log, the optimistic-concurrency store, and the transition verbs that move graph
state forward. Where `vella-core` is pure, frozen data, `vella-runtime` is the
substrate that records, serializes, and concurrency-controls changes to it.

```bash
pip install vella-runtime
```

## What it is

`vella-runtime` builds on `vella-core` and adds **behavior over the data**:

- **Deterministic serialization.** Any set-derived value that is serialized is
  `sorted()`, so reproducible artifacts never depend on hash-seed iteration order.
- **Optimistic concurrency.** Read-modify-write verbs check `expected_version`
  inside a transactional scope and raise `ConcurrencyConflict` on mismatch;
  callers retry.
- **Same front door for everyone.** Internal and external consumers use this exact
  surface — no privileged internal API.
- **Depends downward only.** Runtime depends on `vella-core`; core never depends on
  runtime.

The public surface grows milestone by milestone and is snapshotted by a surface
tripwire so accidental breaking changes fail the gate.

## Start here

- **[API reference](api.md)** — the full public surface, generated directly from
  the source docstrings and type hints.
