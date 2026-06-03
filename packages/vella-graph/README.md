# vella-graph

A read-only graph/traversal projection for the
[Vella](https://github.com/BCGK-2023/vella) SDK. Where `vella-runtime` is *physics*
(the append-only log, the optimistic-concurrency store, and the write verbs that
move state forward), `vella-graph` is a *read-only projection* that folds the
runtime's `observe()` stream into an in-memory, type-partitioned, bidirectional
adjacency index and answers graph/traversal queries from it.

```bash
pip install vella-graph
```

The graph depends on `vella-runtime` and `vella-core` and on nothing higher in the
stack; both lower layers are unaware of it. The runtime exposes no list/scan verb,
so the index is forced: every query is answered from memory, never read-through.
Every query returns `sorted()` ids, so results are deterministic. Its public
surface is snapshotted by a surface tripwire so accidental breaking changes fail
the gate.

The public surface grows milestone by milestone. As of M2 it is the fold builder,
the frozen view, and the materialization mode:

```pycon
>>> import vella.graph
>>> vella.graph.__all__
['GraphProjection', 'GraphView', 'MaterializationMode']

```
