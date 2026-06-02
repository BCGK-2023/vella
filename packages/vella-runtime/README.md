# vella-runtime

The runtime physics/state substrate for the [Vella](https://github.com/BCGK-2023/vella)
graph SDK — the append-only log, the optimistic-concurrency store, and the
transition verbs that move graph state forward. Where `vella-core` is pure,
frozen data, `vella-runtime` is the substrate that records, serializes, and
concurrency-controls changes to it.

```bash
pip install vella-runtime
```

Runtime depends on `vella-core` and on nothing higher in the stack. Its public
surface is snapshotted by a surface tripwire so accidental breaking changes fail
the gate.
