# vella-reconciler

The reconciliation loop for the [Vella](https://github.com/BCGK-2023/vella) graph
SDK — a controller-runtime over the `vella-runtime` contract. Where `vella-runtime`
is *physics* (the append-only log, the optimistic-concurrency store, and the write
verbs that move state forward), `vella-reconciler` is a *control loop* that
observes the log, computes drift between desired and current state, and drives
convergent actions back through the runtime's write verbs.

```bash
pip install vella-reconciler
```

The reconciler depends on `vella-runtime` and `vella-core` and on nothing higher
in the stack; both lower layers are unaware of it. It owns no storage and no clock
of its own — both are injected, which makes convergence deterministically testable
under a manual clock. Its public surface is snapshotted by a surface tripwire so
accidental breaking changes fail the gate.
