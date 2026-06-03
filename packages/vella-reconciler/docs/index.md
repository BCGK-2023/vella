# vella-reconciler

The reconciliation loop for the Vella graph SDK — a controller-runtime over the
`vella-runtime` contract. Where `vella-runtime` is *physics* (the append-only log,
the optimistic-concurrency store, and the write verbs that move state forward),
`vella-reconciler` is a *control loop* that observes the log, computes drift, and
drives convergent actions back through the runtime's write verbs.

```bash
pip install vella-reconciler
```

## What it is

`vella-reconciler` builds on `vella-runtime` and adds a **convergent control loop**
over it:

- **The runtime is physics; the reconciler is a control loop.** It observes the
  log and reconciles toward desired state through the runtime's verbs. It never
  persists world state.
- **Determinism is a property, not a hope.** Every ordering — resync ticks,
  backoff wakes, dead-letter serialization — is explicit and reproducible under an
  injected manual clock. Any set-derived serialized value is `sorted()`.
- **Idle is observable.** Convergence is a race-free predicate the driver
  evaluates, never an emergent side effect of a loop that happened not to fire.
- **Depends downward only.** The reconciler depends on `vella-runtime` and
  `vella-core`; neither depends on it.

The public surface grows milestone by milestone and is snapshotted by a surface
tripwire so accidental breaking changes fail the gate.

## Start here

- **[API reference](api.md)** — the full public surface, generated directly from
  the source docstrings and type hints.
