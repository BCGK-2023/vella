# vella-agent

A self-hosted cognition core for the
[Vella](https://github.com/BCGK-2023/vella) SDK. Where `vella-runtime` is *physics*
(the append-only log, the optimistic-concurrency store, and the write verbs that
move state forward) and `vella-graph` is a *read-only projection* that answers
traversal queries from memory, `vella-agent` is the *cognition core*: a
data-configured agent interpreter that acts only through the runtime's verbs and
perceives only through the graph.

```bash
pip install vella-agent
```

The agent depends on `vella-runtime`, `vella-graph`, and `vella-core` and on
nothing higher in the stack — and **never** on `vella-reconciler`, which is a
sibling rather than a dependency. All three lower layers are unaware of it. The
agent takes no privileged path: an agent run, its steps, tool calls, messages, and
policy are ordinary registered core node types, and the agent acts solely through
the runtime's public write verbs. Its public surface is snapshotted by a surface
tripwire so accidental breaking changes fail the gate.

The public surface grows milestone by milestone. As of M0 the package is the
scaffold: an empty public surface baselined by the tripwire, with the node
type-specs, canonical-turn models, the three Protocol seams, and the FSM
interpreter landing in later milestones:

```pycon
>>> import vella.agent
>>> vella.agent.__all__
[]

```
