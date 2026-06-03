# CLAUDE.md — working in the Vella monorepo

Vella is a namespace monorepo under `packages/` (a `uv` workspace): `vella-core`
(pure-data foundation), `vella-runtime` (v0.1, the state substrate / "physics"),
`vella-graph` (read-only traversal projection), and `vella-reconciler` (the
reconciliation loop) ship as distinct distributions; `vella.agent` and
`vella.vectorstore` are planned siblings. These rules hold everywhere. Each
package has its own CLAUDE.md for its gate and invariants — read it before working
there. The code is the spec; per-package DESIGN.md holds the *why*.

## Principles
- The code is the source of truth; docs are generated from it or executed against
  it, never hand-maintained. A great design works — don't compromise it to satisfy
  a reviewer, and never weaken a locked decision to make a check pass.
- Preserve substantive "why" comments. Reformat freely; don't delete rationale.
- Respect the dependency direction: depend downward, never up. `core` depends on
  nothing; higher layers depend on it, never the reverse.

## Flows
- Work in milestones: build → critic/verify → repair, commit per green milestone.
- Authoring and review are separate passes — never self-approve in the same context.
- Local commits only. Never push, open PRs, or publish without an explicit ask —
  approval for one action doesn't extend to the next.
- Report honestly: if a gate fails, say so with output; never claim "all green."

## Style
- Any set-derived value that gets serialized must be `sorted()` — iteration order
  is nondeterministic and breaks reproducible artifacts.
- Match surrounding conventions; internal and external usage are identical — no
  privileged internal API.

## Testing & gates
Every package keeps its own gate green before every commit — all fail-closed,
identical to that package's CI job. Add a test with every behavior change.
Generated artifacts (schemas, catalogs, API docs) are never hand-edited —
regenerate them. The exact gate commands live in each package's CLAUDE.md.

## Docs
- Author/refresh a package's docs with `/document <package>` (docstrings,
  narrative, doctested tutorials, diagrams, examples). It runs the package's gate
  and lands a **PR** — never self-merged (writer ≠ reviewer).
- Publishing is automatic and deterministic: on merge to `main`,
  `.github/workflows/docs-deploy.yml` discovers every `packages/*/mkdocs.yml`,
  gates and builds each into `…/vella/<slug>/`, and regenerates the aggregate
  landing via `scripts/docs_packages.py`. A new package goes live the moment its
  `mkdocs.yml` reaches `main` — no workflow or landing edit. Never hand-edit the
  landing (`docs-root/index.html` is a layout shell; the cards are generated).

## Packages
- `vella-core` — `packages/vella-core/src/vella/core/CLAUDE.md`
- `vella-runtime` — `packages/vella-runtime/src/vella/runtime/CLAUDE.md`
- `vella-graph` — `packages/vella-graph/src/vella/graph/CLAUDE.md`
- `vella-reconciler` — `packages/vella-reconciler/src/vella/reconciler/CLAUDE.md`
