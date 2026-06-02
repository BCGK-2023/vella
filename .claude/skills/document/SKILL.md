---
name: document
description: Author or update vella-core documentation (docstrings, narrative, doctested tutorials, diagrams-as-code, runnable examples), run the Layer-1 acceptance harness locally, and open a reviewable PR. PR-only — never pushes to main, never auto-merges, never self-approves.
---

# Skill: /document — author vella-core docs, PR-only

## Purpose
Author and maintain `vella-core` documentation as a writer pass that always lands
as a reviewable pull request. Every artifact this skill produces is verified by
the Layer-1 acceptance harness (doctests, lint, coverage, site build, drift
tripwires) before the PR is opened. A *separate* human reviewer approves and
merges — the skill enforces writer/reviewer separation by construction.

Scope: `vella-core` only (the pure-data package under `src/vella/core`). Do NOT
document or import `vella.runtime`.

## When to use
- Mode A — "here's everything I did, document it": you have a description or a
  diff of recent work and want the matching docstrings + narrative + (if
  relevant) a tutorial drafted/updated and proposed as a PR.
- Mode B — "make tutorials/guides/non-text artifacts": you want one or more of
  the three authored artifact types (doctested tutorials, diagrams-as-code,
  runnable examples).

## Hard rules (always)
- **PR-only.** All authored output lands via `gh pr create` on a feature branch.
- **NEVER push to `main`.** NEVER auto-merge. NEVER self-approve. A human
  reviews the PR (writer/reviewer separation — this skill is the writer pass).
- **Zero new runtime dependencies.** This skill lives in `.claude/`, outside the
  package. Docs tooling is dev-only. `pymdown-extensions` already ships with
  `mkdocs-material` — do NOT add it to `dependencies`.
- **Notebooks are deferred.** Author `.py` runnable examples only for now (no
  `.ipynb`, no `nbmake` dependency).

## CRITICAL — edge-example authoring rule (verbatim)
Edge examples in tutorials/doctests MUST use canonical `EdgeTypes.*` constants
(e.g. `EdgeTypes.OWNED_BY`) or explicitly suppress `UnknownEdgeTypeWarning`,
because pyproject sets `filterwarnings=['error::UserWarning']` and
`UnknownEdgeTypeWarning` inherits from `UserWarning` (via `VellaWarning`), so a
non-canonical edge type string raises a FATAL error inside doctests. Node
construction is safe; this applies to edge examples only.

## CRITICAL — hermetic-registry rule
Doctest examples (and runnable examples) that register a type MUST use a local
`Registry()` and pass `registry=` to the `@node_type` decorator (or to
`Registry().register(...)`). NEVER register into the process-global
`default_registry`. This keeps examples hermetic and keeps the fresh-interpreter
invariant green:

```
python -c "from vella.core import default_registry; assert default_registry.names() == []"
```

## Acceptance harness — the 6 Layer-1 gates (run ALL before opening a PR)
Run from the repo root with the venv active (`source .venv/bin/activate`). All
six must pass; if any fails, fix the authored artifact and re-run — do NOT open
the PR on a red gate.

1. `pytest -q`
   Runs doctests (`--doctest-glob=*.md --doctest-modules`, configured in
   `pyproject.toml` `[tool.pytest.ini_options]`; `testpaths` includes `tests`,
   `src/vella/core`, `README.md`, and `docs/tutorials`) plus the `runpy`-based
   `tests/test_examples.py` that executes every `docs/examples/*.py`.
2. `ruff check src/vella/core`
   Google-convention docstring presence/format (`pydocstyle D` rules).
3. `interrogate -c pyproject.toml src/vella/core`
   100% docstring coverage over the public surface.
4. `mkdocs build --strict`
   Renders the site (incl. Mermaid diagrams and tutorials); fails closed on any
   warning (broken refs, unresolved symbols, unrendered fences).
5. `python scripts/export_schema.py --check`
   Schema breaking-change drift tripwire.
6. `python scripts/generate_catalogs.py --check`
   Live-code catalog drift tripwire (edge vocabulary, public-API table, compat
   matrix, type-registry).

## Mode A — "here's everything I did, document it"
Given a description and/or a diff of recent work:
1. Identify the public symbols touched (under `src/vella/core`) and draft/update
   their Google-style docstrings, preserving substantive "why" prose verbatim.
2. Update narrative (`docs/*.md`) where the change alters a documented concept;
   add or update a doctested tutorial if the change introduces a user-facing
   workflow.
3. Apply the edge-example and hermetic-registry rules to every new code block.
4. Create a feature branch (e.g. `docs/<short-topic>`), commit the authored
   changes.
5. Run the full 6-gate acceptance harness above. Fix until green.
6. Open a PR with `gh pr create` (base = the working branch / `main` per repo
   convention; never push directly to `main`). Summarize the docs changes and
   the green harness output in the PR body. Do NOT merge or approve.

Example invocation:
```
/document Mode A: I added a `tenant_id` kwarg to Node.from_data and a new
location_ref helper — document both and open a PR.
```

## Mode B — "make tutorials/guides/non-text artifacts"
Authors any of the three CI-verified artifact types under `docs/`:

- **Doctested tutorial** — `docs/tutorials/*.md`. Every code block is a `>>>`
  doctest, picked up by `--doctest-glob=*.md` (the `docs/tutorials` testpath).
  Register into a LOCAL `Registry()`. Assert only stable properties — never raw
  ids/timestamps; use `# doctest: +ELLIPSIS` where a varying value must appear.
  If an edge is shown, use `EdgeTypes.*`.
- **Diagram-as-code** — a Mermaid fenced block in a `docs/**/*.md` page. Must
  render under `mkdocs build --strict`. Mermaid is enabled via Material's
  `pymdownx.superfences` custom fence (already wired in `mkdocs.yml`); no new
  dependency.
- **Runnable example** — `docs/examples/*.py`, a standalone script runnable via
  `python docs/examples/<name>.py` with exit 0, using a local `Registry()`.
  `tests/test_examples.py` executes each example via `runpy.run_path` as the CI
  gate. Do NOT add `docs/examples` to the `--doctest-modules` testpaths (avoids
  double-execution / non-package import issues).

After authoring, run the 6-gate harness and open a PR (same PR-only rules as
Mode A).

Example invocation:
```
/document Mode B: add a tutorial on authoring a strict node_type, an
architecture diagram of the Node/Edge/State relationships, and a quickstart
runnable example. Open a PR.
```

## Wiring reminders (when adding new artifacts)
- New tutorial under `docs/tutorials/`: ensure `docs/tutorials` is in
  `pyproject.toml` `testpaths`, and add the page to `mkdocs.yml` `nav`.
- New `docs/examples/*.py`: covered automatically by `tests/test_examples.py`
  (`runpy`). Leave `docs/examples` OUT of `testpaths`.
- New Mermaid page: add to `mkdocs.yml` `nav`; the superfences extension is
  already configured.
