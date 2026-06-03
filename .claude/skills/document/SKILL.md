---
name: document
description: Author or update documentation for any Vella package (docstrings, narrative, doctested tutorials, diagrams-as-code, runnable examples), run that package's Layer-1 acceptance gate locally, and open a reviewable PR. PR-only — never pushes to main, never auto-merges, never self-approves. Usage: /document <package> ...
---

# Skill: /document <package> — author Vella package docs, PR-only

## Purpose
Author and maintain a Vella package's documentation as a writer pass that always
lands as a reviewable pull request. Every artifact this skill produces is verified
by that package's Layer-1 acceptance gate (doctests, lint, coverage, strict site
build, drift tripwires) before the PR is opened. A *separate* human reviewer
approves and merges — the skill enforces writer/reviewer separation by
construction.

Publishing is **not** this skill's job. The live site is a deterministic
projection that auto-publishes on merge to `main` (`.github/workflows/docs-deploy.yml`
discovers every `packages/*/mkdocs.yml`). This skill only authors content and
opens the PR; merging is what publishes.

## Argument — which package
Invoke as `/document <package> [mode + instructions]`, where `<package>` is the
`vella.<package>` slug: `core`, `runtime`, `graph`, `reconciler`, … From it:

- distribution dir → `packages/vella-<package>/`
- source package   → `src/vella/<package>`

**Before anything else, read `packages/vella-<package>/src/vella/<package>/CLAUDE.md`.**
Its `## Gate` section is the authoritative gate for that package, and its
`## Invariants` / `## Style` sections define what your examples must respect. The
generic gate and gotchas below mirror those files; if they ever diverge, the
package's own CLAUDE.md wins. Only default to `core` when the user names no
package and the context is unambiguously core.

## When to use
- Mode A — "here's everything I did, document it": you have a description or a
  diff of recent work and want the matching docstrings + narrative + (if
  relevant) a tutorial drafted/updated and proposed as a PR.
- Mode B — "make tutorials/guides/non-text artifacts": you want one or more of
  the three authored artifact types (doctested tutorials, diagrams-as-code,
  runnable examples).

## Hard rules (always, every package)
- **PR-only.** All authored output lands via `gh pr create` on a feature branch.
  (`gh` runs through PowerShell here: `powershell.exe gh pr create ...`.)
- **NEVER push to `main`.** NEVER auto-merge. NEVER self-approve. A human
  reviews the PR (writer/reviewer separation — this skill is the writer pass).
- **Zero new runtime dependencies.** This skill lives in `.claude/`, outside the
  packages. Docs tooling is dev-only. `pymdown-extensions` already ships with
  `mkdocs-material` — do NOT add it to any package's `dependencies`.
- **Document only the target package's public surface, and respect the
  dependency direction.** Never document or assume an *upward* dependency (per
  each package's CLAUDE.md "Scope & deps"). You may *use* a downward dependency
  in examples (e.g. runtime examples build on core) when the package legitimately
  does.
- **Notebooks are deferred.** Author `.py` runnable examples only for now (no
  `.ipynb`, no `nbmake` dependency).

## Acceptance gate — run ALL before opening a PR
Run from `packages/vella-<package>/` with that package's venv active. This generic
gate mirrors every package's CI job and its CLAUDE.md `## Gate`; all must pass. If
any fails, fix the authored artifact and re-run — do NOT open the PR on a red gate.

1. `pytest -q`
   Doctests (`--doctest-glob=*.md --doctest-modules`, per the package's
   `pyproject.toml`) + the package's tests (incl. any `runpy`-based
   `tests/test_examples.py` that executes `docs/examples/*.py`).
2. `mypy`
3. `pyright`
4. `ruff check src/vella/<package>`
   Google-convention docstring presence/format (`pydocstyle D` rules).
5. `interrogate -c pyproject.toml src/vella/<package>`
   100% docstring coverage over the public surface.
6. `mkdocs build --strict`
   Renders the site (incl. Mermaid diagrams and tutorials); fails closed on any
   warning (broken refs, unresolved symbols, unrendered fences).
7. **Every** `scripts/*.py --check` (the PACKAGE's `packages/vella-<package>/scripts/`,
   since you run from there — not the repo-root `scripts/docs_packages.py`, which
   is the site generator exercised by docs-deploy.yml, not a `--check` tripwire).
   Each package's `scripts/` holds only drift tripwires (schema / API / surface /
   catalog projections of live code), and every one accepts `--check`. Run them
   all; none may drift. (core: `export_schema.py` + `generate_catalogs.py`;
   runtime: `export_runtime_surface.py`; graph: `export_graph_surface.py`;
   reconciler: `export_reconciler_surface.py`.)

## Per-package authoring gotchas (doctest/example pitfalls)
Constraints specific to writing *examples* for each package. Always re-read the
package's CLAUDE.md `## Invariants` too — new constraints there are binding even
if not yet mirrored here.

### core
- **Edge-example rule (verbatim).** Edge examples in tutorials/doctests MUST use
  canonical `EdgeTypes.*` constants (e.g. `EdgeTypes.OWNED_BY`) or explicitly
  suppress `UnknownEdgeTypeWarning`, because `pyproject` sets
  `filterwarnings=['error::UserWarning']` and `UnknownEdgeTypeWarning` inherits
  from `UserWarning` (via `VellaWarning`), so a non-canonical edge type string
  raises a FATAL error inside doctests. Node construction is safe; this applies
  to edge examples only.
- **Hermetic-registry rule.** Doctest/runnable examples that register a type MUST
  use a local `Registry()` and pass `registry=` to `@node_type` (or to
  `Registry().register(...)`). NEVER register into the process-global
  `default_registry`. This keeps the fresh-interpreter invariant green:
  `python -c "from vella.core import default_registry; assert default_registry.names() == []"`

### runtime
- **Compare entities via `model_dump(mode="json")`, NEVER Python `==`** — core's
  `_vella_registry` PrivateAttr breaks `==`.
- **Never sort core model fields** (`integrations`, `contributes_to` are
  order-semantic); sort only runtime's own set-derived serialized structures.
- **`observe()` is an async iterator and all verbs are `async`** — examples need
  an async context (`asyncio.run(...)`); use `# doctest: +ELLIPSIS` for any
  varying value. `Cursor` is opaque (`{token}`): pass it back, never compare it.

### graph / reconciler
- No package-specific doctest gotchas recorded yet. Derive constraints from
  `packages/vella-<package>/src/vella/<package>/CLAUDE.md` `## Invariants` — e.g.
  graph queries return `sorted()` ids and the determinism artifact is
  hash-seed-stable, so assert sorted/stable properties only. Add a subsection
  here the first time a doctest pitfall is discovered for these packages.

## Mode A — "here's everything I did, document it"
Given a description and/or a diff of recent work in `<package>`:
1. Identify the public symbols touched (under `src/vella/<package>`) and
   draft/update their Google-style docstrings, preserving substantive "why" prose
   verbatim.
2. Update narrative (`docs/*.md`) where the change alters a documented concept;
   add or update a doctested tutorial if the change introduces a user-facing
   workflow.
3. Apply the per-package authoring gotchas above to every new code block.
4. Create a feature branch (e.g. `docs/<package>-<short-topic>`), commit the
   authored changes.
5. Run the full acceptance gate above. Fix until green.
6. Open a PR with `gh pr create` (base = the working branch / `main` per repo
   convention; never push directly to `main`). Summarize the docs changes and the
   green gate output in the PR body. Do NOT merge or approve.

Example invocation:
```
/document runtime Mode A: I added an `expected_version` kwarg to `edit` and a new
`observe` cursor-resume path — document both and open a PR.
```

## Mode B — "make tutorials/guides/non-text artifacts"
Authors any of the three CI-verified artifact types under `<package>/docs/`:

- **Doctested tutorial** — `docs/tutorials/*.md`. Every code block is a `>>>`
  doctest, picked up by `--doctest-glob=*.md` (the `docs/tutorials` testpath).
  Assert only stable properties — never raw ids/timestamps; use
  `# doctest: +ELLIPSIS` where a varying value must appear. Apply the package's
  authoring gotchas (e.g. core's `EdgeTypes.*` + local `Registry()`).
- **Diagram-as-code** — a Mermaid fenced block in a `docs/**/*.md` page. Must
  render under `mkdocs build --strict`. Mermaid is enabled via Material's
  `pymdownx.superfences` custom fence (already wired in each `mkdocs.yml`); no new
  dependency.
- **Runnable example** — `docs/examples/*.py`, a standalone script runnable via
  `python docs/examples/<name>.py` with exit 0. If the package has a
  `tests/test_examples.py` (`runpy`) gate, it executes each example; do NOT add
  `docs/examples` to the `--doctest-modules` testpaths (avoids double-execution).

After authoring, run the acceptance gate and open a PR (same PR-only rules as
Mode A).

Example invocation:
```
/document core Mode B: add a tutorial on authoring a strict node_type, an
architecture diagram of the Node/Edge/State relationships, and a quickstart
runnable example. Open a PR.
```

## Wiring reminders (when adding new artifacts)
- New tutorial under `docs/tutorials/`: ensure `docs/tutorials` is in the
  package's `pyproject.toml` `testpaths`, and add the page to its `mkdocs.yml`
  `nav`.
- New `docs/examples/*.py`: if the package gates examples via
  `tests/test_examples.py` (`runpy`), it is covered automatically; leave
  `docs/examples` OUT of `testpaths`.
- New Mermaid page: add it to the package's `mkdocs.yml` `nav`; the superfences
  extension is already configured.
- A new sibling package automatically becomes a live subsite once its
  `mkdocs.yml` lands on `main` — no docs-deploy or landing edit is ever needed
  (the landing is generated by `scripts/docs_packages.py`).
