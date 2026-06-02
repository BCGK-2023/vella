# CLAUDE.md — vella-core

Pure-data foundation (`Node`, `Edge`, references, state, tools, registry).
DESIGN.md is the *why*. Inherits the monorepo CLAUDE.md; this adds core specifics.

## Scope & deps
- `vella-core` only — never touch or assume `vella.runtime`.
- Zero runtime deps beyond pydantic + typing_extensions; all tooling is dev-only.

## Style
- Models are frozen and strict (`extra="forbid"`); `model_construct` is locked —
  use the four doors (construct / `evolve` / `parse_*` / `hydrate`).

## Gate (before every commit; mirrors CI)
`pytest` (incl. doctests) · `mypy` · `pyright` · `ruff check src/vella/core` ·
`interrogate -c pyproject.toml src/vella/core` · `mkdocs build --strict` ·
`export_schema.py --check` · `generate_catalogs.py --check`.
Doctest examples register into a local `Registry()`, never the global
`default_registry`. Run `/document` to author docs (gated, PR-only).
