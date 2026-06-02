#!/usr/bin/env python
"""
Documentation-catalog generator + drift tripwire (sibling to ``export_schema.py``).

Emits Markdown catalogs derived from LIVE ``vella.core`` code into
``docs/catalogs/`` so the mkdocs site cannot silently drift from the package:

  1. ``edge_types.md``     -- the canonical edge vocabulary (sorted).
  2. ``public_api.md``     -- every ``vella.core.__all__`` symbol, its kind, and
                              its defining module (the symbol count is computed,
                              never hardcoded).
  3. ``compat_policies.md`` -- the ``CompatPolicy`` Literal values and their
                              direction semantics (value list sourced from the
                              type itself).
  4. ``type_registry.md``  -- every type registered in ``default_registry``.
                              Core registers no concrete types, so this is
                              header-only here -- the same forward-compat
                              situation as ``registry_type_schemas()`` in
                              ``export_schema.py``; it activates when integration
                              packages register their types.
  5. ``index.md``          -- a short intro linking the four pages.

DETERMINISM: ``known_edge_types()`` returns an unordered ``set`` and the
``EdgeTypes`` constants come from ``__dict__`` -- both are ``sorted()`` before
serialization, and every dict serialized to JSON uses ``sort_keys=True``. The
output is byte-identical across repeated runs and across Python 3.11/3.12/3.13.

Usage:
    python scripts/generate_catalogs.py            # (re)write the catalogs
    python scripts/generate_catalogs.py --check     # fail on undeclared drift
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any, get_args

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import vella.core as core  # noqa: E402
from vella.core import (  # noqa: E402
    CompatPolicy,
    EdgeTypes,
    default_registry,
    known_edge_types,
)

ROOT = Path(__file__).resolve().parent.parent
CATALOG_DIR = ROOT / "docs" / "catalogs"

# Submodules of vella.core that may *define* a public symbol. Used to resolve the
# defining module for aliases whose ``__module__`` points at ``typing``/``datetime``
# (e.g. ``CompatPolicy``, ``Reference``, ``UTCDatetime``) and plain constants whose
# ``__module__`` is ``None`` (e.g. ``DEFAULT_TENANT``).
_SUBMODULES = [
    "base", "edge", "embedding", "errors", "history", "integration",
    "node", "parse", "references", "state", "tooling", "_uuid7",
]


_MISSING = object()


def _defining_module(name: str, obj: Any) -> str:
    """Return the ``vella.core.*`` submodule that defines ``name``.

    Prefers the symbol's own ``__module__`` when it already points into
    ``vella.core``; otherwise locates the submodule whose namespace binds this
    exact object (handles aliases/constants re-exported from ``__init__``).
    """
    mod = getattr(obj, "__module__", None)
    if isinstance(mod, str) and mod.startswith("vella.core"):
        return mod
    for sub in _SUBMODULES:
        full = f"vella.core.{sub}"
        module = sys.modules.get(full)
        if module is not None and getattr(module, name, _MISSING) is obj:
            return full
    return "vella.core"


def _symbol_kind(obj: Any) -> str:
    """Classify a public symbol as class / function / alias / constant."""
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return "function"
    # Typing constructs (Literal, Callable, Annotated aliases) re-exported as types.
    if getattr(obj, "__module__", None) in {"typing", "datetime"} or get_args(obj):
        return "alias"
    return "constant"


def _edge_constant_names() -> list[str]:
    """Sorted ``EdgeTypes`` constant names (defensive sort over ``__dict__``)."""
    return sorted(
        k for k, v in vars(EdgeTypes).items()
        if not k.startswith("_") and isinstance(v, str)
    )


def render_edge_types() -> str:
    """Render the edge-vocabulary catalog from live ``vella.core`` code."""
    values = sorted(known_edge_types())
    const_names = _edge_constant_names()
    # Map each canonical value back to the constant that declares it (sorted, 1:1).
    value_to_const = {
        v: k for k, v in vars(EdgeTypes).items()
        if not k.startswith("_") and isinstance(v, str)
    }
    lines = [
        "# Edge vocabulary",
        "",
        "Generated from `vella.core.EdgeTypes` and `known_edge_types()`. Do not "
        "edit by hand; run `python scripts/generate_catalogs.py`.",
        "",
        f"There are {len(values)} canonical edge types. Custom edge type strings "
        "are allowed but trigger a did-you-mean warning to catch typos.",
        "",
        "| Constant | Value |",
        "| --- | --- |",
    ]
    lines += [f"| `EdgeTypes.{value_to_const[v]}` | `{v}` |" for v in values]
    lines.append("")
    return "\n".join(lines)


def render_public_api() -> str:
    """Render the public-API table from ``vella.core.__all__`` (count computed)."""
    names = sorted(core.__all__)
    lines = [
        "# Public API",
        "",
        "Generated from `vella.core.__all__`. Do not edit by hand; run "
        "`python scripts/generate_catalogs.py`.",
        "",
        f"`vella.core` exports {len(names)} public symbols.",
        "",
        "| Symbol | Kind | Module |",
        "| --- | --- | --- |",
    ]
    for name in names:
        obj = getattr(core, name)
        kind = _symbol_kind(obj)
        module = _defining_module(name, obj)
        lines.append(f"| `{name}` | {kind} | `{module}` |")
    lines.append("")
    return "\n".join(lines)


# Direction semantics for each CompatPolicy value. The *value list* is sourced
# from the CompatPolicy Literal itself (below); this maps each value to its
# human-readable semantics (Confluent semantics, mirroring tooling.py).
_BASE_SEMANTICS = {
    "FULL": "Both directions: a new reader reads old data and an old reader reads "
            "new data.",
    "BACKWARD": "A new reader reads old data.",
    "FORWARD": "An old reader reads new data.",
    "NONE": "No compatibility check (schema-on-read).",
}


def _policy_semantics(value: str) -> str:
    """Return the direction-semantics sentence for a ``CompatPolicy`` value."""
    if value.endswith("_TRANSITIVE"):
        base = value[: -len("_TRANSITIVE")]
        return (
            f"{_BASE_SEMANTICS[base]} Checked across the whole version history "
            "(not just the adjacent version)."
        )
    return _BASE_SEMANTICS[value]


def render_compat_policies() -> str:
    """Render the compat-policy matrix from the ``CompatPolicy`` Literal values."""
    values = list(get_args(CompatPolicy))
    lines = [
        "# Compatibility policies",
        "",
        "Generated from the `CompatPolicy` type in `vella.core`. Do not edit by "
        "hand; run `python scripts/generate_catalogs.py`.",
        "",
        "Per-type schema-evolution policy, enforced by the schema tripwire "
        "(`scripts/export_schema.py --check`). Confluent semantics.",
        "",
        "| Policy | Semantics |",
        "| --- | --- |",
    ]
    lines += [f"| `{v}` | {_policy_semantics(v)} |" for v in values]
    lines.append("")
    return "\n".join(lines)


def render_type_registry(registry: Any = None) -> str:
    """Render the registered-type catalog from a registry (default: live global).

    Args:
        registry: Registry to read; defaults to ``default_registry``. Tests pass
            a fresh ``Registry()`` so the render is hermetic and independent of
            any types other in-process tests may have registered on the global.
    """
    reg = registry if registry is not None else default_registry
    names = reg.names()  # already sorted; stable.
    lines = [
        "# Type registry",
        "",
        "Generated from `vella.core.default_registry`. Do not edit by hand; run "
        "`python scripts/generate_catalogs.py`.",
        "",
        "| Type | Compat | Version | Data class | State class |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in names:
        spec = reg.get(name)
        if spec is None:
            continue
        data_cls = spec.data_cls.__name__
        state_cls = spec.state_cls.__name__ if spec.state_cls is not None else "—"
        lines.append(
            f"| `{name}` | `{spec.compat}` | {spec.version} | "
            f"`{data_cls}` | `{state_cls}` |"
        )
    if not names:
        lines.append("")
        lines.append(
            "No types are registered in vella-core itself; integration packages "
            "populate this by importing their `@node_type` definitions."
        )
    lines.append("")
    return "\n".join(lines)


def render_index() -> str:
    """Render the catalog index page linking the four catalogs."""
    return "\n".join([
        "# Catalogs",
        "",
        "Reference catalogs generated directly from live `vella.core` code by "
        "`scripts/generate_catalogs.py` and gated in CI (`--check`). They cannot "
        "silently drift from the package.",
        "",
        "- [Edge vocabulary](edge_types.md) — canonical `EdgeTypes` constants.",
        "- [Public API](public_api.md) — every `vella.core.__all__` symbol.",
        "- [Compatibility policies](compat_policies.md) — the `CompatPolicy` "
        "matrix.",
        "- [Type registry](type_registry.md) — types registered in "
        "`default_registry`.",
        "",
    ])


def render_all(registry: Any = None) -> dict[str, str]:
    """Render every catalog file. Keys are filenames under ``docs/catalogs/``.

    Args:
        registry: Registry for the type-registry catalog; defaults to the global
            ``default_registry``. Tests pass a fresh ``Registry()`` for a
            hermetic render that ignores other tests' in-process registrations.
    """
    return {
        "index.md": render_index(),
        "edge_types.md": render_edge_types(),
        "public_api.md": render_public_api(),
        "compat_policies.md": render_compat_policies(),
        "type_registry.md": render_type_registry(registry),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="fail on undeclared drift between live code and committed catalogs",
    )
    args = parser.parse_args()

    catalogs = render_all()

    if not args.check:
        CATALOG_DIR.mkdir(parents=True, exist_ok=True)
        for filename, content in catalogs.items():
            (CATALOG_DIR / filename).write_text(content, encoding="utf-8")
        print(f"Wrote {len(catalogs)} catalogs to {CATALOG_DIR.relative_to(ROOT)}/")
        return 0

    drift: list[str] = []
    for filename, content in catalogs.items():
        path = CATALOG_DIR / filename
        if not path.exists():
            drift.append(f"  - {filename} (missing)")
        elif path.read_text(encoding="utf-8") != content:
            drift.append(f"  - {filename} (stale)")
    if drift:
        print("Catalog drift detected:", *drift, sep="\n", file=sys.stderr)
        print(
            "Re-run `python scripts/generate_catalogs.py` and commit "
            "docs/catalogs/.",
            file=sys.stderr,
        )
        return 1
    print(f"Catalog check passed ({len(catalogs)} catalogs current).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
