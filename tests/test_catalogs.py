"""Determinism + drift tests for the documentation-catalog generator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from vella.core import Registry

# Import the generator from scripts/ without packaging it (mirrors how
# test_schema_compat.py imports export_schema.py). Importing it has no side
# effects on default_registry — the script only reads the live registry.
_spec = importlib.util.spec_from_file_location(
    "generate_catalogs",
    Path(__file__).resolve().parent.parent / "scripts" / "generate_catalogs.py",
)
assert _spec and _spec.loader
generate_catalogs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate_catalogs)

# A fresh, empty registry. The generator runs in its own clean interpreter in
# CI, but the pytest process shares one global default_registry that OTHER tests
# populate (test_email/test_light); rendering against a clean Registry() makes
# these tests hermetic and match the committed (empty-registry) artifacts.
_CLEAN_REGISTRY = Registry()


def test_render_is_byte_stable_across_two_runs() -> None:
    first = generate_catalogs.render_all(_CLEAN_REGISTRY)
    second = generate_catalogs.render_all(_CLEAN_REGISTRY)
    assert first == second
    # And each render is independently byte-identical (not just object-equal).
    for name in first:
        assert first[name] == second[name], name


def test_render_all_emits_the_expected_catalogs() -> None:
    rendered = generate_catalogs.render_all(_CLEAN_REGISTRY)
    assert set(rendered) == {
        "index.md",
        "edge_types.md",
        "public_api.md",
        "compat_policies.md",
        "type_registry.md",
    }


def test_public_api_count_is_dynamic() -> None:
    import vella.core as core

    rendered = generate_catalogs.render_public_api()
    assert f"{len(core.__all__)} public symbols" in rendered


def test_check_passes_on_committed_tree() -> None:
    catalog_dir = generate_catalogs.CATALOG_DIR
    for filename, content in generate_catalogs.render_all(_CLEAN_REGISTRY).items():
        committed = (catalog_dir / filename).read_text(encoding="utf-8")
        assert committed == content, (
            f"{filename} is stale; run `python scripts/generate_catalogs.py`."
        )
