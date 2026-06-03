#!/usr/bin/env python3
"""Discover publishable packages and project the aggregate Pages landing.

Single source of truth for "which packages does the docs site publish, and what
does the landing say about them". A package is publishable iff it ships a
``packages/<dist>/mkdocs.yml``; its **slug** (the subsite path segment, e.g.
``core`` for ``vella-core``) is the lone namespace package under
``src/vella/<slug>``. Title and blurb come from the mkdocs ``site_name`` /
``site_description`` — never hand-maintained here.

Two modes, one discovery implementation so the deploy matrix and the landing can
never disagree about the package set:

* ``--matrix``   print ``matrix=<json-list>`` for ``$GITHUB_OUTPUT`` (each item
  has ``dir``/``slug``/``name``/``description``); consumed by docs-deploy.yml's
  build matrix.
* ``--landing PATH``   render the aggregate landing HTML to PATH (default
  ``site/index.html``) from ``docs-root/index.html`` as the layout shell.

Determinism: packages are emitted ``sorted()`` by slug, so both the matrix and
the landing are byte-identical across runs and hash seeds — the same rule every
other generated artifact in this repo follows.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "docs-root" / "index.html"
CARD_MARKER = "<!-- PACKAGE_CARDS -->"

# Top-level scalar keys we read out of each mkdocs.yml. We deliberately do NOT
# yaml.safe_load the file: mkdocs configs carry custom ``!!python/name:`` tags
# (the Mermaid superfences fence) that the safe loader rejects. We only need two
# top-level scalars, so a line match against column-0 keys is both sufficient and
# immune to the custom tags below them.
_META_RE = re.compile(r"^(site_name|site_description):\s*(.*)$")


def _scalar(raw: str) -> str:
    """Resolve a top-level YAML scalar value: unquote, or strip a # comment.

    Handles the two forms a mkdocs ``site_*`` key takes in practice — a bare
    scalar (where ``  # ...`` is a trailing comment to drop) and a quoted scalar
    (where a ``#`` inside the quotes is literal). Good enough for these keys; we
    are deliberately not a full YAML parser (see the note on _META_RE above).
    """
    raw = raw.strip()
    if raw[:1] in {'"', "'"}:
        quote = raw[0]
        end = raw.find(quote, 1)
        return raw[1:end] if end != -1 else raw[1:]
    return re.sub(r"\s+#.*$", "", raw).strip()


def _slug_for(dist_dir: Path) -> str:
    """Return the lone ``src/vella/<slug>`` namespace package name for a dist."""
    vella = dist_dir / "src" / "vella"
    candidates = sorted(
        p.name
        for p in vella.iterdir()
        if p.is_dir() and p.name != "__pycache__" and not p.name.startswith(".")
    )
    if len(candidates) != 1:
        raise SystemExit(
            f"{dist_dir.name}: expected exactly one package under src/vella/, "
            f"found {candidates!r}"
        )
    return candidates[0]


def discover() -> list[dict[str, str]]:
    """Discover every publishable package, sorted by slug for determinism."""
    pkgs: list[dict[str, str]] = []
    for mkdocs in REPO_ROOT.glob("packages/*/mkdocs.yml"):
        dist_dir = mkdocs.parent
        meta = {"site_name": dist_dir.name, "site_description": ""}
        for line in mkdocs.read_text(encoding="utf-8").splitlines():
            m = _META_RE.match(line)
            if m:
                meta[m.group(1)] = _scalar(m.group(2))
        pkgs.append(
            {
                "dir": dist_dir.name,
                "slug": _slug_for(dist_dir),
                "name": meta["site_name"],
                "description": meta["site_description"],
            }
        )
    return sorted(pkgs, key=lambda p: p["slug"])


def _render_landing(pkgs: list[dict[str, str]]) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    if CARD_MARKER not in template:
        raise SystemExit(f"{TEMPLATE} is missing the {CARD_MARKER} marker")
    cards = "\n".join(
        "  <div class=\"pkg\">\n"
        f"    <h2><a href=\"{p['slug']}/\">{html.escape(p['name'])}</a></h2>\n"
        f"    <p>{html.escape(p['description'])}</p>\n"
        "  </div>"
        for p in pkgs
    )
    return template.replace(CARD_MARKER, cards)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--matrix",
        action="store_true",
        help="print 'matrix=<json>' for $GITHUB_OUTPUT (deploy build matrix)",
    )
    group.add_argument(
        "--landing",
        nargs="?",
        const="site/index.html",
        metavar="PATH",
        help="render the aggregate landing HTML to PATH (default site/index.html)",
    )
    args = parser.parse_args(argv)

    pkgs = discover()

    if args.matrix:
        print("matrix=" + json.dumps(pkgs, separators=(",", ":"), sort_keys=True))
        return 0

    out = Path(args.landing)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render_landing(pkgs), encoding="utf-8")
    print(f"wrote {out} ({len(pkgs)} package(s): {', '.join(p['slug'] for p in pkgs)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
