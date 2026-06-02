"""Guard: documentation examples are actually executable doctests.

``pytest --doctest-glob=*.md`` (configured in ``pyproject.toml``) executes the
README's example so it cannot silently rot. This guard fails loudly if the README
ever loses its ``>>>`` examples — otherwise "0 doctests collected" would pass green
and the drift-proofing would be a no-op.
"""

from __future__ import annotations

import doctest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_readme_contains_executable_doctests() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    examples = doctest.DocTestParser().get_examples(text)
    assert examples, "README.md must contain at least one >>> doctest example"
