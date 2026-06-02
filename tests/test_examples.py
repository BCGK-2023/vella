"""Execute every runnable example under ``docs/examples`` via runpy.

This is the CI gate for ``docs/examples/*.py``: each script must run to
completion with no exception (the scripts ``assert`` their own outcomes). Run as
``__main__`` so the scripts' ``if __name__ == "__main__":`` blocks execute.

Examples are intentionally NOT on the pytest ``--doctest-modules`` testpath
(``docs/examples`` is not a package and would double-execute); this runpy test is
their sole CI gate.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "docs" / "examples"
_EXAMPLE_SCRIPTS = sorted(_EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize("script", _EXAMPLE_SCRIPTS, ids=lambda p: p.name)
def test_example_runs(script: Path) -> None:
    runpy.run_path(str(script), run_name="__main__")


def test_examples_present() -> None:
    # Guard against the glob silently collecting zero scripts.
    assert _EXAMPLE_SCRIPTS, "expected at least one docs/examples/*.py script"
