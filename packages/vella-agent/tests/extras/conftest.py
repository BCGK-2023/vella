"""Mark every test under ``tests/extras`` with the ``extras`` marker.

These are the out-of-gate adapter smoke tests: they exercise the
``[openrouter]``/``[mcp]`` adapters and require ``httpx``/``mcp``. The
deterministic core gate runs ``-m "not extras"`` (see ``pyproject.toml``), so the
core venv — which installs neither extra — never collects or imports them. Each
test additionally guards with ``pytest.importorskip`` so it SKIPS cleanly rather
than erroring when its extra is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-apply the ``extras`` marker to items collected UNDER THIS dir only.

    A conftest's ``pytest_collection_modifyitems`` is invoked once per session over
    the WHOLE item set (not just this directory's), so we scope by path: only items
    whose file lives under ``tests/extras`` get the marker — never the core suite.
    """
    for item in items:
        path = Path(str(getattr(item, "path", item.fspath))).resolve()
        if _HERE in path.parents:
            item.add_marker(pytest.mark.extras)
