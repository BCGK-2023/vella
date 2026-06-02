"""Pytest collection config.

``--doctest-modules`` (see ``[tool.pytest.ini_options]``) imports and scans every
module under ``src/vella/core`` for docstring examples. Underscore-prefixed modules
(``_uuid7.py``, ``_typevars.py``) are internal-only — not part of the public API and
not documented for external consumers — so they are excluded from doctest collection.
"""

collect_ignore_glob = ["src/vella/core/_*.py"]
