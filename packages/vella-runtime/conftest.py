"""Pytest collection config.

``--doctest-modules`` (see ``[tool.pytest.ini_options]``) imports and scans every
module under ``src/vella/runtime`` for docstring examples. Underscore-prefixed
modules are internal-only — not part of the public API and not documented for
external consumers — so they are excluded from doctest collection.
"""

collect_ignore_glob = ["src/vella/runtime/_*.py"]
