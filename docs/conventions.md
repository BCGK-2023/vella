# Documentation conventions

Docstrings follow [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings),
enforced by ruff's `D` (pydocstyle) rules with `convention = "google"` and by
interrogate at 100% public-surface coverage. Every `__all__` symbol must carry a
docstring, and the API reference is generated from those docstrings — so the code
is the single source of truth for the documentation.
