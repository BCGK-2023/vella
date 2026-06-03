"""Subprocess fixture for the agent determinism artifact (M0 baseline).

Run as a script under a fixed ``PYTHONHASHSEED`` (set by the parent test). It builds
the agent's canonical public-surface snapshot — the same five-section structure the
surface tripwire (``scripts/export_agent_surface.py``) freezes — directly from the
LIVE ``vella.agent`` package, and serializes it to canonical, byte-stable JSON,
printing it to stdout.

The NAMED determinism artifact for M0 is that **canonical surface snapshot**
serialized byte-identically across hash seeds. It is real-but-trivial at M0: the
package's ``__all__`` is empty, so the snapshot is the empty-surface structure. The
load-bearing point is the *mechanism* — the artifact is built from the live package
and dumped with ``sort_keys=True``, and the per-section maps are keyed off
``sorted(vella.agent.__all__)`` (a genuinely set-derived ordering: ``__all__`` is a
list but membership classification iterates it, and as it grows the error-MRO and
``Literal`` value sets it carries are real ``set``/``sorted`` derivations). Removing
a ``sorted()`` here would let hash order leak into the bytes, so this subprocess
test starts guarding determinism from M0 and tightens for free as the surface grows.

The artifact intentionally mirrors the tripwire's ``generate()`` structure WITHOUT
importing it: keeping the helper self-contained (like the graph determinism helper)
avoids coupling the strict-typed package to the build script, while the
``test_determinism`` assertions and the ``--check`` tripwire still both operate over
the same live ``vella.agent.__all__``.

This is a script, not a test module — invoked via ``subprocess.run`` so each run
gets a fresh interpreter with the parent-supplied hash seed. ``PYTHONHASHSEED`` is
read once at interpreter start, so an in-process re-import would NOT reset it; a
subprocess is the only sound way to vary it.
"""

from __future__ import annotations

import json
import typing
from typing import Any

from pydantic import BaseModel

import vella.agent as agent


def _surface() -> dict[str, Any]:
    """Build the five-section public-surface snapshot from the live package.

    ``__all__`` is ``sorted()`` (set-derived ordering must never leak into a
    serialized artifact); each exported exception contributes its sorted base
    classes; each exported ``BaseModel`` contributes its ``Literal`` field value
    sets; each exported ``Literal`` alias contributes its sorted allowed values. At
    M0 ``__all__`` is empty, so every section is empty — but the derivation is the
    genuinely set-derived one the tripwire uses, so the bytes are hash-seed stable
    by construction and tighten as the surface grows.
    """
    exported = sorted(agent.__all__)
    errors: dict[str, list[str]] = {}
    literals: dict[str, list[Any]] = {}
    models: dict[str, list[str]] = {}
    verbs: dict[str, list[str]] = {}
    for name in exported:
        obj = getattr(agent, name)
        if isinstance(obj, type) and issubclass(obj, BaseException):
            errors[name] = sorted(
                f"{base.__module__}.{base.__qualname__}"
                for base in obj.__mro__
                if base is not obj
            )
        elif isinstance(obj, type) and issubclass(obj, BaseModel):
            models[name] = sorted(obj.model_fields)
            for field_name, field in obj.model_fields.items():
                annotation = field.annotation
                if typing.get_origin(annotation) is typing.Literal:
                    literals[f"{name}.{field_name}"] = sorted(typing.get_args(annotation))
        elif typing.get_origin(obj) is typing.Literal:
            literals[name] = sorted(typing.get_args(obj))
        elif isinstance(obj, type):
            verbs[name] = sorted(
                attr for attr in vars(obj) if not attr.startswith("_")
            )
    return {
        "__all__": exported,
        "errors": errors,
        "literals": literals,
        "models": models,
        "verbs": verbs,
    }


def build_artifact() -> str:
    """Serialize the canonical agent surface snapshot to byte-stable JSON.

    Builds the same five-section structure the gate freezes, dumped with
    ``sort_keys=True`` so no set-derived ordering can leak hash order into the bytes.

    Returns:
        The canonical JSON string for the agent public-surface snapshot.
    """
    return json.dumps(_surface(), sort_keys=True, separators=(",", ":"))


def main() -> None:
    """Print the surface artifact as canonical, byte-stable JSON to stdout."""
    print(build_artifact(), end="")


if __name__ == "__main__":
    main()
