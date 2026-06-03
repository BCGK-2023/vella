"""``ReconcileResult`` M4 invariants.

Pins:

1. ``done`` and ``drop`` reject a non-``None`` ``after`` (the model_validator);
2. ``requeue`` accepts an ``after`` value;
3. the model is frozen — field mutation raises ``ValidationError``;
4. the ``kind`` Literal rejects values outside the allowed set;
5. the class-method constructors produce the expected ``kind`` and ``after``;
6. ``requeue`` without an ``after`` defaults to ``None``.

Types are checked at definition time by mypy/pyright (the gate enforces strict
mode over ``tests/``). No pytest-asyncio — this module is entirely synchronous.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vella.reconciler import ReconcileResult

# ---------------------------------------------------------------------------
# Type-level Protocol conformance proof (no runtime overhead, checked by mypy).
# ---------------------------------------------------------------------------
_done: ReconcileResult = ReconcileResult(kind="done")
_requeue: ReconcileResult = ReconcileResult(kind="requeue", after=1.0)
_drop: ReconcileResult = ReconcileResult(kind="drop")


# ---------------------------------------------------------------------------
# model_validator: after on non-requeue is rejected
# ---------------------------------------------------------------------------


def test_done_rejects_after() -> None:
    """``done`` with a non-None ``after`` raises ValidationError."""
    with pytest.raises(ValidationError):
        ReconcileResult(kind="done", after=1.0)


def test_drop_rejects_after() -> None:
    """``drop`` with a non-None ``after`` raises ValidationError."""
    with pytest.raises(ValidationError):
        ReconcileResult(kind="drop", after=0.5)


def test_requeue_accepts_after() -> None:
    """``requeue`` with a positive ``after`` is valid."""
    r = ReconcileResult(kind="requeue", after=2.5)
    assert r.kind == "requeue"
    assert r.after == 2.5


def test_requeue_accepts_none_after() -> None:
    """``requeue`` with ``after=None`` is valid — immediate re-enqueue."""
    r = ReconcileResult(kind="requeue", after=None)
    assert r.kind == "requeue"
    assert r.after is None


def test_done_with_none_after_is_valid() -> None:
    """``done`` with the default ``after=None`` is the normal success path."""
    r = ReconcileResult(kind="done")
    assert r.kind == "done"
    assert r.after is None


def test_drop_with_none_after_is_valid() -> None:
    """``drop`` with the default ``after=None`` is valid."""
    r = ReconcileResult(kind="drop")
    assert r.kind == "drop"
    assert r.after is None


# ---------------------------------------------------------------------------
# Frozen model: mutation raises
# ---------------------------------------------------------------------------


def test_model_is_frozen_kind() -> None:
    """Mutating ``kind`` on a frozen model raises ``ValidationError``."""
    r = ReconcileResult(kind="done")
    with pytest.raises(ValidationError):
        r.kind = "requeue"  # type: ignore[misc]


def test_model_is_frozen_after() -> None:
    """Mutating ``after`` on a frozen model raises ``ValidationError``."""
    r = ReconcileResult(kind="requeue", after=1.0)
    with pytest.raises(ValidationError):
        r.after = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Literal kind: bad values rejected
# ---------------------------------------------------------------------------


def test_bad_kind_rejected() -> None:
    """A ``kind`` value outside the Literal set raises ``ValidationError``."""
    bad: str = "retry"  # runtime string — pydantic validates, mypy cannot narrow
    with pytest.raises(ValidationError):
        ReconcileResult.model_validate({"kind": bad})


def test_empty_kind_rejected() -> None:
    """An empty string ``kind`` raises ``ValidationError``."""
    bad: str = ""
    with pytest.raises(ValidationError):
        ReconcileResult.model_validate({"kind": bad})


# ---------------------------------------------------------------------------
# Class-method constructors
# ---------------------------------------------------------------------------


def test_classmethod_done() -> None:
    """``ReconcileResult.done()`` produces kind="done" with after=None."""
    r = ReconcileResult.done()
    assert r.kind == "done"
    assert r.after is None


def test_classmethod_drop() -> None:
    """``ReconcileResult.drop()`` produces kind="drop" with after=None."""
    r = ReconcileResult.drop()
    assert r.kind == "drop"
    assert r.after is None


def test_classmethod_requeue_with_after() -> None:
    """``ReconcileResult.requeue(after=3.0)`` produces kind="requeue" with after=3.0."""
    r = ReconcileResult.requeue(after=3.0)
    assert r.kind == "requeue"
    assert r.after == 3.0


def test_classmethod_requeue_default_after() -> None:
    """``ReconcileResult.requeue()`` defaults ``after`` to ``None``."""
    r = ReconcileResult.requeue()
    assert r.kind == "requeue"
    assert r.after is None


def test_classmethod_results_are_frozen() -> None:
    """Results from class-method constructors are also frozen."""
    r = ReconcileResult.done()
    with pytest.raises(ValidationError):
        r.kind = "drop"  # type: ignore[misc]
