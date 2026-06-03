"""The outcome a reconcile handler returns to the worker.

:class:`ReconcileResult` is a single frozen ``BaseModel`` (decision C1 in the
plan): one concrete model with a ``kind`` discriminator, rather than a union of
three models. A single concrete model types cleanly under both ``mypy --strict``
and pyright strict and gives pydantic value semantics for free.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator


class ReconcileResult(BaseModel):
    """A handler's verdict for one reconcile pass over an entity.

    The worker interprets ``kind`` as: ``"done"`` clears drift, ``"requeue"``
    re-enqueues the key after ``after`` seconds (counting toward the backoff
    budget), and ``"drop"`` discards the key without dead-lettering. ``after`` is
    meaningful only for ``"requeue"`` and is rejected on the other kinds.

    Attributes:
        kind: The disposition of this pass — one of ``"done"``, ``"requeue"``,
            ``"drop"``.
        after: For ``"requeue"`` only, the delay in seconds before the key is
            eligible again; ``None`` otherwise.

    Examples:
        >>> ReconcileResult(kind="done").kind
        'done'
        >>> ReconcileResult(kind="requeue", after=1.5).after
        1.5
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["done", "requeue", "drop"]
    after: Optional[float] = None

    @model_validator(mode="after")
    def _after_only_on_requeue(self) -> "ReconcileResult":
        """Reject ``after`` set on any kind other than ``"requeue"``.

        Returns:
            The validated model.

        Raises:
            ValueError: If ``after`` is set while ``kind`` is not ``"requeue"``.
        """
        if self.kind != "requeue" and self.after is not None:
            raise ValueError("'after' is only valid when kind == 'requeue'")
        return self
