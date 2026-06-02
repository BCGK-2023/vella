"""Quickstart: build a node, update it copy-on-write, and assert the outcome.

Runnable standalone (``python docs/examples/quickstart.py`` -> exit 0) and
executed in CI via ``tests/test_examples.py`` (``runpy``). Registers into a
*local* ``Registry()`` so the process-global ``default_registry`` stays empty.
"""

from uuid import uuid4

from pydantic import ConfigDict

from vella.core import FlexibleData, Node, Registry, node_type

# Hermetic: a local registry, never the process-global default_registry.
registry = Registry()


@node_type("task", compat="BACKWARD", registry=registry)
class TaskData(FlexibleData):
    """A minimal strict, frozen node-type payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    title: str
    done: bool = False


def main() -> None:
    """Build a node, evolve it copy-on-write, and assert the result."""
    task = Node.from_data(
        TaskData(title="Ship the docs"),
        name="Ship the docs",
        created_by=uuid4(),
    )
    assert task.type == "task"
    assert task.data.title == "Ship the docs"
    assert task.data.done is False

    # Copy-on-write: evolve returns a new node; the original is untouched.
    done = task.evolve(data=TaskData(title="Ship the docs", done=True))
    assert done.data.done is True
    assert task.data.done is False
    assert done is not task

    # The local registry holds only our type; the global default stays empty.
    assert registry.names() == ["task"]


if __name__ == "__main__":
    main()
