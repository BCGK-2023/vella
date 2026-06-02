"""Bind the in-memory adapter to the adapter-agnostic conformance suite.

``InMemoryStore`` is the v0.1 reference ``Store``; running the shared
``StoreConformance`` cases against it proves it satisfies the contract a future
SQL adapter will run unchanged. The binding is a single subclass that points
``store_factory`` at the adapter constructor — mirroring how pytest-style
conformance suites bind implementations.
"""

from __future__ import annotations

from conformance.store_suite import StoreConformance
from vella.runtime._inmemory import InMemoryStore


class TestInMemoryStore(StoreConformance):
    """Run the full Store conformance suite against ``InMemoryStore``."""

    store_factory = staticmethod(InMemoryStore)
