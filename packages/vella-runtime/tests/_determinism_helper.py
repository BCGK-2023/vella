"""Subprocess fixture for the determinism harness.

Run as a script under a fixed ``PYTHONHASHSEED`` (set by the parent test). It
imports the runtime, builds a canonical fixture, and prints its canonical JSON
encoding to stdout. The fixture deliberately contains a ``set``: a set is
serialized via ``sorted()`` so its byte form is independent of the per-process
hash seed. If any serialized value ever derived its order from set/dict hash
iteration instead, the two seeds would diverge and the parent test would fail.

This is a script, not a test module — it is invoked via ``subprocess.run`` so
each run gets a *fresh* interpreter with the parent-supplied hash seed. An
in-process re-import would NOT reset the hash seed (it is fixed once at
interpreter start), so a subprocess is the only sound way to vary it.
"""

from __future__ import annotations

import json

import vella.runtime as runtime


def build_fixture() -> dict[str, object]:
    """Build the canonical fixture exercised across hash seeds.

    Minimal for M1 (a small dict plus a set rendered via ``sorted()``); M5
    replaces the body with a real state-table + log fixture. The set is the
    part under test — it is the value whose ordering a hash seed could perturb.
    """
    tags = {"gamma", "alpha", "beta", "delta"}
    return {
        "surface": sorted(runtime.__all__),
        "tags": sorted(tags),
    }


def main() -> None:
    """Print the fixture as canonical, byte-stable JSON to stdout."""
    print(
        json.dumps(build_fixture(), sort_keys=True, separators=(",", ":")),
        end="",
    )


if __name__ == "__main__":
    main()
