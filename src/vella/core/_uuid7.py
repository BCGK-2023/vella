"""
Inline, monotonic UUIDv7 generator (RFC 9562).

Time-ordered UUIDs give B-tree storage layers near-sequential inserts (far
better index locality than random UUIDv4) and free creation-time sortability.
Implemented inline rather than depending on a backport so ``vella.core`` keeps
its zero-third-party-dependency promise on Python < 3.14.

RFC 9562 §6.2 "Monotonic Random" (method 2): within a single millisecond the
12-bit ``rand_a`` field is used as a counter so successive ids are strictly
ordered (random ordering within a millisecond would otherwise break "latest by
id" and weaken index locality). The counter is process-wide and lock-guarded;
on overflow we borrow into the next millisecond to preserve monotonicity.

Layout (128 bits):
    48  unix timestamp in milliseconds
     4  version (= 7)
    12  rand_a, used here as an intra-millisecond monotonic counter
     2  variant (= 0b10)
    62  rand_b (random)
"""

from __future__ import annotations

import secrets
import threading
import time
from uuid import UUID

_lock = threading.Lock()
_last_ms = -1
_counter = 0
_MAX_COUNTER = 0xFFF  # 12 bits


def uuid7() -> UUID:
    """Generate a time-ordered, intra-millisecond-monotonic UUIDv7."""
    global _last_ms, _counter
    with _lock:
        ms = time.time_ns() // 1_000_000
        if ms > _last_ms:
            _last_ms = ms
            _counter = 0
        else:
            # Same (or backwards) millisecond: advance the counter to stay
            # strictly monotonic; borrow into the next ms on overflow.
            _counter += 1
            if _counter > _MAX_COUNTER:
                _last_ms += 1
                _counter = 0
            ms = _last_ms
        counter = _counter

    rand_b = secrets.randbits(62)
    value = (ms & 0xFFFFFFFFFFFF) << 80
    value |= 0x7 << 76          # version
    value |= counter << 64      # rand_a as monotonic counter
    value |= 0b10 << 62         # variant
    value |= rand_b
    return UUID(int=value)


__all__ = ["uuid7"]
