"""Deterministic canonical JSON + hashing.

Two machines, two Python builds, two runs must produce the *same bytes* for the
same logical object, or the on-chain hash will never re-verify. We follow the
spirit of RFC 8785 (JSON Canonicalization Scheme):

* object keys sorted by UTF-16 code unit (Python's default ``sorted`` on ``str``
  is code-point order, which matches for the BMP subset we emit);
* no insignificant whitespace;
* UTF-8 output, ``ensure_ascii`` disabled;
* integers stay integers; floats are rejected outright -- callers must quantise
  similarity scores to fixed-point integers before canonicalising.

Rejecting floats is deliberate: ``0.1 + 0.2`` style drift is the classic reason a
"verify" step mysteriously fails months later.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = [
    "canonical_bytes",
    "canonical_json",
    "from_fixed",
    "hash_object",
    "sha256_bytes",
    "sha256_hex",
    "to_fixed",
]

FIXED_SCALE = 1_000_000  # similarity scores are stored as ppm integers


def _reject_floats(obj: Any) -> None:
    if isinstance(obj, float):
        raise TypeError(
            "float values are not allowed in canonical JSON; quantise with to_fixed() first"
        )
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"object keys must be str, got {type(k).__name__}")
            _reject_floats(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _reject_floats(v)


def canonical_json(obj: Any) -> str:
    """Return the canonical JSON *string* for ``obj``."""
    _reject_floats(obj)
    return json.dumps(
        obj,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_bytes(obj: Any) -> bytes:
    """Return the canonical JSON encoded as UTF-8 bytes."""
    return canonical_json(obj).encode("utf-8")


def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_object(obj: Any) -> str:
    """SHA-256 hex of the canonical encoding of ``obj``."""
    return sha256_hex(canonical_bytes(obj))


def to_fixed(x: float, scale: int = FIXED_SCALE) -> int:
    """Quantise a float in a stable, rounding-defined way (banker's rounding off)."""
    return int((x * scale) + (0.5 if x >= 0 else -0.5))


def from_fixed(n: int, scale: int = FIXED_SCALE) -> float:
    return n / scale
