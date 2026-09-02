"""Anchor-backend abstraction.

An anchor backend takes the 64-hex-char ``record_hash`` of an
:class:`~facechain.models.EvidenceBundle` and writes it somewhere tamper-evident,
returning an :class:`~facechain.models.AnchorReceipt` that carries everything a
third party needs to read it back. ``verify`` re-reads it and returns a list of
:class:`~facechain.models.Check` rows.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import AnchorReceipt, Check

HASH_RE = "^[0-9a-f]{64}$"


@runtime_checkable
class AnchorBackend(Protocol):
    name: str
    network: str

    def anchor(self, record_hash: str) -> AnchorReceipt: ...

    def verify(self, record_hash: str, receipt: AnchorReceipt) -> list[Check]: ...


def validate_record_hash(record_hash: str) -> str:
    import re

    if not re.match(HASH_RE, record_hash):
        raise ValueError(f"record_hash must be 64 lowercase hex chars, got {record_hash!r}")
    return record_hash
