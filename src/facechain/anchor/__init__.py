"""Blockchain anchoring stage."""

from __future__ import annotations

from .base import AnchorBackend, validate_record_hash
from .factory import build_anchor_backend
from .local_chain import LocalChain
from .merkle import merkle_proof, merkle_root, verify_merkle_proof

__all__ = [
    "AnchorBackend",
    "LocalChain",
    "build_anchor_backend",
    "merkle_proof",
    "merkle_root",
    "validate_record_hash",
    "verify_merkle_proof",
]
