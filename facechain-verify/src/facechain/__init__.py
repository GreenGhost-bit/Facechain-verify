"""facechain-verify.

An end-to-end, content-addressed pipeline:

    face scan  ->  live web/social reverse-image match  ->  tamper-evident
    blockchain anchor  ->  independent re-verification.

Every stage emits an immutable, hash-named artifact. Only the SHA-256 of the
canonical *evidence bundle* is written on-chain, so a third party can re-derive
every hash from the raw artifacts and re-check it against the ledger without
sharing any mutable state with the pipeline that produced it.
"""

from __future__ import annotations

__version__ = "1.0.0"
PIPELINE_VERSION = "facechain-verify/1.0.0"

__all__ = ["PIPELINE_VERSION", "__version__"]
