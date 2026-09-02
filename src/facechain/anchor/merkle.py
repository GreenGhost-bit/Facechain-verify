"""Binary Merkle tree with domain-separated hashing and inclusion proofs.

Domain separation (RFC 6962 style) prevents second-preimage attacks that swap an
internal node for a leaf:

    leaf(x)        = SHA-256( 0x00 || x )
    node(l, r)     = SHA-256( 0x01 || l || r )

An odd level duplicates its last node. All public functions take/return
lowercase hex so proofs serialise cleanly into the evidence bundle.
"""

from __future__ import annotations

import hashlib

from ..models import MerkleProofStep

_LEAF = b"\x00"
_NODE = b"\x01"


def leaf_hash(data: bytes) -> str:
    return hashlib.sha256(_LEAF + data).hexdigest()


def _node(left_hex: str, right_hex: str) -> str:
    return hashlib.sha256(_NODE + bytes.fromhex(left_hex) + bytes.fromhex(right_hex)).hexdigest()


def merkle_root(leaves_hex: list[str]) -> str:
    """Root of a tree over already-``leaf_hash``-ed leaves. Empty -> SHA-256("")."""
    if not leaves_hex:
        return hashlib.sha256(b"").hexdigest()
    level = list(leaves_hex)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [_node(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def merkle_proof(leaves_hex: list[str], index: int) -> list[MerkleProofStep]:
    """Sibling path from leaf ``index`` up to the root."""
    if not 0 <= index < len(leaves_hex):
        raise IndexError(f"leaf index {index} out of range for {len(leaves_hex)} leaves")
    path: list[MerkleProofStep] = []
    level = list(leaves_hex)
    idx = index
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        if idx % 2:
            path.append(MerkleProofStep(sibling=level[idx - 1], position="left"))
        else:
            path.append(MerkleProofStep(sibling=level[idx + 1], position="right"))
        level = [_node(level[i], level[i + 1]) for i in range(0, len(level), 2)]
        idx //= 2
    return path


def verify_merkle_proof(leaf_hex: str, proof: list[MerkleProofStep], root_hex: str) -> bool:
    """Recompute the root from ``leaf_hex`` + ``proof`` and compare (constant-ish)."""
    acc = leaf_hex
    for step in proof:
        acc = _node(step.sibling, acc) if step.position == "left" else _node(acc, step.sibling)
    return _consteq(acc, root_hex)


def _consteq(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    diff = 0
    for x, y in zip(a, b, strict=False):
        diff |= ord(x) ^ ord(y)
    return diff == 0
