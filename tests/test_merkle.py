from __future__ import annotations

import hashlib

import pytest

from facechain.anchor.merkle import (
    leaf_hash,
    merkle_proof,
    merkle_root,
    verify_merkle_proof,
)


def _leaves(n: int) -> list[str]:
    return [leaf_hash(f"record-{i}".encode()) for i in range(n)]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 13])
def test_every_leaf_has_a_valid_inclusion_proof(n: int):
    leaves = _leaves(n)
    root = merkle_root(leaves)
    for i in range(n):
        proof = merkle_proof(leaves, i)
        assert verify_merkle_proof(leaves[i], proof, root)


def test_single_leaf_root_is_the_leaf():
    leaves = _leaves(1)
    assert merkle_root(leaves) == leaves[0]
    assert verify_merkle_proof(leaves[0], merkle_proof(leaves, 0), leaves[0])


def test_tampered_leaf_fails_verification():
    leaves = _leaves(4)
    root = merkle_root(leaves)
    proof = merkle_proof(leaves, 2)
    forged = leaf_hash(b"record-999")
    assert not verify_merkle_proof(forged, proof, root)


def test_tampered_proof_step_fails():
    leaves = _leaves(4)
    root = merkle_root(leaves)
    proof = merkle_proof(leaves, 1)
    proof[0].sibling = "0" * 64
    assert not verify_merkle_proof(leaves[1], proof, root)


def test_domain_separation_leaf_vs_node():
    # a leaf hash must never collide with an internal node hash of the same bytes
    x = b"abc"
    leaf = leaf_hash(x)
    node_like = hashlib.sha256(b"\x01" + bytes.fromhex(leaf) + bytes.fromhex(leaf)).hexdigest()
    assert leaf != node_like


def test_root_changes_if_any_leaf_changes():
    a = merkle_root(_leaves(6))
    changed = _leaves(6)
    changed[3] = leaf_hash(b"different")
    assert a != merkle_root(changed)


def test_out_of_range_index_raises():
    with pytest.raises(IndexError):
        merkle_proof(_leaves(3), 3)


def test_empty_root_is_sha256_of_empty():
    assert merkle_root([]) == hashlib.sha256(b"").hexdigest()
