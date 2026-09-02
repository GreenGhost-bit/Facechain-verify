from __future__ import annotations

import json
from pathlib import Path

import pytest

from facechain.anchor.local_chain import LocalChain, _leading_zero_bits, compute_block_hash
from facechain.errors import ChainIntegrityError

H1 = "a" * 64
H2 = "b" * 64
H3 = "c" * 64


def test_genesis_is_deterministic(tmp_path: Path):
    c1 = LocalChain(tmp_path / "x")
    c2_root = tmp_path / "y"
    c2 = LocalChain(c2_root)
    g1, g2 = c1.blocks()[0], c2.blocks()[0]
    assert g1["hash"] == g2["hash"]
    assert g1["prev_hash"] == "0" * 64
    assert g1["index"] == 0


def test_anchor_appends_linked_block_and_verifies(tmp_path: Path):
    chain = LocalChain(tmp_path / "c")
    receipt = chain.anchor(H1)
    assert receipt.block_index == 1
    assert receipt.idempotent_hit is False
    blocks = chain.blocks()
    assert len(blocks) == 2
    assert blocks[1]["prev_hash"] == blocks[0]["hash"]
    chain.verify_chain()  # must not raise
    assert chain.verify(H1, receipt)[0].ok  # chain_integrity check


def test_reanchor_same_hash_is_idempotent(tmp_path: Path):
    chain = LocalChain(tmp_path / "c")
    r1 = chain.anchor(H1)
    r2 = chain.anchor(H1)
    assert r2.idempotent_hit is True
    assert r2.block_hash == r1.block_hash
    assert len(chain.blocks()) == 2  # no new block


def test_distinct_hashes_get_distinct_blocks(tmp_path: Path):
    chain = LocalChain(tmp_path / "c")
    chain.anchor(H1)
    chain.anchor(H2)
    chain.anchor(H3)
    assert [b["index"] for b in chain.blocks()] == [0, 1, 2, 3]
    chain.verify_chain()


def test_rejects_non_hex_record_hash(tmp_path: Path):
    chain = LocalChain(tmp_path / "c")
    with pytest.raises(ValueError):
        chain.anchor("not-a-hash")


def test_tampering_a_record_is_detected(tmp_path: Path):
    chain = LocalChain(tmp_path / "c")
    chain.anchor(H1)
    blocks = chain.blocks()
    blocks[1]["records"][0] = H2
    chain.path.write_text(
        "\n".join(json.dumps(b, sort_keys=True, separators=(",", ":")) for b in blocks) + "\n"
    )
    with pytest.raises(ChainIntegrityError, match="Merkle root mismatch"):
        LocalChain(tmp_path / "c").verify_chain()


def test_tampering_a_block_hash_is_detected(tmp_path: Path):
    chain = LocalChain(tmp_path / "c")
    chain.anchor(H1)
    blocks = chain.blocks()
    blocks[1]["timestamp"] = "2099-01-01T00:00:00Z"  # content changed, hash not recomputed
    chain.path.write_text(
        "\n".join(json.dumps(b, sort_keys=True, separators=(",", ":")) for b in blocks) + "\n"
    )
    with pytest.raises(ChainIntegrityError, match="hash mismatch"):
        LocalChain(tmp_path / "c").verify_chain()


def test_breaking_the_backlink_is_detected(tmp_path: Path):
    chain = LocalChain(tmp_path / "c")
    chain.anchor(H1)
    chain.anchor(H2)
    blocks = chain.blocks()
    blocks[2]["prev_hash"] = "0" * 64
    blocks[2]["hash"] = compute_block_hash(blocks[2])  # keep self-hash consistent
    chain.path.write_text(
        "\n".join(json.dumps(b, sort_keys=True, separators=(",", ":")) for b in blocks) + "\n"
    )
    with pytest.raises(ChainIntegrityError, match="prev_hash"):
        LocalChain(tmp_path / "c").verify_chain()


def test_proof_of_work_difficulty_is_enforced(tmp_path: Path):
    chain = LocalChain(tmp_path / "c", difficulty_bits=8)
    receipt = chain.anchor(H1)
    assert receipt.block_hash is not None
    assert _leading_zero_bits(receipt.block_hash) >= 8
    chain.verify_chain()


def test_merkle_inclusion_check_in_backend_verify(tmp_path: Path):
    chain = LocalChain(tmp_path / "c")
    r = chain.anchor(H1)
    names = {c.name: c.ok for c in chain.verify(H1, r)}
    assert names["local.chain_integrity"]
    assert names["local.record_on_chain"]
    assert names["local.merkle_inclusion"]
    assert names["local.receipt_block_hash"]


def test_leading_zero_bits_helper():
    assert _leading_zero_bits("0" * 64) == 256
    assert _leading_zero_bits("8" + "0" * 63) == 0
    assert _leading_zero_bits("1" + "0" * 63) == 3
    assert _leading_zero_bits("00" + "f" * 62) == 8
