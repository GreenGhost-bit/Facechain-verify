"""Offline unit tests for the EVM backend's pure helpers.

The broadcast/verify paths that need a live JSON-RPC node are exercised manually
against a funded testnet key (see README) -- not in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from facechain.anchor.evm_backend import _IndexCache, decode_calldata, encode_calldata

REC = "0123456789abcdef" * 4  # 64 hex chars


def test_calldata_round_trip():
    data = encode_calldata(REC)
    assert data[:4] == b"FCV1"
    assert len(data) == 36
    assert decode_calldata(data) == REC


def test_decode_rejects_bad_magic():
    assert decode_calldata(b"XXXX" + bytes.fromhex(REC)) is None


def test_decode_rejects_wrong_length():
    assert decode_calldata(b"FCV1" + b"\x00" * 10) is None
    assert decode_calldata(b"") is None


def test_encode_validates_hash():
    with pytest.raises(ValueError):
        encode_calldata("nope")


def test_index_cache_persists(tmp_path: Path):
    cache = _IndexCache(tmp_path / "evm-11155111.index.json")
    assert cache.get(REC) is None
    cache.put(REC, {"tx_hash": "0xabc", "block_number": 42})
    again = _IndexCache(tmp_path / "evm-11155111.index.json")
    assert again.get(REC) == {"tx_hash": "0xabc", "block_number": 42}


def test_index_cache_survives_corruption(tmp_path: Path):
    p = tmp_path / "evm-1.index.json"
    p.write_text("{ not json")
    cache = _IndexCache(p)
    assert cache.load() == {}
