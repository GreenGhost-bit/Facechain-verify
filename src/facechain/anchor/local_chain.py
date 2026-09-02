"""Local hash-linked Merkle ledger -- the default anchor backend.

Not a Python ``dict`` pretending to be a chain. Each anchored ``record_hash``
becomes a leaf in a new block:

* blocks are hash-linked (``prev_hash``) back to a deterministic genesis;
* each block commits to its records through a Merkle root;
* an optional proof-of-work (``difficulty`` leading zero bits) makes silent
  rewrites expensive;
* the whole ledger is an append-only ``blocks.jsonl`` with an atomic cross-
  platform lock;
* ``verify_chain`` independently recomputes every Merkle root, every block hash,
  and every back-link, and pinpoints the first broken block.

Idempotency: anchoring a ``record_hash`` that is already in the ledger returns
the existing inclusion proof instead of appending a duplicate.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..canonical import canonical_bytes, sha256_hex
from ..errors import ChainIntegrityError
from ..logging import LOG
from ..models import AnchorReceipt, Check, MerkleProofStep
from .base import validate_record_hash
from .merkle import leaf_hash, merkle_proof, merkle_root, verify_merkle_proof

GENESIS_PREV = "0" * 64
_CHAIN_MAGIC = "facechain-local-chain/1"


@contextmanager
def _dir_lock(path: Path, *, timeout_s: float = 15.0, poll_s: float = 0.05) -> Iterator[None]:
    """Cross-platform advisory lock via atomic ``mkdir``."""
    lock = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            lock.mkdir(parents=False, exist_ok=False)
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"could not acquire chain lock {lock}") from None
            time.sleep(poll_s)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):  # pragma: no cover - best effort
            lock.rmdir()


def _block_signing_payload(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": block["index"],
        "timestamp": block["timestamp"],
        "prev_hash": block["prev_hash"],
        "merkle_root": block["merkle_root"],
        "difficulty": block["difficulty"],
        "records": block["records"],
        "nonce": block["nonce"],
    }


def compute_block_hash(block: dict[str, Any]) -> str:
    return sha256_hex(canonical_bytes(_block_signing_payload(block)))


def _leading_zero_bits(hex_digest: str) -> int:
    bits = 0
    for ch in hex_digest:
        v = int(ch, 16)
        if v == 0:
            bits += 4
            continue
        bits += (4 - v.bit_length())
        break
    return bits


class LocalChain:
    """Append-only hash-linked Merkle ledger persisted under ``root``."""

    name = "local"

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        difficulty_bits: int = 0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "blocks.jsonl"
        self.difficulty_bits = int(difficulty_bits)
        self.network = f"local-merkle-chain(diff={self.difficulty_bits})"
        self._clock = clock or time.time
        if not self.path.exists():
            self._bootstrap()

    # -- persistence ----------------------------------------------------
    def _bootstrap(self) -> None:
        with _dir_lock(self.path):
            if self.path.exists():
                return
            genesis = {
                "index": 0,
                "timestamp": "1970-01-01T00:00:00Z",  # fixed => reproducible genesis
                "prev_hash": GENESIS_PREV,
                "records": [],
                "merkle_root": merkle_root([]),
                "difficulty": 0,
                "nonce": 0,
                "magic": _CHAIN_MAGIC,
            }
            genesis["hash"] = compute_block_hash(genesis)
            self._append_raw(genesis)
            LOG.info("chain.genesis", root=str(self.root), hash=genesis["hash"])

    def _append_raw(self, block: dict[str, Any]) -> None:
        line = json.dumps(block, sort_keys=True, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def blocks(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw:
                out.append(json.loads(raw))
        return out

    def head(self) -> dict[str, Any]:
        return self.blocks()[-1]

    # -- mining -------------------------------------------------------
    def _mine(self, block: dict[str, Any]) -> dict[str, Any]:
        if self.difficulty_bits <= 0:
            block["nonce"] = 0
            block["hash"] = compute_block_hash(block)
            return block
        nonce = 0
        while True:
            block["nonce"] = nonce
            digest = compute_block_hash(block)
            if _leading_zero_bits(digest) >= self.difficulty_bits:
                block["hash"] = digest
                return block
            nonce += 1

    # -- public API ----------------------------------------------------
    def find_record(self, record_hash: str) -> tuple[dict[str, Any], int] | None:
        for block in self.blocks():
            if record_hash in block["records"]:
                return block, block["records"].index(record_hash)
        return None

    def anchor(self, record_hash: str) -> AnchorReceipt:
        record_hash = validate_record_hash(record_hash)
        with _dir_lock(self.path):
            existing = self.find_record(record_hash)
            if existing is not None:
                block, leaf_index = existing
                LOG.info("chain.anchor.idempotent", record_hash=record_hash, block=block["index"])
                return self._receipt(block, leaf_index, record_hash, idempotent=True)

            chain = self.blocks()
            prev = chain[-1]
            leaves = [leaf_hash(bytes.fromhex(record_hash))]
            block = {
                "index": prev["index"] + 1,
                "timestamp": _rfc3339(self._clock()),
                "prev_hash": prev["hash"],
                "records": [record_hash],
                "merkle_root": merkle_root(leaves),
                "difficulty": self.difficulty_bits,
                "nonce": 0,
                "magic": _CHAIN_MAGIC,
            }
            block = self._mine(block)
            self._append_raw(block)
            LOG.info(
                "chain.anchor",
                record_hash=record_hash,
                block=block["index"],
                block_hash=block["hash"],
                difficulty=self.difficulty_bits,
            )
            return self._receipt(block, 0, record_hash, idempotent=False)

    def _receipt(
        self, block: dict[str, Any], leaf_index: int, record_hash: str, *, idempotent: bool
    ) -> AnchorReceipt:
        leaves = [leaf_hash(bytes.fromhex(r)) for r in block["records"]]
        proof = merkle_proof(leaves, leaf_index)
        return AnchorReceipt(
            backend="local",
            network=f"local-merkle-chain(diff={block['difficulty']})",
            record_hash=record_hash,
            idempotent_hit=idempotent,
            ref={
                "chain_root": str(self.root),
                "block_index": block["index"],
                "block_hash": block["hash"],
                "block_timestamp": block["timestamp"],
                "prev_hash": block["prev_hash"],
                "merkle_root": block["merkle_root"],
                "leaf_index": leaf_index,
            },
            block_index=block["index"],
            block_hash=block["hash"],
            merkle_root=block["merkle_root"],
            leaf_index=leaf_index,
            merkle_proof=proof,
        )

    def verify_chain(self) -> None:
        """Full structural re-check. Raises :class:`ChainIntegrityError` on the
        first inconsistency."""
        chain = self.blocks()
        if not chain:
            raise ChainIntegrityError("ledger is empty (no genesis)")
        if chain[0]["prev_hash"] != GENESIS_PREV or chain[0]["index"] != 0:
            raise ChainIntegrityError("genesis block malformed", detail=chain[0].get("index"))
        prev_hash = None
        for i, block in enumerate(chain):
            if block["index"] != i:
                raise ChainIntegrityError(f"block {i} has wrong index {block['index']}")
            leaves = [leaf_hash(bytes.fromhex(r)) for r in block["records"]]
            if merkle_root(leaves) != block["merkle_root"]:
                raise ChainIntegrityError(f"block {i} Merkle root mismatch")
            if compute_block_hash(block) != block["hash"]:
                raise ChainIntegrityError(f"block {i} hash mismatch (content tampered)")
            if i > 0 and block["prev_hash"] != prev_hash:
                raise ChainIntegrityError(f"block {i} prev_hash does not link to block {i - 1}")
            declared_diff = int(block.get("difficulty", 0))
            if declared_diff > 0 and _leading_zero_bits(block["hash"]) < declared_diff:
                raise ChainIntegrityError(f"block {i} fails its stated proof-of-work")
            prev_hash = block["hash"]

    def verify(self, record_hash: str, receipt: AnchorReceipt) -> list[Check]:
        """Backend-interface verification: chain integrity + inclusion proof."""
        checks: list[Check] = []
        try:
            self.verify_chain()
            checks.append(Check(name="local.chain_integrity", ok=True, detail="full chain re-hashed"))
        except ChainIntegrityError as exc:
            checks.append(Check(name="local.chain_integrity", ok=False, detail=str(exc)))
            return checks

        found = self.find_record(record_hash)
        checks.append(
            Check(
                name="local.record_on_chain",
                ok=found is not None,
                detail="" if found else "record_hash not present in any block",
                expected=record_hash,
            )
        )
        if found is None:
            return checks
        block, leaf_index = found

        block_hash_ok = block["hash"] == (receipt.block_hash or receipt.ref.get("block_hash"))
        checks.append(
            Check(
                name="local.receipt_block_hash",
                ok=block_hash_ok,
                expected=str(receipt.block_hash),
                actual=block["hash"],
            )
        )

        leaves = [leaf_hash(bytes.fromhex(r)) for r in block["records"]]
        proof: list[MerkleProofStep] = receipt.merkle_proof or merkle_proof(leaves, leaf_index)
        this_leaf = leaf_hash(bytes.fromhex(record_hash))
        proof_ok = verify_merkle_proof(this_leaf, proof, block["merkle_root"])
        checks.append(
            Check(
                name="local.merkle_inclusion",
                ok=proof_ok,
                detail=f"leaf {leaf_index} of block {block['index']}",
                expected=block["merkle_root"],
            )
        )
        return checks


def _rfc3339(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))
