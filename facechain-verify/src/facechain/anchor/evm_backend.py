"""EVM public-testnet anchor backend (opt-in).

Two anchoring modes, chosen automatically:

* **registry** -- if ``FACECHAIN_EVM_REGISTRY_ADDRESS`` is set, call
  ``anchor(bytes32)`` on a deployed :file:`contracts/EvidenceRegistry.sol`;
  re-verification reads the stored block number back via ``recordBlock``.
* **calldata** -- otherwise send a 0-value self-transaction whose input data is
  ``b"FCV1" || record_hash`` (36 bytes). Re-verification pulls the transaction by
  hash and re-parses the calldata. Works on any EVM chain with zero setup.

Idempotency: a local ``evm-<chainid>.index.json`` maps ``record_hash -> txhash``;
a repeat anchor returns the cached receipt without broadcasting.

The pure helpers (``encode_calldata`` / ``decode_calldata`` / the index cache)
are unit-tested offline; the broadcast path is exercised manually against a
funded testnet key (see README).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..errors import BackendUnavailableError
from ..logging import LOG
from ..models import AnchorReceipt, Check
from .base import validate_record_hash

_MAGIC = b"FCV1"

# Minimal ABI for contracts/EvidenceRegistry.sol
REGISTRY_ABI: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "anchor",
        "stateMutability": "nonpayable",
        "inputs": [{"name": "recordHash", "type": "bytes32"}],
        "outputs": [],
    },
    {
        "type": "function",
        "name": "recordBlock",
        "stateMutability": "view",
        "inputs": [{"name": "recordHash", "type": "bytes32"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "type": "function",
        "name": "recordTimestamp",
        "stateMutability": "view",
        "inputs": [{"name": "recordHash", "type": "bytes32"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "type": "event",
        "name": "Anchored",
        "anonymous": False,
        "inputs": [
            {"name": "recordHash", "type": "bytes32", "indexed": True},
            {"name": "submitter", "type": "address", "indexed": True},
            {"name": "timestamp", "type": "uint256", "indexed": False},
        ],
    },
]


def encode_calldata(record_hash: str) -> bytes:
    return _MAGIC + bytes.fromhex(validate_record_hash(record_hash))


def decode_calldata(data: bytes) -> str | None:
    if len(data) != len(_MAGIC) + 32 or not data.startswith(_MAGIC):
        return None
    return data[len(_MAGIC):].hex()


class _IndexCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text("utf-8"))
        except json.JSONDecodeError:  # pragma: no cover - corrupt cache
            return {}

    def get(self, record_hash: str) -> dict[str, Any] | None:
        return self.load().get(record_hash)

    def put(self, record_hash: str, entry: dict[str, Any]) -> None:
        data = self.load()
        data[record_hash] = entry
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


class EVMChainAnchor:
    name = "evm"

    def __init__(
        self,
        *,
        rpc_url: str | None,
        private_key: str | None,
        registry_address: str | None = None,
        chain_id: int | None = None,
        cache_dir: str | Path = "chaindata",
        confirmations: int = 1,
        tx_timeout_s: int = 180,
    ) -> None:
        if not rpc_url or not private_key:
            raise BackendUnavailableError(
                "EVM backend needs FACECHAIN_EVM_RPC_URL and FACECHAIN_EVM_PRIVATE_KEY"
            )
        try:
            from web3 import Web3
        except ImportError as exc:  # pragma: no cover - optional dep
            raise BackendUnavailableError(
                "web3 is not installed; run: pip install 'facechain-verify[evm]'"
            ) from exc

        self._w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
        if not self._w3.is_connected():
            raise BackendUnavailableError(f"cannot reach EVM RPC at {rpc_url}")
        self._acct = self._w3.eth.account.from_key(private_key)
        self._chain_id = int(chain_id or self._w3.eth.chain_id)
        self._confirmations = confirmations
        self._tx_timeout_s = tx_timeout_s
        self._registry_address = (
            self._w3.to_checksum_address(registry_address) if registry_address else None
        )
        self._mode = "registry" if self._registry_address else "calldata"
        self.network = f"evm:{self._chain_id}:{self._mode}"
        self._cache = _IndexCache(Path(cache_dir) / f"evm-{self._chain_id}.index.json")
        LOG.info(
            "evm.ready",
            chain_id=self._chain_id,
            mode=self._mode,
            account=self._acct.address,
            registry=self._registry_address,
        )

    @classmethod
    def available(cls) -> bool:
        try:
            import web3  # noqa: F401

            return True
        except ImportError:
            return False

    # -- anchor ------------------------------------------------------
    def anchor(self, record_hash: str) -> AnchorReceipt:
        record_hash = validate_record_hash(record_hash)
        cached = self._cache.get(record_hash)
        if cached:
            LOG.info("evm.anchor.idempotent", record_hash=record_hash, tx=cached["tx_hash"])
            return self._receipt_from_entry(record_hash, cached, idempotent=True)

        if self._mode == "registry":
            tx_hash = self._send_registry(record_hash)
        else:
            tx_hash = self._send_calldata(record_hash)

        rcpt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=self._tx_timeout_s)
        if rcpt["status"] != 1:
            raise BackendUnavailableError(f"anchor tx reverted: {tx_hash.hex()}")

        entry = {
            "tx_hash": rcpt["transactionHash"].hex(),
            "block_number": int(rcpt["blockNumber"]),
            "chain_id": self._chain_id,
            "mode": self._mode,
            "contract": self._registry_address,
            "from": self._acct.address,
            "anchored_epoch": int(time.time()),
        }
        self._cache.put(record_hash, entry)
        LOG.info("evm.anchor", record_hash=record_hash, **{k: entry[k] for k in ("tx_hash", "block_number")})
        return self._receipt_from_entry(record_hash, entry, idempotent=False)

    def _send_calldata(self, record_hash: str) -> Any:
        tx: dict[str, Any] = {
            "chainId": self._chain_id,
            "from": self._acct.address,
            "to": self._acct.address,
            "value": 0,
            "nonce": self._w3.eth.get_transaction_count(self._acct.address),
            "data": "0x" + encode_calldata(record_hash).hex(),
        }
        tx["gas"] = self._w3.eth.estimate_gas(tx)  # type: ignore[arg-type]
        tx.update(self._fee_fields())
        signed = self._acct.sign_transaction(tx)
        return self._w3.eth.send_raw_transaction(signed.raw_transaction)

    def _send_registry(self, record_hash: str) -> Any:
        contract = self._w3.eth.contract(address=self._registry_address, abi=REGISTRY_ABI)
        fn = contract.functions.anchor(bytes.fromhex(record_hash))
        tx = fn.build_transaction(
            {  # type: ignore[arg-type]
                "chainId": self._chain_id,
                "from": self._acct.address,
                "nonce": self._w3.eth.get_transaction_count(self._acct.address),
                **self._fee_fields(),
            }
        )
        signed = self._acct.sign_transaction(tx)
        return self._w3.eth.send_raw_transaction(signed.raw_transaction)

    def _fee_fields(self) -> dict[str, Any]:
        try:
            base = self._w3.eth.get_block("latest")["baseFeePerGas"]
            tip = self._w3.eth.max_priority_fee
            return {"maxFeePerGas": base * 2 + tip, "maxPriorityFeePerGas": tip}
        except Exception:  # pragma: no cover - pre-EIP-1559 chains
            return {"gasPrice": self._w3.eth.gas_price}

    # -- verify -----------------------------------------------------
    def verify(self, record_hash: str, receipt: AnchorReceipt) -> list[Check]:
        record_hash = validate_record_hash(record_hash)
        checks: list[Check] = []
        tx_hash = receipt.ref.get("tx_hash")
        if not tx_hash:
            checks.append(Check(name="evm.receipt", ok=False, detail="receipt has no tx_hash"))
            return checks

        try:
            tx = self._w3.eth.get_transaction(tx_hash)
            rcpt = self._w3.eth.get_transaction_receipt(tx_hash)
        except Exception as exc:
            checks.append(Check(name="evm.tx_lookup", ok=False, detail=repr(exc), expected=tx_hash))
            return checks

        checks.append(Check(name="evm.tx_mined", ok=rcpt["status"] == 1, actual=str(rcpt["status"])))
        head = self._w3.eth.block_number
        confs = head - int(rcpt["blockNumber"]) + 1
        checks.append(
            Check(
                name="evm.confirmations",
                ok=confs >= self._confirmations,
                detail=f"{confs} confirmation(s)",
            )
        )

        if receipt.ref.get("mode", self._mode) == "calldata":
            data = bytes(tx["input"])
            parsed = decode_calldata(data)
            checks.append(
                Check(
                    name="evm.calldata_matches_record",
                    ok=parsed == record_hash,
                    expected=record_hash,
                    actual=str(parsed),
                )
            )
        else:
            contract = self._w3.eth.contract(
                address=self._w3.to_checksum_address(receipt.ref["contract"]), abi=REGISTRY_ABI
            )
            stored_block = contract.functions.recordBlock(bytes.fromhex(record_hash)).call()
            checks.append(
                Check(
                    name="evm.registry_has_record",
                    ok=stored_block == int(rcpt["blockNumber"]),
                    expected=str(rcpt["blockNumber"]),
                    actual=str(stored_block),
                )
            )
        return checks

    def _receipt_from_entry(
        self, record_hash: str, entry: dict[str, Any], *, idempotent: bool
    ) -> AnchorReceipt:
        return AnchorReceipt(
            backend="evm",
            network=f"evm:{entry['chain_id']}:{entry['mode']}",
            record_hash=record_hash,
            idempotent_hit=idempotent,
            ref=dict(entry),
        )
