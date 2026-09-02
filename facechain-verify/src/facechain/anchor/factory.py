"""Anchor-backend selection."""

from __future__ import annotations

from ..config import Settings
from ..errors import BackendUnavailableError
from .base import AnchorBackend
from .local_chain import LocalChain


def build_anchor_backend(settings: Settings) -> AnchorBackend:
    kind = settings.anchor_backend
    if kind == "local":
        return LocalChain(
            settings.chain_dir / "local",
            difficulty_bits=settings.chain_difficulty_bits,
        )
    if kind == "evm":
        from .evm_backend import EVMChainAnchor

        return EVMChainAnchor(
            rpc_url=settings.evm_rpc_url,
            private_key=settings.evm_private_key,
            registry_address=settings.evm_registry_address,
            chain_id=settings.evm_chain_id,
            cache_dir=str(settings.chain_dir),
        )
    raise BackendUnavailableError(f"unknown anchor backend {kind!r}")
