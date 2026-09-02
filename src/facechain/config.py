"""Typed, validated runtime configuration.

Precedence (low -> high): dataclass defaults < ``.env`` file < process environment
< explicit constructor kwargs (CLI flags). No third-party settings loader: a tiny
``.env`` parser keeps the dependency surface minimal.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .errors import ConfigError

_ENV_PREFIX = "FACECHAIN_"


def load_dotenv(path: str | os.PathLike[str] = ".env") -> dict[str, str]:
    """Parse a minimal ``KEY=VALUE`` .env file. Missing file -> empty dict."""
    p = Path(path)
    out: dict[str, str] = {}
    if not p.is_file():
        return out
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            out[key] = value
    return out


class Settings(BaseModel):
    """Effective configuration for one pipeline invocation."""

    model_config = {"frozen": True, "extra": "forbid"}

    # -- workspace ----------------------------------------------------------
    runs_dir: Path = Field(default=Path("runs"))
    chain_dir: Path = Field(default=Path("chaindata"))
    corpus_dir: Path = Field(default=Path("data/corpus"))

    # -- imaging safety ---------------------------------------------------
    max_image_bytes: int = Field(default=25 * 1024 * 1024, ge=1024)
    max_image_pixels: int = Field(default=40_000_000, ge=10_000)
    min_face_pixels: int = Field(default=48, ge=16)

    # -- face matching --------------------------------------------------
    face_engine: str = Field(default="auto")  # auto | opencv | numpy | insightface
    match_threshold: float = Field(default=0.86, ge=0.0, le=1.0)
    ambiguous_margin: float = Field(default=0.04, ge=0.0, le=1.0)

    # -- search --------------------------------------------------------
    search_providers: tuple[str, ...] = Field(default=("wikimedia", "local"))
    max_candidates_per_provider: int = Field(default=12, ge=1, le=100)
    serpapi_key: str | None = None
    google_credentials: str | None = None
    http_contact: str = "facechain-verify (contact: unset)"
    http_timeout_s: float = Field(default=20.0, gt=0)
    http_max_redirects: int = Field(default=3, ge=0, le=10)

    # -- anchoring -----------------------------------------------------
    anchor_backend: str = Field(default="local")  # local | evm
    chain_difficulty_bits: int = Field(default=0, ge=0, le=28)
    evm_rpc_url: str | None = None
    evm_private_key: str | None = None
    evm_registry_address: str | None = None
    evm_chain_id: int | None = None

    @field_validator("search_providers", mode="before")
    @classmethod
    def _split_providers(cls, v: Any) -> Any:
        if isinstance(v, str):
            return tuple(p.strip() for p in v.split(",") if p.strip())
        return v

    @field_validator("face_engine")
    @classmethod
    def _known_engine(cls, v: str) -> str:
        allowed = {"auto", "opencv", "numpy", "insightface"}
        if v not in allowed:
            raise ValueError(f"face_engine must be one of {sorted(allowed)}")
        return v

    @field_validator("anchor_backend")
    @classmethod
    def _known_backend(cls, v: str) -> str:
        allowed = {"local", "evm"}
        if v not in allowed:
            raise ValueError(f"anchor_backend must be one of {sorted(allowed)}")
        return v

    # -- construction --------------------------------------------------
    @classmethod
    def load(cls, env_file: str | os.PathLike[str] = ".env", **overrides: Any) -> Settings:
        """Build settings from .env + os.environ, then apply explicit overrides."""
        merged: dict[str, str] = {}
        merged.update(load_dotenv(env_file))
        merged.update(os.environ)

        def pick(*keys: str) -> str | None:
            for k in keys:
                if merged.get(k):
                    return merged[k]
            return None

        data: dict[str, Any] = {}
        if (v := pick(f"{_ENV_PREFIX}RUNS_DIR")) is not None:
            data["runs_dir"] = Path(v)
        if (v := pick(f"{_ENV_PREFIX}CHAIN_DIR")) is not None:
            data["chain_dir"] = Path(v)
        if (v := pick(f"{_ENV_PREFIX}CORPUS_DIR")) is not None:
            data["corpus_dir"] = Path(v)
        if (v := pick(f"{_ENV_PREFIX}FACE_ENGINE")) is not None:
            data["face_engine"] = v
        if (v := pick(f"{_ENV_PREFIX}MATCH_THRESHOLD")) is not None:
            data["match_threshold"] = float(v)
        if (v := pick(f"{_ENV_PREFIX}SEARCH_PROVIDERS")) is not None:
            data["search_providers"] = v
        if (v := pick(f"{_ENV_PREFIX}SERPAPI_KEY", "SERPAPI_KEY")) is not None:
            data["serpapi_key"] = v
        if (v := pick("GOOGLE_APPLICATION_CREDENTIALS")) is not None:
            data["google_credentials"] = v
        if (v := pick(f"{_ENV_PREFIX}HTTP_CONTACT")) is not None:
            data["http_contact"] = v
        if (v := pick(f"{_ENV_PREFIX}ANCHOR_BACKEND")) is not None:
            data["anchor_backend"] = v
        if (v := pick(f"{_ENV_PREFIX}CHAIN_DIFFICULTY_BITS")) is not None:
            data["chain_difficulty_bits"] = int(v)
        if (v := pick(f"{_ENV_PREFIX}EVM_RPC_URL")) is not None:
            data["evm_rpc_url"] = v
        if (v := pick(f"{_ENV_PREFIX}EVM_PRIVATE_KEY")) is not None:
            data["evm_private_key"] = v
        if (v := pick(f"{_ENV_PREFIX}EVM_REGISTRY_ADDRESS")) is not None:
            data["evm_registry_address"] = v
        if (v := pick(f"{_ENV_PREFIX}EVM_CHAIN_ID")) is not None:
            data["evm_chain_id"] = int(v)

        data.update({k: v for k, v in overrides.items() if v is not None})
        try:
            return cls(**data)
        except Exception as exc:
            raise ConfigError("invalid configuration", detail=str(exc)) from exc

    def with_(self, **overrides: Any) -> Settings:
        clean = {k: v for k, v in overrides.items() if v is not None}
        return self.model_copy(update=clean)
