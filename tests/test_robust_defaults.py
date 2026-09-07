"""Tests for the robust default search stack."""

from __future__ import annotations

from pathlib import Path

from facechain.config import ROBUST_SEARCH_PROVIDERS, Settings
from facechain.search.factory import build_providers


def test_robust_default_providers_constant() -> None:
    assert ROBUST_SEARCH_PROVIDERS == ("serpapi", "multiris", "wikimedia", "local")
    s = Settings.load(env_file="/nonexistent")
    assert s.search_providers == ROBUST_SEARCH_PROVIDERS


def test_build_providers_skips_unavailable_keeps_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("FACECHAIN_SERPAPI_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    settings = Settings.load(
        env_file="/nonexistent",
        serpapi_key=None,
        corpus_dir=tmp_path / "empty_corpus",
    )
    # Default stack lists serpapi; without a key it must be skipped, not crash.
    providers = build_providers(settings, strict=False)
    names = [p.name for p in providers]
    assert "serpapi" not in names
    assert "wikimedia" in names or "local" in names or "multiris" in names
