from __future__ import annotations

from pathlib import Path

import pytest

from facechain.config import Settings, load_dotenv
from facechain.errors import ConfigError


def test_dotenv_parsing(tmp_path: Path):
    p = tmp_path / ".env"
    p.write_text(
        "# a comment\n"
        "FACECHAIN_SERPAPI_KEY = abc123 \n"
        'FACECHAIN_HTTP_CONTACT="me@example.com"\n'
        "BLANK=\n"
        "no_equals_line\n"
    )
    env = load_dotenv(p)
    assert env["FACECHAIN_SERPAPI_KEY"] == "abc123"
    assert env["FACECHAIN_HTTP_CONTACT"] == "me@example.com"
    assert "no_equals_line" not in env


def test_env_file_feeds_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FACECHAIN_SERPAPI_KEY", raising=False)
    p = tmp_path / ".env"
    p.write_text("FACECHAIN_SERPAPI_KEY=k-from-env\nFACECHAIN_MATCH_THRESHOLD=0.7\n")
    s = Settings.load(env_file=str(p))
    assert s.serpapi_key == "k-from-env"
    assert s.match_threshold == pytest.approx(0.7)


def test_explicit_override_beats_env(tmp_path: Path):
    p = tmp_path / ".env"
    p.write_text("FACECHAIN_MATCH_THRESHOLD=0.7\n")
    s = Settings.load(env_file=str(p), match_threshold=0.9)
    assert s.match_threshold == pytest.approx(0.9)


def test_providers_string_is_split():
    s = Settings.load(env_file="/nonexistent", search_providers="wikimedia, local ,serpapi")
    assert s.search_providers == ("wikimedia", "local", "serpapi")


def test_invalid_engine_raises_config_error():
    with pytest.raises(ConfigError):
        Settings.load(env_file="/nonexistent", face_engine="magic")


def test_invalid_threshold_raises_config_error():
    with pytest.raises(ConfigError):
        Settings.load(env_file="/nonexistent", match_threshold=5.0)


def test_settings_is_frozen():
    s = Settings.load(env_file="/nonexistent")
    with pytest.raises(Exception):
        s.match_threshold = 0.1  # type: ignore[misc]


def test_seed_demo_corpus_creates_entries(tmp_path: Path):
    from facechain.corpus import seed_demo_corpus

    s = Settings.load(env_file="/nonexistent", corpus_dir=tmp_path / "corpus")
    n = seed_demo_corpus(s, repo_root=Path(__file__).parent.parent)
    assert n >= 3
    imgs = list((tmp_path / "corpus").glob("*.jpg"))
    metas = list((tmp_path / "corpus").glob("*.json"))
    assert len(imgs) == n == len(metas)
    import json

    meta = json.loads(metas[0].read_text())
    assert meta["post_url"].startswith("http")
