"""Wikidata P18 corpus builder (network calls stubbed)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from facechain.config import Settings
from facechain.corpus import _wikidata_p18, fetch_corpus_wikidata

_PIXEL_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300"
    + "08060607060508070707090909080a0c140d0c0b0b0c191213"
    + "0f141d1a1f1e1d1a1c1c20242e2720222c231c1c28372"
    + "9"  # (truncated placeholder; decode not exercised — load_image_bytes is stubbed)
)


class _FakeFetcher:
    """Minimal SafeFetcher stand-in: canned Wikidata JSON + a fixed image."""

    def __init__(self, image: bytes) -> None:
        self._image = image
        self.calls: list[str] = []

    def __enter__(self):  # type: ignore[no-untyped-def]
        return self

    def __exit__(self, *exc):  # type: ignore[no-untyped-def]
        return False

    def get_json(self, url: str, *, params: dict[str, str]) -> dict:
        self.calls.append(params.get("action", ""))
        if params.get("action") == "wbsearchentities":
            return {"search": [{"id": "Q76", "label": params["search"]}]}
        if params.get("action") == "wbgetclaims":
            return {
                "claims": {
                    "P18": [
                        {"mainsnak": {"datavalue": {"value": "Barack Obama portrait.jpg"}}}
                    ]
                }
            }
        return {}

    def fetch_image(self, url: str):  # type: ignore[no-untyped-def]
        self.calls.append(f"img:{url}")
        return MagicMock(content=self._image, final_url=url)


def test_wikidata_p18_resolves_entity_and_filename() -> None:
    f = _FakeFetcher(_PIXEL_JPEG)
    got = _wikidata_p18(f, "Barack Obama")  # type: ignore[arg-type]
    assert got is not None
    qid, filename, url = got
    assert qid == "Q76"
    assert filename == "Barack Obama portrait.jpg"
    assert url == "https://www.wikidata.org/wiki/Q76"


def test_fetch_corpus_wikidata_writes_entry(tmp_path: Path, monkeypatch) -> None:
    import facechain.corpus as corpus_mod

    fake = _FakeFetcher(_PIXEL_JPEG)
    monkeypatch.setattr(corpus_mod, "SafeFetcher", lambda **kw: fake)
    # Skip real image decoding; we only assert the entry + metadata are written.
    monkeypatch.setattr(
        corpus_mod, "load_image_bytes",
        lambda *a, **kw: MagicMock(fingerprint=MagicMock(mime="image/jpeg")),
    )
    settings = Settings.load(env_file="/nonexistent", corpus_dir=tmp_path / "corpus")
    n = fetch_corpus_wikidata(settings, ["Barack Obama"])
    assert n == 1
    metas = list((tmp_path / "corpus").glob("*.json"))
    assert len(metas) == 1
    meta = json.loads(metas[0].read_text())
    assert meta["title"] == "Barack Obama"
    assert meta["wikidata"] == "Q76"
    assert meta["post_url"] == "https://www.wikidata.org/wiki/Q76"
