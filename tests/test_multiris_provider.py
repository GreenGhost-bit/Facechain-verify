"""Unit tests for the multiris (PicImageSearch) provider mapping helpers."""

from __future__ import annotations

from types import SimpleNamespace

from facechain.search.multiris_provider import (
    MultiRisProvider,
    _dedupe,
    candidate_from_hit,
    extract_bing,
    extract_google_lens,
    extract_tineye,
    extract_yandex,
)


def test_candidate_from_hit_requires_http_image() -> None:
    assert candidate_from_hit(engine="yandex", post_url="https://x", image_url="") is None
    assert candidate_from_hit(engine="yandex", post_url="https://x", image_url="ftp://z") is None
    c = candidate_from_hit(
        engine="yandex",
        post_url="https://example.com/post",
        image_url="https://cdn.example.com/a.jpg",
        title="Hello",
    )
    assert c is not None
    assert c.provider == "multiris/yandex"
    assert c.post_url.endswith("/post")
    assert c.image_url.endswith("/a.jpg")


def test_extract_yandex_maps_raw_items() -> None:
    resp = SimpleNamespace(
        raw=[
            SimpleNamespace(
                url="https://news.example/a",
                thumbnail="https://img.example/a.jpg",
                title="A",
                source="news.example",
                content="",
            ),
            SimpleNamespace(url="https://x", thumbnail="", title="bad", source="", content=""),
        ]
    )
    out = extract_yandex(resp, limit=10)
    assert len(out) == 1
    assert out[0].provider == "multiris/yandex"
    assert out[0].image_url == "https://img.example/a.jpg"


def test_extract_bing_prefers_pages_including() -> None:
    resp = SimpleNamespace(
        pages_including=[
            SimpleNamespace(
                url="https://site/p",
                image_url="https://cdn/p.jpg",
                name="Page",
                thumbnail="",
            )
        ],
        visual_search=[
            SimpleNamespace(
                url="https://site/v",
                image_url="https://cdn/v.jpg",
                name="Visual",
                thumbnail="",
            )
        ],
        raw=[],
    )
    out = extract_bing(resp, limit=10)
    assert [c.image_url for c in out] == ["https://cdn/p.jpg", "https://cdn/v.jpg"]
    assert out[0].provider == "multiris/bing"


def test_extract_tineye_and_google_lens() -> None:
    tineye = SimpleNamespace(
        raw=[
            SimpleNamespace(
                url="https://blog/x",
                image_url="https://cdn/x.jpg",
                thumbnail="https://cdn/x-thumb.jpg",
                domain="blog.example",
                title="",
            )
        ]
    )
    lens = SimpleNamespace(
        raw=[
            SimpleNamespace(
                url="https://linkedin.com/in/someone",
                thumbnail="https://media.licdn.com/photo.jpg",
                title="Someone",
                site_name="LinkedIn",
            )
        ]
    )
    t = extract_tineye(tineye, limit=5)
    g = extract_google_lens(lens, limit=5)
    assert len(t) == 1 and t[0].provider == "multiris/tineye"
    assert len(g) == 1 and g[0].provider == "multiris/google_lens"
    assert "linkedin" in g[0].post_url


def test_dedupe_keeps_first() -> None:
    a = candidate_from_hit(
        engine="yandex",
        post_url="https://a",
        image_url="https://cdn/same.jpg",
        title="1",
    )
    b = candidate_from_hit(
        engine="bing",
        post_url="https://b",
        image_url="https://cdn/same.jpg",
        title="2",
    )
    assert a and b
    out = _dedupe([a, b])
    assert len(out) == 1
    assert out[0].provider == "multiris/yandex"


def test_multiris_available_reflects_import(monkeypatch) -> None:
    from facechain.config import Settings
    from facechain.search import multiris_provider as mod

    monkeypatch.setattr(mod, "_picimage_available", lambda: False)
    assert MultiRisProvider.available(Settings()) is False
    monkeypatch.setattr(mod, "_picimage_available", lambda: True)
    assert MultiRisProvider.available(Settings()) is True


def test_multiris_search_soft_fails_engines(monkeypatch, tmp_path) -> None:
    from facechain.config import Settings
    from facechain.search import multiris_provider as mod
    from facechain.search.base import ProbeContext

    monkeypatch.setattr(mod, "_picimage_available", lambda: True)

    def run_engine(name: str, file_path, *, limit: int):
        if name == "yandex":
            return [
                candidate_from_hit(
                    engine="yandex",
                    post_url="https://y",
                    image_url="https://cdn/y.jpg",
                )
            ]
        return []

    monkeypatch.setattr(mod, "_run_engine", run_engine)

    class DummyEngine:
        name = "dummy"

    settings = Settings(max_candidates_per_provider=5)
    # minimal ProbeContext — fetcher/face unused by provider
    probe = ProbeContext(
        image_bytes=b"\xff\xd8\xff" + b"\x00" * 32,  # jpeg-ish header
        rgb=__import__("numpy").zeros((8, 8, 3), dtype="uint8"),
        embedding=__import__("numpy").zeros(8, dtype="float32"),
        settings=settings,
        fetcher=None,  # type: ignore[arg-type]
        face_engine=DummyEngine(),  # type: ignore[arg-type]
    )
    hits = list(MultiRisProvider().search(probe))
    assert len(hits) == 1
    assert hits[0].provider == "multiris/yandex"
