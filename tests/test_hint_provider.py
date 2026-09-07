"""Unit tests for the hint (name / profile-URL) candidate provider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from facechain.search.hint_provider import (
    HintProvider,
    extract_meta_image_urls,
    hint_queries,
    looks_like_url,
    normalize_hint_url,
    slug_to_display_name,
)


def test_looks_like_url_and_normalize() -> None:
    assert looks_like_url("https://www.linkedin.com/in/rakesh-munikoti-816535353/")
    assert looks_like_url("linkedin.com/in/rakesh-munikoti-816535353")
    assert not looks_like_url("Rakesh Munikoti")
    assert normalize_hint_url("linkedin.com/in/foo-bar") == "https://linkedin.com/in/foo-bar"
    assert normalize_hint_url("https://x.com/a") == "https://x.com/a"


def test_slug_to_display_name() -> None:
    assert slug_to_display_name("rakesh-munikoti-816535353") == "rakesh munikoti"
    assert slug_to_display_name("karthikeya-atthavelli") == "karthikeya atthavelli"


def test_hint_queries_from_linkedin_url() -> None:
    q = hint_queries("https://www.linkedin.com/in/rakesh-munikoti-816535353/")
    assert q[0].startswith("https://")
    assert "rakesh munikoti" in q
    assert any("site:linkedin.com" in x for x in q)


def test_hint_queries_from_name() -> None:
    q = hint_queries("Rakesh Munikoti")
    assert q[0] == "Rakesh Munikoti"
    assert "Rakesh Munikoti site:linkedin.com" in q


def test_extract_meta_image_urls_both_attr_orders() -> None:
    html = """
    <html><head>
      <meta property="og:image" content="https://cdn.example/a.jpg" />
      <meta content='https://cdn.example/b.jpg' property='twitter:image' />
      <link rel="image_src" href="https://cdn.example/c.jpg" />
      <meta property="og:title" content="nope" />
    </head></html>
    """
    urls = extract_meta_image_urls(html, base_url="https://example.com/p")
    assert urls == [
        "https://cdn.example/a.jpg",
        "https://cdn.example/b.jpg",
        "https://cdn.example/c.jpg",
    ]


def test_extract_meta_resolves_relative() -> None:
    html = '<meta property="og:image" content="/img/face.png">'
    urls = extract_meta_image_urls(html, base_url="https://example.com/in/x/")
    assert urls == ["https://example.com/img/face.png"]


def test_hint_provider_skips_without_hint() -> None:
    probe = SimpleNamespace(hint=None, settings=SimpleNamespace(max_candidates_per_provider=8))
    assert list(HintProvider().search(probe)) == []  # type: ignore[arg-type]


def test_hint_provider_og_from_page() -> None:
    html = '<meta property="og:image" content="https://media.licdn.com/head.jpg">'
    fetcher = MagicMock()
    fetcher.get_html.return_value = html
    settings = SimpleNamespace(max_candidates_per_provider=8, serpapi_key=None)
    probe = SimpleNamespace(
        hint="https://www.linkedin.com/in/rakesh-munikoti-816535353/",
        settings=settings,
        fetcher=fetcher,
    )
    out = list(HintProvider().search(probe))  # type: ignore[arg-type]
    assert len(out) == 1
    assert out[0].provider == "hint/og"
    assert out[0].image_url.endswith("head.jpg")
    assert "linkedin.com/in/rakesh" in out[0].post_url


def test_hint_provider_serpapi_images() -> None:
    fetcher = MagicMock()
    fetcher.get_json.return_value = {
        "images_results": [
            {
                "original": "https://cdn.example/rakesh.jpg",
                "link": "https://www.linkedin.com/in/rakesh-munikoti-816535353/",
                "title": "Rakesh",
            }
        ]
    }
    settings = SimpleNamespace(max_candidates_per_provider=8, serpapi_key="test-key")
    probe = SimpleNamespace(hint="Rakesh Munikoti", settings=settings, fetcher=fetcher)
    out = list(HintProvider().search(probe))  # type: ignore[arg-type]
    assert out
    assert out[0].provider == "hint/serpapi"
    assert out[0].image_url.endswith("rakesh.jpg")
    assert fetcher.get_json.called
