"""Unit tests for probe auto-hosting and SerpAPI auto-host search path."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from facechain.errors import ProviderError
from facechain.search.probe_host import _sniff_name_and_type, host_probe_image
from facechain.search.serpapi_provider import SerpApiProvider


def test_sniff_png_and_jpeg() -> None:
    assert _sniff_name_and_type(b"\x89PNG\r\n\x1a\n....") == ("probe.png", "image/png")
    assert _sniff_name_and_type(b"\xff\xd8\xff....") == ("probe.jpg", "image/jpeg")


def test_host_probe_image_uses_first_successful_host() -> None:
    fake = MagicMock()
    with patch("facechain.search.probe_host.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value = fake
        with patch(
            "facechain.search.probe_host._host_catbox",
            return_value="https://files.catbox.moe/abc.png",
        ) as catbox:
            with patch("facechain.search.probe_host._host_litterbox") as litter:
                url = host_probe_image(b"\x89PNG\r\n\x1a\nxxxx")
    assert url == "https://files.catbox.moe/abc.png"
    catbox.assert_called_once()
    litter.assert_not_called()


def test_host_probe_image_falls_through_and_errors() -> None:
    with patch("facechain.search.probe_host.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value = MagicMock()
        with patch("facechain.search.probe_host._host_catbox", return_value=None):
            with patch("facechain.search.probe_host._host_litterbox", return_value=None):
                with patch("facechain.search.probe_host._host_tmpfiles", return_value=None):
                    with pytest.raises(ProviderError, match="failed to host"):
                        host_probe_image(b"\xff\xd8\xffxxxx")


def test_tmpfiles_dl_rewrite() -> None:
    client = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "status": "success",
        "data": {"url": "https://tmpfiles.org/abc123/probe.png"},
    }
    client.post.return_value = resp
    from facechain.search.probe_host import _host_tmpfiles

    url = _host_tmpfiles(client, b"\xff\xd8\xff", "probe.jpg", "image/jpeg")
    assert url == "https://tmpfiles.org/dl/abc123/probe.png"


def test_serpapi_auto_hosts_when_no_probe_url() -> None:
    fetcher = MagicMock()
    fetcher.get_json.return_value = {
        "visual_matches": [
            {
                "title": "Elvish Yadav",
                "link": "https://www.instagram.com/reel/abc/",
                "thumbnail": "https://cdn.example/elvish.jpg",
                "source": "Instagram",
            }
        ]
    }
    settings = SimpleNamespace(
        serpapi_key="k",
        max_candidates_per_provider=12,
        http_timeout_s=20.0,
    )
    probe = SimpleNamespace(
        image_bytes=b"\x89PNG\r\n\x1a\nxxxx",
        settings=settings,
        fetcher=fetcher,
        extra={},
    )
    with patch(
        "facechain.search.serpapi_provider.host_probe_image",
        return_value="https://files.catbox.moe/probe.png",
    ) as host:
        out = list(SerpApiProvider().search(probe))  # type: ignore[arg-type]
    host.assert_called_once()
    assert probe.extra["probe_image_url"] == "https://files.catbox.moe/probe.png"
    assert len(out) == 1
    assert out[0].title == "Elvish Yadav"
    assert "elvish.jpg" in out[0].image_url
    assert fetcher.get_json.call_args.kwargs["params"]["url"] == "https://files.catbox.moe/probe.png"
    assert fetcher.get_json.call_args.kwargs["params"]["engine"] == "google_lens"


def test_serpapi_uses_explicit_probe_url_without_hosting() -> None:
    fetcher = MagicMock()
    fetcher.get_json.return_value = {"visual_matches": []}
    settings = SimpleNamespace(serpapi_key="k", max_candidates_per_provider=12, http_timeout_s=20.0)
    probe = SimpleNamespace(
        image_bytes=b"\xff\xd8\xff",
        settings=settings,
        fetcher=fetcher,
        extra={"probe_image_url": "https://example.com/mine.jpg"},
    )
    with patch("facechain.search.serpapi_provider.host_probe_image") as host:
        list(SerpApiProvider().search(probe))  # type: ignore[arg-type]
    host.assert_not_called()
    assert fetcher.get_json.call_args.kwargs["params"]["url"] == "https://example.com/mine.jpg"
