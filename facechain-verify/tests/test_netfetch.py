from __future__ import annotations

import socket

import pytest

from facechain.errors import UnsafeURLError
from facechain.netfetch import _sniff_image, assert_url_is_safe


@pytest.fixture
def fake_dns(monkeypatch: pytest.MonkeyPatch):
    """Map any hostname to a caller-chosen IP without touching the network."""
    mapping: dict[str, str] = {}

    def _getaddrinfo(host, port, *a, **kw):
        ip = mapping.get(host, "93.184.216.34")  # a real public unicast address
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port or 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _getaddrinfo)
    return mapping


@pytest.mark.parametrize(
    "host,ip",
    [
        ("evil.internal", "127.0.0.1"),
        ("evil.internal", "10.0.0.5"),
        ("evil.internal", "192.168.1.1"),
        ("evil.internal", "169.254.169.254"),  # cloud metadata endpoint
        ("evil.internal", "0.0.0.0"),
        ("evil.internal", "::1"),
        ("evil.internal", "fd00::1"),
    ],
)
def test_rejects_non_public_targets(fake_dns, host: str, ip: str):
    fake_dns[host] = ip
    with pytest.raises(UnsafeURLError):
        assert_url_is_safe(f"https://{host}/image.jpg")


def test_allows_public_target(fake_dns):
    fake_dns["cdn.example.com"] = "93.184.216.34"
    assert_url_is_safe("https://cdn.example.com/a.png")  # no raise


def test_rejects_non_http_schemes():
    with pytest.raises(UnsafeURLError):
        assert_url_is_safe("file:///etc/passwd")
    with pytest.raises(UnsafeURLError):
        assert_url_is_safe("ftp://example.com/x")


def test_rejects_url_without_host():
    with pytest.raises(UnsafeURLError):
        assert_url_is_safe("https:///nohost")


def test_rejects_unresolvable_host(monkeypatch: pytest.MonkeyPatch):
    def _boom(*a, **kw):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", _boom)
    with pytest.raises(UnsafeURLError):
        assert_url_is_safe("https://does-not-exist.invalid/x")


def test_image_magic_sniffer():
    assert _sniff_image(b"\xff\xd8\xff\xe0abcd") == "image/jpeg"
    assert _sniff_image(b"\x89PNG\r\n\x1a\n....") == "image/png"
    assert _sniff_image(b"RIFF____WEBPVP8 ") == "image/webp"
    assert _sniff_image(b"not an image") is None


@pytest.mark.network
def test_real_public_fetch_of_bundled_style_image():
    """Sanity check against a real CDN; skip with -m 'not network'."""
    from facechain.netfetch import SafeFetcher

    url = "https://upload.wikimedia.org/wikipedia/commons/9/9d/Barack_Obama.jpg"
    with SafeFetcher(contact="pytest") as f:
        res = f.fetch_image(url)
    assert res.content[:3] == b"\xff\xd8\xff"
    assert res.content_type == "image/jpeg"
