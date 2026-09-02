"""SSRF-hardened HTTP client for pulling untrusted candidate images off the web.

Guarantees enforced on *every* hop (including redirects):

* scheme is ``http``/``https`` only;
* the host resolves exclusively to public, non-loopback, non-link-local,
  non-multicast, non-reserved unicast addresses;
* redirects are followed manually and re-validated, capped at ``max_redirects``;
* the response body is streamed and aborted past ``max_bytes``;
* the ``Content-Type`` is on an allow-list (or the payload sniffs as an image).

Residual caveat: a determined DNS-rebinding attacker could flip the record
between our resolve check and httpx's own connect. For a research pipeline that
only fetches images this is an acceptable, documented risk; a production build
would pin the socket to the vetted IP.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from .errors import UnsafeURLError
from .logging import LOG

_ALLOWED_SCHEMES = {"http", "https"}
_IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "image/gif",
}
_MAGIC = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    content: bytes
    content_type: str
    elapsed_ms: float


def _ip_is_public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def assert_url_is_safe(url: str) -> None:
    """Raise :class:`UnsafeURLError` unless ``url`` is safe to fetch."""
    parts = urlsplit(url)
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        raise UnsafeURLError(f"scheme {parts.scheme!r} not allowed", detail=url)
    host = parts.hostname
    if not host:
        raise UnsafeURLError("URL has no host", detail=url)
    try:
        infos = socket.getaddrinfo(host, parts.port or 0, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"host does not resolve: {exc}", detail=url) from exc
    resolved = {str(info[4][0]) for info in infos}
    if not resolved:
        raise UnsafeURLError("host resolved to nothing", detail=url)
    bad = sorted(ip for ip in resolved if not _ip_is_public(ip))
    if bad:
        raise UnsafeURLError(f"host resolves to non-public address(es): {bad}", detail=url)


def _sniff_image(prefix: bytes) -> str | None:
    for magic, mime in _MAGIC:
        if prefix.startswith(magic):
            return mime
    if prefix[:4] == b"RIFF" and prefix[8:12] == b"WEBP":
        return "image/webp"
    return None


class SafeFetcher:
    """Reusable SSRF-guarded fetcher. One instance per pipeline run."""

    def __init__(
        self,
        *,
        contact: str = "facechain-verify",
        timeout_s: float = 20.0,
        max_redirects: int = 3,
        max_bytes: int = 25 * 1024 * 1024,
    ) -> None:
        self._timeout = timeout_s
        self._max_redirects = max_redirects
        self._max_bytes = max_bytes
        self._ua = f"facechain-verify/1.0 (+https://github.com/; {contact})"
        self._client = httpx.Client(
            follow_redirects=False,
            timeout=timeout_s,
            headers={"User-Agent": self._ua, "Accept": "*/*"},
        )

    def __enter__(self) -> SafeFetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # -- JSON (for search-provider APIs) ----------------------------------
    def get_json(self, url: str, *, params: dict[str, str] | None = None) -> object:
        assert_url_is_safe(url)
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    # -- images ---------------------------------------------------------
    def fetch_image(self, url: str) -> FetchResult:
        """Fetch an image with full redirect re-validation and a streamed cap."""
        current = url
        for hop in range(self._max_redirects + 1):
            assert_url_is_safe(current)
            with self._client.stream("GET", current) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location", "")
                    if not location:
                        raise UnsafeURLError("redirect without Location", detail=current)
                    current = str(httpx.URL(current).join(location))
                    LOG.debug("netfetch.redirect", hop=hop, to=current)
                    continue
                resp.raise_for_status()
                declared = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_bytes(chunk_size=65536):
                    total += len(chunk)
                    if total > self._max_bytes:
                        raise UnsafeURLError(
                            f"response exceeds {self._max_bytes} bytes", detail=current
                        )
                    chunks.append(chunk)
                body = b"".join(chunks)
                sniffed = _sniff_image(body[:16])
                if declared not in _IMAGE_CONTENT_TYPES and sniffed is None:
                    raise UnsafeURLError(
                        f"not an image (content-type={declared!r}, magic mismatch)",
                        detail=current,
                    )
                return FetchResult(
                    url=url,
                    final_url=current,
                    content=body,
                    content_type=sniffed or declared,
                    elapsed_ms=round(resp.elapsed.total_seconds() * 1000, 2),
                )
        raise UnsafeURLError(f"too many redirects (> {self._max_redirects})", detail=url)
