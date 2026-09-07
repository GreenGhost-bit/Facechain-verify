"""Ephemeral public hosting so SerpAPI Google Lens can crawl the probe image.

SerpAPI cannot accept a raw file upload — it only accepts a public ``url``.
The Elvish-Yadav identification path that worked in practice was:

1. host the probe bytes on a short-lived public file host;
2. call SerpAPI ``google_lens`` with that URL;
3. rank returned visual matches with the same face engine.

This module only does step 1. Hosts are tried in order; the first success wins.
"""

from __future__ import annotations

import httpx

from ..errors import ProviderError
from ..logging import LOG

_TIMEOUT_S = 60.0


def _sniff_name_and_type(image_bytes: bytes) -> tuple[str, str]:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "probe.jpg", "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "probe.png", "image/png"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "probe.webp", "image/webp"
    return "probe.jpg", "image/jpeg"


def _host_catbox(client: httpx.Client, image_bytes: bytes, filename: str, content_type: str) -> str | None:
    resp = client.post(
        "https://catbox.moe/user/api.php",
        data={"reqtype": "fileupload"},
        files={"fileToUpload": (filename, image_bytes, content_type)},
    )
    text = (resp.text or "").strip()
    if resp.status_code == 200 and text.startswith("http"):
        return text
    LOG.warning("probe_host.catbox_failed", status=resp.status_code, body=text[:160])
    return None


def _host_litterbox(client: httpx.Client, image_bytes: bytes, filename: str, content_type: str) -> str | None:
    resp = client.post(
        "https://litterbox.catbox.moe/resources/internals/api.php",
        data={"reqtype": "fileupload", "time": "1h"},
        files={"fileToUpload": (filename, image_bytes, content_type)},
    )
    text = (resp.text or "").strip()
    if resp.status_code == 200 and text.startswith("http"):
        return text
    LOG.warning("probe_host.litterbox_failed", status=resp.status_code, body=text[:160])
    return None


def _host_tmpfiles(client: httpx.Client, image_bytes: bytes, filename: str, content_type: str) -> str | None:
    resp = client.post(
        "https://tmpfiles.org/api/v1/upload",
        files={"file": (filename, image_bytes, content_type)},
    )
    if resp.status_code != 200:
        LOG.warning("probe_host.tmpfiles_failed", status=resp.status_code, body=resp.text[:160])
        return None
    try:
        payload = resp.json()
    except Exception:
        LOG.warning("probe_host.tmpfiles_failed", status=resp.status_code, body=resp.text[:160])
        return None
    url = ""
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            url = str(data.get("url") or "")
    if not url.startswith("http"):
        return None
    # HTML landing page → direct download path Google / SerpAPI can fetch.
    return url.replace("tmpfiles.org/", "tmpfiles.org/dl/", 1)


def host_probe_image(image_bytes: bytes, *, timeout_s: float = _TIMEOUT_S) -> str:
    """Upload ``image_bytes`` and return a public HTTP(S) URL.

    Raises :class:`~facechain.errors.ProviderError` if every hoster fails.
    """
    if not image_bytes:
        raise ProviderError("cannot host empty probe image")
    filename, content_type = _sniff_name_and_type(image_bytes)
    errors: list[str] = []
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        for name, fn in (
            ("catbox", _host_catbox),
            ("litterbox", _host_litterbox),
            ("tmpfiles", _host_tmpfiles),
        ):
            try:
                url = fn(client, image_bytes, filename, content_type)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                LOG.warning("probe_host.failed", host=name, error=str(exc))
                continue
            if url:
                LOG.info("probe_host.ok", host=name, url=url)
                return url
            errors.append(f"{name}: no url")
    raise ProviderError(
        "failed to host probe image for SerpAPI Lens "
        f"({'; '.join(errors) or 'no hosters tried'})"
    )
