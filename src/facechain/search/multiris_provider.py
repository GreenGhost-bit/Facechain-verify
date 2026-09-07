"""Multi-engine reverse image search via PicImageSearch.

Architecture borrowed from kitUIN/PicImageSearch (MIT): fan out the probe to
several public reverse-image engines, then let facechain's aggregator decide the
winner by face-embedding similarity (GReverse-style face filter, but with our
InsightFace / OpenCV engines).

Engines used (file upload -- no public probe URL required):

* Yandex Images
* Bing Visual Search
* TinEye
* Google Lens

Anime / adult scrapers (SauceNAO, IQDB, TraceMoe, E-Hentai, …) are intentionally
skipped. Private Instagram / Facebook content is still unreachable; this only
improves coverage of *public, indexed* pages.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..config import Settings
from ..logging import LOG
from .base import ProbeContext, RawCandidate

# Soft per-run score budget is enforced in the aggregator; each engine still
# respects ``settings.max_candidates_per_provider``.
_ENGINES: tuple[str, ...] = ("yandex", "bing", "tineye", "google_lens")


def _picimage_available() -> bool:
    try:
        import PicImageSearch  # noqa: F401
        from PicImageSearch.sync import Bing, GoogleLens, Tineye, Yandex  # noqa: F401

        return True
    except Exception:
        return False


def _attr(obj: Any, *names: str, default: str = "") -> str:
    for name in names:
        val = getattr(obj, name, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return default


def candidate_from_hit(
    *,
    engine: str,
    post_url: str,
    image_url: str,
    title: str = "",
    snippet: str = "",
) -> RawCandidate | None:
    """Build a :class:`RawCandidate` or return ``None`` if URLs are unusable."""
    image = (image_url or "").strip()
    post = (post_url or "").strip() or image
    if not image:
        return None
    if not (image.startswith("http://") or image.startswith("https://")):
        return None
    return RawCandidate(
        provider=f"multiris/{engine}",
        post_url=post if post.startswith("http") else image,
        image_url=image,
        title=title or "",
        snippet=snippet or engine,
    )


def extract_yandex(resp: Any, *, limit: int) -> list[RawCandidate]:
    out: list[RawCandidate] = []
    for item in list(getattr(resp, "raw", None) or [])[:limit]:
        c = candidate_from_hit(
            engine="yandex",
            post_url=_attr(item, "url"),
            image_url=_attr(item, "thumbnail", "image_url"),
            title=_attr(item, "title"),
            snippet=_attr(item, "source", "content"),
        )
        if c:
            out.append(c)
    return out


def extract_bing(resp: Any, *, limit: int) -> list[RawCandidate]:
    out: list[RawCandidate] = []
    # Prefer pages that include the image (closer to exact reverse hits), then similar.
    buckets: list[Any] = []
    buckets.extend(getattr(resp, "pages_including", None) or [])
    buckets.extend(getattr(resp, "visual_search", None) or [])
    buckets.extend(getattr(resp, "raw", None) or [])
    for item in buckets:
        if len(out) >= limit:
            break
        c = candidate_from_hit(
            engine="bing",
            post_url=_attr(item, "url", "hostPageUrl"),
            image_url=_attr(item, "image_url", "contentUrl", "thumbnail"),
            title=_attr(item, "title", "name"),
            snippet="bing",
        )
        if c:
            out.append(c)
    return out


def extract_tineye(resp: Any, *, limit: int) -> list[RawCandidate]:
    out: list[RawCandidate] = []
    for item in list(getattr(resp, "raw", None) or [])[:limit]:
        c = candidate_from_hit(
            engine="tineye",
            post_url=_attr(item, "url"),
            image_url=_attr(item, "image_url", "thumbnail"),
            title=_attr(item, "domain", "title"),
            snippet="tineye",
        )
        if c:
            out.append(c)
    return out


def extract_google_lens(resp: Any, *, limit: int) -> list[RawCandidate]:
    out: list[RawCandidate] = []
    items = list(getattr(resp, "raw", None) or [])
    # Exact-match response type exposes the same ``raw`` list of items.
    for item in items[:limit]:
        c = candidate_from_hit(
            engine="google_lens",
            post_url=_attr(item, "url"),
            image_url=_attr(item, "thumbnail", "image_url", "image"),
            title=_attr(item, "title"),
            snippet=_attr(item, "site_name", "source") or "google_lens",
        )
        if c:
            out.append(c)
    return out


def _dedupe(cands: list[RawCandidate]) -> list[RawCandidate]:
    seen: set[str] = set()
    out: list[RawCandidate] = []
    for c in cands:
        key = c.key()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _run_engine(name: str, file_path: Path, *, limit: int) -> list[RawCandidate]:
    """Call one PicImageSearch sync engine; never raises to the caller."""
    try:
        from PicImageSearch.sync import Bing, GoogleLens, Tineye, Yandex
    except Exception as exc:
        LOG.warning("search.multiris.import_failed", error=str(exc))
        return []

    try:
        if name == "yandex":
            resp = Yandex().search(file=str(file_path))
            return extract_yandex(resp, limit=limit)
        if name == "bing":
            resp = Bing().search(file=str(file_path))
            return extract_bing(resp, limit=limit)
        if name == "tineye":
            resp = Tineye().search(file=str(file_path))
            return extract_tineye(resp, limit=limit)
        if name == "google_lens":
            resp = GoogleLens(search_type="all", hl="en", country="US").search(file=str(file_path))
            return extract_google_lens(resp, limit=limit)
    except Exception as exc:
        LOG.warning("search.multiris.engine_failed", engine=name, error=str(exc))
        return []
    return []


class MultiRisProvider:
    """Fan-out reverse-image provider backed by PicImageSearch."""

    name = "multiris"

    @classmethod
    def available(cls, settings: Settings) -> bool:
        return _picimage_available()

    def search(self, probe: ProbeContext) -> Iterable[RawCandidate]:
        if not _picimage_available():
            LOG.warning("search.multiris.skipped", reason="PicImageSearch not installed")
            return []

        limit = probe.settings.max_candidates_per_provider
        suffix = ".jpg"
        # Preserve a plausible extension for engines that sniff content-type from name.
        raw = probe.reverse_image_bytes()
        if raw[:8] == b"\x89PNG\r\n\x1a\n":
            suffix = ".png"
        elif raw[:2] == b"\xff\xd8":
            suffix = ".jpg"
        elif raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
            suffix = ".webp"

        collected: list[RawCandidate] = []
        with tempfile.TemporaryDirectory(prefix="facechain-ris-") as tmp:
            path = Path(tmp) / f"probe{suffix}"
            path.write_bytes(raw)
            for engine in _ENGINES:
                with LOG.span("search.multiris.engine", engine=engine) as sp:
                    hits = _run_engine(engine, path, limit=limit)
                    sp["hits"] = len(hits)
                    LOG.info("search.multiris", engine=engine, candidates=len(hits))
                    collected.extend(hits)

        return _dedupe(collected)
