"""Inject candidate headshots from a ``--hint`` (name or profile URL).

Reverse-image engines only score photos that are already indexed. LinkedIn and
other private CDNs often are not — so a correct identity never enters the pool.
This provider bridges that gap: when the user supplies a name or profile URL,
we fetch likely headshot URLs (page ``og:image`` / SerpAPI Google Images) and
hand them to the same ArcFace ranker. The hint never decides the match; it only
widens the candidate set.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from urllib.parse import unquote, urljoin, urlsplit

from ..config import Settings
from ..logging import LOG
from .base import ProbeContext, RawCandidate
from .serpapi_provider import SerpApiProvider

_ENDPOINT = "https://serpapi.com/search.json"
_META_PROPS = {
    "og:image",
    "og:image:secure_url",
    "twitter:image",
    "twitter:image:src",
}
_ATTR_RE = re.compile(
    r"""([^\s=]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""",
    re.IGNORECASE,
)
_META_TAG_RE = re.compile(r"<meta\s+[^>]+>", re.IGNORECASE)
_LINK_TAG_RE = re.compile(r"<link\s+[^>]+>", re.IGNORECASE)
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_LINKEDIN_IN_RE = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/([^/?#]+)/?",
    re.IGNORECASE,
)


def looks_like_url(hint: str) -> bool:
    text = hint.strip()
    if _URL_RE.match(text):
        return True
    return bool(_LINKEDIN_IN_RE.fullmatch(text) or text.lower().startswith("linkedin.com/"))


def normalize_hint_url(hint: str) -> str:
    text = hint.strip()
    if _URL_RE.match(text):
        return text
    if text.lower().startswith("linkedin.com/"):
        return "https://" + text
    m = _LINKEDIN_IN_RE.search(text)
    if m:
        return f"https://www.linkedin.com/in/{m.group(1)}/"
    return text


def _tag_attrs(tag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in _ATTR_RE.finditer(tag):
        key = m.group(1).lower()
        val = m.group(2) if m.group(2) is not None else (
            m.group(3) if m.group(3) is not None else (m.group(4) or "")
        )
        out[key] = val
    return out


def extract_meta_image_urls(html: str, *, base_url: str) -> list[str]:
    """Pull ``og:image`` / Twitter card / image_src URLs from HTML."""
    found: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        url = urljoin(base_url, raw.strip())
        if not url.lower().startswith(("http://", "https://")):
            return
        key = url.lower()
        if key in seen:
            return
        seen.add(key)
        found.append(url)

    for tag in _META_TAG_RE.findall(html):
        attrs = _tag_attrs(tag)
        prop = (attrs.get("property") or attrs.get("name") or "").strip().lower()
        if prop in _META_PROPS:
            content = attrs.get("content") or ""
            if content:
                add(content)

    for tag in _LINK_TAG_RE.findall(html):
        attrs = _tag_attrs(tag)
        rel = (attrs.get("rel") or "").strip().lower()
        if rel in {"image_src", "icon", "apple-touch-icon"}:
            href = attrs.get("href") or ""
            if href and rel == "image_src":
                add(href)

    return found


def slug_to_display_name(slug: str) -> str:
    """``rakesh-munikoti-816535353`` → ``rakesh munikoti``."""
    cleaned = unquote(slug).strip().strip("/")
    cleaned = re.sub(r"-\d{5,}$", "", cleaned)
    return re.sub(r"[-_]+", " ", cleaned).strip()


def hint_queries(hint: str) -> list[str]:
    """Build Google Images queries from a free-text hint or profile URL."""
    text = hint.strip()
    if not text:
        return []
    if looks_like_url(text):
        url = normalize_hint_url(text)
        queries = [url]
        m = _LINKEDIN_IN_RE.search(url)
        if m:
            name = slug_to_display_name(m.group(1))
            if name:
                queries.append(name)
                queries.append(f"{name} site:linkedin.com")
                queries.append(f'"{name}" linkedin')
        return list(dict.fromkeys(queries))
    return list(dict.fromkeys([text, f"{text} site:linkedin.com"]))


def _http_image(url: str) -> bool:
    return url.lower().startswith(("http://", "https://"))


class HintProvider:
    """Gather headshot candidates from ``probe.hint`` only."""

    name = "hint"

    @classmethod
    def available(cls, settings: Settings) -> bool:
        return True

    def search(self, probe: ProbeContext) -> Iterable[RawCandidate]:
        hint = (probe.hint or "").strip()
        if not hint:
            LOG.info("search.hint.skipped", reason="no --hint")
            return []

        limit = probe.settings.max_candidates_per_provider
        out: list[RawCandidate] = []
        seen: set[str] = set()

        def push(cand: RawCandidate) -> None:
            key = cand.key()
            if not key or key in seen:
                return
            seen.add(key)
            out.append(cand)

        if looks_like_url(hint):
            for cand in self._from_profile_page(probe, normalize_hint_url(hint), limit=limit):
                push(cand)

        if probe.settings.serpapi_key:
            for query in hint_queries(hint):
                if len(out) >= limit:
                    break
                for cand in self._from_serpapi_images(
                    probe, query, limit=max(1, limit - len(out))
                ):
                    push(cand)
                    if len(out) >= limit:
                        break
        elif not out:
            LOG.warning(
                "search.hint.no_serpapi",
                reason="set FACECHAIN_SERPAPI_KEY for name→image search; "
                "profile URL og:image still tried when hint is a URL",
            )

        LOG.info("search.hint", hint=hint[:80], candidates=len(out))
        return out[:limit]

    def _from_profile_page(
        self, probe: ProbeContext, url: str, *, limit: int
    ) -> list[RawCandidate]:
        try:
            html = probe.fetcher.get_html(url)
        except Exception as exc:
            LOG.warning("search.hint.page_failed", url=url, error=str(exc))
            return []

        images = extract_meta_image_urls(html, base_url=url)
        host = (urlsplit(url).hostname or "").lower()
        out: list[RawCandidate] = []
        for image in images:
            if not _http_image(image):
                continue
            out.append(
                RawCandidate(
                    provider="hint/og",
                    post_url=url,
                    image_url=image,
                    title=f"og:image from {host}",
                    snippet=host,
                )
            )
            if len(out) >= limit:
                break
        LOG.info("search.hint.og", url=url, images=len(out))
        return out

    def _from_serpapi_images(
        self, probe: ProbeContext, query: str, *, limit: int
    ) -> list[RawCandidate]:
        key = probe.settings.serpapi_key
        if not key:
            return []
        params = {
            "engine": "google_images",
            "api_key": key,
            "q": query,
            "hl": "en",
        }
        try:
            data = probe.fetcher.get_json(_ENDPOINT, params=params)
        except Exception as exc:
            LOG.warning("search.hint.serpapi_failed", query=query[:80], error=str(exc))
            return []

        parsed = SerpApiProvider.parse(data, provider="hint/serpapi")
        # Prefer social / LinkedIn hits when ranking within this provider's list.
        LOG.info("search.hint.serpapi", query=query[:80], candidates=len(parsed))
        return parsed[:limit]
