"""Keyless live provider: Wikimedia Commons.

Commons has no image-similarity endpoint, so this is a *scripted search*: it
issues a real full-text query against the live MediaWiki API, pulls the actual
image files and their human-facing File: pages, and hands them to the aggregator,
which decides the match by face embedding. A ``--hint`` (a name or descriptive
keywords) sharpens the candidate pool; with no hint it runs a broad
portrait sweep.

No API key, no account. Honours the API etiquette guidelines: descriptive
User-Agent, ``maxlag`` fallback, small result pages.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..config import Settings
from ..errors import ProviderError
from ..logging import LOG
from .base import ProbeContext, RawCandidate

_API = "https://commons.wikimedia.org/w/api.php"
_BROAD_QUERIES = (
    'portrait "headshot" filetype:bitmap',
    "official portrait person face",
    "person portrait photograph face -logo -map",
)


class WikimediaProvider:
    name = "wikimedia"

    def __init__(self, *, api_url: str = _API) -> None:
        self._api = api_url

    @classmethod
    def available(cls, settings: Settings) -> bool:
        return True

    def _query(self, probe: ProbeContext, srsearch: str, limit: int) -> list[dict[str, Any]]:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": srsearch,
            "gsrnamespace": "6",  # File:
            "gsrlimit": str(limit),
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "iiurlwidth": "800",
            "maxlag": "5",
        }
        try:
            data = probe.fetcher.get_json(self._api, params=params)
        except Exception as exc:
            raise ProviderError(f"wikimedia query failed: {exc}") from exc
        if not isinstance(data, dict):
            return []
        pages = data.get("query", {}).get("pages", {})
        return list(pages.values()) if isinstance(pages, dict) else []

    def search(self, probe: ProbeContext) -> Iterable[RawCandidate]:
        limit = probe.settings.max_candidates_per_provider
        queries: list[str]
        queries = [f"{probe.hint} portrait", probe.hint] if probe.hint else list(_BROAD_QUERIES)

        seen: set[str] = set()
        out: list[RawCandidate] = []
        for q in queries:
            if len(out) >= limit:
                break
            for page in self._query(probe, q, limit):
                infos = page.get("imageinfo") or []
                if not infos:
                    continue
                info = infos[0]
                mime = str(info.get("mime", ""))
                if not mime.startswith("image/"):
                    continue
                img_url = str(info.get("thumburl") or info.get("url") or "")
                page_url = str(info.get("descriptionurl") or info.get("url") or "")
                if not img_url or img_url in seen:
                    continue
                seen.add(img_url)
                title = str(page.get("title", "")).removeprefix("File:")
                artist = _plain(info.get("extmetadata", {}), "Artist")
                out.append(
                    RawCandidate(
                        provider=self.name,
                        post_url=page_url,
                        image_url=img_url,
                        title=title,
                        snippet=artist,
                    )
                )
                if len(out) >= limit:
                    break
        LOG.info("search.wikimedia", queries=len(queries), candidates=len(out), hinted=bool(probe.hint))
        return out


def _plain(extmeta: object, key: str) -> str:
    if isinstance(extmeta, dict):
        node = extmeta.get(key)
        if isinstance(node, dict):
            value = node.get("value", "")
            if isinstance(value, str):
                # strip the most common HTML wrapper Commons returns
                import re

                return re.sub(r"<[^>]+>", "", value).strip()[:200]
    return ""
