"""True reverse-image search via SerpAPI (Google Lens / Yandex Images).

This is the provider that returns real *social-media* posts (Instagram, X,
Facebook, LinkedIn, TikTok, ...). It needs a free SerpAPI key in
``FACECHAIN_SERPAPI_KEY``. The probe image must be reachable by a URL: pass one
via ``probe.extra["probe_image_url"]`` (e.g. an imgur/tmpfiles link you control),
otherwise the provider is skipped with a clear log line -- SerpAPI cannot accept
a raw upload.

The response parser is unit-tested against a recorded fixture so behaviour is
pinned even without network.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..config import Settings
from ..errors import ProviderError
from ..logging import LOG
from .base import ProbeContext, RawCandidate

_ENDPOINT = "https://serpapi.com/search.json"
_SOCIAL_HOSTS = (
    "instagram.com", "twitter.com", "x.com", "facebook.com", "fb.com",
    "linkedin.com", "tiktok.com", "reddit.com", "flickr.com", "youtube.com",
    "threads.net", "mastodon", "bsky.app", "vk.com", "weibo.com",
)


class SerpApiProvider:
    name = "serpapi"

    def __init__(self, *, engine: str = "google_lens") -> None:
        self._engine = engine

    @classmethod
    def available(cls, settings: Settings) -> bool:
        return bool(settings.serpapi_key)

    def search(self, probe: ProbeContext) -> Iterable[RawCandidate]:
        key = probe.settings.serpapi_key
        if not key:
            return []
        probe_url = probe.extra.get("probe_image_url")
        if not probe_url:
            LOG.warning(
                "search.serpapi.skipped",
                reason="no probe_image_url; SerpAPI needs a public URL for the probe",
            )
            return []

        params = {
            "engine": self._engine,
            "api_key": key,
            "url": probe_url,
            "hl": "en",
        }
        try:
            data = probe.fetcher.get_json(_ENDPOINT, params=params)
        except Exception as exc:
            raise ProviderError(f"serpapi request failed: {exc}") from exc

        candidates = list(self.parse(data, provider=self.name))
        LOG.info("search.serpapi", engine=self._engine, candidates=len(candidates))
        return candidates[: probe.settings.max_candidates_per_provider]

    @staticmethod
    def parse(data: object, *, provider: str = "serpapi") -> list[RawCandidate]:
        if not isinstance(data, dict):
            return []
        rows: list[dict[str, Any]] = []
        for field_name in ("visual_matches", "image_results", "inline_images", "organic_results"):
            node = data.get(field_name)
            if isinstance(node, list):
                rows.extend(x for x in node if isinstance(x, dict))

        out: list[RawCandidate] = []
        seen: set[str] = set()
        for row in rows:
            link = str(row.get("link") or row.get("source") or "")
            nested = row.get("original_image")
            nested_link = nested.get("link", "") if isinstance(nested, dict) else ""
            image = str(
                row.get("original")
                or row.get("thumbnail")
                or row.get("image")
                or nested_link
            )
            if not image or image in seen:
                continue
            seen.add(image)
            out.append(
                RawCandidate(
                    provider=provider,
                    post_url=link or image,
                    image_url=image,
                    title=str(row.get("title", "")),
                    snippet=str(row.get("source", "")),
                )
            )
        # social posts first -- they are what the task asks for
        out.sort(key=lambda c: (not _is_social(c.post_url), c.post_url))
        return out


def _is_social(url: str) -> bool:
    low = url.lower()
    return any(host in low for host in _SOCIAL_HOSTS)
