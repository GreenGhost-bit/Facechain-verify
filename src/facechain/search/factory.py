"""Search-provider selection."""

from __future__ import annotations

from ..config import ROBUST_SEARCH_PROVIDERS, Settings
from ..errors import ProviderError
from ..logging import LOG
from .base import SearchProvider
from .face_index_provider import FaceIndexProvider
from .hint_provider import HintProvider
from .local_index_provider import LocalIndexProvider
from .multiris_provider import MultiRisProvider
from .serpapi_provider import SerpApiProvider
from .wikimedia_provider import WikimediaProvider

_REGISTRY: dict[str, type[SearchProvider]] = {
    "serpapi": SerpApiProvider,
    "wikimedia": WikimediaProvider,
    "faceindex": FaceIndexProvider,
    "local": LocalIndexProvider,
    "multiris": MultiRisProvider,
    "hint": HintProvider,
}

# Providers that take a corpus-dir constructor arg rather than no args.
_CORPUS_PROVIDERS = {"local", "faceindex"}


def build_providers(settings: Settings, *, strict: bool = False) -> list[SearchProvider]:
    """Instantiate the configured providers, silently dropping unavailable ones
    (unless ``strict``)."""
    built: list[SearchProvider] = []
    skipped: list[str] = []
    for name in settings.search_providers:
        cls = _REGISTRY.get(name)
        if cls is None:
            raise ProviderError(f"unknown search provider {name!r}; known: {sorted(_REGISTRY)}")
        if not cls.available(settings):
            msg = f"provider {name!r} is not available in this configuration"
            if strict:
                raise ProviderError(msg)
            skipped.append(name)
            LOG.warning("search.provider.unavailable", provider=name)
            continue
        if name in _CORPUS_PROVIDERS:
            built.append(cls(settings.corpus_dir))  # type: ignore[call-arg]
        else:
            built.append(cls())
    if not built:
        asked = set(settings.search_providers)
        hints = []
        if "faceindex" in asked or "local" in asked:
            hints.append(
                "for offline matching: `facechain fetch-corpus --seed-demo` (or "
                "--name \"Some Person\") then `facechain build-index`"
            )
        if "wikimedia" in asked:
            hints.append("for 'wikimedia': allow network and pass --hint \"Name\" to steer it")
        if "serpapi" in asked:
            hints.append("for 'serpapi': set FACECHAIN_SERPAPI_KEY (+ --allow-public-host)")
        if "multiris" in asked:
            hints.append("for 'multiris': pip install -e '.[ris]'")
        raise ProviderError(
            "no usable search provider was available.\n  " + "\n  ".join(hints or ["configure one"])
        )
    LOG.info(
        "search.providers",
        providers=[p.name for p in built],
        skipped=skipped,
        robust_default=list(settings.search_providers) == list(ROBUST_SEARCH_PROVIDERS),
    )
    return built
