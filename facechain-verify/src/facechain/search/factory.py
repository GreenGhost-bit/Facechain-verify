"""Search-provider selection."""

from __future__ import annotations

from ..config import Settings
from ..errors import ProviderError
from ..logging import LOG
from .base import SearchProvider
from .local_index_provider import LocalIndexProvider
from .serpapi_provider import SerpApiProvider
from .wikimedia_provider import WikimediaProvider

_REGISTRY: dict[str, type[SearchProvider]] = {
    "serpapi": SerpApiProvider,
    "wikimedia": WikimediaProvider,
    "local": LocalIndexProvider,
}


def build_providers(settings: Settings, *, strict: bool = False) -> list[SearchProvider]:
    """Instantiate the configured providers, silently dropping unavailable ones
    (unless ``strict``)."""
    built: list[SearchProvider] = []
    for name in settings.search_providers:
        cls = _REGISTRY.get(name)
        if cls is None:
            raise ProviderError(f"unknown search provider {name!r}; known: {sorted(_REGISTRY)}")
        if not cls.available(settings):
            msg = f"provider {name!r} is not available in this configuration"
            if strict:
                raise ProviderError(msg)
            LOG.warning("search.provider.unavailable", provider=name)
            continue
        if name == "local":
            built.append(LocalIndexProvider(settings.corpus_dir))
        else:
            built.append(cls())
    if not built:
        raise ProviderError(
            "no usable search provider. Configure FACECHAIN_SERPAPI_KEY, allow network "
            "for 'wikimedia', or populate the corpus with `facechain fetch-corpus`."
        )
    LOG.info("search.providers", providers=[p.name for p in built])
    return built
