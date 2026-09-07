"""Search-provider selection."""

from __future__ import annotations

from ..config import ROBUST_SEARCH_PROVIDERS, Settings
from ..errors import ProviderError
from ..logging import LOG
from .base import SearchProvider
from .hint_provider import HintProvider
from .local_index_provider import LocalIndexProvider
from .multiris_provider import MultiRisProvider
from .serpapi_provider import SerpApiProvider
from .wikimedia_provider import WikimediaProvider

_REGISTRY: dict[str, type[SearchProvider]] = {
    "serpapi": SerpApiProvider,
    "wikimedia": WikimediaProvider,
    "local": LocalIndexProvider,
    "multiris": MultiRisProvider,
    "hint": HintProvider,
}


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
        if name == "local":
            built.append(LocalIndexProvider(settings.corpus_dir))
        else:
            built.append(cls())
    if not built:
        raise ProviderError(
            "no usable search provider. Install PicImageSearch (`pip install -e '.[ris]'`) "
            "for 'multiris', set FACECHAIN_SERPAPI_KEY for 'serpapi', pass --hint for 'hint', "
            "allow network for 'wikimedia', or populate the corpus with `facechain fetch-corpus`."
        )
    LOG.info(
        "search.providers",
        providers=[p.name for p in built],
        skipped=skipped,
        robust_default=list(settings.search_providers) == list(ROBUST_SEARCH_PROVIDERS),
    )
    return built
