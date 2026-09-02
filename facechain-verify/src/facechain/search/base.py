"""Search-provider abstractions.

A provider's only job is to *gather* real candidate images from a live source.
It never decides the match -- the :class:`~facechain.search.aggregator.SearchAggregator`
re-encodes every candidate's face and ranks purely by embedding cosine
similarity against the probe. This separation is what makes the pipeline a
genuine search rather than a lookup: swapping providers cannot change *which*
candidate wins, only which candidates are considered.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..config import Settings
from ..face.base import FaceEngine
from ..netfetch import SafeFetcher


@dataclass(frozen=True)
class RawCandidate:
    """A search hit before the pipeline has looked at its pixels."""

    provider: str
    post_url: str
    image_url: str
    title: str = ""
    snippet: str = ""

    def key(self) -> str:
        return self.image_url.strip().lower() or self.post_url.strip().lower()


@dataclass
class ProbeContext:
    """Everything a provider might need about the probe face."""

    image_bytes: bytes
    rgb: np.ndarray
    embedding: np.ndarray
    settings: Settings
    fetcher: SafeFetcher
    face_engine: FaceEngine
    hint: str | None = None
    extra: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class SearchProvider(Protocol):
    name: str

    @classmethod
    def available(cls, settings: Settings) -> bool: ...

    def search(self, probe: ProbeContext) -> Iterable[RawCandidate]: ...
