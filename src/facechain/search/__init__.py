"""Web / social-media search stage."""

from __future__ import annotations

from .aggregator import AggregateResult, ScoredCandidate, SearchAggregator
from .base import ProbeContext, RawCandidate, SearchProvider
from .factory import build_providers

__all__ = [
    "AggregateResult",
    "ProbeContext",
    "RawCandidate",
    "ScoredCandidate",
    "SearchAggregator",
    "SearchProvider",
    "build_providers",
]
