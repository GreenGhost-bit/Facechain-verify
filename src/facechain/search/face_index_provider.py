"""Offline provider that retrieves corpus candidates by face-embedding cosine.

Unlike :mod:`facechain.search.local_index_provider` (which forwards the whole
corpus), this one ranks the corpus against the probe embedding via
:class:`facechain.search.face_index.FaceIndex` and forwards only the top-K. The
aggregator still re-encodes and re-scores every forwarded candidate, so the
"embedding alone decides the match" invariant holds -- this provider only makes
the candidate set small and relevant.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..config import Settings
from ..logging import LOG
from .base import ProbeContext, RawCandidate
from .face_index import FaceIndex

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# Drop obvious non-matches before they reach the aggregator's re-encode.
_PREFILTER_FLOOR = 0.15


class FaceIndexProvider:
    name = "faceindex"

    def __init__(self, corpus_dir: str | Path) -> None:
        self.corpus_dir = Path(corpus_dir)

    @classmethod
    def available(cls, settings: Settings) -> bool:
        d = settings.corpus_dir
        return d.is_dir() and any(
            p.suffix.lower() in _IMAGE_SUFFIXES for p in d.iterdir() if p.is_file()
        )

    def search(self, probe: ProbeContext) -> Iterable[RawCandidate]:
        index = FaceIndex(self.corpus_dir, probe.face_engine).load_or_build()
        if index.size == 0:
            LOG.info("search.faceindex.empty", corpus=str(self.corpus_dir))
            return []
        top_k = probe.settings.max_candidates_per_provider
        hits = index.query(probe.embedding, top_k=top_k)
        out: list[RawCandidate] = []
        for hit in hits:
            if hit.score < _PREFILTER_FLOOR:
                continue
            img = (self.corpus_dir / hit.file).resolve()
            out.append(
                RawCandidate(
                    provider=self.name,
                    post_url=hit.post_url or img.as_uri(),
                    image_url=img.as_uri(),
                    title=hit.title,
                    snippet=f"{hit.source} (face-index cos {hit.score:.3f})",
                )
            )
        LOG.info(
            "search.faceindex",
            corpus=str(self.corpus_dir),
            indexed=index.size,
            returned=len(out),
            best=round(hits[0].score, 4) if hits else None,
        )
        return out
