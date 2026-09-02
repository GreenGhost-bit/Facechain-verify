"""Offline provider backed by a locally cached corpus.

The corpus under ``data/corpus/`` is *not* hand-picked per probe: it is built once
by ``facechain fetch-corpus``, which live-pulls real images (and their real
source URLs) from Wikimedia Commons. At search time this provider simply exposes
every corpus entry as a candidate; the aggregator still decides the match by
face embedding. This keeps the pipeline fully runnable -- and deterministic --
with no network, which is what the automated tests and the offline screen
recording use.

Each corpus entry is ``<name>.jpg`` (or .png/.webp) plus a sibling
``<name>.json`` = ``{"post_url": ..., "title": ..., "source": ...}``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ..config import Settings
from ..logging import LOG
from .base import ProbeContext, RawCandidate

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class LocalIndexProvider:
    name = "local"

    def __init__(self, corpus_dir: str | Path) -> None:
        self.corpus_dir = Path(corpus_dir)

    @classmethod
    def available(cls, settings: Settings) -> bool:
        d = settings.corpus_dir
        return d.is_dir() and any(
            p.suffix.lower() in _IMAGE_SUFFIXES for p in d.iterdir() if p.is_file()
        )

    def iter_entries(self) -> Iterable[tuple[Path, dict[str, str]]]:
        if not self.corpus_dir.is_dir():
            return
        for img in sorted(self.corpus_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            meta_path = img.with_suffix(".json")
            meta: dict[str, str] = {}
            if meta_path.is_file():
                try:
                    loaded = json.loads(meta_path.read_text("utf-8"))
                    if isinstance(loaded, dict):
                        meta = {str(k): str(v) for k, v in loaded.items()}
                except json.JSONDecodeError:
                    LOG.warning("search.local.bad_meta", path=str(meta_path))
            yield img, meta

    def search(self, probe: ProbeContext) -> Iterable[RawCandidate]:
        out: list[RawCandidate] = []
        for img, meta in self.iter_entries():
            uri = img.resolve().as_uri()
            out.append(
                RawCandidate(
                    provider=self.name,
                    post_url=meta.get("post_url", uri),
                    image_url=uri,
                    title=meta.get("title", img.stem),
                    snippet=meta.get("source", "local corpus (fetched via `facechain fetch-corpus`)"),
                )
            )
        LOG.info("search.local", candidates=len(out), corpus=str(self.corpus_dir))
        return out
