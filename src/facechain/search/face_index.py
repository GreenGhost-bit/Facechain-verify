"""Persistent face-embedding index over the offline corpus.

The plain ``local`` provider hands *every* corpus entry to the aggregator, which
then re-fetches and re-encodes all of them. This module instead pre-computes one
face embedding per corpus image, caches it, and at query time ranks by cosine
against the probe embedding -- a genuine face-vector retrieval step. Only the
top-K survivors are handed on, so the aggregator's expensive re-encode runs on a
short list instead of the whole corpus.

Brute-force ``float32`` matmul: dependency-free and instant for the corpus sizes
this tool builds (tens to low thousands). Swap in FAISS/hnswlib here if a corpus
ever grows past that.

Cache layout, next to the corpus::

    <corpus>/.faceindex/<engine-name>.npz        # {"vecs": (N, D) float32}
    <corpus>/.faceindex/<engine-name>.meta.json  # [{sha256, file, post_url, ...}]

Entries are keyed by image content hash, so a corpus edit only re-encodes what
changed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..face import encode_candidate
from ..face.base import FaceEngine
from ..imaging import load_image_bytes
from ..logging import LOG

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_INDEX_DIRNAME = ".faceindex"


@dataclass(frozen=True)
class IndexHit:
    score: float
    file: str
    post_url: str
    title: str
    source: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _meta_for(img: Path) -> dict[str, str]:
    sidecar = img.with_suffix(".json")
    meta: dict[str, str] = {}
    if sidecar.is_file():
        try:
            loaded = json.loads(sidecar.read_text("utf-8"))
            if isinstance(loaded, dict):
                meta = {str(k): str(v) for k, v in loaded.items()}
        except json.JSONDecodeError:
            LOG.warning("faceindex.bad_meta", path=str(sidecar))
    return meta


class FaceIndex:
    """A cached matrix of corpus face embeddings for one engine."""

    def __init__(self, corpus_dir: Path, engine: FaceEngine) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.engine = engine
        self._dir = self.corpus_dir / _INDEX_DIRNAME
        self._vecs = np.zeros((0, 0), dtype=np.float32)
        self._rows: list[dict[str, str]] = []

    # -- persistence --------------------------------------------------
    @property
    def _npz(self) -> Path:
        return self._dir / f"{self.engine.name}.npz"

    @property
    def _meta(self) -> Path:
        return self._dir / f"{self.engine.name}.meta.json"

    def _load_cache(self) -> dict[str, np.ndarray]:
        if not (self._npz.is_file() and self._meta.is_file()):
            return {}
        try:
            data = np.load(self._npz)
            rows = json.loads(self._meta.read_text("utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
        vecs = data["vecs"]
        if not isinstance(rows, list) or len(rows) != vecs.shape[0]:
            return {}
        return {str(r["sha256"]): vecs[i] for i, r in enumerate(rows) if "sha256" in r}

    def _save_cache(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._npz, vecs=self._vecs)
        self._meta.write_text(json.dumps(self._rows, indent=2), encoding="utf-8")

    # -- build ------------------------------------------------------
    def build(self, *, rebuild: bool = False) -> int:
        """Encode every corpus image (reusing cached vectors) and persist."""
        cached = {} if rebuild else self._load_cache()
        images = sorted(
            p for p in self.corpus_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )
        vecs: list[np.ndarray] = []
        rows: list[dict[str, str]] = []
        encoded = 0
        for img in images:
            digest = _sha256_file(img)
            vec = cached.get(digest)
            if vec is None:
                try:
                    loaded = load_image_bytes(img.read_bytes(), source=str(img))
                    emb = encode_candidate(loaded.rgb, engine=self.engine)
                except Exception as exc:  # unreadable / no face
                    LOG.warning("faceindex.encode_failed", file=img.name, error=str(exc))
                    continue
                if emb is None:
                    LOG.info("faceindex.no_face", file=img.name)
                    continue
                vec = np.asarray(emb, dtype=np.float32)
                encoded += 1
            meta = _meta_for(img)
            vecs.append(vec)
            rows.append(
                {
                    "sha256": digest,
                    "file": img.name,
                    "post_url": meta.get("post_url", img.resolve().as_uri()),
                    "title": meta.get("title", img.stem),
                    "source": meta.get("source", "local corpus"),
                }
            )
        self._vecs = np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, 0), np.float32)
        self._rows = rows
        self._save_cache()
        LOG.info(
            "faceindex.built",
            engine=self.engine.name,
            entries=len(rows),
            newly_encoded=encoded,
            corpus=str(self.corpus_dir),
        )
        return len(rows)

    def load_or_build(self) -> FaceIndex:
        if not self.corpus_dir.is_dir():
            return self
        # A cheap freshness check: rebuild whenever the image set changed.
        self.build(rebuild=False)
        return self

    # -- query ----------------------------------------------------
    def query(self, probe_embedding: np.ndarray, *, top_k: int = 12) -> list[IndexHit]:
        if self._vecs.size == 0:
            return []
        p = np.asarray(probe_embedding, dtype=np.float32).ravel()
        n = float(np.linalg.norm(p))
        if n == 0.0:
            return []
        p = p / n
        if p.shape[0] != self._vecs.shape[1]:
            LOG.warning(
                "faceindex.dim_mismatch",
                probe=p.shape[0], index=self._vecs.shape[1], engine=self.engine.name,
            )
            return []
        sims = self._vecs @ p
        order = np.argsort(-sims)[:top_k]
        return [
            IndexHit(
                score=float(sims[i]),
                file=self._rows[i]["file"],
                post_url=self._rows[i]["post_url"],
                title=self._rows[i]["title"],
                source=self._rows[i]["source"],
            )
            for i in order
        ]

    @property
    def size(self) -> int:
        return self._vecs.shape[0]
