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
    face_bbox: list[int] | None = None

    def reverse_image_bytes(self) -> bytes:
        """Bytes to hand a reverse-image engine.

        Google Lens & co. latch onto whatever object is most distinctive -- often
        the shirt or the background, not the face. Sending a tight crop around the
        detected face removes those distractors. Falls back to the full frame
        when there is no bbox or ``settings.reverse_image_face_crop`` is off.
        """
        if not (self.settings.reverse_image_face_crop and self.face_bbox):
            return self.image_bytes
        try:
            import io

            from PIL import Image

            x, y, w, h = self.face_bbox
            ih, iw = self.rgb.shape[:2]
            mx, my = int(w * 0.55), int(h * 0.55)  # generous: keep some context
            x0, y0 = max(0, x - mx), max(0, y - my)
            x1, y1 = min(iw, x + w + mx), min(ih, y + h + my)
            crop = self.rgb[y0:y1, x0:x1]
            if crop.size == 0:
                return self.image_bytes
            buf = io.BytesIO()
            Image.fromarray(crop, "RGB").save(buf, format="JPEG", quality=92)
            return buf.getvalue()
        except Exception:
            return self.image_bytes


@runtime_checkable
class SearchProvider(Protocol):
    name: str

    @classmethod
    def available(cls, settings: Settings) -> bool: ...

    def search(self, probe: ProbeContext) -> Iterable[RawCandidate]: ...
