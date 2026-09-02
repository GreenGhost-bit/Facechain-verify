"""OpenCV Haar-cascade detector + LBPH descriptor (default engine).

Installs cleanly on CPython 3.11-3.14 via ``opencv-python-headless>=4.9,<5``
(OpenCV 5 removed the legacy ``CascadeClassifier`` API). The frontal-face cascade
is loaded from :mod:`facechain.face.cascade` (OpenCV's own bundled copy, or a
compressed fallback); encoding uses the shared
:mod:`facechain.face.descriptor` LBPH implementation.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from .base import DetectedFace
from .cascade import cascade_path
from .descriptor import EMBED_DIM, lbph_embedding


@lru_cache(maxsize=1)
def _load_cascade() -> Any:
    import cv2

    clf = cv2.CascadeClassifier(cascade_path())
    if clf.empty():  # pragma: no cover - would mean a corrupt asset
        raise RuntimeError(f"failed to load Haar cascade from {cascade_path()!r}")
    return clf


class OpenCVFaceEngine:
    name = "opencv-haar-lbph"
    embed_dim = EMBED_DIM

    def __init__(self) -> None:
        import cv2

        self.version = f"opencv-{cv2.__version__}"
        self._cv2: Any = cv2
        self._cascade: Any = _load_cascade()

    @classmethod
    def available(cls) -> bool:
        try:
            import cv2

            return hasattr(cv2, "CascadeClassifier")
        except Exception:
            return False

    def detect(self, rgb: np.ndarray) -> list[DetectedFace]:
        cv2 = self._cv2
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        raw = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        faces = [DetectedFace(int(x), int(y), int(w), int(h), 1.0) for (x, y, w, h) in raw]
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces

    def embed(self, rgb: np.ndarray, face: DetectedFace) -> np.ndarray:
        return lbph_embedding(rgb, face)
