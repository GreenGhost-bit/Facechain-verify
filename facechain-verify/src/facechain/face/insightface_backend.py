"""Optional InsightFace / ArcFace engine (512-d CNN embedding).

Enabled only when ``pip install 'facechain-verify[insightface]'`` succeeded and
the ONNX model pack can be loaded. Detection + recognition both come from the
``buffalo_l`` pack. This is the recommended engine for real identity matching;
the pipeline falls back to the classical LBPH engines when it is absent.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from .base import DetectedFace


@lru_cache(maxsize=1)
def _get_app() -> Any:
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 => CPU
    return app


class InsightFaceEngine:
    name = "insightface-arcface"
    embed_dim = 512

    def __init__(self) -> None:
        self._app: Any = _get_app()
        try:
            import insightface

            self.version = f"insightface-{insightface.__version__}"
        except Exception:
            self.version = "insightface-unknown"

    @classmethod
    def available(cls) -> bool:
        try:
            _get_app()
            return True
        except Exception:
            return False

    def _analyse(self, rgb: np.ndarray) -> list[Any]:
        bgr = rgb[..., ::-1].copy()
        return list(self._app.get(bgr))

    def detect(self, rgb: np.ndarray) -> list[DetectedFace]:
        faces = self._analyse(rgb)
        out: list[DetectedFace] = []
        for f in faces:
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            score = float(getattr(f, "det_score", 1.0))
            out.append(DetectedFace(x1, y1, max(1, x2 - x1), max(1, y2 - y1), score))
        out.sort(key=lambda d: d.area, reverse=True)
        return out

    def embed(self, rgb: np.ndarray, face: DetectedFace) -> np.ndarray:
        faces = self._analyse(rgb)
        if not faces:
            raise RuntimeError("insightface produced no embedding for the given crop")
        target = _closest(faces, face)
        vec = np.asarray(target.normed_embedding, dtype=np.float32)
        n = float(np.linalg.norm(vec))
        return vec / n if n > 0 else vec


def _closest(faces: list[Any], face: DetectedFace) -> Any:
    fx, fy = face.x + face.w / 2, face.y + face.h / 2
    best = faces[0]
    best_d = float("inf")
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        d = (cx - fx) ** 2 + (cy - fy) ** 2
        if d < best_d:
            best, best_d = f, d
    return best
