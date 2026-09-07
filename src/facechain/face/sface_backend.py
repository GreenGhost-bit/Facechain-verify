"""YuNet detector + SFace recogniser -- the recommended default engine.

Runs entirely on ``opencv-python-headless`` (>= 4.9) using OpenCV's built-in DNN
face API -- no ``onnxruntime``, no ``insightface``, CPU-only:

* **YuNet** (``cv2.FaceDetectorYN``) -- frontal + near-profile detection with five
  facial landmarks, robust to tilt / lighting / small faces where the legacy
  Haar cascade fails.
* **SFace** (``cv2.FaceRecognizerSF``) -- landmark-aligned 128-D identity
  embedding. On the bundled fixtures a genuine same-person pair scores ~0.98
  cosine while impostors sit below ~0.25 -- a clean, calibratable margin, unlike
  the classical LBPH descriptor.

Models: YuNet is vendored (tiny); SFace is downloaded once and checksum-pinned by
:mod:`facechain.face.onnx_zoo`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from .base import DetectedFace
from .onnx_zoo import sface_is_cached, sface_path, yunet_path

_INPUT = (320, 320)
_SCORE_THRESH = 0.6
_NMS_THRESH = 0.3
_TOP_K = 50


@lru_cache(maxsize=1)
def _detector() -> Any:
    import cv2

    return cv2.FaceDetectorYN.create(
        yunet_path(), "", _INPUT,
        score_threshold=_SCORE_THRESH, nms_threshold=_NMS_THRESH, top_k=_TOP_K,
    )


@lru_cache(maxsize=1)
def _recognizer() -> Any:
    import cv2

    return cv2.FaceRecognizerSF.create(sface_path(), "")


def _row_from_face(face: DetectedFace) -> np.ndarray | None:
    """Rebuild YuNet's 15-float detection row (bbox + 5 landmarks + score)."""
    if not face.landmarks or len(face.landmarks) < 5:
        return None
    flat: list[float] = [float(face.x), float(face.y), float(face.w), float(face.h)]
    for px, py in face.landmarks[:5]:
        flat.extend((float(px), float(py)))
    flat.append(float(min(1.0, max(0.0, face.det_score))))
    return np.asarray([flat], dtype=np.float32)


class SFaceEngine:
    name = "yunet-sface"
    embed_dim = 128

    def __init__(self) -> None:
        import cv2

        self.version = f"yunet-2023mar/sface-2021dec/opencv-{cv2.__version__}"
        self._cv2 = cv2
        self._det = _detector()
        self._rec = _recognizer()

    @classmethod
    def available(cls) -> bool:
        try:
            import cv2

            if not (hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")):
                return False
            # Detector is vendored; only claim availability if the recogniser
            # is already local -- callers that want the download use the factory.
            return sface_is_cached()
        except Exception:
            return False

    # -- detection -------------------------------------------------------
    def detect(self, rgb: np.ndarray) -> list[DetectedFace]:
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        h, w = bgr.shape[:2]
        self._det.setInputSize((w, h))
        _n, faces = self._det.detect(bgr)
        out: list[DetectedFace] = []
        if faces is None:
            return out
        for row in faces:
            x, y, bw, bh = (float(v) for v in row[:4])
            lms = tuple(
                (float(row[4 + 2 * i]), float(row[5 + 2 * i])) for i in range(5)
            )
            score = float(row[14])
            out.append(
                DetectedFace(
                    round(x), round(y),
                    max(1, round(bw)), max(1, round(bh)),
                    score, lms,
                )
            )
        out.sort(key=lambda f: f.area, reverse=True)
        return out

    # -- embedding ------------------------------------------------------
    def embed(self, rgb: np.ndarray, face: DetectedFace) -> np.ndarray:
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        row = _row_from_face(face)
        if row is None:
            # No landmarks (face came from another detector) -- re-detect and
            # pick the row that best overlaps the requested box.
            row = self._closest_row(bgr, face)
        # Last resort when even re-detection fails: a plain centre crop.
        aligned = self._fallback_crop(bgr, face) if row is None else self._rec.alignCrop(bgr, row)
        feat = np.asarray(self._rec.feature(aligned), dtype=np.float32).ravel()
        n = float(np.linalg.norm(feat))
        return feat / n if n > 0 else feat

    def _closest_row(self, bgr: np.ndarray, face: DetectedFace) -> np.ndarray | None:
        h, w = bgr.shape[:2]
        self._det.setInputSize((w, h))
        _n, faces = self._det.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        fx, fy = face.x + face.w / 2.0, face.y + face.h / 2.0
        best, best_d = None, float("inf")
        for row in faces:
            cx, cy = float(row[0]) + float(row[2]) / 2.0, float(row[1]) + float(row[3]) / 2.0
            d = (cx - fx) ** 2 + (cy - fy) ** 2
            if d < best_d:
                best, best_d = row, d
        return None if best is None else np.asarray([best], dtype=np.float32)

    def _fallback_crop(self, bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        h, w = bgr.shape[:2]
        f = face.clipped(w, h)
        crop = bgr[f.y:f.y + f.h, f.x:f.x + f.w]
        if crop.size == 0:
            crop = bgr
        return self._cv2.resize(crop, (112, 112), interpolation=self._cv2.INTER_AREA)
