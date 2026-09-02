"""Face detection + encoding stage.

``encode_probe`` turns a :class:`~facechain.imaging.LoadedImage` into a
:class:`~facechain.models.FaceRecord` plus the raw L2-normalised embedding used
downstream for candidate ranking.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ..canonical import to_fixed
from ..errors import NoFaceFoundError
from ..imaging import LoadedImage
from ..logging import LOG
from ..models import FaceRecord
from .base import DetectedFace, EncodedFace, FaceEngine, cosine
from .descriptor import sharpness_quality
from .factory import build_face_engine

__all__ = [
    "DetectedFace",
    "EncodedFace",
    "FaceEngine",
    "build_face_engine",
    "cosine",
    "embedding_sha256",
    "encode_candidate",
    "encode_probe",
]


def embedding_sha256(vec: np.ndarray) -> str:
    """Hash the embedding in a byte-stable way (float32, C-order, rounded)."""
    rounded = np.round(vec.astype(np.float64), 6).astype(np.float32)
    return hashlib.sha256(rounded.tobytes(order="C")).hexdigest()


def _select_face(
    faces: list[DetectedFace], *, min_pixels: int, image_wh: tuple[int, int]
) -> tuple[DetectedFace, bool, str]:
    w, h = image_wh
    usable = [f.clipped(w, h) for f in faces if min(f.w, f.h) >= min_pixels]
    if not usable:
        raise NoFaceFoundError(
            "no face large enough was detected",
            detail={"raw_count": len(faces), "min_pixels": min_pixels},
        )
    usable.sort(key=lambda f: f.area, reverse=True)
    primary = usable[0]
    ambiguous = False
    note = ""
    if len(usable) > 1:
        runner = usable[1]
        ratio = runner.area / primary.area
        if ratio >= 0.8:
            ambiguous = True
            note = (
                f"{len(usable)} comparably sized faces detected "
                f"(runner-up is {ratio:.0%} of the primary); using the largest"
            )
    return primary, ambiguous, note


def encode_probe(
    image: LoadedImage,
    *,
    engine: FaceEngine | None = None,
    engine_preference: str = "auto",
    min_face_pixels: int = 48,
) -> tuple[FaceRecord, np.ndarray, bool]:
    """Detect + encode the dominant face in ``image``.

    Returns ``(face_record, embedding, ambiguous)``.
    """
    eng = engine or build_face_engine(engine_preference)
    h, w = image.rgb.shape[:2]

    with LOG.span("face.detect", engine=eng.name) as sp:
        faces = eng.detect(image.rgb)
        sp["faces"] = len(faces)
    if not faces:
        raise NoFaceFoundError("face engine detected no faces", detail={"engine": eng.name})

    primary, ambiguous, note = _select_face(
        faces, min_pixels=min_face_pixels, image_wh=(w, h)
    )

    with LOG.span("face.embed", engine=eng.name):
        vec = np.asarray(eng.embed(image.rgb, primary), dtype=np.float32)
    n = float(np.linalg.norm(vec))
    if n > 0:
        vec = vec / n

    quality = sharpness_quality(image.rgb, primary)
    record = FaceRecord(
        engine=eng.name,
        engine_version=eng.version,
        bbox=primary.bbox,
        detection_score_ppm=to_fixed(min(1.0, float(primary.det_score) / 10.0)),
        quality_ppm=to_fixed(quality),
        embedding_dim=int(vec.size),
        embedding_sha256=embedding_sha256(vec),
        all_bboxes=[f.clipped(w, h).bbox for f in faces],
        note=note,
    )
    LOG.info(
        "face.encoded",
        engine=eng.name,
        bbox=primary.bbox,
        quality=quality,
        ambiguous=ambiguous,
        dim=int(vec.size),
    )
    return record, vec, ambiguous


def encode_candidate(
    rgb: np.ndarray, *, engine: FaceEngine, min_face_pixels: int = 32
) -> np.ndarray | None:
    """Encode the dominant face of a candidate image, or ``None`` if no usable face."""
    faces = engine.detect(rgb)
    h, w = rgb.shape[:2]
    usable = [f.clipped(w, h) for f in faces if min(f.w, f.h) >= min_face_pixels]
    if not usable:
        return None
    usable.sort(key=lambda f: f.area, reverse=True)
    vec = np.asarray(engine.embed(rgb, usable[0]), dtype=np.float32)
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec
