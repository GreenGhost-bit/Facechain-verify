"""Face detection + encoding stage.

``encode_probe`` turns a :class:`~facechain.imaging.LoadedImage` into a
:class:`~facechain.models.FaceRecord` plus the raw L2-normalised embedding used
downstream for candidate ranking.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ..canonical import to_fixed
from ..enhance import enhance_rgb_for_faces
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
    enhance: bool = True,
) -> tuple[FaceRecord, np.ndarray, bool]:
    """Detect + encode the dominant face in ``image``.

    Returns ``(face_record, embedding, ambiguous)``.

    When ``enhance`` is true (default), a failed or soft detection is retried on
    a mildly sharpened / contrast-stretched copy — helps candid phone photos.
    """
    eng = engine or build_face_engine(engine_preference)

    def _attempt(
        rgb: np.ndarray,
    ) -> tuple[list[DetectedFace], np.ndarray | None, tuple[DetectedFace, bool, str] | None, float]:
        faces_local = eng.detect(rgb)
        if not faces_local:
            return [], None, None, 0.0
        try:
            primary_local, ambiguous_local, note_local = _select_face(
                faces_local,
                min_pixels=min_face_pixels,
                image_wh=(rgb.shape[1], rgb.shape[0]),
            )
        except NoFaceFoundError:
            return faces_local, None, None, 0.0
        vec_local = np.asarray(eng.embed(rgb, primary_local), dtype=np.float32)
        n = float(np.linalg.norm(vec_local))
        if n > 0:
            vec_local = vec_local / n
        quality_local = sharpness_quality(rgb, primary_local)
        return faces_local, vec_local, (primary_local, ambiguous_local, note_local), quality_local

    with LOG.span("face.detect", engine=eng.name) as sp:
        faces, vec, meta, quality = _attempt(image.rgb)
        sp["faces"] = len(faces)
        used_enhance = False
        h, w = image.rgb.shape[0], image.rgb.shape[1]
        if enhance and (vec is None or quality < 0.35):
            try:
                enhanced = enhance_rgb_for_faces(image.rgb)
            except Exception as exc:
                LOG.warning("enhance.failed", error=str(exc))
                enhanced = None
            if enhanced is not None:
                faces2, vec2, meta2, quality2 = _attempt(enhanced)
                sp["faces_enhanced"] = len(faces2)
                better = vec is None and vec2 is not None
                better = better or (
                    vec is not None
                    and vec2 is not None
                    and (
                        quality2 > quality + 0.05
                        or (len(faces2) > len(faces) and quality2 >= quality)
                    )
                )
                if better and vec2 is not None and meta2 is not None:
                    faces, vec, meta, quality = faces2, vec2, meta2, quality2
                    used_enhance = True
                    h, w = enhanced.shape[0], enhanced.shape[1]

    if vec is None or meta is None:
        raise NoFaceFoundError("face engine detected no faces", detail={"engine": eng.name})

    primary, ambiguous, note = meta
    if used_enhance:
        note = (note + "; " if note else "") + "probe enhanced for clarity"

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
        enhanced=used_enhance,
    )
    return record, vec, ambiguous


def encode_candidate(
    rgb: np.ndarray, *, engine: FaceEngine, min_face_pixels: int = 32, enhance: bool = True
) -> np.ndarray | None:
    """Encode the dominant face of a candidate image, or ``None`` if no usable face."""

    def _embed(arr: np.ndarray) -> np.ndarray | None:
        faces = engine.detect(arr)
        h, w = arr.shape[:2]
        usable = [f.clipped(w, h) for f in faces if min(f.w, f.h) >= min_face_pixels]
        if not usable:
            return None
        usable.sort(key=lambda f: f.area, reverse=True)
        vec = np.asarray(engine.embed(arr, usable[0]), dtype=np.float32)
        n = float(np.linalg.norm(vec))
        return vec / n if n > 0 else vec

    out = _embed(rgb)
    if out is not None or not enhance:
        return out
    try:
        return _embed(enhance_rgb_for_faces(rgb))
    except Exception:
        return None
