"""Face-engine selection."""

from __future__ import annotations

from ..errors import FaceEngineUnavailableError
from ..logging import LOG
from .base import FaceEngine

_PREFERENCE = ("insightface", "opencv", "numpy")


def _construct(kind: str) -> FaceEngine:
    if kind == "opencv":
        from .opencv_backend import OpenCVFaceEngine

        return OpenCVFaceEngine()
    if kind == "numpy":
        from .numpy_vj_backend import NumpyViolaJonesEngine

        return NumpyViolaJonesEngine()
    if kind == "insightface":
        from .insightface_backend import InsightFaceEngine

        return InsightFaceEngine()
    raise FaceEngineUnavailableError(f"unknown face engine {kind!r}")


def _is_available(kind: str) -> bool:
    try:
        if kind == "opencv":
            from .opencv_backend import OpenCVFaceEngine

            return OpenCVFaceEngine.available()
        if kind == "numpy":
            return True
        if kind == "insightface":
            from .insightface_backend import InsightFaceEngine

            return InsightFaceEngine.available()
    except Exception:
        return False
    return False


def build_face_engine(preference: str = "auto") -> FaceEngine:
    """Return a ready face engine.

    ``preference`` is ``auto`` or one of ``insightface`` / ``opencv`` / ``numpy``.
    ``auto`` tries them in descending order of quality and falls back.
    """
    if preference != "auto":
        if not _is_available(preference):
            raise FaceEngineUnavailableError(
                f"requested face engine {preference!r} is not available; "
                f"install its extra or choose another"
            )
        engine = _construct(preference)
        LOG.info("face.engine", engine=engine.name, version=engine.version, requested=preference)
        return engine

    for kind in _PREFERENCE:
        if _is_available(kind):
            engine = _construct(kind)
            LOG.info("face.engine", engine=engine.name, version=engine.version, requested="auto")
            return engine
    raise FaceEngineUnavailableError("no face engine is available")
