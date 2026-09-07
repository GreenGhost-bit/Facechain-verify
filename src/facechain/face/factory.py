"""Face-engine selection.

``auto`` walks the engines best-first and returns the first that initialises:

    insightface (ArcFace, if the extra is installed)
    sface       (YuNet + SFace via OpenCV DNN -- the recommended default;
                 downloads a 37 MB model once, then fully offline)
    opencv      (Haar + classical LBPH/HOG descriptor)
    numpy       (pure-NumPy Viola-Jones + the same descriptor)
"""

from __future__ import annotations

from ..errors import FaceEngineUnavailableError
from ..logging import LOG
from .base import FaceEngine

_PREFERENCE = ("insightface", "sface", "opencv", "numpy")

# When True, ``auto`` (and an explicit ``--engine sface``) may fetch the SFace
# model on first use. Set False for a hermetic/offline run.
ALLOW_SFACE_DOWNLOAD = True


def _construct(kind: str) -> FaceEngine:
    if kind == "sface":
        from .sface_backend import SFaceEngine

        return SFaceEngine()
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


def _sface_ready(*, allow_download: bool) -> bool:
    try:
        import cv2

        if not (hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")):
            return False
        from .onnx_zoo import sface_is_cached

        return True if allow_download else sface_is_cached()
    except Exception:
        return False


def _is_available(kind: str) -> bool:
    try:
        if kind == "sface":
            return _sface_ready(allow_download=ALLOW_SFACE_DOWNLOAD)
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

    ``preference`` is ``auto`` or one of ``insightface`` / ``sface`` / ``opencv``
    / ``numpy``. ``auto`` tries them best-first and falls back on any failure.
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
        if not _is_available(kind):
            continue
        try:
            engine = _construct(kind)
        except Exception as exc:  # model download failed, native init blew up, ...
            LOG.warning("face.engine.init_failed", engine=kind, error=str(exc))
            continue
        LOG.info("face.engine", engine=engine.name, version=engine.version, requested="auto")
        return engine
    raise FaceEngineUnavailableError("no face engine is available")
