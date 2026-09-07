"""Locate the OpenCV Zoo ONNX models used by the ``sface`` engine.

Two models, both run by OpenCV's built-in DNN face API (``cv2.FaceDetectorYN`` /
``cv2.FaceRecognizerSF``) -- no ``onnxruntime``, no ``insightface``:

* **YuNet** (~227 KB) -- frontal+profile face detector that also returns five
  landmarks. Vendored in ``face/models/`` because it is tiny.
* **SFace** (~37 MB) -- 128-D face-recognition embedding. Too big to vendor; it
  is downloaded once from the OpenCV Zoo, checksum-pinned, and cached in a
  per-user directory (same pattern as :mod:`facechain.face.cascade`).

Set ``FACECHAIN_CACHE_DIR`` to control where the download lands. Set
``FACECHAIN_SFACE_PATH`` to point at a pre-downloaded copy and skip the network.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

# OpenCV Zoo stores these as git-LFS objects; pin to the commit that last
# touched them and fetch via the LFS media endpoint so the bytes never move.
_ZOO_COMMIT = "25f423d0e04c31a17254620e58febd7386da523b"
_ZOO_MEDIA = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{_ZOO_COMMIT}/models"

_YUNET_NAME = "face_detection_yunet_2023mar.onnx"
_YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

_SFACE_NAME = "face_recognition_sface_2021dec.onnx"
_SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
_SFACE_URL = f"{_ZOO_MEDIA}/face_recognition_sface/{_SFACE_NAME}"
_SFACE_BYTES = 38_696_353


def _cache_dir() -> Path:
    root = os.environ.get("FACECHAIN_CACHE_DIR") or (Path(tempfile.gettempdir()) / "facechain")
    d = Path(root)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _verify(data: bytes, expected_sha: str, name: str) -> None:
    got = hashlib.sha256(data).hexdigest()
    if got != expected_sha:
        raise RuntimeError(
            f"{name}: checksum mismatch (expected {expected_sha}, got {got}); refusing to use it"
        )


@lru_cache(maxsize=1)
def yunet_path() -> str:
    """Filesystem path to the vendored YuNet detector (verified on first use)."""
    override = os.environ.get("FACECHAIN_YUNET_PATH")
    if override and Path(override).is_file():
        return override
    res = files("facechain.face.models").joinpath(_YUNET_NAME)
    data = res.read_bytes()
    _verify(data, _YUNET_SHA256, _YUNET_NAME)
    # importlib.resources may hand back a non-filesystem path on exotic loaders;
    # materialise a copy in that case so OpenCV can open it by name.
    try:
        p = Path(str(res))
        if p.is_file():
            return str(p)
    except (TypeError, ValueError):  # pragma: no cover - zipimport etc.
        pass
    target = _cache_dir() / _YUNET_NAME
    if not target.is_file() or target.stat().st_size != len(data):
        target.write_bytes(data)
    return str(target)


@lru_cache(maxsize=1)
def sface_path() -> str:
    """Path to the SFace recogniser, downloading + caching it once if needed."""
    override = os.environ.get("FACECHAIN_SFACE_PATH")
    if override and Path(override).is_file():
        return override

    target = _cache_dir() / f"{_SFACE_NAME}.{_SFACE_SHA256[:16]}.onnx"
    if target.is_file() and target.stat().st_size == _SFACE_BYTES:
        return str(target)

    req = urllib.request.Request(_SFACE_URL, headers={"User-Agent": "facechain-verify"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    _verify(data, _SFACE_SHA256, _SFACE_NAME)
    tmp = target.with_suffix(".onnx.tmp")
    tmp.write_bytes(data)
    tmp.replace(target)
    return str(target)


def sface_is_cached() -> bool:
    """True if the SFace model can be produced without hitting the network."""
    override = os.environ.get("FACECHAIN_SFACE_PATH")
    if override and Path(override).is_file():
        return True
    target = _cache_dir() / f"{_SFACE_NAME}.{_SFACE_SHA256[:16]}.onnx"
    return target.is_file() and target.stat().st_size == _SFACE_BYTES


def ensure_models(*, download: bool = True) -> dict[str, str]:
    """Resolve both model paths. With ``download=False`` raise if SFace is absent."""
    if not download and not sface_is_cached():
        raise RuntimeError("SFace model not cached and download disabled")
    return {"yunet": yunet_path(), "sface": sface_path()}
