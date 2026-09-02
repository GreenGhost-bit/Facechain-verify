"""Locate the frontal-face Haar cascade without vendoring a huge XML blob.

Priority order:

1. The copy shipped inside ``opencv-python`` / ``opencv-python-headless``
   (``cv2.data.haarcascades``) -- present whenever OpenCV is installed, which is
   the default. Nothing to unpack.
2. A gzip-compressed fallback vendored at ``face/models/*.xml.gz`` (~130 KB),
   decompressed once into a per-user cache directory. This keeps the pure
   ``numpy + Pillow`` install (no OpenCV) fully working and offline.

The raw ~900 KB XML is deliberately **not** committed: GitHub refuses to render
files that large in the browser, and it needlessly bloats clones.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import tempfile
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

_CASCADE_NAME = "haarcascade_frontalface_default.xml"
_GZ_RESOURCE = f"{_CASCADE_NAME}.gz"


def _from_opencv() -> Path | None:
    try:
        import cv2
    except Exception:  # OpenCV is optional
        return None
    data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if not data_dir:
        return None
    candidate = Path(data_dir) / _CASCADE_NAME
    return candidate if candidate.is_file() else None


@lru_cache(maxsize=1)
def cascade_bytes() -> bytes:
    """Raw XML bytes of the cascade (from OpenCV if available, else the vendored gz)."""
    opencv_path = _from_opencv()
    if opencv_path is not None:
        return opencv_path.read_bytes()
    compressed = files("facechain.face.models").joinpath(_GZ_RESOURCE).read_bytes()
    return gzip.decompress(compressed)


@lru_cache(maxsize=1)
def cascade_path() -> str:
    """Filesystem path to the cascade XML (OpenCV needs a path, not bytes).

    Returns the OpenCV-bundled file directly when present; otherwise materialises
    the decompressed fallback in a stable, content-addressed cache location.
    """
    opencv_path = _from_opencv()
    if opencv_path is not None:
        return str(opencv_path)

    data = cascade_bytes()
    digest = hashlib.sha256(data).hexdigest()[:16]
    cache_dir = Path(os.environ.get("FACECHAIN_CACHE_DIR", Path(tempfile.gettempdir()) / "facechain"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{_CASCADE_NAME}.{digest}.xml"
    if not target.is_file() or target.stat().st_size != len(data):
        tmp = target.with_suffix(".xml.tmp")
        tmp.write_bytes(data)
        tmp.replace(target)
    return str(target)
