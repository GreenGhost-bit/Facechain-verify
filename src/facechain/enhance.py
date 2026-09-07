"""Mild clarity / contrast helpers for candid phone photos.

Reverse-image + ArcFace work best on reasonably sharp, well-lit faces. Random
candids are often soft, dark, or small. These transforms are intentionally
conservative: they must not invent facial structure, only recover contrast and
edge energy so the detector and embedder see a clearer signal.

The probe path tries the original first, then an enhanced copy if detection
fails or sharpness quality is poor — and keeps whichever yields the better face.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .logging import LOG

# Upscale tiny probes so InsightFace's detector has more pixels to work with.
_MIN_SIDE_FOR_DETECT = 480
_MAX_UPSCALE = 5.0


def enhance_rgb_for_faces(rgb: np.ndarray) -> np.ndarray:
    """Return a lightly sharpened / contrast-stretched copy of ``rgb`` (H,W,3 uint8)."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("enhance_rgb_for_faces expects HxWx3 uint8 RGB")

    img = Image.fromarray(rgb, mode="RGB")
    w, h = img.size
    min_side = min(w, h)
    if 0 < min_side < _MIN_SIDE_FOR_DETECT:
        scale = min(_MIN_SIDE_FOR_DETECT / float(min_side), _MAX_UPSCALE)
        img = img.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            resample=Image.Resampling.LANCZOS,
        )

    # Stretch tonal range (helps underexposed beach / night candids).
    img = ImageOps.autocontrast(img, cutoff=1)
    # Mild unsharp — enough for soft phone JPEGs, not enough to halo.
    img = img.filter(ImageFilter.UnsharpMask(radius=1.4, percent=140, threshold=2))
    out = np.asarray(img, dtype=np.uint8)
    LOG.debug(
        "enhance.applied",
        in_shape=list(rgb.shape),
        out_shape=list(out.shape),
    )
    return out
