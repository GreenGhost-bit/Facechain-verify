"""Classical face descriptor shared by the OpenCV and pure-NumPy backends.

A local-binary-pattern histogram (LBPH) descriptor: crop with margin -> resize to
a fixed 128x128 patch -> histogram-equalise -> 8-neighbour LBP -> per-cell
64-bin histograms over an 8x8 grid -> L2-normalised concatenation. This is the
pre-deep-learning face descriptor (Ahonen et al. 2006); it is deterministic,
needs no model download, and -- as measured on the bundled fixtures -- keeps a
same-identity re-encode near cos 0.90 while impostors sit around 0.60-0.80.

InsightFace/ArcFace, when installed, replaces this with a 512-d CNN embedding.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import DetectedFace

PATCH = 128
GRID = 8
BINS = 64
HOG_BINS = 9
# LBPH histograms + coarse intensity map + HOG-lite orientation histograms,
# each block L2-normalised then concatenated and L2-normalised again.
EMBED_DIM = GRID * GRID * BINS + GRID * GRID + GRID * GRID * HOG_BINS  # 4096 + 64 + 576 = 4736


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(y, 0, 255).astype(np.uint8)


def _equalize(gray: np.ndarray) -> np.ndarray:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    cdf = hist.cumsum()
    nz = cdf[cdf > 0]
    if nz.size == 0:
        return gray
    lut = np.round((cdf - nz[0]) / max(gray.size - nz[0], 1) * 255.0)
    return np.clip(lut, 0, 255).astype(np.uint8)[gray]


def align_crop(rgb: np.ndarray, face: DetectedFace, *, margin: float = 0.18) -> np.ndarray:
    """Return a PATCH x PATCH histogram-equalised grey crop of the face."""
    h, w = rgb.shape[:2]
    f = face.clipped(w, h)
    mx, my = int(f.w * margin), int(f.h * margin)
    x0, y0 = max(0, f.x - mx), max(0, f.y - my)
    x1, y1 = min(w, f.x + f.w + mx), min(h, f.y + f.h + my)
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        crop = rgb
    gray = _to_gray(crop)
    resized = np.asarray(
        Image.fromarray(gray, "L").resize((PATCH, PATCH), Image.Resampling.LANCZOS)
    )
    return _equalize(resized)


def _lbp(gray: np.ndarray) -> np.ndarray:
    """8-neighbour local binary pattern; drops the 1px border. Returns uint8 codes."""
    g = gray.astype(np.int16)
    center = g[1:-1, 1:-1]
    codes = np.zeros_like(center, dtype=np.uint8)
    neigh = [
        g[0:-2, 0:-2], g[0:-2, 1:-1], g[0:-2, 2:],
        g[1:-1, 2:], g[2:, 2:], g[2:, 1:-1],
        g[2:, 0:-2], g[1:-1, 0:-2],
    ]
    for i, nb in enumerate(neigh):
        codes |= ((nb >= center).astype(np.uint8) << i)
    return codes


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def lbph_embedding(rgb: np.ndarray, face: DetectedFace) -> np.ndarray:
    """Composite face descriptor, shape ``(EMBED_DIM,)`` float32.

    Three complementary, per-cell, block-normalised feature families over an 8x8
    grid: LBP texture histograms, a coarse intensity map, and HOG-lite gradient
    orientation histograms. Blends texture (robust to lighting) with shape
    (helps cross-photo identity)."""
    patch = align_crop(rgb, face).astype(np.float64)
    coded = _lbp(patch.astype(np.uint8))  # (126, 126)

    # gradients for HOG-lite
    gx_img = np.zeros_like(patch)
    gy_img = np.zeros_like(patch)
    gx_img[:, 1:-1] = patch[:, 2:] - patch[:, :-2]
    gy_img[1:-1, :] = patch[2:, :] - patch[:-2, :]
    mag = np.hypot(gx_img, gy_img)
    ori = (np.degrees(np.arctan2(gy_img, gx_img)) % 180.0)

    lbp_edges = np.linspace(0, 256, BINS + 1)
    hog_edges = np.linspace(0, 180, HOG_BINS + 1)
    lcell = coded.shape[0] // GRID
    pcell = PATCH // GRID

    lbp_blocks: list[np.ndarray] = []
    intensity: list[float] = []
    hog_blocks: list[np.ndarray] = []
    for gy in range(GRID):
        for gx in range(GRID):
            sub = coded[gy * lcell:(gy + 1) * lcell, gx * lcell:(gx + 1) * lcell]
            hist, _ = np.histogram(sub, bins=lbp_edges)
            lbp_blocks.append(_l2(hist.astype(np.float64)))

            psub = patch[gy * pcell:(gy + 1) * pcell, gx * pcell:(gx + 1) * pcell]
            intensity.append(float(psub.mean()) / 255.0)

            msub = mag[gy * pcell:(gy + 1) * pcell, gx * pcell:(gx + 1) * pcell]
            osub = ori[gy * pcell:(gy + 1) * pcell, gx * pcell:(gx + 1) * pcell]
            hh, _ = np.histogram(osub, bins=hog_edges, weights=msub)
            hog_blocks.append(_l2(hh))

    vec = np.concatenate(
        [
            np.concatenate(lbp_blocks),
            _l2(np.asarray(intensity, dtype=np.float64)),
            np.concatenate(hog_blocks),
        ]
    ).astype(np.float32)
    return _l2(vec).astype(np.float32)


def sharpness_quality(rgb: np.ndarray, face: DetectedFace) -> float:
    """Heuristic in ``[0, 1]``: blends face size and Laplacian-variance sharpness."""
    h, w = rgb.shape[:2]
    f = face.clipped(w, h)
    size_score = min(1.0, min(f.w, f.h) / 96.0)
    crop = rgb[f.y:f.y + f.h, f.x:f.x + f.w]
    gray = _to_gray(crop if crop.size else rgb).astype(np.float64)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return round(0.5 * size_score, 6)
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    sharp_score = min(1.0, float(np.var(lap)) / 500.0)
    return round(0.5 * size_score + 0.5 * sharp_score, 6)
