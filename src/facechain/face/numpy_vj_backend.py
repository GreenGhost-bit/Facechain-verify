"""Pure-NumPy Viola-Jones face detector (zero binary dependencies).

Parses the OpenCV frontal-face Haar cascade (via :mod:`facechain.face.cascade`,
which uses OpenCV's own bundled copy or a compressed fallback) and evaluates it
with a vectorised integral-image sweep over an image pyramid. Used automatically
when OpenCV is not importable, so the pipeline still runs on a bare
``numpy + Pillow`` install.

This is a *portability fallback*, not the recommended engine: it is slower
(~0.3-2 s / image) and its boxes are looser than OpenCV's, which lowers
match fidelity by a few points of cosine similarity. Install
``opencv-python-headless`` (or the ``[opencv]`` extra) for the default engine.
Encoding reuses the shared LBP/HOG descriptor.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

from .base import DetectedFace
from .cascade import cascade_bytes
from .descriptor import EMBED_DIM, lbph_embedding


@dataclass(frozen=True)
class _Feature:
    rects: tuple[tuple[int, int, int, int, float], ...]  # x, y, w, h, weight


@dataclass(frozen=True)
class _Stump:
    feature: int
    threshold: float
    left: float
    right: float


@dataclass(frozen=True)
class _Stage:
    threshold: float
    stumps: tuple[_Stump, ...]


@dataclass(frozen=True)
class _Cascade:
    window: int
    features: tuple[_Feature, ...]
    stages: tuple[_Stage, ...]


@lru_cache(maxsize=1)
def _load_cascade() -> _Cascade:
    root = ET.fromstring(cascade_bytes())
    cascade = root.find("cascade")
    if cascade is None:
        cascade = root.find(".//cascade")
    if cascade is None:  # pragma: no cover - malformed asset
        raise RuntimeError("cascade element not found in vendored XML")
    win = int((cascade.findtext("width") or "24").strip())

    features: list[_Feature] = []
    for feat in cascade.findall("./features/_"):
        rects: list[tuple[int, int, int, int, float]] = []
        for rect in feat.findall("./rects/_"):
            parts = (rect.text or "").split()
            x, y, w, h = (int(p) for p in parts[:4])
            weight = float(parts[4])
            rects.append((x, y, w, h, weight))
        features.append(_Feature(tuple(rects)))

    stages: list[_Stage] = []
    for st in cascade.findall("./stages/_"):
        st_thr = float((st.findtext("stageThreshold") or "0").strip())
        stumps: list[_Stump] = []
        for wc in st.findall("./weakClassifiers/_"):
            inodes = (wc.findtext("internalNodes") or "").split()
            leaves = (wc.findtext("leafValues") or "").split()
            feat_idx = int(inodes[2])
            thr = float(inodes[3])
            stumps.append(_Stump(feat_idx, thr, float(leaves[0]), float(leaves[1])))
        stages.append(_Stage(st_thr, tuple(stumps)))

    return _Cascade(win, tuple(features), tuple(stages))


def _integral(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ii = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float64)
    ii2 = np.zeros_like(ii)
    ii[1:, 1:] = img.cumsum(0).cumsum(1)
    ii2[1:, 1:] = (img.astype(np.float64) ** 2).cumsum(0).cumsum(1)
    return ii, ii2


def _rect_sum(
    ii: np.ndarray, ys: np.ndarray, xs: np.ndarray, x: int, y: int, w: int, h: int
) -> np.ndarray:
    """Sum over a rect offset (x, y, w, h) for every window origin in ys/xs."""
    a = ii[ys + y, xs + x]
    b = ii[ys + y, xs + x + w]
    c = ii[ys + y + h, xs + x]
    d = ii[ys + y + h, xs + x + w]
    return d - b - c + a


class NumpyViolaJonesEngine:
    name = "numpy-violajones-lbph"
    embed_dim = EMBED_DIM
    version = "vj-opencv-frontalface-default"

    def __init__(self) -> None:
        self._cascade = _load_cascade()

    @classmethod
    def available(cls) -> bool:
        return True  # numpy + Pillow are hard dependencies

    def detect(
        self,
        rgb: np.ndarray,
        *,
        scale_factor: float = 1.3,
        step: int = 3,
        min_neighbors: int = 2,
    ) -> list[DetectedFace]:
        faces = self._detect_raw(rgb, scale_factor, step, min_neighbors)
        if not faces:
            # Portability fallback recall is lower than OpenCV's; retry once denser.
            faces = self._detect_raw(rgb, 1.2, 2, 1)
        return faces

    def _detect_raw(
        self, rgb: np.ndarray, scale_factor: float, step: int, min_neighbors: int
    ) -> list[DetectedFace]:
        casc = self._cascade
        win = casc.window
        gray = _luma(rgb)
        h0, w0 = gray.shape
        detections: list[tuple[int, int, int, int]] = []

        scale = 1.0
        while min(h0, w0) / scale >= win:
            new_w = max(win, round(w0 / scale))
            new_h = max(win, round(h0 / scale))
            small = np.asarray(
                Image.fromarray(gray, "L").resize((new_w, new_h), Image.Resampling.BILINEAR),
                dtype=np.float64,
            )
            detections.extend(
                (int(x * scale), int(y * scale), int(win * scale), int(win * scale))
                for (x, y) in self._scan_level(small, step)
            )
            scale *= scale_factor

        return _group(detections, min_neighbors)

    def _scan_level(self, img: np.ndarray, step: int) -> list[tuple[int, int]]:
        casc = self._cascade
        win = casc.window
        h, w = img.shape
        if h < win or w < win:
            return []
        ii, ii2 = _integral(img)
        ys = np.arange(0, h - win + 1, step)
        xs = np.arange(0, w - win + 1, step)
        mesh_x, mesh_y = np.meshgrid(xs, ys)
        gx = mesh_x.ravel()
        gy = mesh_y.ravel()

        inv_area = 1.0 / (win * win)
        win_sum = _rect_sum(ii, gy, gx, 0, 0, win, win)
        win_sq = _rect_sum(ii2, gy, gx, 0, 0, win, win)
        mean = win_sum * inv_area
        var = win_sq * inv_area - mean * mean
        std = np.sqrt(np.where(var > 0, var, 1.0))

        alive = np.arange(gx.size)
        for stage in casc.stages:
            if alive.size == 0:
                break
            cx, cy, cstd = gx[alive], gy[alive], std[alive]
            acc = np.zeros(alive.size, dtype=np.float64)
            for stump in stage.stumps:
                feat = casc.features[stump.feature]
                val = np.zeros(alive.size, dtype=np.float64)
                for (rx, ry, rw, rh, weight) in feat.rects:
                    val += weight * _rect_sum(ii, cy, cx, rx, ry, rw, rh)
                val *= inv_area
                acc += np.where(val < stump.threshold * cstd, stump.left, stump.right)
            keep = acc >= stage.threshold
            alive = alive[keep]

        return [(int(gx[i]), int(gy[i])) for i in alive]

    def embed(self, rgb: np.ndarray, face: DetectedFace) -> np.ndarray:
        return lbph_embedding(rgb, face)


def _luma(rgb: np.ndarray) -> np.ndarray:
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.clip(y, 0, 255).astype(np.uint8)


def _group(boxes: list[tuple[int, int, int, int]], min_neighbors: int) -> list[DetectedFace]:
    """Rudimentary agglomerative NMS: cluster boxes whose centres and sizes are close."""
    if not boxes:
        return []
    used = [False] * len(boxes)
    clusters: list[list[tuple[int, int, int, int]]] = []
    for i, bi in enumerate(boxes):
        if used[i]:
            continue
        group = [bi]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            bj = boxes[j]
            if _similar(bi, bj):
                group.append(bj)
                used[j] = True
        clusters.append(group)

    out: list[DetectedFace] = []
    for group in clusters:
        if len(group) < min_neighbors:
            continue
        arr = np.array(group, dtype=np.float64)
        x, y, w, h = arr.mean(0)
        out.append(DetectedFace(int(x), int(y), int(w), int(h), float(len(group))))
    out.sort(key=lambda f: f.area, reverse=True)
    return out


def _similar(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    tol = 0.5 * (aw + bw) / 2
    center_close = abs((ax + aw / 2) - (bx + bw / 2)) <= tol and abs(
        (ay + ah / 2) - (by + bh / 2)
    ) <= tol
    size_close = 0.6 <= (aw / max(bw, 1)) <= 1.7
    return center_close and size_close
