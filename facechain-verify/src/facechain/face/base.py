"""Face-engine abstractions shared by every backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class DetectedFace:
    """An axis-aligned face box in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    det_score: float = 1.0

    @property
    def area(self) -> int:
        return int(self.w) * int(self.h)

    @property
    def bbox(self) -> list[int]:
        return [int(self.x), int(self.y), int(self.w), int(self.h)]

    def clipped(self, width: int, height: int) -> DetectedFace:
        x = max(0, min(int(self.x), width - 1))
        y = max(0, min(int(self.y), height - 1))
        w = max(1, min(int(self.w), width - x))
        h = max(1, min(int(self.h), height - y))
        return DetectedFace(x, y, w, h, self.det_score)


@dataclass
class EncodedFace:
    """A detected face plus its L2-normalised embedding."""

    face: DetectedFace
    embedding: np.ndarray  # float32, ||v|| == 1
    engine: str
    engine_version: str
    quality: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class FaceEngine(Protocol):
    """Every backend implements exactly this."""

    name: str
    version: str

    @classmethod
    def available(cls) -> bool: ...

    def detect(self, rgb: np.ndarray) -> list[DetectedFace]: ...

    def embed(self, rgb: np.ndarray, face: DetectedFace) -> np.ndarray: ...


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors, clamped to [-1, 1]."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
