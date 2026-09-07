"""Per-engine score calibration.

Cosine similarity is not comparable across engines: a genuine SFace/ArcFace pair
sits near 0.4-1.0 while the classical LBPH descriptor puts genuine pairs near
0.90. Each engine therefore carries:

* ``threshold`` -- the accept/reject cut, chosen near FAR 1e-3;
* ``(center, scale)`` -- a logistic that maps cosine -> P(same identity), so the
  CLI / report can show a real confidence and a ``match / likely / no-match``
  band instead of a bare number.

The defaults come from each model's published operating point plus the bundled
fixture pairs; ``python -m facechain.bench calibrate`` re-fits them from a labelled
pair set and rewrites this table's values in ``calibration.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

_BUILTIN: dict[str, dict[str, float]] = {
    # SFace: OpenCV's reference cosine threshold is 0.363; fixtures separate
    # ~0.98 genuine vs <0.25 impostor, so 0.40 buys precision cheaply.
    "yunet-sface": {"threshold": 0.40, "center": 0.40, "scale": 0.06},
    # ArcFace (insightface buffalo_l, normed embeddings).
    "insightface-arcface": {"threshold": 0.42, "center": 0.42, "scale": 0.08},
    # Classical texture descriptor: genuine pairs ride high, margin is thin.
    "opencv-haar-lbph": {"threshold": 0.86, "center": 0.86, "scale": 0.03},
    "numpy-violajones-lbph": {"threshold": 0.86, "center": 0.86, "scale": 0.03},
}

# Bands on the calibrated probability.
_LIKELY_P = 0.5
_MATCH_P = 0.85


def _load_overrides() -> dict[str, dict[str, float]]:
    path = Path(__file__).with_name("calibration.json")
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def params_for(engine_name: str) -> dict[str, float]:
    merged = dict(_BUILTIN.get(engine_name, {"threshold": 0.5, "center": 0.5, "scale": 0.08}))
    merged.update(_load_overrides().get(engine_name, {}))
    return merged


def default_threshold(engine_name: str) -> float:
    return float(params_for(engine_name)["threshold"])


def probability(engine_name: str, cosine: float) -> float:
    """Calibrated P(same identity) in ``[0, 1]`` for a cosine score."""
    p = params_for(engine_name)
    z = (cosine - p["center"]) / max(p["scale"], 1e-6)
    return 1.0 / (1.0 + math.exp(-z))


def band(engine_name: str, cosine: float) -> str:
    """``"match"`` | ``"likely"`` | ``"no-match"`` from the calibrated probability."""
    prob = probability(engine_name, cosine)
    if prob >= _MATCH_P:
        return "match"
    if prob >= _LIKELY_P:
        return "likely"
    return "no-match"
