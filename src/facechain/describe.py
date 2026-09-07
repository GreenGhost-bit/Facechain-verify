"""Describe an arbitrary image: deterministic attributes + an optional VLM caption.

Two layers:

* **attributes** -- always available, no model: dimensions, aspect, brightness,
  Laplacian-variance sharpness, dominant colours, and (via the face engine) how
  many faces and how big/clear the largest one is.
* **caption** -- a free-text sentence from a local vision-language model
  (default ``Salesforce/blip-image-captioning-base``). Needs the ``[describe]``
  extra (``torch`` + ``transformers``); the model is downloaded once on first
  use and then runs fully offline on CPU.

This is advisory context only. It never touches the face-match decision, the
evidence bundle, or ``record_hash``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .errors import ConfigError
from .imaging import load_image_bytes
from .logging import LOG

_DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"


def caption_available() -> bool:
    import importlib.util as ilu

    return all(ilu.find_spec(m) is not None for m in ("torch", "transformers"))


@lru_cache(maxsize=2)
def _captioner(model_id: str) -> Any:
    """Return a ``(image) -> str`` callable backed by a local BLIP model on CPU."""
    if not caption_available():
        raise ConfigError(
            "image captioning needs the '[describe]' extra: pip install -e '.[describe]'"
        )
    import torch  # noqa: F401  (ensures the backend is importable)
    from transformers import BlipForConditionalGeneration, BlipProcessor

    LOG.info("describe.model_loading", model=model_id)
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    model.eval()  # type: ignore[no-untyped-call]

    def _caption(img: Image.Image) -> str:
        import torch as _t

        inputs = processor(images=img, return_tensors="pt")
        with _t.no_grad():
            ids = model.generate(**inputs, max_new_tokens=40)
        text = processor.decode(ids[0], skip_special_tokens=True)  # type: ignore[no-untyped-call]
        return str(text).strip()

    return _caption


# --------------------------------------------------------------------------
# deterministic attributes
# --------------------------------------------------------------------------
def _dominant_colours(rgb: np.ndarray, k: int = 4) -> list[dict[str, Any]]:
    img = Image.fromarray(rgb, "RGB").resize((80, 80)).quantize(
        colors=k, method=Image.Quantize.FASTOCTREE
    )
    pal = list(img.getpalette() or [])
    counts: list[tuple[int, int]] = []
    for entry in img.getcolors() or []:
        if isinstance(entry, tuple) and len(entry) == 2:
            cnt, palette_idx = entry
            if isinstance(cnt, int) and isinstance(palette_idx, int):
                counts.append((cnt, palette_idx))
    counts.sort(reverse=True)
    total = sum(c for c, _ in counts) or 1
    out: list[dict[str, Any]] = []
    for count, i in counts[:k]:
        r, g, b = pal[i * 3], pal[i * 3 + 1], pal[i * 3 + 2]
        out.append({"hex": f"#{r:02x}{g:02x}{b:02x}", "fraction": round(count / total, 3)})
    return out


def _sharpness(gray: np.ndarray) -> float:
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    return float(np.var(lap))


def image_attributes(rgb: np.ndarray, *, face_engine: Any | None = None) -> dict[str, Any]:
    h, w = rgb.shape[:2]
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float64)
    attrs: dict[str, Any] = {
        "width": int(w),
        "height": int(h),
        "aspect_ratio": round(w / h, 3) if h else 0.0,
        "megapixels": round(w * h / 1e6, 2),
        "mean_brightness": round(float(gray.mean()) / 255.0, 3),
        "sharpness_lapvar": round(_sharpness(gray), 1),
        "is_low_light": bool(gray.mean() < 60),
        "is_blurry": bool(_sharpness(gray) < 100),
        "dominant_colours": _dominant_colours(rgb),
    }
    if face_engine is not None:
        try:
            faces = face_engine.detect(rgb)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("describe.detect_failed", error=str(exc))
            faces = []
        attrs["faces_detected"] = len(faces)
        if faces:
            big = max(faces, key=lambda f: f.area)
            attrs["largest_face"] = {
                "bbox": big.bbox,
                "fraction_of_frame": round(big.area / (w * h), 3) if w * h else 0.0,
                "detection_score": round(float(big.det_score), 3),
            }
    return attrs


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def describe_image(
    source: str | Path | np.ndarray,
    *,
    with_caption: bool = True,
    with_faces: bool = True,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Return ``{"attributes": {...}, "caption": str|None, "caption_model": str|None}``."""
    if isinstance(source, np.ndarray):
        rgb = source
    else:
        rgb = load_image_bytes(Path(source).read_bytes(), source=str(source)).rgb

    engine = None
    if with_faces:
        try:
            from .face.factory import build_face_engine

            engine = build_face_engine("auto")
        except Exception as exc:  # pragma: no cover - engineless env
            LOG.warning("describe.no_engine", error=str(exc))

    result: dict[str, Any] = {
        "attributes": image_attributes(rgb, face_engine=engine),
        "caption": None,
        "caption_model": None,
    }

    if with_caption:
        mid = model_id or os.environ.get("FACECHAIN_CAPTION_MODEL") or _DEFAULT_MODEL
        try:
            cap = _captioner(mid)
            text = cap(Image.fromarray(rgb, "RGB"))
            result["caption"] = text or None
            result["caption_model"] = mid
        except ConfigError as exc:
            result["caption_error"] = str(exc)
        except Exception as exc:  # model download / inference failure
            LOG.warning("describe.caption_failed", model=mid, error=str(exc))
            result["caption_error"] = f"{type(exc).__name__}: {exc}"

    return result
