"""ProbeContext.reverse_image_bytes(): face crop for reverse-image engines."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from facechain.config import Settings
from facechain.imaging import load_image_path
from facechain.search.base import ProbeContext

SAMPLES = Path(__file__).parent.parent / "samples"


def _ctx(settings: Settings, bbox: list[int] | None) -> ProbeContext:
    img = load_image_path(SAMPLES / "probe_obama.jpg")
    return ProbeContext(
        image_bytes=img.raw_bytes,
        rgb=img.rgb,
        embedding=np.zeros(128, dtype="float32"),
        settings=settings,
        fetcher=None,  # type: ignore[arg-type]
        face_engine=None,  # type: ignore[arg-type]
        face_bbox=bbox,
    )


def test_default_sends_a_tighter_face_crop() -> None:
    s = Settings.load(env_file="/nonexistent")
    ctx = _ctx(s, [94, 72, 246, 353])
    crop = ctx.reverse_image_bytes()
    assert crop != ctx.image_bytes
    cw, ch = Image.open(io.BytesIO(crop)).size
    fw, fh = Image.open(io.BytesIO(ctx.image_bytes)).size
    assert cw <= fw and ch <= fh
    # the crop still contains the face box plus margin, so it isn't tiny
    assert cw >= 246 and ch >= 353


def test_full_frame_when_flag_off() -> None:
    s = Settings.load(env_file="/nonexistent", reverse_image_face_crop=False)
    ctx = _ctx(s, [94, 72, 246, 353])
    assert ctx.reverse_image_bytes() == ctx.image_bytes


def test_full_frame_when_no_bbox() -> None:
    s = Settings.load(env_file="/nonexistent")
    ctx = _ctx(s, None)
    assert ctx.reverse_image_bytes() == ctx.image_bytes


def test_env_toggle() -> None:
    s = Settings.load(env_file="/nonexistent", **{})
    assert s.reverse_image_face_crop is True
    import os

    os.environ["FACECHAIN_REVERSE_IMAGE_FACE_CROP"] = "0"
    try:
        s2 = Settings.load(env_file="/nonexistent")
        assert s2.reverse_image_face_crop is False
    finally:
        del os.environ["FACECHAIN_REVERSE_IMAGE_FACE_CROP"]
