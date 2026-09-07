"""Image description: deterministic attributes always; VLM caption when installed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from facechain.describe import caption_available, describe_image, image_attributes
from facechain.imaging import load_image_path

SAMPLES = Path(__file__).parent.parent / "samples"
FIXTURES = Path(__file__).parent / "fixtures"


def test_attributes_are_deterministic_and_sane() -> None:
    rgb = load_image_path(SAMPLES / "probe_obama.jpg").rgb
    a1 = image_attributes(rgb)
    a2 = image_attributes(rgb)
    assert a1 == a2
    assert a1["width"] == 444 and a1["height"] == 600
    assert 0.0 <= a1["mean_brightness"] <= 1.0
    assert a1["sharpness_lapvar"] > 0
    assert 1 <= len(a1["dominant_colours"]) <= 4
    assert all(c["hex"].startswith("#") and 0 <= c["fraction"] <= 1 for c in a1["dominant_colours"])


def test_low_light_and_blur_flags() -> None:
    dark = np.full((64, 64, 3), 10, dtype=np.uint8)
    a = image_attributes(dark)
    assert a["is_low_light"] is True
    assert a["is_blurry"] is True


def test_describe_without_caption_has_no_model_download() -> None:
    r = describe_image(SAMPLES / "probe_obama.jpg", with_caption=False, with_faces=False)
    assert r["caption"] is None
    assert r["caption_model"] is None
    assert "attributes" in r


def test_describe_reports_face_count() -> None:
    r = describe_image(FIXTURES / "two_faces.jpg", with_caption=False, with_faces=True)
    # two_faces.jpg has two portraits side by side
    assert r["attributes"].get("faces_detected", 0) >= 1


@pytest.mark.slow
@pytest.mark.skipif(not caption_available(), reason="[describe] extra (torch+transformers) not installed")
def test_vlm_caption_produces_text() -> None:
    r = describe_image(SAMPLES / "probe_obama.jpg", with_caption=True, with_faces=False)
    assert r.get("caption"), r.get("caption_error")
    assert isinstance(r["caption"], str) and len(r["caption"]) > 3
    assert r["caption_model"]
