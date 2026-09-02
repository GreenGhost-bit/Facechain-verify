from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from facechain.errors import ImageDecodeError, ImageTooLargeError
from facechain.imaging import (
    dhash_hex,
    hamming_hex,
    load_image_bytes,
    load_image_path,
    phash_hex,
)


def test_loads_valid_jpeg_and_populates_fingerprint(jpeg_bytes: bytes):
    img = load_image_bytes(jpeg_bytes, source="x.jpg")
    fp = img.fingerprint
    assert fp.mime == "image/jpeg"
    assert fp.width == 64 and fp.height == 64
    assert len(fp.sha256) == 64
    assert len(fp.phash) == 16 and len(fp.dhash) == 16
    assert img.rgb.shape == (64, 64, 3)
    assert img.gray.shape == (64, 64)


def test_rejects_empty_payload():
    with pytest.raises(ImageDecodeError):
        load_image_bytes(b"", source="empty")


def test_rejects_non_image_bytes():
    with pytest.raises(ImageDecodeError):
        load_image_bytes(b"this is definitely not an image", source="junk")


def test_rejects_oversize_bytes(jpeg_bytes: bytes):
    with pytest.raises(ImageTooLargeError):
        load_image_bytes(jpeg_bytes, max_bytes=10, source="big")


def test_rejects_too_many_pixels():
    buf = io.BytesIO()
    Image.new("RGB", (2000, 2000), (1, 2, 3)).save(buf, format="PNG")
    with pytest.raises(ImageTooLargeError):
        load_image_bytes(buf.getvalue(), max_pixels=1_000_000, source="wide")


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(ImageDecodeError):
        load_image_path(tmp_path / "nope.jpg")


def test_perceptual_hash_survives_reencode(probe_obama: Path, tmp_path: Path, reencoder):
    original = load_image_path(probe_obama).fingerprint
    variant_path = reencoder(probe_obama, tmp_path / "v.jpg", scale=0.7, quality=60, rotate=0)
    variant = load_image_path(variant_path).fingerprint
    assert original.sha256 != variant.sha256  # exact bytes changed
    assert hamming_hex(original.phash, variant.phash) <= 12
    assert hamming_hex(original.dhash, variant.dhash) <= 14
    ok, diag = variant.matches(original, max_hamming=16)
    assert ok, diag


def test_phash_dhash_are_deterministic(probe_obama: Path):
    a = load_image_path(probe_obama)
    b = load_image_path(probe_obama)
    assert a.fingerprint.phash == b.fingerprint.phash
    assert phash_hex(a.rgb) == phash_hex(b.rgb)
    assert dhash_hex(a.rgb) == dhash_hex(b.rgb)


def test_distinct_images_have_distant_phash(probe_obama: Path, fx: Path):
    obama = load_image_path(probe_obama).fingerprint
    lincoln = load_image_path(fx / "probe_lincoln.jpg").fingerprint
    assert hamming_hex(obama.phash, lincoln.phash) >= 10
