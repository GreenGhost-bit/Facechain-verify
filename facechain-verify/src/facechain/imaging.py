"""Image loading, hardening, and fingerprinting.

Responsibilities:

* decode untrusted image bytes safely (size cap, pixel cap, format allow-list,
  two-pass ``verify()`` + reload);
* normalise to a deterministic RGB ``numpy`` array with EXIF orientation applied
  and all other metadata stripped;
* produce a :class:`~facechain.models.Fingerprint` (SHA-256 + 64-bit pHash +
  64-bit dHash) that is stable across benign re-encoding.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import ImageDecodeError, ImageTooLargeError
from .models import Fingerprint

# Pillow's own guard against decompression bombs; we set an explicit ceiling and
# also check dimensions ourselves for a precise error message.
Image.MAX_IMAGE_PIXELS = 60_000_000

_ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF"}
_MIME_BY_FORMAT = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
    "GIF": "image/gif",
}


class LoadedImage:
    """A safely decoded image plus its provenance hashes."""

    __slots__ = ("fingerprint", "raw_bytes", "rgb")

    def __init__(self, rgb: np.ndarray, fingerprint: Fingerprint, raw_bytes: bytes) -> None:
        self.rgb = rgb
        self.fingerprint = fingerprint
        self.raw_bytes = raw_bytes

    @property
    def gray(self) -> np.ndarray:
        """ITU-R BT.601 luma, uint8, shape (H, W)."""
        r, g, b = self.rgb[..., 0], self.rgb[..., 1], self.rgb[..., 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        return np.clip(y, 0, 255).astype(np.uint8)


def load_image_bytes(
    data: bytes,
    *,
    max_bytes: int = 25 * 1024 * 1024,
    max_pixels: int = 40_000_000,
    source: str = "<bytes>",
) -> LoadedImage:
    """Decode ``data`` into a :class:`LoadedImage` or raise a typed error."""
    if not data:
        raise ImageDecodeError("empty image payload", detail=source)
    if len(data) > max_bytes:
        raise ImageTooLargeError(
            f"image is {len(data)} bytes, limit is {max_bytes}", detail=source
        )

    # Pass 1: structural validation on a throwaway handle.
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageDecodeError("could not decode image", detail=f"{source}: {exc}") from exc

    fmt = (probe.format or "").upper()
    if fmt not in _ALLOWED_FORMATS:
        raise ImageDecodeError(f"unsupported image format {fmt!r}", detail=source)

    # Pass 2: real decode (verify() leaves the image unusable).
    try:
        img: Image.Image = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img) or img  # bake in orientation, drop the tag
    except (UnidentifiedImageError, OSError, ValueError) as exc:  # pragma: no cover - rare
        raise ImageDecodeError("could not re-open image", detail=f"{source}: {exc}") from exc

    w, h = img.size
    if w * h > max_pixels:
        raise ImageTooLargeError(
            f"image is {w}x{h} = {w * h} px, limit is {max_pixels}", detail=source
        )
    if w < 8 or h < 8:
        raise ImageDecodeError(f"image too small: {w}x{h}", detail=source)

    rgb = np.ascontiguousarray(np.asarray(img.convert("RGB"), dtype=np.uint8))
    fingerprint = Fingerprint(
        sha256=hashlib.sha256(data).hexdigest(),
        phash=phash_hex(rgb),
        dhash=dhash_hex(rgb),
        byte_len=len(data),
        width=w,
        height=h,
        mime=_MIME_BY_FORMAT.get(fmt, "application/octet-stream"),
    )
    return LoadedImage(rgb=rgb, fingerprint=fingerprint, raw_bytes=data)


def load_image_path(
    path: str | Path,
    *,
    max_bytes: int = 25 * 1024 * 1024,
    max_pixels: int = 40_000_000,
) -> LoadedImage:
    p = Path(path)
    if not p.is_file():
        raise ImageDecodeError("input file does not exist", detail=str(p))
    return load_image_bytes(
        p.read_bytes(), max_bytes=max_bytes, max_pixels=max_pixels, source=str(p)
    )


# ---------------------------------------------------------------------------
# Perceptual hashes (dependency-free numpy implementations)
# ---------------------------------------------------------------------------
def _to_gray_small(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(rgb, "RGB").convert("L").resize(size, Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float64)


def _dct_1d(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    k = np.arange(n)
    basis = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    return basis @ matrix


def phash_hex(rgb: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """DCT-based perceptual hash (Zauner 2010). Returns 16 hex chars (64 bits)."""
    img_size = hash_size * highfreq_factor
    pixels = _to_gray_small(rgb, (img_size, img_size))
    dct = _dct_1d(_dct_1d(pixels).T).T
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:, 1:])  # exclude the DC term from the threshold
    bits = (low > med).flatten()
    return _bits_to_hex(bits)


def dhash_hex(rgb: np.ndarray, hash_size: int = 8) -> str:
    """Gradient (difference) hash. Returns 16 hex chars (64 bits)."""
    pixels = _to_gray_small(rgb, (hash_size + 1, hash_size))
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return _bits_to_hex(bits)


def _bits_to_hex(bits: np.ndarray) -> str:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    width = (len(bits) + 3) // 4
    return f"{value:0{width}x}"


def hamming_hex(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b)) * 4
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def fingerprint_bytes(data: bytes, *, source: str = "<bytes>") -> Fingerprint:
    """Fingerprint arbitrary image bytes without keeping the pixel array."""
    return load_image_bytes(data, source=source).fingerprint
