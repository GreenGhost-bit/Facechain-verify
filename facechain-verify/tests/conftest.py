"""Shared fixtures.

The bundled ``tests/fixtures/*.jpg`` are public-domain portraits (see
``samples/SOURCES.json``). Tests that need a face engine use the OpenCV backend
and are skipped if it is unavailable.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from facechain.config import Settings

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLES = Path(__file__).parent.parent / "samples"


def _opencv_ok() -> bool:
    try:
        from facechain.face.opencv_backend import OpenCVFaceEngine

        return OpenCVFaceEngine.available()
    except Exception:
        return False


requires_opencv = pytest.mark.skipif(not _opencv_ok(), reason="opencv face engine unavailable")


@pytest.fixture
def probe_obama() -> Path:
    return SAMPLES / "probe_obama.jpg"


@pytest.fixture
def probe_repost() -> Path:
    return SAMPLES / "probe_repost.jpg"


@pytest.fixture
def fx() -> Path:
    return FIXTURES


@pytest.fixture
def opencv_engine():  # type: ignore[no-untyped-def]
    from facechain.face.opencv_backend import OpenCVFaceEngine

    if not OpenCVFaceEngine.available():
        pytest.skip("opencv unavailable")
    return OpenCVFaceEngine()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings.load(
        env_file=str(tmp_path / "nonexistent.env"),
        runs_dir=tmp_path / "runs",
        chain_dir=tmp_path / "chain",
        corpus_dir=tmp_path / "corpus",
        search_providers="local",
        anchor_backend="local",
    )


@pytest.fixture
def seeded_corpus(settings: Settings) -> Settings:
    """Copy the bundled fixtures into the tmp corpus with source metadata."""
    from facechain.corpus import seed_demo_corpus

    seed_demo_corpus(settings, repo_root=Path(__file__).parent.parent)
    return settings


@pytest.fixture
def blank_png(tmp_path: Path) -> Path:
    p = tmp_path / "blank.png"
    Image.new("RGB", (256, 256), (200, 200, 200)).save(p)
    return p


@pytest.fixture
def jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    arr = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype("uint8")
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def reencode(src: Path, dst: Path, *, scale: float = 0.8, quality: int = 70, rotate: int = 0) -> Path:
    img = Image.open(src).convert("RGB")
    img = img.resize((int(img.width * scale), int(img.height * scale)))
    if rotate:
        img = img.rotate(rotate, expand=False, fillcolor=(255, 255, 255))
    img.save(dst, quality=quality)
    return dst


@pytest.fixture
def reencoder():  # type: ignore[no-untyped-def]
    return reencode


@pytest.fixture
def two_face_image() -> Path:
    """Two similarly-sized bundled faces on one canvas -> exercises ambiguity handling.

    Pre-built at ``tests/fixtures/two_faces.jpg`` (Eisenhower + Kennedy crops) so
    detection is deterministic across OpenCV versions.
    """
    p = FIXTURES / "two_faces.jpg"
    if not p.is_file():  # pragma: no cover - regenerate if missing
        a = Image.open(FIXTURES / "corpus_eisenhower.jpg").convert("RGB").resize((320, 320))
        b = Image.open(FIXTURES / "corpus_kennedy.jpg").convert("RGB").resize((320, 320))
        canvas = Image.new("RGB", (720, 360), (250, 250, 250))
        canvas.paste(a, (20, 20))
        canvas.paste(b, (380, 20))
        canvas.save(p, quality=92)
    return p
