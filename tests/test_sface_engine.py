"""YuNet + SFace engine: detection, landmark-aligned embeddings, identity margin.

Skipped automatically unless OpenCV's DNN face API is present and the SFace model
is already cached locally (the CI job runs ``facechain fetch-models`` first).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from facechain.face import calibration, cosine, encode_candidate, encode_probe
from facechain.imaging import load_image_path

SAMPLES = Path(__file__).parent.parent / "samples"
FIXTURES = Path(__file__).parent / "fixtures"


def _sface_ok() -> bool:
    try:
        from facechain.face.sface_backend import SFaceEngine

        return SFaceEngine.available()
    except Exception:
        return False


requires_sface = pytest.mark.skipif(not _sface_ok(), reason="sface engine/model unavailable")


@pytest.fixture
def sface_engine():  # type: ignore[no-untyped-def]
    from facechain.face.sface_backend import SFaceEngine

    if not SFaceEngine.available():
        pytest.skip("sface unavailable")
    return SFaceEngine()


@requires_sface
def test_detects_landmarks(sface_engine) -> None:
    img = load_image_path(SAMPLES / "probe_obama.jpg")
    faces = sface_engine.detect(img.rgb)
    assert faces, "expected at least one detected face"
    lm = faces[0].landmarks
    assert lm is not None and len(lm) == 5
    assert all(0 <= x <= img.rgb.shape[1] and 0 <= y <= img.rgb.shape[0] for x, y in lm)


@requires_sface
def test_embedding_shape_and_norm(sface_engine) -> None:
    img = load_image_path(SAMPLES / "probe_obama.jpg")
    rec, emb, ambiguous = encode_probe(img, engine=sface_engine)
    assert emb.shape == (128,)
    assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)
    assert rec.engine == "yunet-sface"
    assert ambiguous is False


@requires_sface
def test_embedding_is_deterministic(sface_engine) -> None:
    img = load_image_path(SAMPLES / "probe_obama.jpg")
    _, e1, _ = encode_probe(img, engine=sface_engine)
    _, e2, _ = encode_probe(img, engine=sface_engine)
    assert cosine(e1, e2) == pytest.approx(1.0, abs=1e-6)


@requires_sface
def test_identity_margin_beats_lbph(sface_engine) -> None:
    """Genuine same-person pair scores far above every impostor pair."""
    probe = load_image_path(SAMPLES / "probe_obama.jpg")
    _, probe_emb, _ = encode_probe(probe, engine=sface_engine)

    genuine = encode_candidate(
        load_image_path(FIXTURES / "corpus_obama_reencode.jpg").rgb, engine=sface_engine
    )
    impostors = [
        encode_candidate(load_image_path(FIXTURES / name).rgb, engine=sface_engine)
        for name in ("corpus_kennedy.jpg", "corpus_reagan.jpg", "corpus_eisenhower.jpg")
    ]
    g = cosine(probe_emb, genuine)
    worst_impostor = max(cosine(probe_emb, e) for e in impostors)
    assert g > 0.9
    assert worst_impostor < 0.4
    assert g - worst_impostor > 0.4  # a wide, calibratable gap


@requires_sface
def test_calibration_bands(sface_engine) -> None:
    name = sface_engine.name
    assert calibration.default_threshold(name) == pytest.approx(0.40, abs=1e-6)
    assert calibration.band(name, 0.98) == "match"
    assert calibration.band(name, 0.30) == "no-match"
    assert 0.0 <= calibration.probability(name, 0.5) <= 1.0
