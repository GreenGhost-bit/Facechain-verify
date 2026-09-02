from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests.conftest import requires_opencv

from facechain.errors import NoFaceFoundError
from facechain.face import cosine, embedding_sha256, encode_probe
from facechain.face.base import DetectedFace
from facechain.face.descriptor import EMBED_DIM, lbph_embedding
from facechain.imaging import load_image_path


@requires_opencv
def test_detects_the_bundled_face(opencv_engine, probe_obama: Path):
    img = load_image_path(probe_obama)
    faces = opencv_engine.detect(img.rgb)
    assert len(faces) >= 1
    assert faces[0].area >= 60 * 60


@requires_opencv
def test_embedding_is_unit_norm_and_right_shape(opencv_engine, probe_obama: Path):
    img = load_image_path(probe_obama)
    rec, emb, ambiguous = encode_probe(img, engine=opencv_engine)
    assert emb.shape == (EMBED_DIM,)
    assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)
    assert rec.embedding_dim == EMBED_DIM
    assert rec.embedding_sha256 == embedding_sha256(emb)
    assert ambiguous is False


@requires_opencv
def test_embedding_is_deterministic(opencv_engine, probe_obama: Path):
    img = load_image_path(probe_obama)
    _, e1, _ = encode_probe(img, engine=opencv_engine)
    _, e2, _ = encode_probe(img, engine=opencv_engine)
    assert embedding_sha256(e1) == embedding_sha256(e2)
    assert cosine(e1, e2) == pytest.approx(1.0, abs=1e-6)


@requires_opencv
def test_same_photo_reencode_beats_impostors(opencv_engine, probe_obama: Path, fx: Path, tmp_path, reencoder):
    img = load_image_path(probe_obama)
    _, probe_emb, _ = encode_probe(img, engine=opencv_engine)

    variant = reencoder(probe_obama, tmp_path / "v.jpg", scale=0.72, quality=70, rotate=-3)
    _, near_emb, _ = encode_probe(load_image_path(variant), engine=opencv_engine)
    near = cosine(probe_emb, near_emb)

    impostor_scores = []
    for name in ("corpus_eisenhower.jpg", "corpus_kennedy.jpg", "corpus_reagan.jpg"):
        _, e, _ = encode_probe(load_image_path(fx / name), engine=opencv_engine)
        impostor_scores.append(cosine(probe_emb, e))

    assert near >= 0.88
    assert near > max(impostor_scores) + 0.05
    assert max(impostor_scores) < 0.86  # default threshold cleanly rejects impostors


@requires_opencv
def test_blank_image_raises_no_face(opencv_engine, blank_png: Path):
    with pytest.raises(NoFaceFoundError):
        encode_probe(load_image_path(blank_png), engine=opencv_engine)


@requires_opencv
def test_two_comparable_faces_flag_ambiguous(opencv_engine, two_face_image: Path):
    rec, _, ambiguous = encode_probe(load_image_path(two_face_image), engine=opencv_engine)
    assert len(rec.all_bboxes) >= 2
    assert ambiguous is True
    assert "comparably sized" in rec.note


def test_lbph_embedding_pure_numpy_shape():
    rng = np.random.default_rng(1)
    rgb = (rng.random((200, 200, 3)) * 255).astype(np.uint8)
    v = lbph_embedding(rgb, DetectedFace(20, 20, 150, 150))
    assert v.shape == (EMBED_DIM,)
    assert np.isfinite(v).all()
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)


def test_numpy_viola_jones_engine_detects_bundled_face(probe_obama: Path):
    from facechain.face.numpy_vj_backend import NumpyViolaJonesEngine

    img = load_image_path(probe_obama)
    faces = NumpyViolaJonesEngine().detect(img.rgb)
    assert len(faces) >= 1
