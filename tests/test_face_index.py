"""Face-embedding corpus index + the faceindex provider."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from facechain.config import Settings
from facechain.search.face_index import FaceIndex
from facechain.search.face_index_provider import FaceIndexProvider

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLES = Path(__file__).parent.parent / "samples"


def _engine():  # type: ignore[no-untyped-def]
    from facechain.face.factory import build_face_engine

    try:
        return build_face_engine("auto")
    except Exception:
        pytest.skip("no face engine available")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    d = tmp_path / "corpus"
    d.mkdir()
    for name, title in [
        ("corpus_obama_reencode.jpg", "Barack Obama"),
        ("corpus_kennedy.jpg", "John F. Kennedy"),
        ("corpus_reagan.jpg", "Ronald Reagan"),
        ("corpus_eisenhower.jpg", "Dwight D. Eisenhower"),
    ]:
        (d / name).write_bytes((FIXTURES / name).read_bytes())
        (d / name).with_suffix(".json").write_text(
            json.dumps({"post_url": f"https://example/{name}", "title": title, "source": "test"})
        )
    return d


def test_build_persists_and_reloads(corpus: Path) -> None:
    eng = _engine()
    idx = FaceIndex(corpus, eng)
    n = idx.build()
    assert n == 4
    assert (corpus / ".faceindex" / f"{eng.name}.npz").is_file()
    assert (corpus / ".faceindex" / f"{eng.name}.meta.json").is_file()

    # a fresh instance loads from cache without re-encoding
    reloaded = FaceIndex(corpus, eng)
    reloaded.build()
    assert reloaded.size == 4


def test_query_ranks_true_identity_first(corpus: Path) -> None:
    from facechain.face import encode_probe
    from facechain.imaging import load_image_path

    eng = _engine()
    idx = FaceIndex(corpus, eng).load_or_build()
    _, probe_emb, _ = encode_probe(load_image_path(SAMPLES / "probe_obama.jpg"), engine=eng)

    hits = idx.query(probe_emb, top_k=4)
    assert hits, "expected ranked hits"
    assert hits[0].title == "Barack Obama"
    if eng.name == "yunet-sface":
        assert hits[0].score > 0.9
        assert hits[1].score < 0.5  # wide margin to the next identity


def test_provider_returns_topk_candidates(corpus: Path) -> None:
    from types import SimpleNamespace

    from facechain.face import encode_probe
    from facechain.imaging import load_image_path

    eng = _engine()
    FaceIndex(corpus, eng).build()
    _, probe_emb, _ = encode_probe(load_image_path(SAMPLES / "probe_obama.jpg"), engine=eng)

    settings = Settings.load(env_file="/nonexistent", corpus_dir=corpus)
    probe = SimpleNamespace(embedding=probe_emb, face_engine=eng, settings=settings)
    cands = list(FaceIndexProvider(corpus).search(probe))  # type: ignore[arg-type]
    assert cands
    assert cands[0].provider == "faceindex"
    assert "Barack Obama" in cands[0].title
    assert cands[0].image_url.startswith("file://")


def test_provider_unavailable_on_empty_corpus(tmp_path: Path) -> None:
    settings = Settings.load(env_file="/nonexistent", corpus_dir=tmp_path / "nope")
    assert FaceIndexProvider.available(settings) is False
