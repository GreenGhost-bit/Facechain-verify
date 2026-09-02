from __future__ import annotations

import pytest
from pydantic import ValidationError

from facechain.models import (
    Candidate,
    EvidenceBundle,
    FaceRecord,
    Fingerprint,
    MatchResult,
    SearchSummary,
    VerificationReport,
)


def _fp(sha: str = "a" * 64) -> Fingerprint:
    return Fingerprint(sha256=sha, phash="0" * 16, dhash="0" * 16, byte_len=10,
                       width=10, height=10, mime="image/jpeg")


def _bundle() -> EvidenceBundle:
    face = FaceRecord(engine="opencv", engine_version="4", bbox=[0, 0, 10, 10],
                      detection_score_ppm=1_000_000, quality_ppm=800_000,
                      embedding_dim=4, embedding_sha256="b" * 64)
    cand = Candidate(provider="local", post_url="https://x/y", image_url="https://x/y.jpg",
                     similarity_ppm=950_000, image_fingerprint=_fp("c" * 64))
    return EvidenceBundle(
        run_id="r1",
        probe_image_fingerprint=_fp(),
        probe_face=face,
        probe_embedding_sha256="b" * 64,
        search=SearchSummary(providers_run=["local"], providers_ok=["local"],
                             candidates_seen=1, candidates_scored=1),
        match=MatchResult(threshold_ppm=860_000, best=cand, ranked=[cand]),
    )


def test_record_hash_is_stable_and_excludes_itself():
    b = _bundle().finalized()
    assert b.record_hash
    assert b.verify_self()
    # recomputing from a reload gives the same hash
    reloaded = EvidenceBundle.model_validate_json(b.model_dump_json())
    assert reloaded.compute_record_hash() == b.record_hash


def test_mutating_any_field_changes_the_record_hash():
    b = _bundle().finalized()
    mutated = b.model_copy(deep=True)
    mutated.match.best.post_url = "https://evil/swap"
    assert mutated.compute_record_hash() != b.record_hash


def test_fingerprint_matches_exact_and_perceptual():
    a = _fp("a" * 64)
    exact_same = _fp("a" * 64)
    ok, diag = a.matches(exact_same)
    assert ok and diag["exact_sha256"]

    near = Fingerprint(sha256="z" * 64, phash="0000000000000001", dhash="0" * 16,
                       byte_len=10, width=10, height=10, mime="image/jpeg")
    ok, diag = a.matches(near, max_hamming=4)
    assert ok and not diag["exact_sha256"] and diag["phash_hamming"] == 1

    far = Fingerprint(sha256="z" * 64, phash="ffffffffffffffff", dhash="ffffffffffffffff",
                      byte_len=10, width=10, height=10, mime="image/jpeg")
    ok, _ = a.matches(far, max_hamming=4)
    assert not ok


def test_verification_report_ok_requires_all_checks():
    r = VerificationReport(run_id="r", record_hash="h", backend="local")
    assert r.ok is False  # no checks yet
    r.add("a", ok=True)
    r.add("b", ok=True)
    assert r.ok is True
    r.add("c", ok=False)
    assert r.ok is False


def test_bundle_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        FaceRecord(engine="x", engine_version="1", bbox=[0, 0, 1, 1],
                   detection_score_ppm=0, quality_ppm=0, embedding_dim=1,
                   embedding_sha256="a", surprise="!")
