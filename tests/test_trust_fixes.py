"""Regression guards for the notarised-evidence integrity fixes.

* the advisory ``identity_*`` fields (scraped from live page text) must NOT
  change ``record_hash`` -- the record stays byte-reproducible;
* ``decided_by`` must tell the truth about whether text-consensus moved the pick.
"""

from __future__ import annotations

from facechain.models import (
    Candidate,
    EvidenceBundle,
    FaceRecord,
    Fingerprint,
    MatchResult,
    SearchSummary,
)


def _bundle(**match_kw: object) -> EvidenceBundle:
    cand = Candidate(provider="local", post_url="https://example/p", image_url="https://example/i")
    return EvidenceBundle(
        run_id="r1",
        probe_image_fingerprint=Fingerprint(
            sha256="a" * 64, phash="0" * 16, dhash="0" * 16,
            byte_len=10, width=8, height=8, mime="image/jpeg",
        ),
        probe_face=FaceRecord(
            engine="yunet-sface", engine_version="v", bbox=[0, 0, 8, 8],
            detection_score_ppm=1, quality_ppm=1, embedding_dim=128,
            embedding_sha256="b" * 64,
        ),
        probe_embedding_sha256="b" * 64,
        search=SearchSummary(
            providers_run=["local"], providers_ok=["local"],
            candidates_seen=1, candidates_scored=1,
        ),
        match=MatchResult(threshold_ppm=400_000, best=cand, ranked=[cand], **match_kw),
    )


def test_identity_fields_excluded_from_record_hash() -> None:
    plain = _bundle().finalized()
    with_identity = _bundle(
        identity_guess="Some Person",
        identity_confidence_ppm=730_000,
        identity_note="4/10 top hits mention 'Some Person'",
    ).finalized()
    assert plain.record_hash == with_identity.record_hash
    assert with_identity.verify_self()


def test_record_hash_still_covers_the_real_finding() -> None:
    base = _bundle().finalized()
    moved = _bundle()
    moved = moved.model_copy(update={"probe_embedding_sha256": "c" * 64})
    assert moved.finalized().record_hash != base.record_hash


def test_decided_by_accepts_consensus_literal() -> None:
    b = _bundle(decided_by="embedding_cosine+text_consensus").finalized()
    assert b.match.decided_by == "embedding_cosine+text_consensus"
    # and it is part of the hash (it describes how the finding was made)
    other = _bundle(decided_by="embedding_cosine").finalized()
    assert b.record_hash != other.record_hash
