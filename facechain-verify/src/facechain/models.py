"""Pydantic schema for every artifact the pipeline persists.

The :class:`EvidenceBundle` is the single object that gets notarised: its
canonical SHA-256 (``record_hash``) is the only value written on-chain. All
numeric quantities are integers (pixels, or parts-per-million for scores) so the
canonical encoding is byte-stable -- see :mod:`facechain.canonical`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, NonNegativeInt

from . import PIPELINE_VERSION
from .canonical import hash_object

SCHEMA_VERSION = "1.0.0"


def utc_now_rfc3339() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class StrictModel(BaseModel):
    model_config = {"extra": "forbid", "frozen": False}


class Fingerprint(StrictModel):
    """Exact + perceptual identity of an image blob."""

    sha256: str
    phash: str  # 16-hex-char 64-bit perceptual hash (DCT)
    dhash: str  # 16-hex-char 64-bit difference hash
    byte_len: NonNegativeInt
    width: NonNegativeInt
    height: NonNegativeInt
    mime: str

    def matches(self, other: Fingerprint, *, max_hamming: int = 10) -> tuple[bool, dict[str, Any]]:
        """Exact-or-perceptual equality with a diagnostic breakdown."""
        exact = self.sha256 == other.sha256
        ph = _hamming_hex(self.phash, other.phash)
        dh = _hamming_hex(self.dhash, other.dhash)
        perceptual = ph <= max_hamming and dh <= max_hamming
        return (exact or perceptual), {
            "exact_sha256": exact,
            "phash_hamming": ph,
            "dhash_hamming": dh,
            "perceptual_ok": perceptual,
            "max_hamming": max_hamming,
        }


class FaceRecord(StrictModel):
    """A detected + encoded face."""

    engine: str
    engine_version: str
    bbox: list[NonNegativeInt] = Field(min_length=4, max_length=4)  # x, y, w, h
    detection_score_ppm: int
    quality_ppm: int
    embedding_dim: NonNegativeInt
    embedding_sha256: str
    all_bboxes: list[list[NonNegativeInt]] = Field(default_factory=list)
    note: str = ""


class Candidate(StrictModel):
    """One search hit, after the pipeline has scored it against the probe."""

    provider: str
    post_url: str  # the human-facing page / social post
    image_url: str
    title: str = ""
    snippet: str = ""
    fetched: bool = False
    image_fingerprint: Fingerprint | None = None
    similarity_ppm: int = 0
    rank: NonNegativeInt = 0
    note: str = ""


class MatchResult(StrictModel):
    threshold_ppm: int
    decided_by: Literal["embedding_cosine"] = "embedding_cosine"
    ambiguous: bool = False
    ambiguity_note: str = ""
    best: Candidate
    ranked: list[Candidate] = Field(default_factory=list)


class SearchSummary(StrictModel):
    providers_run: list[str]
    providers_ok: list[str]
    providers_failed: dict[str, str] = Field(default_factory=dict)
    candidates_seen: NonNegativeInt
    candidates_scored: NonNegativeInt


class EvidenceBundle(StrictModel):
    """The notarised object. ``record_hash`` covers every other field."""

    schema_version: str = SCHEMA_VERSION
    pipeline_version: str = PIPELINE_VERSION
    run_id: str
    created_at: str = Field(default_factory=utc_now_rfc3339)

    probe_image_fingerprint: Fingerprint
    probe_face: FaceRecord
    probe_embedding_sha256: str

    search: SearchSummary
    match: MatchResult

    record_hash: str = ""

    # Fields excluded from the notarised payload: ``record_hash`` is the output;
    # ``run_id`` is a local directory name; ``created_at`` is a wall-clock stamp
    # whose trusted version is supplied by the blockchain block/tx itself. The
    # hash therefore covers the *finding* (probe + face + match), which makes it
    # content-addressed and safely idempotent across re-runs.
    _HASH_EXCLUDE: ClassVar[set[str]] = {"record_hash", "run_id", "created_at"}

    def _payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude=self._HASH_EXCLUDE)

    def compute_record_hash(self) -> str:
        return hash_object(self._payload())

    def finalized(self) -> EvidenceBundle:
        """Return a copy with ``record_hash`` populated."""
        return self.model_copy(update={"record_hash": self.compute_record_hash()})

    def verify_self(self) -> bool:
        return bool(self.record_hash) and self.record_hash == self.compute_record_hash()


class MerkleProofStep(StrictModel):
    sibling: str
    position: Literal["left", "right"]


class AnchorReceipt(StrictModel):
    """Where and how the ``record_hash`` was written, plus everything needed to
    read it back independently."""

    backend: str
    network: str
    record_hash: str
    anchored_at: str = Field(default_factory=utc_now_rfc3339)
    idempotent_hit: bool = False
    ref: dict[str, Any] = Field(default_factory=dict)

    # local-chain specifics (also mirrored into ``ref`` for uniformity)
    block_index: int | None = None
    block_hash: str | None = None
    merkle_root: str | None = None
    leaf_index: int | None = None
    merkle_proof: list[MerkleProofStep] = Field(default_factory=list)


class Check(StrictModel):
    name: str
    ok: bool
    detail: str = ""
    expected: str = ""
    actual: str = ""


class VerificationReport(StrictModel):
    run_id: str
    record_hash: str
    verified_at: str = Field(default_factory=utc_now_rfc3339)
    backend: str
    checks: list[Check] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    def add(
        self, name: str, ok: bool, *, detail: str = "", expected: str = "", actual: str = ""
    ) -> Check:
        c = Check(name=name, ok=ok, detail=detail, expected=expected, actual=actual)
        self.checks.append(c)
        return c


class RunManifest(StrictModel):
    """Top-level index for a ``runs/<run_id>/`` directory."""

    run_id: str
    created_at: str = Field(default_factory=utc_now_rfc3339)
    pipeline_version: str = PIPELINE_VERSION
    input_path: str
    settings_digest: str
    record_hash: str
    anchor_backend: str
    anchor_network: str
    artifacts: dict[str, str] = Field(default_factory=dict)
    status: Literal["ok", "no_match", "error"] = "ok"
    error: str = ""


def _hamming_hex(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b)) * 4
    return bin(int(a, 16) ^ int(b, 16)).count("1")
