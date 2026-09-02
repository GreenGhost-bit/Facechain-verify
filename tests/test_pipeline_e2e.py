"""Full offline pipeline + independent verification, incl. tamper detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.conftest import FIXTURES, requires_opencv

from facechain.config import Settings
from facechain.pipeline import load_bundle, run_pipeline
from facechain.verify import verify_run

pytestmark = requires_opencv


@pytest.fixture
def repost_probe(tmp_path: Path, reencoder) -> Path:
    # a re-encoded copy of a corpus image == the "reposted on social media" case
    return reencoder(FIXTURES / "corpus_reagan.jpg", tmp_path / "probe.jpg", scale=0.75, quality=68, rotate=-2)


@pytest.fixture
def offline_settings(tmp_path: Path) -> Settings:
    s = Settings.load(
        env_file=str(tmp_path / "none.env"),
        runs_dir=tmp_path / "runs",
        chain_dir=tmp_path / "chain",
        corpus_dir=tmp_path / "corpus",
        search_providers="local",
        anchor_backend="local",
    )
    from facechain.corpus import seed_demo_corpus

    seed_demo_corpus(s, repo_root=Path(__file__).parent.parent)
    return s


def test_pipeline_runs_end_to_end_and_self_verifies(offline_settings: Settings, repost_probe: Path):
    result = run_pipeline(repost_probe, offline_settings, verify_after=True)
    assert result.status == "ok"
    assert result.bundle is not None and result.receipt is not None
    assert result.verification is not None and result.verification.ok

    rd = result.run_dir
    for artifact in ("manifest.json", "probe.jpg", "face.json", "embedding.npy",
                     "evidence.json", "receipt.json", "verification.json", "telemetry.jsonl"):
        assert (rd / artifact).is_file(), artifact
    assert any((rd / "candidates").iterdir())

    bundle = load_bundle(rd)
    assert bundle.verify_self()
    assert bundle.record_hash == result.receipt.record_hash
    assert "reagan" in bundle.match.best.post_url.lower() or "Reagan" in bundle.match.best.title


def test_independent_verify_passes_on_fresh_run(offline_settings: Settings, repost_probe: Path):
    result = run_pipeline(repost_probe, offline_settings, verify_after=False)
    report = verify_run(result.run_dir, offline_settings, live_refetch=False)
    assert report.ok, [c.model_dump() for c in report.checks if not c.ok]
    names = {c.name for c in report.checks}
    assert {"evidence.self_consistent", "match.face_recheck", "local.merkle_inclusion"} <= names


def test_verify_detects_tampered_evidence(offline_settings: Settings, repost_probe: Path):
    result = run_pipeline(repost_probe, offline_settings, verify_after=False)
    ev_path = result.run_dir / "evidence.json"
    doc = json.loads(ev_path.read_text())
    doc["match"]["best"]["post_url"] = "https://evil.example/swapped"
    ev_path.write_text(json.dumps(doc))

    report = verify_run(result.run_dir, offline_settings, live_refetch=False)
    assert not report.ok
    assert not next(c for c in report.checks if c.name == "evidence.self_consistent").ok


def test_verify_detects_tampered_chain(offline_settings: Settings, repost_probe: Path):
    result = run_pipeline(repost_probe, offline_settings, verify_after=False)
    ledger = offline_settings.chain_dir / "local" / "blocks.jsonl"
    lines = ledger.read_text().splitlines()
    blk = json.loads(lines[1])
    blk["records"][0] = "d" * 64
    lines[1] = json.dumps(blk, sort_keys=True, separators=(",", ":"))
    ledger.write_text("\n".join(lines) + "\n")

    report = verify_run(result.run_dir, offline_settings, live_refetch=False)
    assert not report.ok
    assert not next(c for c in report.checks if c.name == "local.chain_integrity").ok


def test_rerun_same_input_is_idempotent_on_chain(offline_settings: Settings, repost_probe: Path):
    r1 = run_pipeline(repost_probe, offline_settings, verify_after=False)
    r2 = run_pipeline(repost_probe, offline_settings, verify_after=False)
    assert r1.bundle.record_hash == r2.bundle.record_hash
    assert r2.receipt.idempotent_hit is True
    ledger = offline_settings.chain_dir / "local" / "blocks.jsonl"
    assert len(ledger.read_text().splitlines()) == 2  # genesis + one block only


def test_no_match_writes_manifest_and_returns_status(offline_settings: Settings):
    # Lincoln is not in the seeded corpus -> genuine "no confident match"
    result = run_pipeline(FIXTURES / "probe_lincoln.jpg", offline_settings, verify_after=False)
    assert result.status == "no_match"
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["status"] == "no_match"
    assert (result.run_dir / "no_match_debug.json").is_file()
