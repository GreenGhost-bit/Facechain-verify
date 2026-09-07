"""Ed25519 operator signature over the record hash."""

from __future__ import annotations

from pathlib import Path

import pytest

from facechain.signing import (
    available,
    generate_keypair,
    load_or_create_key,
    sign_record,
    verify_signature,
)

pytestmark = pytest.mark.skipif(not available(), reason="cryptography not installed")

_HASH = "1d9be96d67108b5e4941010d1e23a2f3d751717ae2306392df3652d046e884f2"


def test_sign_and_verify_roundtrip() -> None:
    priv, _pub = generate_keypair()
    sig = sign_record(_HASH, priv)
    assert sig["alg"] == "Ed25519"
    assert sig["record_hash"] == _HASH
    assert verify_signature(sig, _HASH) is True


def test_verify_rejects_wrong_hash() -> None:
    priv, _pub = generate_keypair()
    sig = sign_record(_HASH, priv)
    assert verify_signature(sig, "00" * 32) is False


def test_verify_rejects_tampered_signature() -> None:
    priv, _pub = generate_keypair()
    sig = sign_record(_HASH, priv)
    sig["signature"] = ("0" if sig["signature"][0] != "0" else "1") + sig["signature"][1:]
    assert verify_signature(sig, _HASH) is False


def test_load_or_create_key_persists(tmp_path: Path) -> None:
    kp = tmp_path / "op.key"
    priv1, pub1, created1 = load_or_create_key(kp)
    assert created1 is True and kp.is_file()
    priv2, pub2, created2 = load_or_create_key(kp)
    assert created2 is False
    assert (priv1, pub1) == (priv2, pub2)


def test_signed_run_adds_verification_check(tmp_path: Path) -> None:
    """A signed pipeline run gains an `operator.signature` PASS on re-verify."""
    from facechain.config import Settings
    from facechain.corpus import seed_demo_corpus
    from facechain.pipeline import run_pipeline
    from facechain.verify import verify_run

    s = Settings.load(
        env_file="/nonexistent", runs_dir=tmp_path / "r", chain_dir=tmp_path / "c",
        corpus_dir=tmp_path / "corp", search_providers="local", anchor_backend="local",
    )
    seed_demo_corpus(s, repo_root=Path(__file__).parent.parent)
    res = run_pipeline(
        "samples/probe_obama.jpg", s, verify_after=False,
        sign_key=str(tmp_path / "op.key"),
    )
    assert (res.run_dir / "signature.json").is_file()
    report = verify_run(res.run_dir, s, live_refetch=False)
    names = {c.name: c.ok for c in report.checks}
    assert names.get("operator.signature") is True
    assert report.ok
