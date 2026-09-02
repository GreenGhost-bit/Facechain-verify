from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.conftest import FIXTURES, requires_opencv

from facechain.cli import main


def _common(tmp_path: Path) -> list[str]:
    return [
        "--runs-dir", str(tmp_path / "runs"),
        "--chain-dir", str(tmp_path / "chain"),
        "--corpus-dir", str(tmp_path / "corpus"),
    ]


def test_version_command(capsys: pytest.CaptureFixture[str]):
    assert main(["version"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["version"]
    assert "serpapi_key" not in out["effective_settings"]


@requires_opencv
def test_identify_command_json(capsys: pytest.CaptureFixture[str], probe_obama: Path):
    assert main(["identify", str(probe_obama), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["face"]["embedding_dim"] > 0
    assert len(payload["image_fingerprint"]["sha256"]) == 64


@requires_opencv
def test_run_then_verify_via_cli(tmp_path: Path, reencoder, capsys: pytest.CaptureFixture[str]):
    probe = reencoder(FIXTURES / "corpus_reagan.jpg", tmp_path / "p.jpg", scale=0.75, quality=68)
    # seed corpus
    assert main(["fetch-corpus", "--seed-demo", *_common(tmp_path)]) == 0
    capsys.readouterr()

    rc = main(["run", str(probe), "--providers", "local", "--anchor", "local", "--json", *_common(tmp_path)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["verified"] is True
    run_dir = out["run_dir"]

    rc = main(["verify", run_dir, "--no-network", *_common(tmp_path)])
    assert rc == 0


@requires_opencv
def test_chain_show_and_verify_and_tamper(tmp_path: Path, reencoder, capsys: pytest.CaptureFixture[str]):
    probe = reencoder(FIXTURES / "corpus_kennedy.jpg", tmp_path / "p.jpg", scale=0.8, quality=70)
    main(["fetch-corpus", "--seed-demo", *_common(tmp_path)])
    main(["run", str(probe), "--providers", "local", "--anchor", "local", *_common(tmp_path)])
    capsys.readouterr()

    assert main(["chain", "verify", *_common(tmp_path)]) == 0
    assert "CHAIN INTEGRITY: OK" in capsys.readouterr().out

    assert main(["chain", "tamper", *_common(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["chain", "verify", *_common(tmp_path)]) == 5  # anchor stage exit code


def test_unknown_command_exits_nonzero():
    with pytest.raises(SystemExit) as ei:
        main(["frobnicate"])
    assert ei.value.code != 0
