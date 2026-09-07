"""Benchmark harness: metric sanity + artifact generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from facechain.bench import _roc, run_bench, score_engine
from facechain.face.factory import _is_available

REPO = Path(__file__).resolve().parents[1]


def test_roc_perfect_separation() -> None:
    import numpy as np

    genuine = np.array([0.9, 0.95, 0.8])
    impostor = np.array([0.1, 0.2, 0.05])
    pts, auc, eer = _roc(genuine, impostor)
    assert auc == pytest.approx(1.0, abs=1e-9)
    assert eer == pytest.approx(0.0, abs=1e-9)
    assert pts[0] == (0.0, 0.0) or pts[0][0] == 0.0


@pytest.mark.skipif(not _is_available("sface"), reason="sface engine unavailable")
def test_score_engine_sface_separates_identities() -> None:
    res = score_engine("sface", corpus=None, repo_root=REPO, augment=False)
    assert res is not None
    assert res.n_genuine >= 1 and res.n_impostor >= 1
    assert res.margin > 0.4          # deep embedding: wide genuine/impostor gap
    assert res.auc >= 0.99
    assert res.mean_impostor < 0.4
    # robustness sweep ran and the clean pair matches
    assert res.robustness and res.robustness[0][0] == "(none)"
    assert res.robustness[0][2] is True


@pytest.mark.skipif(not _is_available("sface"), reason="sface engine unavailable")
def test_run_bench_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    # Only benchmark sface to keep the test quick.
    import facechain.bench as bench_mod

    monkeypatch.setattr(bench_mod, "_ENGINES", ("sface",))
    md = run_bench(out_dir=tmp_path, augment=False, repo_root=REPO)
    assert md.is_file()
    assert (tmp_path / "roc.svg").read_text().startswith("<svg")
    assert (tmp_path / "results.json").is_file()
    assert "yunet-sface" in md.read_text()
