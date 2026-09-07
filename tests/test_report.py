"""Self-contained HTML report generation."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

import pytest

from facechain.config import Settings
from facechain.corpus import seed_demo_corpus
from facechain.pipeline import run_pipeline
from facechain.report import build_report


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags = 0
        self.error: str | None = None

    def handle_starttag(self, tag: str, attrs: object) -> None:
        self.tags += 1


@pytest.fixture
def a_run(tmp_path: Path):  # type: ignore[no-untyped-def]
    s = Settings.load(
        env_file="/nonexistent", runs_dir=tmp_path / "r", chain_dir=tmp_path / "c",
        corpus_dir=tmp_path / "corp", search_providers="local", anchor_backend="local",
    )
    seed_demo_corpus(s, repo_root=Path(__file__).parent.parent)
    return run_pipeline("samples/probe_obama.jpg", s, verify_after=True)


def test_run_autogenerates_report(a_run) -> None:
    assert (a_run.run_dir / "report.html").is_file()


def test_report_is_self_contained_and_parses(a_run) -> None:
    html_path = build_report(a_run.run_dir)
    text = html_path.read_text("utf-8")
    # no external resource references
    assert "http://" not in text.split("<body")[0]  # head has no remote links
    assert 'src="http' not in text and "@import" not in text
    assert "data:image/" in text  # images inlined
    for token in (a_run.bundle.record_hash, "yunet-sface", "VERIFIED", "record_hash"):
        assert token in text
    p = _Parser()
    p.feed(text)
    assert p.tags > 30
