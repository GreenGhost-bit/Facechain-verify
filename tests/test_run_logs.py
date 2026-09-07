"""Tests for per-run log files."""

from __future__ import annotations

from pathlib import Path

from facechain.logging import LOG, open_run_logs


def test_open_run_logs_writes_json_and_text(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    bundle = open_run_logs(run_dir)
    LOG.attach_run_logs(bundle)
    try:
        LOG.info("test.event", step=1, ok=True)
        LOG.warning("test.warn", reason="demo")
    finally:
        LOG.detach_run_logs(bundle)
        bundle.close()

    text = (run_dir / "run.log").read_text(encoding="utf-8")
    jsonl = (run_dir / "telemetry.jsonl").read_text(encoding="utf-8")
    assert "test.event" in text
    assert "step=1" in text
    assert "test.warn" in text
    assert '"event":"test.event"' in jsonl.replace(" ", "")
    assert jsonl.strip().count("\n") + 1 >= 2
