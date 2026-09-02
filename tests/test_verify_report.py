from __future__ import annotations

from facechain.models import VerificationReport
from facechain.verify import format_report


def test_format_report_renders_pass_and_fail():
    r = VerificationReport(run_id="run-1", record_hash="deadbeef", backend="local")
    r.add("evidence.self_consistent", ok=True, detail="ok")
    r.add("match.face_recheck", ok=False, detail="cosine 0.1", expected=">=0.86", actual="0.10")
    text = format_report(r)
    assert "[PASS] evidence.self_consistent" in text
    assert "[FAIL] match.face_recheck" in text
    assert "expected='>=0.86' actual='0.10'" in text
    assert "OVERALL: FAILED (1/2 checks passed)" in text


def test_format_report_all_pass():
    r = VerificationReport(run_id="r", record_hash="h", backend="evm")
    r.add("a", ok=True)
    r.add("b", ok=True)
    assert "OVERALL: VERIFIED (2/2 checks passed)" in format_report(r)
