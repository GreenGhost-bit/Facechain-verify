"""Render a run directory into a single self-contained ``report.html``.

Everything is inlined (CSS + base64 images), so the file can be opened offline or
mailed as-is. It reads only the artifacts the pipeline already wrote -- no
recomputation, no network.
"""

from __future__ import annotations

import base64
import html
import json
import mimetypes
from pathlib import Path
from typing import Any

from .face.calibration import band, probability
from .models import EvidenceBundle, VerificationReport

_CSS = """
:root { color-scheme: light dark; --fg:#0f172a; --mut:#64748b; --line:#e2e8f0;
        --ok:#16a34a; --bad:#dc2626; --warn:#d97706; --card:#ffffff; --bg:#f8fafc; }
@media (prefers-color-scheme: dark) { :root {
  --fg:#e2e8f0; --mut:#94a3b8; --line:#1e293b; --card:#0f172a; --bg:#020617; } }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--fg);
  font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
.wrap { max-width:900px; margin:0 auto; padding:32px 20px 64px; }
h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:15px; margin:28px 0 10px;
  text-transform:uppercase; letter-spacing:.06em; color:var(--mut); }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:16px 18px; margin:10px 0; }
.verdict { display:flex; gap:14px; align-items:baseline; flex-wrap:wrap; }
.big { font-size:28px; font-weight:700; } .pill { padding:2px 10px; border-radius:999px;
  font-weight:600; font-size:12px; border:1px solid currentColor; }
.match { color:var(--ok); } .likely { color:var(--warn); } .nomatch { color:var(--bad); }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
img.shot { width:100%; border-radius:8px; border:1px solid var(--line); background:#0002; }
table { border-collapse:collapse; width:100%; font-size:13px; }
td,th { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--mut); font-weight:600; }
code,.mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.mono.sm { font-size:12px; word-break:break-all; } .kv td:first-child { color:var(--mut); width:170px; }
.ok { color:var(--ok); } .bad { color:var(--bad); } .thumb { width:64px; height:64px; object-fit:cover;
  border-radius:6px; border:1px solid var(--line); }
.foot { color:var(--mut); font-size:12px; margin-top:40px; }
a { color:inherit; }
"""


def _data_uri(path: Path) -> str:
    if not path.is_file():
        return ""
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _esc(x: object) -> str:
    return html.escape(str(x), quote=True)


def _kv_rows(pairs: list[tuple[str, str]]) -> str:
    return "".join(f"<tr><td>{_esc(k)}</td><td class='mono sm'>{v}</td></tr>" for k, v in pairs)


def _verdict_block(bundle: EvidenceBundle, verified: bool | None) -> str:
    m = bundle.match
    best = m.best
    sim = best.similarity_ppm / 1e6
    engine = bundle.probe_face.engine
    prob = probability(engine, sim)
    tier = band(engine, sim)
    cls = {"match": "match", "likely": "likely", "no-match": "nomatch"}[tier]
    vstate = (
        "<span class='pill ok'>chain verified</span>" if verified
        else "<span class='pill bad'>verification failed</span>" if verified is False
        else ""
    )
    who = f"<div class='mut'>WHO (advisory): <b>{_esc(m.identity_guess)}</b> " \
          f"(consensus {m.identity_confidence_ppm / 1e6:.2f})</div>" if m.identity_guess else ""
    return f"""
<div class="card verdict">
  <span class="big {cls}">{sim:.3f}</span>
  <span class="pill {cls}">{tier.upper()}</span>
  <span class="mut">P(same identity) &approx; {prob:.2f} &nbsp;|&nbsp; threshold {m.threshold_ppm / 1e6:.2f}
    &nbsp;|&nbsp; decided by <code>{_esc(m.decided_by)}</code></span>
  {vstate}
</div>
<div class="card">
  <b>Best match</b> &nbsp; <span class="mut">via</span> <code>{_esc(best.provider)}</code><br>
  <a href="{_esc(best.post_url)}">{_esc(best.post_url)}</a>
  {who}
</div>"""


def _candidates_table(run_dir: Path) -> str:
    p = run_dir / "candidates.json"
    if not p.is_file():
        return ""
    try:
        ranked = json.loads(p.read_text("utf-8")).get("ranked", [])
    except json.JSONDecodeError:
        return ""
    rows = []
    for c in ranked[:10]:
        art = c.get("local_artifact")
        thumb = f"<img class='thumb' src='{_data_uri(run_dir / art)}'>" if art else ""
        sim = (c.get("similarity_ppm") or 0) / 1e6
        rows.append(
            f"<tr><td>{c.get('rank', '')}</td><td>{thumb}</td>"
            f"<td class='mono'>{sim:+.4f}</td><td>{_esc(c.get('provider', ''))}</td>"
            f"<td>{_esc((c.get('title') or '')[:60])}<br>"
            f"<a class='mono sm' href='{_esc(c.get('post_url', ''))}'>{_esc(c.get('post_url', ''))}</a>"
            f"{('<br><span class=mut>' + _esc(c['note']) + '</span>') if c.get('note') else ''}</td></tr>"
        )
    return (
        "<h2>Ranked candidates</h2><div class='card'><table>"
        "<tr><th>#</th><th></th><th>cosine</th><th>provider</th><th>source</th></tr>"
        + "".join(rows) + "</table></div>"
    )


def _checks_table(report: VerificationReport | None) -> str:
    if report is None:
        return ""
    rows = []
    for c in report.checks:
        mark = "<span class='ok'>PASS</span>" if c.ok else "<span class='bad'>FAIL</span>"
        rows.append(
            f"<tr><td>{mark}</td><td class='mono'>{_esc(c.name)}</td>"
            f"<td>{_esc(c.detail)}</td></tr>"
        )
    overall = "<span class='ok'>VERIFIED</span>" if report.ok else "<span class='bad'>FAILED</span>"
    passed = sum(1 for c in report.checks if c.ok)
    return (
        f"<h2>Independent re-verification</h2><div class='card'>"
        f"<p>{overall} &nbsp; ({passed}/{len(report.checks)} checks passed)</p>"
        f"<table><tr><th></th><th>check</th><th>detail</th></tr>{''.join(rows)}</table></div>"
    )


def build_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    bundle = EvidenceBundle.model_validate_json((run_dir / "evidence.json").read_text("utf-8"))
    receipt: dict[str, Any] = json.loads((run_dir / "receipt.json").read_text("utf-8"))
    manifest: dict[str, Any] = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    vpath = run_dir / "verification.json"
    vreport = (
        VerificationReport.model_validate_json(vpath.read_text("utf-8")) if vpath.is_file() else None
    )
    verified = None if vreport is None else vreport.ok

    fp = bundle.probe_image_fingerprint
    pf = bundle.probe_face
    best = bundle.match.best
    matched_art = None
    try:
        ranked = json.loads((run_dir / "candidates.json").read_text("utf-8")).get("ranked", [])
        if ranked and ranked[0].get("local_artifact"):
            matched_art = run_dir / ranked[0]["local_artifact"]
    except (OSError, json.JSONDecodeError):
        pass

    probe_img = next((run_dir / f"probe{e}" for e in (".jpg", ".png", ".webp", ".bmp")
                      if (run_dir / f"probe{e}").is_file()), run_dir / "probe.jpg")

    anchor_rows = _kv_rows([
        ("backend", _esc(receipt.get("backend"))),
        ("network", _esc(receipt.get("network"))),
        ("record_hash", _esc(receipt.get("record_hash"))),
        ("block index", _esc(receipt.get("block_index"))),
        ("block hash", _esc(receipt.get("block_hash"))),
        ("merkle root", _esc(receipt.get("merkle_root"))),
        ("leaf index", _esc(receipt.get("leaf_index"))),
        ("tx / ref", _esc(json.dumps(receipt.get("ref", {})))),
    ])
    probe_rows = _kv_rows([
        ("engine", f"{_esc(pf.engine)} <span class='mut'>{_esc(pf.engine_version)}</span>"),
        ("face bbox", _esc(pf.bbox)),
        ("detection score", f"{pf.detection_score_ppm / 1e6:.3f}"),
        ("quality", f"{pf.quality_ppm / 1e6:.3f}"),
        ("embedding", f"dim {pf.embedding_dim} &nbsp; sha256 {_esc(pf.embedding_sha256)}"),
        ("image sha256", _esc(fp.sha256)),
        ("image pHash / dHash", f"{_esc(fp.phash)} / {_esc(fp.dhash)}"),
        ("dimensions", f"{fp.width}&times;{fp.height} &nbsp; {_esc(fp.mime)}"),
    ])
    match_fp = best.image_fingerprint
    match_rows = _kv_rows([
        ("provider", _esc(best.provider)),
        ("similarity", f"{best.similarity_ppm / 1e6:+.4f}"),
        ("source page", f"<a href='{_esc(best.post_url)}'>{_esc(best.post_url)}</a>"),
        ("image url", _esc(best.image_url)),
        ("image sha256", _esc(match_fp.sha256) if match_fp else "&mdash;"),
    ])

    status = manifest.get("status", "?")
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>facechain report {_esc(bundle.run_id)}</title><style>{_CSS}</style></head><body><div class="wrap">
<h1>facechain-verify &mdash; run report</h1>
<div class="mut mono sm">run {_esc(bundle.run_id)} &nbsp;|&nbsp; {_esc(bundle.created_at)}
 &nbsp;|&nbsp; status <b>{_esc(status)}</b> &nbsp;|&nbsp; {_esc(bundle.pipeline_version)}</div>

{_verdict_block(bundle, verified)}

<h2>Probe &amp; match</h2>
<div class="grid2">
  <div class="card"><b>Probe</b> (aligned crop inset)<br>
    <img class="shot" src="{_data_uri(probe_img)}">
    <img class="thumb" style="margin-top:8px" src="{_data_uri(run_dir / 'face_crop.png')}">
    <table class="kv">{probe_rows}</table></div>
  <div class="card"><b>Matched image</b><br>
    <img class="shot" src="{_data_uri(matched_art) if matched_art else ''}">
    <table class="kv">{match_rows}</table></div>
</div>

{_candidates_table(run_dir)}

<h2>Evidence &amp; anchor</h2>
<div class="card"><table class="kv">
  <tr><td>record_hash</td><td class="mono sm"><b>{_esc(bundle.record_hash)}</b></td></tr>
  <tr><td>settings_digest</td><td class="mono sm">{_esc(manifest.get('settings_digest'))}</td></tr>
  {anchor_rows}
</table></div>

{_checks_table(vreport)}

<p class="foot">Generated from the run directory only &mdash; no recomputation, no network.
Open <code>verification.json</code> / <code>evidence.json</code> for the raw records, or run
<code>facechain verify {_esc(bundle.run_id)}</code> to re-check independently.</p>
</div></body></html>"""

    out = run_dir / "report.html"
    out.write_text(doc, encoding="utf-8")
    return out
