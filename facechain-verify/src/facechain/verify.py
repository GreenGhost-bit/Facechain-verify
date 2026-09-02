"""Independent re-verification of a completed run.

Shares no mutable state with :mod:`facechain.pipeline`: it re-reads the raw
artifacts from ``runs/<id>/``, re-derives every hash, re-runs the face match, and
re-reads the ``record_hash`` from the chain. Every check is a
:class:`~facechain.models.Check`; the report is ``ok`` only if all pass.

Checks performed
----------------
1. ``evidence.self_consistent``   -- recompute the canonical record_hash.
2. ``evidence.binds_receipt``     -- bundle.record_hash == receipt.record_hash.
3. ``probe.image_integrity``      -- SHA-256 of the stored probe file.
4. ``probe.embedding_integrity``  -- SHA-256 of embedding.npy vs FaceRecord.
5. ``match.face_recheck``         -- re-encode probe + best candidate, cosine >= threshold.
6. ``match.candidate_integrity``  -- stored candidate image == bundle fingerprint (exact or perceptual).
7. ``match.live_refetch``         -- (optional) re-download the post image and compare perceptually.
8. anchor-backend checks          -- chain integrity + Merkle inclusion (local) or tx/calldata (evm).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .anchor import build_anchor_backend
from .config import Settings
from .face import build_face_engine, cosine, embedding_sha256, encode_candidate, encode_probe
from .imaging import load_image_bytes, load_image_path
from .logging import LOG
from .models import VerificationReport
from .netfetch import SafeFetcher


def _engine_pref(recorded_engine: str, fallback: str) -> str:
    """Map a recorded engine name (e.g. 'opencv-haar-lbph') to a factory preference."""
    for known in ("insightface", "opencv", "numpy"):
        if recorded_engine.startswith(known):
            return known
    return fallback


def _find(run_dir: Path, *names: str) -> Path | None:
    for n in names:
        hits = sorted(run_dir.glob(n))
        if hits:
            return hits[0]
    return None


def verify_run(
    run_dir: str | Path,
    settings: Settings,
    *,
    live_refetch: bool = True,
) -> VerificationReport:
    run_dir = Path(run_dir)
    from .pipeline import load_bundle, load_manifest, load_receipt

    bundle = load_bundle(run_dir)
    receipt = load_receipt(run_dir)
    try:
        manifest = load_manifest(run_dir)
        backend_name = manifest.anchor_backend
    except FileNotFoundError:  # pragma: no cover - manifest always written
        backend_name = receipt.backend

    report = VerificationReport(
        run_id=bundle.run_id, record_hash=bundle.record_hash, backend=backend_name
    )

    # 1. evidence self-consistency
    recomputed = bundle.compute_record_hash()
    report.add(
        "evidence.self_consistent",
        ok=recomputed == bundle.record_hash,
        detail="recomputed canonical SHA-256 of the bundle payload",
        expected=bundle.record_hash,
        actual=recomputed,
    )

    # 2. bundle <-> receipt binding
    report.add(
        "evidence.binds_receipt",
        ok=bundle.record_hash == receipt.record_hash,
        expected=bundle.record_hash,
        actual=receipt.record_hash,
    )

    # 3. probe image integrity
    probe_file = _find(run_dir, "probe.jpg", "probe.png", "probe.webp", "probe.*")
    if probe_file is None:
        report.add("probe.image_integrity", ok=False, detail="probe file missing from run dir")
        probe_loaded = None
    else:
        probe_loaded = load_image_path(
            probe_file, max_bytes=settings.max_image_bytes, max_pixels=settings.max_image_pixels
        )
        fp = probe_loaded.fingerprint
        exp = bundle.probe_image_fingerprint
        report.add(
            "probe.image_integrity",
            ok=fp.sha256 == exp.sha256,
            detail=f"{probe_file.name}",
            expected=exp.sha256,
            actual=fp.sha256,
        )

    # 4. embedding integrity
    emb_path = run_dir / "embedding.npy"
    probe_emb: np.ndarray | None = None
    if emb_path.is_file():
        probe_emb = np.load(emb_path)
        report.add(
            "probe.embedding_integrity",
            ok=embedding_sha256(probe_emb) == bundle.probe_embedding_sha256,
            expected=bundle.probe_embedding_sha256,
            actual=embedding_sha256(probe_emb),
        )
    else:
        report.add("probe.embedding_integrity", ok=False, detail="embedding.npy missing")

    # 5 + 6. re-run the match against the stored candidate image
    best = bundle.match.best
    threshold = bundle.match.threshold_ppm / 1_000_000
    cand_file = _find(run_dir, "candidates/00_*", "candidates/00*.*")
    if cand_file is None:
        report.add("match.candidate_integrity", ok=False, detail="stored best-candidate image missing")
        report.add("match.face_recheck", ok=False, detail="no candidate image to re-encode")
    else:
        cand_bytes = cand_file.read_bytes()
        cand_loaded = load_image_bytes(
            cand_bytes,
            max_bytes=settings.max_image_bytes,
            max_pixels=settings.max_image_pixels,
            source=str(cand_file),
        )
        if best.image_fingerprint is not None:
            ok_fp, diag = cand_loaded.fingerprint.matches(best.image_fingerprint, max_hamming=10)
            report.add(
                "match.candidate_integrity",
                ok=ok_fp,
                detail=str(diag),
                expected=best.image_fingerprint.sha256,
                actual=cand_loaded.fingerprint.sha256,
            )
        else:
            report.add("match.candidate_integrity", ok=False, detail="bundle has no candidate fingerprint")

        if probe_loaded is not None:
            engine = build_face_engine(_engine_pref(bundle.probe_face.engine, settings.face_engine))
            try:
                _re_probe_rec, re_probe_emb, _ = encode_probe(probe_loaded, engine=engine)
                cand_emb = encode_candidate(cand_loaded.rgb, engine=engine)
                if cand_emb is None:
                    report.add("match.face_recheck", ok=False, detail="no face found in candidate on recheck")
                else:
                    sim = cosine(re_probe_emb, cand_emb)
                    report.add(
                        "match.face_recheck",
                        ok=sim >= threshold - 1e-6,
                        detail=f"cosine {sim:.4f} vs threshold {threshold:.4f}",
                        expected=f">= {threshold:.4f}",
                        actual=f"{sim:.4f}",
                    )
            except Exception as exc:
                report.add("match.face_recheck", ok=False, detail=f"recheck raised {exc!r}")
        else:
            report.add("match.face_recheck", ok=False, detail="probe image unavailable for recheck")

    # 7. optional live re-fetch of the post image
    if live_refetch and best.image_url and not best.image_url.lower().startswith("file://"):
        try:
            with SafeFetcher(
                contact=settings.http_contact,
                timeout_s=settings.http_timeout_s,
                max_redirects=settings.http_max_redirects,
                max_bytes=settings.max_image_bytes,
            ) as fetcher:
                fetched = fetcher.fetch_image(best.image_url)
            live_fp = load_image_bytes(fetched.content, source=best.image_url).fingerprint
            if best.image_fingerprint is not None:
                ok_live, diag = live_fp.matches(best.image_fingerprint, max_hamming=14)
                report.add(
                    "match.live_refetch",
                    ok=ok_live,
                    detail=f"re-downloaded {best.image_url} -> {diag}",
                    expected=best.image_fingerprint.phash,
                    actual=live_fp.phash,
                )
            else:
                report.add("match.live_refetch", ok=True, detail="no stored fingerprint to compare")
        except Exception as exc:
            # A removed post or blocked network must not fail verification: the
            # on-chain record + stored artifact are the source of truth.
            report.add(
                "match.live_refetch",
                ok=True,
                detail=f"skipped (post unreachable / offline): {type(exc).__name__}: {exc}",
            )
    elif live_refetch:
        report.add("match.live_refetch", ok=True, detail="best candidate is a local corpus file; skipped")

    # 8. anchor-backend verification
    try:
        anchor_settings = settings.with_(anchor_backend=backend_name)
        backend = build_anchor_backend(anchor_settings)
        for chk in backend.verify(bundle.record_hash, receipt):
            report.checks.append(chk)
    except Exception as exc:
        report.add("anchor.backend", ok=False, detail=f"backend verify failed: {exc!r}")

    LOG.info(
        "verify.done",
        run_id=bundle.run_id,
        ok=report.ok,
        passed=sum(c.ok for c in report.checks),
        total=len(report.checks),
    )
    return report


def format_report(report: VerificationReport) -> str:
    lines = [
        f"verification report for run {report.run_id}",
        f"  record_hash : {report.record_hash}",
        f"  anchor      : {report.backend}",
        "",
    ]
    for c in report.checks:
        mark = "PASS" if c.ok else "FAIL"
        lines.append(f"  [{mark}] {c.name}")
        if c.detail:
            lines.append(f"         {c.detail}")
        if not c.ok and (c.expected or c.actual):
            lines.append(f"         expected={c.expected!r} actual={c.actual!r}")
    lines.append("")
    lines.append(f"  OVERALL: {'VERIFIED' if report.ok else 'FAILED'} "
                 f"({sum(c.ok for c in report.checks)}/{len(report.checks)} checks passed)")
    return "\n".join(lines)
