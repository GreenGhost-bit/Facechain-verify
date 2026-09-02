"""End-to-end orchestration: face scan -> search -> anchor.

Produces a self-describing ``runs/<run_id>/`` directory:

    manifest.json        index + settings digest + artifact map + status
    probe.<ext>          exact bytes of the input image
    probe_fingerprint.json
    face.json            FaceRecord
    face_crop.png        the aligned crop that was encoded
    embedding.npy        raw L2-normalised probe embedding
    candidates.json      every scored candidate, ranked
    candidates/NN_*.<ext>  the fetched candidate images
    evidence.json        the notarised EvidenceBundle (record_hash inside)
    receipt.json         AnchorReceipt (how to read the hash back)
    verification.json    result of an immediate independent re-verification
    telemetry.jsonl      structured, timed log of the whole run
"""

from __future__ import annotations

import io
import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .anchor import build_anchor_backend
from .canonical import hash_object
from .config import Settings
from .errors import FaceChainError, NoMatchFoundError
from .face import build_face_engine, encode_probe
from .face.base import DetectedFace
from .face.descriptor import align_crop
from .imaging import load_image_path
from .logging import LOG, get_logger
from .models import (
    AnchorReceipt,
    EvidenceBundle,
    RunManifest,
    VerificationReport,
)
from .netfetch import SafeFetcher
from .search import ProbeContext, SearchAggregator, build_providers

_SECRET_FIELDS = {"serpapi_key", "evm_private_key", "google_credentials"}


@dataclass
class RunResult:
    run_dir: Path
    status: str
    bundle: EvidenceBundle | None = None
    receipt: AnchorReceipt | None = None
    verification: VerificationReport | None = None
    error: str = ""


def _run_id(input_path: Path) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tag = hash_object({"p": str(input_path), "n": secrets.token_hex(4)})[:8]
    return f"{stamp}-{tag}"


def settings_digest(settings: Settings) -> str:
    # Coerce every value to a string: this digest only needs to be stable and
    # secret-free, and floats are intentionally rejected by the canonical hasher.
    payload = {
        k: str(v)
        for k, v in sorted(settings.model_dump().items())
        if k not in _SECRET_FIELDS
    }
    payload["_secrets_present"] = ",".join(
        sorted(k for k in _SECRET_FIELDS if getattr(settings, k, None))
    )
    return hash_object(payload)


def _save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = obj.model_dump_json(indent=2) if hasattr(obj, "model_dump_json") else json.dumps(
        obj, indent=2, sort_keys=True, default=str
    )
    path.write_text(text, encoding="utf-8")


def _ext_for_mime(mime: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
        "image/gif": ".gif",
        "image/tiff": ".tiff",
    }.get(mime, ".bin")


def run_pipeline(
    input_path: str | Path,
    settings: Settings,
    *,
    hint: str | None = None,
    probe_image_url: str | None = None,
    verify_after: bool = True,
) -> RunResult:
    """Execute the full pipeline. Raises :class:`FaceChainError` subclasses on
    unrecoverable stage failures (after writing a manifest)."""
    input_path = Path(input_path)
    run_id = _run_id(input_path)
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "candidates").mkdir(exist_ok=True)

    log = get_logger("facechain.run").bind(run_id=run_id)
    telemetry = (run_dir / "telemetry.jsonl").open("w", encoding="utf-8")
    log.add_sink(telemetry)
    LOG.add_sink(telemetry)

    digest = settings_digest(settings)
    manifest = RunManifest(
        run_id=run_id,
        input_path=str(input_path),
        settings_digest=digest,
        record_hash="",
        anchor_backend=settings.anchor_backend,
        anchor_network="",
    )

    try:
        with log.span("pipeline", input=str(input_path)):
            # 1. probe --------------------------------------------------
            with log.span("stage.load"):
                probe = load_image_path(
                    input_path,
                    max_bytes=settings.max_image_bytes,
                    max_pixels=settings.max_image_pixels,
                )
            probe_ext = _ext_for_mime(probe.fingerprint.mime)
            (run_dir / f"probe{probe_ext}").write_bytes(probe.raw_bytes)
            _save_json(run_dir / "probe_fingerprint.json", probe.fingerprint)
            manifest.artifacts["probe"] = f"probe{probe_ext}"
            manifest.artifacts["probe_fingerprint"] = "probe_fingerprint.json"

            # 2. face -------------------------------------------------
            engine = build_face_engine(settings.face_engine)
            with log.span("stage.face", engine=engine.name):
                face_record, embedding, _ambiguous = encode_probe(
                    probe,
                    engine=engine,
                    min_face_pixels=settings.min_face_pixels,
                )
            _save_json(run_dir / "face.json", face_record)
            np.save(run_dir / "embedding.npy", embedding)
            x, y, w, h = face_record.bbox
            crop = align_crop(probe.rgb, DetectedFace(x, y, w, h))
            Image.fromarray(crop, "L").save(run_dir / "face_crop.png")
            manifest.artifacts.update(
                {"face": "face.json", "embedding": "embedding.npy", "face_crop": "face_crop.png"}
            )

            # 3. search ---------------------------------------------
            providers = build_providers(settings)
            with SafeFetcher(
                contact=settings.http_contact,
                timeout_s=settings.http_timeout_s,
                max_redirects=settings.http_max_redirects,
                max_bytes=settings.max_image_bytes,
            ) as fetcher:
                ctx = ProbeContext(
                    image_bytes=probe.raw_bytes,
                    rgb=probe.rgb,
                    embedding=embedding,
                    settings=settings,
                    fetcher=fetcher,
                    face_engine=engine,
                    hint=hint,
                    extra={"probe_image_url": probe_image_url} if probe_image_url else {},
                )
                with log.span("stage.search", providers=[p.name for p in providers]):
                    agg = SearchAggregator(providers).run(ctx)

            # persist candidates
            cand_index = []
            for i, sc in enumerate(agg.scored):
                entry = sc.model.model_dump(mode="json")
                if sc.image_bytes is not None:
                    ext = _ext_for_mime(
                        sc.model.image_fingerprint.mime if sc.model.image_fingerprint else ""
                    )
                    fname = f"candidates/{i:02d}_{sc.raw.provider}{ext}"
                    (run_dir / fname).write_bytes(sc.image_bytes)
                    entry["local_artifact"] = fname
                cand_index.append(entry)
            _save_json(run_dir / "candidates.json", {"ranked": cand_index})
            manifest.artifacts["candidates"] = "candidates.json"

            # 4. evidence bundle ---------------------------------
            bundle = EvidenceBundle(
                run_id=run_id,
                probe_image_fingerprint=probe.fingerprint,
                probe_face=face_record,
                probe_embedding_sha256=face_record.embedding_sha256,
                search=agg.summary,
                match=agg.match,
            ).finalized()
            _save_json(run_dir / "evidence.json", bundle)
            manifest.artifacts["evidence"] = "evidence.json"
            manifest.record_hash = bundle.record_hash
            LOG.info("pipeline.record_hash", record_hash=bundle.record_hash)

            # 5. anchor -----------------------------------------
            backend = build_anchor_backend(settings)
            with log.span("stage.anchor", backend=backend.name):
                receipt = backend.anchor(bundle.record_hash)
            _save_json(run_dir / "receipt.json", receipt)
            manifest.artifacts["receipt"] = "receipt.json"
            manifest.anchor_network = receipt.network

            # 6. immediate independent verification ----------
            verification = None
            if verify_after:
                from .verify import verify_run

                with log.span("stage.verify"):
                    verification = verify_run(run_dir, settings, live_refetch=False)
                _save_json(run_dir / "verification.json", verification)
                manifest.artifacts["verification"] = "verification.json"

            manifest.status = "ok"
            _save_json(run_dir / "manifest.json", manifest)
            LOG.info(
                "pipeline.done",
                run_dir=str(run_dir),
                record_hash=bundle.record_hash,
                anchored=receipt.network,
                verified=None if verification is None else verification.ok,
            )
            return RunResult(
                run_dir=run_dir,
                status="ok",
                bundle=bundle,
                receipt=receipt,
                verification=verification,
            )

    except NoMatchFoundError as exc:
        manifest.status = "no_match"
        manifest.error = str(exc)
        _save_json(run_dir / "manifest.json", manifest)
        if isinstance(exc.detail, dict):
            _save_json(run_dir / "no_match_debug.json", exc.detail)
        LOG.warning("pipeline.no_match", run_dir=str(run_dir), detail=str(exc))
        return RunResult(run_dir=run_dir, status="no_match", error=str(exc))
    except FaceChainError as exc:
        manifest.status = "error"
        manifest.error = f"{exc.code}: {exc}"
        _save_json(run_dir / "manifest.json", manifest)
        LOG.error("pipeline.error", run_dir=str(run_dir), code=exc.code, detail=str(exc))
        raise
    finally:
        LOG.remove_sink(telemetry)
        telemetry.close()


def load_bundle(run_dir: Path) -> EvidenceBundle:
    return EvidenceBundle.model_validate_json((run_dir / "evidence.json").read_text("utf-8"))


def load_receipt(run_dir: Path) -> AnchorReceipt:
    return AnchorReceipt.model_validate_json((run_dir / "receipt.json").read_text("utf-8"))


def load_manifest(run_dir: Path) -> RunManifest:
    return RunManifest.model_validate_json((run_dir / "manifest.json").read_text("utf-8"))


# re-exported for tests / callers that build a probe crop image
def probe_crop_png_bytes(rgb: np.ndarray, bbox: list[int]) -> bytes:
    x, y, w, h = bbox
    crop = align_crop(rgb, DetectedFace(x, y, w, h))
    buf = io.BytesIO()
    Image.fromarray(crop, "L").save(buf, format="PNG")
    return buf.getvalue()
