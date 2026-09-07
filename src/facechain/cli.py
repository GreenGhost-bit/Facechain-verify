"""``facechain`` command-line interface.

Subcommands
-----------
    identify      detect + encode the face in an image (no search, no chain)
    search        run the live search and print the ranked candidates
    run           full pipeline: face -> search -> anchor -> verify
    verify        independently re-verify a completed run directory
    chain         inspect / integrity-check / tamper-test the local ledger
    fetch-corpus  build the offline search corpus (live pull) or seed the demo one
    version       print version and effective configuration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import PIPELINE_VERSION, __version__
from .config import Settings
from .errors import FaceChainError


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--engine", choices=["auto", "sface", "opencv", "numpy", "insightface"],
                   default=None,
                   help="face engine (default: auto -> insightface > sface > opencv > numpy)")
    p.add_argument("--providers", default=None,
                   help="comma-separated search providers "
                        "(default: serpapi,multiris,wikimedia,faceindex; "
                        "unavailable ones are skipped)")
    p.add_argument("--anchor", choices=["local", "evm"], default=None,
                   help="blockchain anchor backend (default: local)")
    p.add_argument("--threshold", type=float, default=None,
                   help="face-match cosine threshold "
                        "(default: calibrated per engine - sface 0.40, arcface 0.42, lbph 0.86)")
    p.add_argument("--allow-public-host", action="store_true",
                   help="permit uploading the probe to a public file host so SerpAPI "
                        "Lens can crawl it (off by default; needed only for serpapi "
                        "without --probe-image-url)")
    p.add_argument("--difficulty", type=int, default=None,
                   help="local-chain proof-of-work leading zero bits (default: 0)")
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--chain-dir", default=None)
    p.add_argument("--corpus-dir", default=None)
    p.add_argument("--hint", default=None,
                   help="name or profile URL; focuses Wikimedia and auto-enables "
                        "the 'hint' provider (og:image + Google Images candidates)")
    p.add_argument("--probe-image-url", default=None,
                   help="public URL of the probe image for serpapi "
                        "(optional: auto-hosted on catbox/tmpfiles if omitted)")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON on stdout")


# When multiris / SerpAPI Lens fans out, score a longer tail of candidates.
_MULTIRIS_MIN_PER_PROVIDER = 24
_SERPAPI_MIN_PER_PROVIDER = 36


def _settings_from_args(args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    if getattr(args, "engine", None):
        overrides["face_engine"] = args.engine
    if getattr(args, "providers", None):
        overrides["search_providers"] = args.providers
    if getattr(args, "anchor", None):
        overrides["anchor_backend"] = args.anchor
    threshold_explicit = getattr(args, "threshold", None) is not None
    if threshold_explicit:
        overrides["match_threshold"] = args.threshold
    if getattr(args, "difficulty", None) is not None:
        overrides["chain_difficulty_bits"] = args.difficulty
    if getattr(args, "runs_dir", None):
        overrides["runs_dir"] = Path(args.runs_dir)
    if getattr(args, "chain_dir", None):
        overrides["chain_dir"] = Path(args.chain_dir)
    if getattr(args, "corpus_dir", None):
        overrides["corpus_dir"] = Path(args.corpus_dir)

    if getattr(args, "allow_public_host", False):
        overrides["allow_public_probe_host"] = True

    settings = Settings.load(**overrides)
    # Threshold is resolved per-engine downstream (calibration table); here we
    # only widen candidate budgets for the fan-out providers and wire --hint.
    settings = _apply_search_candidate_budgets(settings)
    return _ensure_hint_provider(settings, getattr(args, "hint", None))


def _ensure_hint_provider(settings: Settings, hint: str | None) -> Settings:
    """When ``--hint`` is set, append the hint provider if the user omitted it."""
    if not hint or not str(hint).strip():
        return settings
    if "hint" in settings.search_providers:
        return settings
    return settings.with_(search_providers=(*settings.search_providers, "hint"))


def _apply_search_candidate_budgets(settings: Settings) -> Settings:
    """Raise scoring budgets when Lens / multiris are in the configured stack."""
    settings = _apply_multiris_candidate_budget(settings)
    return _apply_serpapi_candidate_budget(settings)


def _apply_multiris_candidate_budget(settings: Settings) -> Settings:
    """Raise per-provider candidate cap when the multi-engine RIS provider is active."""
    if "multiris" not in settings.search_providers:
        return settings
    if settings.max_candidates_per_provider >= _MULTIRIS_MIN_PER_PROVIDER:
        return settings
    return settings.with_(max_candidates_per_provider=_MULTIRIS_MIN_PER_PROVIDER)


def _apply_serpapi_candidate_budget(settings: Settings) -> Settings:
    """Raise per-provider candidate cap for SerpAPI Google Lens (often 40-60 visual matches)."""
    if "serpapi" not in settings.search_providers:
        return settings
    if settings.max_candidates_per_provider >= _SERPAPI_MIN_PER_PROVIDER:
        return settings
    return settings.with_(max_candidates_per_provider=_SERPAPI_MIN_PER_PROVIDER)


# --------------------------------------------------------------------------
# subcommand handlers
# --------------------------------------------------------------------------
def _cmd_identify(args: argparse.Namespace) -> int:
    from .face import build_face_engine, encode_probe
    from .imaging import load_image_path

    settings = _settings_from_args(args)
    image = load_image_path(args.image, max_bytes=settings.max_image_bytes,
                            max_pixels=settings.max_image_pixels)
    engine = build_face_engine(settings.face_engine)
    record, embedding, ambiguous = encode_probe(image, engine=engine,
                                                min_face_pixels=settings.min_face_pixels)
    payload = {
        "input": str(args.image),
        "image_fingerprint": image.fingerprint.model_dump(),
        "face": record.model_dump(),
        "embedding_preview": [round(float(x), 5) for x in embedding[:8]],
        "ambiguous": ambiguous,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"engine        : {record.engine} ({record.engine_version})")
        print(f"faces found   : {len(record.all_bboxes)}  primary bbox={record.bbox}")
        print(f"quality       : {record.quality_ppm / 1e6:.3f}")
        print(f"embedding     : dim={record.embedding_dim} sha256={record.embedding_sha256}")
        print(f"image sha256  : {image.fingerprint.sha256}")
        print(f"image phash   : {image.fingerprint.phash}   dhash: {image.fingerprint.dhash}")
        if ambiguous:
            print(f"WARNING       : {record.note}")
    return 0


def _print_closest_near_miss(detail: object) -> None:
    """Print the strongest below-threshold hit when search returns NO MATCH."""
    if not isinstance(detail, dict):
        return
    closest = detail.get("closest")
    ranked = detail.get("ranked") or []
    if not isinstance(closest, dict) and ranked:
        top = ranked[0] if isinstance(ranked[0], dict) else None
        if top:
            closest = {
                "similarity": (top.get("similarity_ppm") or 0) / 1e6,
                "provider": top.get("provider"),
                "title": top.get("title"),
                "post_url": top.get("post_url"),
                "image_url": top.get("image_url"),
            }
    if not isinstance(closest, dict):
        return
    sim: Any = closest.get("similarity")
    try:
        sim_s = f"{float(sim):.4f}"
    except (TypeError, ValueError):
        sim_s = str(sim)
    print(f"CLOSEST      sim={sim_s}  [{closest.get('provider') or '?'}]", flush=True)
    if closest.get("title"):
        print(f"             {closest['title']}", flush=True)
    if closest.get("post_url"):
        print(f"             source: {closest['post_url']}", flush=True)
    if closest.get("image_url"):
        print(f"             image:  {closest['image_url']}", flush=True)
    who = detail.get("identity_guess") or ""
    if who:
        conf = detail.get("identity_confidence")
        conf_s = f"  (consensus {float(conf):.2f})" if conf is not None else ""
        print(f"WHO (near)   {who}{conf_s}", flush=True)
    for i, row in enumerate(ranked[1:4] if isinstance(ranked, list) else []):
        if not isinstance(row, dict):
            continue
        rsim = (row.get("similarity_ppm") or 0) / 1e6
        print(
            f"  runner-up {i + 2}: sim={rsim:+.4f}  [{row.get('provider')}]  "
            f"{(row.get('title') or '')[:50]}  {row.get('post_url')}",
            flush=True,
        )


def _cmd_search(args: argparse.Namespace) -> int:
    from .face import build_face_engine, encode_probe
    from .imaging import load_image_path
    from .logging import LOG, open_run_logs
    from .netfetch import SafeFetcher
    from .pipeline import _run_id, _save_json
    from .search import ProbeContext, SearchAggregator, build_providers

    settings = _settings_from_args(args)
    input_path = Path(args.image)
    run_id = _run_id(input_path)
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle_logs = open_run_logs(run_dir)
    LOG.attach_run_logs(bundle_logs)
    LOG.info(
        "search.start",
        input=str(input_path),
        run_dir=str(run_dir),
        run_log=str(bundle_logs.run_log_path),
        telemetry=str(bundle_logs.telemetry_path),
        providers=list(settings.search_providers),
        engine=settings.face_engine,
        threshold=settings.match_threshold,
    )
    if not args.json:
        print(f"run dir       : {run_dir}", flush=True)
        print(f"live log      : {bundle_logs.run_log_path}", flush=True)
        print(f"              (tail -f {bundle_logs.run_log_path})", flush=True)

    try:
        image = load_image_path(args.image, max_bytes=settings.max_image_bytes,
                                max_pixels=settings.max_image_pixels)
        LOG.info("search.probe_loaded", bytes=len(image.raw_bytes),
                 size=f"{image.fingerprint.width}x{image.fingerprint.height}")
        engine = build_face_engine(settings.face_engine)
        settings = settings.with_calibrated_threshold(engine.name)
        _, embedding, _ = encode_probe(image, engine=engine, min_face_pixels=settings.min_face_pixels)
        providers = build_providers(settings)
        with SafeFetcher(contact=settings.http_contact, timeout_s=settings.http_timeout_s,
                         max_redirects=settings.http_max_redirects,
                         max_bytes=settings.max_image_bytes) as fetcher:
            _probe_extra = (
                {"probe_image_url": args.probe_image_url} if args.probe_image_url else {}
            )
            ctx = ProbeContext(image_bytes=image.raw_bytes, rgb=image.rgb, embedding=embedding,
                               settings=settings, fetcher=fetcher, face_engine=engine, hint=args.hint,
                               extra=_probe_extra)
            try:
                result = SearchAggregator(providers).run(ctx)
            except FaceChainError as exc:
                _detail = (
                    exc.detail
                    if isinstance(exc.detail, (dict, list, str, type(None)))
                    else repr(exc.detail)
                )
                _save_json(run_dir / "no_match_debug.json", {
                    "error": exc.code,
                    "message": str(exc),
                    "detail": _detail,
                })
                LOG.warning("search.no_match", error=str(exc), run_dir=str(run_dir))
                if args.json:
                    print(json.dumps({
                        "error": exc.code,
                        "message": str(exc),
                        "run_dir": str(run_dir),
                        "closest": exc.detail.get("closest") if isinstance(exc.detail, dict) else None,
                        "identity_guess": (
                            exc.detail.get("identity_guess") if isinstance(exc.detail, dict) else None
                        ),
                        "ranked": exc.detail.get("ranked") if isinstance(exc.detail, dict) else None,
                    }, indent=2))
                else:
                    print(f"NO MATCH: {exc}", file=sys.stderr, flush=True)
                    _print_closest_near_miss(exc.detail)
                    print(f"logs: {bundle_logs.run_log_path}", file=sys.stderr, flush=True)
                return exc.exit_code

        _save_json(run_dir / "match.json", result.match)
        _save_json(run_dir / "summary.json", result.summary)
        LOG.info(
            "search.done",
            best_sim=round(result.match.best.similarity_ppm / 1e6, 4),
            who=result.match.identity_guess or None,
            run_dir=str(run_dir),
        )

        if args.json:
            payload = json.loads(result.match.model_dump_json())
            payload["run_dir"] = str(run_dir)
            payload["run_log"] = str(bundle_logs.run_log_path)
            print(json.dumps(payload, indent=2))
        else:
            print(f"scored {result.summary.candidates_scored} candidate(s) "
                  f"from {', '.join(result.summary.providers_ok) or 'no providers'}")
            for m in result.match.ranked[:10]:
                print(f"  #{m.rank:<2} sim={m.similarity_ppm / 1e6:+.4f}  [{m.provider}]  {m.post_url}")
                if m.note:
                    print(f"       note: {m.note}")
            b = result.match.best
            print(f"\nBEST MATCH  sim={b.similarity_ppm / 1e6:.4f}  {b.post_url}")
            if result.match.identity_guess:
                conf = result.match.identity_confidence_ppm / 1e6
                print(f"WHO         {result.match.identity_guess}  (consensus {conf:.2f})")
                if result.match.identity_note:
                    print(f"            {result.match.identity_note}")
            if result.match.ambiguous:
                print(f"AMBIGUOUS   {result.match.ambiguity_note}")
            print(f"\nlogs        : {bundle_logs.run_log_path}")
            print(f"telemetry   : {bundle_logs.telemetry_path}")
        return 0
    finally:
        LOG.detach_run_logs(bundle_logs)
        bundle_logs.close()


def _cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_pipeline
    from .verify import format_report

    settings = _settings_from_args(args)
    sign_key = None
    if getattr(args, "sign", False):
        import os

        sign_key = args.key or os.environ.get("FACECHAIN_SIGNING_KEY") or str(
            Path(settings.chain_dir) / "operator_ed25519.key"
        )
    result = run_pipeline(args.image, settings, hint=args.hint,
                          probe_image_url=args.probe_image_url,
                          verify_after=not args.no_verify,
                          sign_key=sign_key,
                          describe=getattr(args, "describe", False))
    if result.status == "no_match":
        if args.json:
            print(json.dumps({"status": "no_match", "run_dir": str(result.run_dir),
                              "message": result.error}, indent=2))
        else:
            print(f"NO MATCH  (artifacts in {result.run_dir})")
            print(result.error)
            debug = result.run_dir / "no_match_debug.json"
            if debug.is_file():
                try:
                    detail = json.loads(debug.read_text(encoding="utf-8"))
                    # pipeline saves the exception detail dict directly or nested
                    if isinstance(detail, dict) and "closest" not in detail and "ranked" in detail:
                        pass
                    elif isinstance(detail, dict) and "detail" in detail:
                        detail = detail["detail"]
                    _print_closest_near_miss(detail)
                except Exception:
                    pass
            print(f"logs: {result.run_dir / 'run.log'}")
        return 4

    assert result.bundle is not None and result.receipt is not None
    if args.json:
        print(json.dumps({
            "status": result.status,
            "run_dir": str(result.run_dir),
            "record_hash": result.bundle.record_hash,
            "match": result.bundle.match.best.model_dump(),
            "anchor": result.receipt.model_dump(),
            "verified": None if result.verification is None else result.verification.ok,
        }, indent=2, default=str))
    else:
        b = result.bundle
        print(f"run dir       : {result.run_dir}")
        print(f"face engine   : {b.probe_face.engine}   bbox={b.probe_face.bbox}")
        print(f"BEST MATCH    : sim={b.match.best.similarity_ppm / 1e6:.4f}  "
              f"[{b.match.best.provider}]  {b.match.best.post_url}")
        if b.match.identity_guess:
            print(
                f"WHO           : {b.match.identity_guess}  "
                f"(consensus {b.match.identity_confidence_ppm / 1e6:.2f})"
            )
        print(f"record_hash   : {b.record_hash}")
        print(f"anchored on   : {result.receipt.network}")
        print(f"logs          : {result.run_dir / 'run.log'}")
        print(f"telemetry     : {result.run_dir / 'telemetry.jsonl'}")
        if result.receipt.block_hash:
            print(f"  block #{result.receipt.block_index}  hash={result.receipt.block_hash}")
            print(f"  merkle_root {result.receipt.merkle_root}  (idempotent={result.receipt.idempotent_hit})")
        if result.receipt.ref.get("tx_hash"):
            print(f"  tx {result.receipt.ref['tx_hash']}  block {result.receipt.ref.get('block_number')}")
        if result.verification is not None:
            print()
            print(format_report(result.verification))
    return 0 if (result.verification is None or result.verification.ok) else 6


def _cmd_verify(args: argparse.Namespace) -> int:
    from .verify import format_report, verify_run

    settings = _settings_from_args(args)
    report = verify_run(args.run_dir, settings, live_refetch=not args.no_network)
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(format_report(report))
    return 0 if report.ok else 6


def _cmd_chain(args: argparse.Namespace) -> int:
    from .anchor.local_chain import LocalChain
    from .errors import ChainIntegrityError

    settings = _settings_from_args(args)
    chain = LocalChain(settings.chain_dir / "local", difficulty_bits=settings.chain_difficulty_bits)

    if args.chain_cmd == "show":
        blocks = chain.blocks()
        if args.json:
            print(json.dumps(blocks, indent=2))
        else:
            for blk in blocks:
                print(f"#{blk['index']:<4} {blk['timestamp']}  records={len(blk['records'])}  "
                      f"hash={blk['hash'][:16]}...  prev={blk['prev_hash'][:16]}...")
                for r in blk["records"]:
                    print(f"        record {r}")
        return 0

    if args.chain_cmd == "verify":
        try:
            chain.verify_chain()
        except ChainIntegrityError as exc:
            print(f"CHAIN INTEGRITY: FAILED -- {exc}", file=sys.stderr)
            return 5
        print(f"CHAIN INTEGRITY: OK  ({len(chain.blocks())} blocks, "
              f"head={chain.head()['hash']})")
        return 0

    if args.chain_cmd == "tamper":
        # Deliberate corruption for the tamper-evidence demo. Never run on real data.
        blocks = chain.blocks()
        if len(blocks) < 2:
            print("nothing to tamper with (chain has only genesis)", file=sys.stderr)
            return 1
        target = blocks[args.block if args.block is not None else len(blocks) - 1]
        original = target["records"][0]
        target["records"][0] = ("f" * 64) if original != "f" * 64 else ("0" * 64)
        chain.path.write_text(
            "\n".join(json.dumps(b, sort_keys=True, separators=(",", ":")) for b in blocks) + "\n",
            encoding="utf-8",
        )
        print(f"tampered block #{target['index']}: flipped record 0")
        print("now run:  facechain chain verify   (expect FAILED)")
        return 0

    return 2


def _cmd_fetch_corpus(args: argparse.Namespace) -> int:
    from .corpus import fetch_corpus, fetch_corpus_wikidata, seed_demo_corpus

    settings = _settings_from_args(args)
    if args.seed_demo:
        n = seed_demo_corpus(settings)
        print(f"seeded {n} demo entries into {settings.corpus_dir}")
        return 0
    names = list(args.name or [])
    if args.names_file:
        names += [
            ln.strip() for ln in Path(args.names_file).read_text("utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    if names:
        n = fetch_corpus_wikidata(settings, names, overwrite=args.overwrite)
        print(f"fetched {n}/{len(names)} Wikidata portraits into {settings.corpus_dir}")
        return 0 if n > 0 else 1
    n = fetch_corpus(settings, queries=args.query or None, per_query=args.per_query,
                     overwrite=args.overwrite)
    print(f"fetched {n} corpus entries into {settings.corpus_dir}")
    return 0 if n > 0 else 1


def _cmd_build_index(args: argparse.Namespace) -> int:
    """Pre-encode the corpus into a face-embedding index for the 'faceindex' provider."""
    from .face import build_face_engine
    from .search.face_index import FaceIndex

    settings = _settings_from_args(args)
    engine = build_face_engine(settings.face_engine)
    n = FaceIndex(settings.corpus_dir, engine).build(rebuild=bool(getattr(args, "rebuild", False)))
    print(f"indexed {n} corpus face(s) for engine '{engine.name}' under {settings.corpus_dir}")
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    from .describe import describe_image

    result = describe_image(
        args.image,
        with_caption=not args.no_caption,
        with_faces=not args.no_faces,
        model_id=args.model,
    )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0
    a = result["attributes"]
    if result.get("caption"):
        print(f"CAPTION   {result['caption']}   [{result.get('caption_model')}]")
    elif result.get("caption_error"):
        print(f"CAPTION   (unavailable: {result['caption_error']})")
    print(f"SIZE      {a['width']}x{a['height']}  {a['megapixels']} MP  aspect {a['aspect_ratio']}")
    print(f"LIGHT     brightness {a['mean_brightness']}  "
          f"{'low-light ' if a['is_low_light'] else ''}"
          f"{'blurry' if a['is_blurry'] else 'sharp'} (lapvar {a['sharpness_lapvar']})")
    print("COLOURS   " + ", ".join(f"{c['hex']} {c['fraction']:.0%}" for c in a["dominant_colours"]))
    if "faces_detected" in a:
        print(f"FACES     {a['faces_detected']}"
              + (f"  largest {a['largest_face']['bbox']} "
                 f"({a['largest_face']['fraction_of_frame']:.0%} of frame)"
                 if a.get("largest_face") else ""))
    return 0


def _cmd_keygen(args: argparse.Namespace) -> int:
    from .signing import load_or_create_key

    path = args.out or "operator_ed25519.key"
    priv_hex, pub_hex, created = load_or_create_key(path)
    del priv_hex
    print(f"{'created' if created else 'loaded'}: {path}")
    print(f"public key (hex): {pub_hex}")
    print("keep the key file secret; share only the public key for verification")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Print what's installed / configured and what each capability needs."""
    import importlib.util as ilu
    import platform

    settings = _settings_from_args(args)
    rows: list[tuple[str, str, str]] = []

    def add(name: str, ok: bool | None, note: str) -> None:
        mark = {True: "ok", False: "MISSING", None: "--"}[ok]
        rows.append((name, mark, note))

    def have(mod: str) -> bool:
        return ilu.find_spec(mod) is not None

    print(f"facechain-verify {__version__}   python {platform.python_version()}   {platform.platform()}")
    print()

    for m in ("numpy", "PIL", "pydantic", "httpx"):
        add(f"core: {m}", have(m), "required")

    # face engines
    from .face.factory import _is_available as _face_ok

    add("engine: insightface (ArcFace)", _face_ok("insightface"),
        "pip install -e '.[insightface]'  (optional, best)")
    try:
        from .face.onnx_zoo import sface_is_cached
        cached = sface_is_cached()
    except Exception:
        cached = False
    add("engine: sface (YuNet+SFace)", _face_ok("sface"),
        "DEFAULT" + ("" if cached else "  -- run: facechain fetch-models"))
    add("engine: opencv (Haar+LBPH)", _face_ok("opencv"), "pip install -e '.[opencv]'")
    add("engine: numpy (pure-python)", True, "always available (slow fallback)")

    # search providers
    add("search: serpapi (Google Lens)", bool(settings.serpapi_key),
        "set FACECHAIN_SERPAPI_KEY" + ("" if settings.allow_public_probe_host
                                       else "  (+ --allow-public-host or --probe-image-url)"))
    add("search: multiris (Yandex/Bing/..)", have("PicImageSearch"), "pip install -e '.[ris]'")
    add("search: wikimedia", True, "keyless, needs network")
    corpus_n = 0
    if settings.corpus_dir.is_dir():
        corpus_n = sum(1 for p in settings.corpus_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    idx = (settings.corpus_dir / ".faceindex").is_dir()
    add("search: faceindex (offline)", corpus_n > 0,
        f"{corpus_n} corpus image(s), index {'built' if idx else 'NOT built -- facechain build-index'}")

    # anchoring
    add("anchor: local Merkle chain", True, "DEFAULT, no secrets")
    add("anchor: evm testnet", have("web3") and bool(settings.evm_rpc_url),
        "pip install -e '.[evm]' + FACECHAIN_EVM_RPC_URL / _PRIVATE_KEY")

    # optional extras
    add("sign: ed25519 operator signature", have("cryptography"),
        "pip install cryptography  (optional; facechain run --sign)")
    add("describe: local VLM caption", have("torch") and have("transformers"),
        "pip install -e '.[describe]'  (optional; facechain describe)")

    w = max(len(r[0]) for r in rows)
    for name, mark, note in rows:
        print(f"  {name:<{w}}  {mark:<8}  {note}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .report import build_report

    out = build_report(args.run_dir)
    print(f"report written: {out}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from .bench import run_bench

    md = run_bench(
        corpus=Path(args.corpus) if args.corpus else None,
        out_dir=Path(args.out) if args.out else None,
        augment=not args.no_augment,
        engines=tuple(e.strip() for e in args.engines.split(",")) if args.engines else None,
    )
    print(md.read_text(encoding="utf-8"))
    print(f"\n(written to {md.parent}/)")
    return 0


def _cmd_fetch_models(args: argparse.Namespace) -> int:
    """Pre-download + checksum-verify the ONNX face models (YuNet + SFace)."""
    from .face.onnx_zoo import ensure_models

    paths = ensure_models(download=True)
    for name, path in paths.items():
        print(f"{name:6}: {path}")
    print("ok - the 'sface' engine is now available offline")
    return 0


def _cmd_version(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    info = {
        "version": __version__,
        "pipeline_version": PIPELINE_VERSION,
        "effective_settings": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in settings.model_dump().items()
            if k not in {"serpapi_key", "evm_private_key", "google_credentials"}
        },
    }
    print(json.dumps(info, indent=2))
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facechain",
        description="Face scan -> live web/social match -> tamper-evident blockchain anchor.",
    )
    parser.add_argument("--version", action="version", version=f"facechain-verify {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("identify", help="detect + encode a face")
    p.add_argument("image")
    _add_common(p)
    p.set_defaults(func=_cmd_identify)

    p = sub.add_parser("search", help="run the live search only")
    p.add_argument("image")
    _add_common(p)
    p.set_defaults(func=_cmd_search)

    p = sub.add_parser("run", help="full end-to-end pipeline")
    p.add_argument("image")
    _add_common(p)
    p.add_argument("--no-verify", action="store_true", help="skip the immediate re-verification")
    p.add_argument("--sign", action="store_true",
                   help="Ed25519-sign the record_hash (needs '.[sign]'); "
                        "key from --key / FACECHAIN_SIGNING_KEY, else auto-created under chaindata/")
    p.add_argument("--key", default=None, help="path to the hex Ed25519 private key for --sign")
    p.add_argument("--describe", action="store_true",
                   help="also write describe.json (VLM caption + attributes; needs '.[describe]')")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("verify", help="independently re-verify a run directory")
    p.add_argument("run_dir")
    _add_common(p)
    p.add_argument("--no-network", action="store_true", help="skip the live post re-fetch check")
    p.set_defaults(func=_cmd_verify)

    p = sub.add_parser("report", help="render a run directory into a self-contained report.html")
    p.add_argument("run_dir")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("doctor", help="show installed engines / providers / anchors and what each needs")
    _add_common(p)
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("keygen", help="create an Ed25519 operator signing key")
    p.add_argument("--out", default=None, help="key file path (default: operator_ed25519.key)")
    p.set_defaults(func=_cmd_keygen)

    p = sub.add_parser("describe", help="describe any image: attributes + local VLM caption")
    p.add_argument("image")
    p.add_argument("--no-caption", action="store_true", help="skip the VLM caption (attributes only)")
    p.add_argument("--no-faces", action="store_true", help="skip face detection")
    p.add_argument("--model", default=None, help="caption model id (default: BLIP base)")
    p.add_argument("--json", action="store_true", help="emit JSON")
    p.set_defaults(func=_cmd_describe)

    p = sub.add_parser("chain", help="inspect the local ledger")
    _add_common(p)
    csub = p.add_subparsers(dest="chain_cmd", required=True)
    for name, chelp in (
        ("show", "print every block"),
        ("verify", "full integrity re-check"),
        ("tamper", "[demo] corrupt a block to show detection"),
    ):
        cp = csub.add_parser(name, help=chelp)
        _add_common(cp)
        if name == "tamper":
            cp.add_argument("--block", type=int, default=None, help="block index (default: head)")
    p.set_defaults(func=_cmd_chain)

    p = sub.add_parser("fetch-corpus", help="build the offline search corpus")
    _add_common(p)
    p.add_argument("--query", action="append", help="repeatable Commons full-text query")
    p.add_argument("--per-query", type=int, default=3)
    p.add_argument("--name", action="append",
                   help="repeatable person name -> one canonical Wikidata P18 portrait")
    p.add_argument("--names-file", default=None,
                   help="text file of names (one per line) for Wikidata P18 lookup")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed-demo", action="store_true",
                   help="copy the repo's bundled public-domain fixtures instead of pulling live")
    p.set_defaults(func=_cmd_fetch_corpus)

    p = sub.add_parser("fetch-models", help="pre-download the ONNX face models (YuNet + SFace)")
    p.set_defaults(func=_cmd_fetch_models)

    p = sub.add_parser("build-index", help="encode the corpus into a face-embedding index")
    _add_common(p)
    p.add_argument("--rebuild", action="store_true", help="ignore cached vectors and re-encode all")
    p.set_defaults(func=_cmd_build_index)

    p = sub.add_parser("bench", help="benchmark every face engine (ROC/AUC/EER + robustness)")
    p.add_argument("--corpus", default=None, help="labelled folder: <identity>__<n>.jpg")
    p.add_argument("--out", default=None, help="output dir (default: bench/)")
    p.add_argument("--no-augment", action="store_true", help="natural pairs only")
    p.add_argument("--engines", default=None, help="comma list to restrict (e.g. sface,opencv)")
    p.set_defaults(func=_cmd_bench)

    p = sub.add_parser("version", help="print version + effective config")
    _add_common(p)
    p.set_defaults(func=_cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page; make sure our output
    # (help text, captions, log lines) never dies on a non-ASCII character.
    import contextlib

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with contextlib.suppress(ValueError, OSError):  # already-detached stream
                reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FaceChainError as exc:
        print(f"error [{exc.code}]: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
