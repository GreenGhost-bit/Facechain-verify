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
    p.add_argument("--engine", choices=["auto", "opencv", "numpy", "insightface"], default=None,
                   help="face engine (default: auto)")
    p.add_argument("--providers", default=None,
                   help="comma-separated search providers (default: wikimedia,local)")
    p.add_argument("--anchor", choices=["local", "evm"], default=None,
                   help="blockchain anchor backend (default: local)")
    p.add_argument("--threshold", type=float, default=None,
                   help="face-match cosine threshold (default: 0.86)")
    p.add_argument("--difficulty", type=int, default=None,
                   help="local-chain proof-of-work leading zero bits (default: 0)")
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--chain-dir", default=None)
    p.add_argument("--corpus-dir", default=None)
    p.add_argument("--hint", default=None, help="text hint (a name/keywords) to focus the search")
    p.add_argument("--probe-image-url", default=None,
                   help="public URL of the probe image (required by the serpapi provider)")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON on stdout")


def _settings_from_args(args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    if getattr(args, "engine", None):
        overrides["face_engine"] = args.engine
    if getattr(args, "providers", None):
        overrides["search_providers"] = args.providers
    if getattr(args, "anchor", None):
        overrides["anchor_backend"] = args.anchor
    if getattr(args, "threshold", None) is not None:
        overrides["match_threshold"] = args.threshold
    if getattr(args, "difficulty", None) is not None:
        overrides["chain_difficulty_bits"] = args.difficulty
    if getattr(args, "runs_dir", None):
        overrides["runs_dir"] = Path(args.runs_dir)
    if getattr(args, "chain_dir", None):
        overrides["chain_dir"] = Path(args.chain_dir)
    if getattr(args, "corpus_dir", None):
        overrides["corpus_dir"] = Path(args.corpus_dir)
    return Settings.load(**overrides)


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


def _cmd_search(args: argparse.Namespace) -> int:
    from .face import build_face_engine, encode_probe
    from .imaging import load_image_path
    from .netfetch import SafeFetcher
    from .search import ProbeContext, SearchAggregator, build_providers

    settings = _settings_from_args(args)
    image = load_image_path(args.image, max_bytes=settings.max_image_bytes,
                            max_pixels=settings.max_image_pixels)
    engine = build_face_engine(settings.face_engine)
    _, embedding, _ = encode_probe(image, engine=engine, min_face_pixels=settings.min_face_pixels)
    providers = build_providers(settings)
    with SafeFetcher(contact=settings.http_contact, timeout_s=settings.http_timeout_s,
                     max_redirects=settings.http_max_redirects,
                     max_bytes=settings.max_image_bytes) as fetcher:
        ctx = ProbeContext(image_bytes=image.raw_bytes, rgb=image.rgb, embedding=embedding,
                           settings=settings, fetcher=fetcher, face_engine=engine, hint=args.hint,
                           extra={"probe_image_url": args.probe_image_url} if args.probe_image_url else {})
        try:
            result = SearchAggregator(providers).run(ctx)
        except FaceChainError as exc:
            if args.json:
                print(json.dumps({"error": exc.code, "message": str(exc)}, indent=2))
            else:
                print(f"NO MATCH: {exc}", file=sys.stderr)
            return exc.exit_code

    if args.json:
        print(result.match.model_dump_json(indent=2))
    else:
        print(f"scored {result.summary.candidates_scored} candidate(s) "
              f"from {', '.join(result.summary.providers_ok) or 'no providers'}")
        for m in result.match.ranked[:10]:
            print(f"  #{m.rank:<2} sim={m.similarity_ppm / 1e6:+.4f}  [{m.provider}]  {m.post_url}")
            if m.note:
                print(f"       note: {m.note}")
        b = result.match.best
        print(f"\nBEST MATCH  sim={b.similarity_ppm / 1e6:.4f}  {b.post_url}")
        if result.match.ambiguous:
            print(f"AMBIGUOUS   {result.match.ambiguity_note}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_pipeline
    from .verify import format_report

    settings = _settings_from_args(args)
    result = run_pipeline(args.image, settings, hint=args.hint,
                          probe_image_url=args.probe_image_url,
                          verify_after=not args.no_verify)
    if result.status == "no_match":
        if args.json:
            print(json.dumps({"status": "no_match", "run_dir": str(result.run_dir),
                              "message": result.error}, indent=2))
        else:
            print(f"NO MATCH  (artifacts in {result.run_dir})")
            print(result.error)
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
        print(f"record_hash   : {b.record_hash}")
        print(f"anchored on   : {result.receipt.network}")
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
    from .corpus import fetch_corpus, seed_demo_corpus

    settings = _settings_from_args(args)
    if args.seed_demo:
        n = seed_demo_corpus(settings)
        print(f"seeded {n} demo entries into {settings.corpus_dir}")
        return 0
    n = fetch_corpus(settings, queries=args.query or None, per_query=args.per_query,
                     overwrite=args.overwrite)
    print(f"fetched {n} corpus entries into {settings.corpus_dir}")
    return 0 if n > 0 else 1


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
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("verify", help="independently re-verify a run directory")
    p.add_argument("run_dir")
    _add_common(p)
    p.add_argument("--no-network", action="store_true", help="skip the live post re-fetch check")
    p.set_defaults(func=_cmd_verify)

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
    p.add_argument("--query", action="append", help="repeatable search query")
    p.add_argument("--per-query", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed-demo", action="store_true",
                   help="copy the repo's bundled public-domain fixtures instead of pulling live")
    p.set_defaults(func=_cmd_fetch_corpus)

    p = sub.add_parser("version", help="print version + effective config")
    _add_common(p)
    p.set_defaults(func=_cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
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
