"""Face-engine benchmark: ROC / AUC / EER / TAR@FAR + a robustness sweep.

``facechain bench`` (or ``python -m facechain.bench``) scores every available
engine on a labelled pair set and writes a Markdown table plus a dependency-free
SVG ROC curve to ``bench/``.

Pair set, in order of preference:

* ``--corpus DIR`` -- images named ``<identity>__<anything>.jpg``; genuine pairs
  share the ``<identity>`` prefix, impostor pairs do not. This is the real
  headline number -- point it at LFW pairs or your own labelled folder.
* otherwise the bundled ``tests/fixtures`` micro-set: the natural
  same-person pairs (Obama, Eisenhower) plus, for every portrait, augmentation
  pairs (JPEG q30, 0.5x resize, +/-8 deg rotate, greyscale, gamma, crop-pad).
  Small, but enough to show the SFace/ArcFace margin vs the classical descriptor.

Nothing here touches the pipeline or the chain; it only exercises
``detect`` + ``embed``.
"""

from __future__ import annotations

import io
import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .face import cosine, encode_candidate
from .face.base import FaceEngine
from .face.calibration import default_threshold
from .face.factory import _construct, _is_available
from .imaging import load_image_bytes
from .logging import LOG

_ENGINES = ("insightface", "sface", "opencv", "numpy")
_FAR_TARGETS = (0.01, 0.001)


# --------------------------------------------------------------------------
# augmentations
# --------------------------------------------------------------------------
def _reencode(img: Image.Image, *, fmt: str = "JPEG", quality: int = 30) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


_AUGMENTS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "jpeg_q30": lambda im: _reencode(im, quality=30),
    "resize_0.5x": lambda im: im.resize((max(1, im.width // 2), max(1, im.height // 2))),
    "rotate_+8": lambda im: im.rotate(8, expand=False, fillcolor=(127, 127, 127)),
    "rotate_-8": lambda im: im.rotate(-8, expand=False, fillcolor=(127, 127, 127)),
    "greyscale": lambda im: ImageOps.grayscale(im).convert("RGB"),
    "gamma_1.6": lambda im: Image.eval(im, lambda p: int(255 * (p / 255) ** 1.6)),
    "crop_pad": lambda im: ImageOps.expand(
        im.crop((int(im.width * 0.08), int(im.height * 0.08),
                 int(im.width * 0.92), int(im.height * 0.92))),
        border=12, fill=(127, 127, 127),
    ),
}


# --------------------------------------------------------------------------
# pair building
# --------------------------------------------------------------------------
@dataclass
class Pair:
    a: np.ndarray
    b: np.ndarray
    genuine: bool
    tag: str


def _rgb(path: Path) -> np.ndarray:
    return load_image_bytes(path.read_bytes(), source=str(path)).rgb


def _embed(engine: FaceEngine, rgb: np.ndarray) -> np.ndarray | None:
    try:
        return encode_candidate(rgb, engine=engine)
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("bench.embed_failed", error=str(exc))
        return None


_FIXTURE_IDENTITY = {
    "probe_obama": "obama",
    "corpus_obama_reencode": "obama",
    "probe_repost": "eisenhower",
    "corpus_eisenhower": "eisenhower",
    "corpus_kennedy": "kennedy",
    "corpus_reagan": "reagan",
    "probe_lincoln": "lincoln",
}


def _gather_images(corpus: Path | None, repo_root: Path) -> dict[str, list[Path]]:
    """identity -> list of image paths."""
    by_id: dict[str, list[Path]] = {}
    if corpus and corpus.is_dir():
        for p in sorted(corpus.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            ident = p.stem.split("__", 1)[0].split("_")[0].lower()
            by_id.setdefault(ident, []).append(p)
        return by_id
    fixtures = repo_root / "tests" / "fixtures"
    samples = repo_root / "samples"
    for stem, ident in _FIXTURE_IDENTITY.items():
        for base in (fixtures, samples):
            for ext in (".jpg", ".png"):
                cand = base / f"{stem}{ext}"
                if cand.is_file():
                    by_id.setdefault(ident, []).append(cand)
    return by_id


def build_pairs(
    engine: FaceEngine, *, corpus: Path | None, repo_root: Path, augment: bool
) -> list[Pair]:
    by_id = _gather_images(corpus, repo_root)
    # cache embeddings of the natural images
    emb_cache: dict[Path, np.ndarray] = {}
    for paths in by_id.values():
        for p in paths:
            e = _embed(engine, _rgb(p))
            if e is not None:
                emb_cache[p] = e

    pairs: list[Pair] = []
    # genuine: natural same-identity pairs
    for ident, paths in by_id.items():
        usable = [p for p in paths if p in emb_cache]
        for a, b in itertools.combinations(usable, 2):
            pairs.append(Pair(emb_cache[a], emb_cache[b], True, f"natural:{ident}"))
    # genuine: augmentation pairs
    if augment:
        for ident, paths in by_id.items():
            for p in paths:
                if p not in emb_cache:
                    continue
                base_img = Image.fromarray(_rgb(p), "RGB")
                for name, fn in _AUGMENTS.items():
                    try:
                        aug_rgb = np.asarray(fn(base_img).convert("RGB"), dtype=np.uint8)
                    except Exception:  # pragma: no cover
                        continue
                    e = _embed(engine, aug_rgb)
                    if e is not None:
                        pairs.append(Pair(emb_cache[p], e, True, f"aug:{ident}:{name}"))
    # impostor: all cross-identity natural combinations
    ids = list(by_id)
    for id_a, id_b in itertools.combinations(ids, 2):
        for pa in by_id[id_a]:
            for pb in by_id[id_b]:
                if pa in emb_cache and pb in emb_cache:
                    pairs.append(Pair(emb_cache[pa], emb_cache[pb], False, f"imp:{id_a}/{id_b}"))
    return pairs


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
@dataclass
class EngineResult:
    engine: str
    version: str
    threshold: float
    n_genuine: int
    n_impostor: int
    auc: float
    eer: float
    mean_genuine: float
    mean_impostor: float
    margin: float
    tar_at_far: dict[str, float] = field(default_factory=dict)
    acc_at_threshold: float = 0.0
    roc: list[tuple[float, float]] = field(default_factory=list)  # (far, tar)
    robustness: list[tuple[str, float, bool]] = field(default_factory=list)


def _roc(genuine: np.ndarray, impostor: np.ndarray) -> tuple[list[tuple[float, float]], float, float]:
    thresholds = np.unique(np.concatenate([genuine, impostor, [1.0, -1.0]]))[::-1]
    pts: list[tuple[float, float]] = []
    for t in thresholds:
        tar = float(np.mean(genuine >= t)) if genuine.size else 0.0
        far = float(np.mean(impostor >= t)) if impostor.size else 0.0
        pts.append((far, tar))
    pts.sort()
    # AUC via trapezoid on (far, tar)
    auc = 0.0
    for (f0, t0), (f1, t1) in itertools.pairwise(pts):
        auc += (f1 - f0) * (t0 + t1) / 2.0
    # EER: where FAR ~= FRR (1 - TAR)
    eer = 1.0
    for far, tar in pts:
        frr = 1.0 - tar
        eer = min(eer, max(far, frr))
    return pts, float(auc), float(eer)


def _tar_at_far(genuine: np.ndarray, impostor: np.ndarray, far_target: float) -> float:
    if impostor.size == 0 or genuine.size == 0:
        return 0.0
    t = np.quantile(impostor, 1.0 - far_target)
    return float(np.mean(genuine >= t))


def score_engine(
    kind: str, *, corpus: Path | None, repo_root: Path, augment: bool
) -> EngineResult | None:
    if not _is_available(kind):
        return None
    try:
        engine = _construct(kind)
    except Exception as exc:
        LOG.warning("bench.engine_unavailable", engine=kind, error=str(exc))
        return None

    pairs = build_pairs(engine, corpus=corpus, repo_root=repo_root, augment=augment)
    genuine = np.array([cosine(p.a, p.b) for p in pairs if p.genuine], dtype=np.float64)
    impostor = np.array([cosine(p.a, p.b) for p in pairs if not p.genuine], dtype=np.float64)
    if genuine.size == 0 or impostor.size == 0:
        LOG.warning("bench.no_pairs", engine=engine.name)
        return None

    thr = default_threshold(engine.name)
    roc, auc, eer = _roc(genuine, impostor)
    acc = float(
        (np.sum(genuine >= thr) + np.sum(impostor < thr)) / (genuine.size + impostor.size)
    )
    res = EngineResult(
        engine=engine.name,
        version=engine.version,
        threshold=thr,
        n_genuine=int(genuine.size),
        n_impostor=int(impostor.size),
        auc=auc,
        eer=eer,
        mean_genuine=float(genuine.mean()),
        mean_impostor=float(impostor.mean()),
        margin=float(genuine.mean() - impostor.mean()),
        tar_at_far={f"{ft:g}": _tar_at_far(genuine, impostor, ft) for ft in _FAR_TARGETS},
        acc_at_threshold=acc,
        roc=roc,
    )
    res.robustness = _robustness(engine, repo_root, thr)
    return res


def _robustness(engine: FaceEngine, repo_root: Path, threshold: float) -> list[tuple[str, float, bool]]:
    """Genuine Obama pair, one side perturbed -> does it still clear threshold?"""
    base = repo_root / "samples" / "probe_obama.jpg"
    other = repo_root / "tests" / "fixtures" / "corpus_obama_reencode.jpg"
    if not (base.is_file() and other.is_file()):
        return []
    e_other = _embed(engine, _rgb(other))
    if e_other is None:
        return []
    base_img = Image.fromarray(_rgb(base), "RGB")
    rows: list[tuple[str, float, bool]] = [("(none)", 0.0, False)]
    e_base = _embed(engine, np.asarray(base_img, dtype=np.uint8))
    if e_base is not None:
        c = cosine(e_base, e_other)
        rows[0] = ("(none)", c, c >= threshold)
    for name, fn in _AUGMENTS.items():
        try:
            aug = np.asarray(fn(base_img).convert("RGB"), dtype=np.uint8)
        except Exception:  # pragma: no cover
            continue
        e = _embed(engine, aug)
        if e is None:
            rows.append((name, float("nan"), False))
            continue
        c = cosine(e, e_other)
        rows.append((name, c, c >= threshold))
    return rows


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _svg_roc(results: list[EngineResult]) -> str:
    W, H, pad = 420, 420, 48
    colours = ["#2563eb", "#16a34a", "#d97706", "#dc2626", "#7c3aed"]
    plot_w, plot_h = W - 2 * pad, H - 2 * pad

    def xy(far: float, tar: float) -> tuple[float, float]:
        return pad + far * plot_w, H - pad - tar * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif" font-size="11">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<rect x="{pad}" y="{pad}" width="{plot_w}" height="{plot_h}" '
        f'fill="#f8fafc" stroke="#cbd5e1"/>',
        f'<line x1="{pad}" y1="{H - pad}" x2="{W - pad}" y2="{pad}" '
        f'stroke="#e2e8f0" stroke-dasharray="4 3"/>',
        f'<text x="{W / 2}" y="{H - 12}" text-anchor="middle">false accept rate</text>',
        f'<text x="14" y="{H / 2}" text-anchor="middle" '
        f'transform="rotate(-90 14 {H / 2})">true accept rate</text>',
    ]
    for i, r in enumerate(results):
        col = colours[i % len(colours)]
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(f, t) for f, t in r.roc))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        parts.append(
            f'<text x="{pad + 8}" y="{pad + 16 + i * 15}" fill="{col}">'
            f'{r.engine}  AUC={r.auc:.3f}  EER={r.eer:.2f}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _markdown(results: list[EngineResult], *, source: str) -> str:
    lines = [
        "# Face-engine benchmark",
        "",
        f"_Pair source: **{source}**_",
        "",
        "| engine | genuine | impostor | mean gen | mean imp | margin | AUC | EER | "
        "TAR@FAR=1e-2 | acc@thr | thr |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in results:
        lines.append(
            f"| `{r.engine}` | {r.n_genuine} | {r.n_impostor} | {r.mean_genuine:.3f} | "
            f"{r.mean_impostor:.3f} | **{r.margin:.3f}** | **{r.auc:.3f}** | {r.eer:.3f} | "
            f"{r.tar_at_far.get('0.01', 0.0):.3f} | {r.acc_at_threshold:.3f} | {r.threshold:.2f} |"
        )
    lines += ["", "![ROC](roc.svg)", "", "## Robustness (genuine Obama pair, one side perturbed)", ""]
    for r in results:
        lines += [f"### `{r.engine}` (threshold {r.threshold:.2f})", "",
                  "| perturbation | cosine | still matches |", "|---|--:|:--:|"]
        for name, cos_v, ok in r.robustness:
            lines.append(f"| {name} | {cos_v:.3f} | {'yes' if ok else 'NO'} |")
        lines.append("")
    return "\n".join(lines)


def run_bench(
    *, corpus: Path | None = None, out_dir: Path | None = None, augment: bool = True,
    repo_root: Path | None = None, engines: tuple[str, ...] | None = None,
) -> Path:
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    out_dir = out_dir or (repo_root / "bench")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[EngineResult] = []
    for kind in (engines or _ENGINES):
        with LOG.span("bench.engine", engine=kind):
            r = score_engine(kind, corpus=corpus, repo_root=repo_root, augment=augment)
        if r is not None:
            results.append(r)
    if not results:
        raise RuntimeError("no face engine available to benchmark")

    if corpus:
        source = str(corpus)
    else:
        source = "bundled tests/fixtures micro-set" + (" (augmented)" if augment else " (natural pairs only)")
    (out_dir / "roc.svg").write_text(_svg_roc(results), encoding="utf-8")
    (out_dir / "results.md").write_text(_markdown(results, source=source), encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps(
            [
                {k: v for k, v in r.__dict__.items() if k != "roc"} | {"roc_points": r.roc}
                for r in results
            ],
            indent=2, default=float,
        ),
        encoding="utf-8",
    )
    LOG.info("bench.done", engines=[r.engine for r in results], out=str(out_dir))
    return out_dir / "results.md"


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="facechain bench")
    ap.add_argument("--corpus", default=None, help="labelled folder: <identity>__<n>.jpg")
    ap.add_argument("--out", default=None, help="output dir (default: bench/)")
    ap.add_argument("--no-augment", action="store_true", help="natural pairs only")
    ap.add_argument("--engines", default=None,
                    help="comma list to restrict (e.g. sface,opencv); default: all available")
    args = ap.parse_args(argv)
    md = run_bench(
        corpus=Path(args.corpus) if args.corpus else None,
        out_dir=Path(args.out) if args.out else None,
        augment=not args.no_augment,
        engines=tuple(e.strip() for e in args.engines.split(",")) if args.engines else None,
    )
    print(md.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
