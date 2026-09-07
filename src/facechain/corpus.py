"""Build the offline search corpus by live-pulling from Wikimedia Commons.

``facechain fetch-corpus --query "..." --query "..."`` issues real MediaWiki
searches, downloads the actual image files under the SSRF policy, and writes each
as ``<slug>.<ext>`` + a sibling ``<slug>.json`` recording the real File: page URL
and licence. The corpus is content the pipeline *searches over* offline -- it is
gathered live and is not tailored to any particular probe.

``--seed-demo`` instead copies the repo's bundled public-domain fixtures into the
corpus (with their recorded source URLs) so the pipeline can be demonstrated with
no network at all.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from .config import Settings
from .imaging import load_image_bytes
from .logging import LOG
from .netfetch import SafeFetcher

_API = "https://commons.wikimedia.org/w/api.php"
_WD_API = "https://www.wikidata.org/w/api.php"
_COMMONS_FILEPATH = "https://commons.wikimedia.org/wiki/Special:FilePath/"
_DEFAULT_QUERIES = (
    "Barack Obama official portrait",
    "Angela Merkel portrait",
    "Cristiano Ronaldo face",
    "Marie Curie portrait",
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "item"


def _license(info: dict[str, object]) -> str:
    meta = info.get("extmetadata")
    if isinstance(meta, dict):
        node = meta.get("LicenseShortName")
        if isinstance(node, dict) and isinstance(node.get("value"), str):
            return str(node["value"])
    return "see Commons file page"


def _ext(mime: str) -> str:
    return {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(mime, ".jpg")


def fetch_corpus(
    settings: Settings,
    *,
    queries: list[str] | None = None,
    per_query: int = 3,
    overwrite: bool = False,
) -> int:
    dest = settings.corpus_dir
    dest.mkdir(parents=True, exist_ok=True)
    queries = queries or list(_DEFAULT_QUERIES)
    saved = 0
    with SafeFetcher(
        contact=settings.http_contact,
        timeout_s=settings.http_timeout_s,
        max_redirects=settings.http_max_redirects,
        max_bytes=settings.max_image_bytes,
    ) as fetcher:
        for query in queries:
            params = {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": "6",
                "gsrlimit": str(per_query),
                "prop": "imageinfo",
                "iiprop": "url|mime|extmetadata",
                "iiurlwidth": "900",
                "maxlag": "5",
            }
            try:
                data = fetcher.get_json(_API, params=params)
            except Exception as exc:
                LOG.warning("corpus.query_failed", query=query, error=str(exc))
                continue
            pages = (data or {}).get("query", {}).get("pages", {}) if isinstance(data, dict) else {}
            for page in pages.values():
                infos = page.get("imageinfo") or []
                if not infos:
                    continue
                info = infos[0]
                mime = str(info.get("mime", ""))
                if not mime.startswith("image/"):
                    continue
                img_url = str(info.get("thumburl") or info.get("url") or "")
                page_url = str(info.get("descriptionurl") or info.get("url") or "")
                title = str(page.get("title", "")).removeprefix("File:")
                slug = _slug(f"{query}-{title}")
                target = dest / f"{slug}{_ext(mime)}"
                if target.exists() and not overwrite:
                    continue
                try:
                    fetched = fetcher.fetch_image(img_url)
                    load_image_bytes(fetched.content, source=img_url)  # validate decodes
                except Exception as exc:
                    LOG.warning("corpus.download_failed", url=img_url, error=str(exc))
                    continue
                target.write_bytes(fetched.content)
                lic = _license(info)
                target.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "post_url": page_url,
                            "title": title,
                            "source": f"Wikimedia Commons ({lic})",
                            "query": query,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                saved += 1
                LOG.info("corpus.saved", file=target.name, from_query=query)
    LOG.info("corpus.done", saved=saved, dir=str(dest))
    return saved


def _wikidata_p18(fetcher: SafeFetcher, name: str) -> tuple[str, str, str] | None:
    """Resolve ``name`` -> (entity_id, P18 image filename, entity_url) via Wikidata.

    This is a precise, one-portrait-per-person lookup: the P18 ("image") claim on
    the person's Wikidata item, which is what infoboxes use.
    """
    try:
        found = fetcher.get_json(
            _WD_API,
            params={
                "action": "wbsearchentities", "format": "json",
                "language": "en", "type": "item", "limit": "1", "search": name,
            },
        )
    except Exception as exc:
        LOG.warning("corpus.wikidata.search_failed", name=name, error=str(exc))
        return None
    hits = found.get("search", []) if isinstance(found, dict) else []
    if not hits:
        LOG.info("corpus.wikidata.no_entity", name=name)
        return None
    qid = str(hits[0].get("id", ""))
    if not qid:
        return None
    try:
        claims = fetcher.get_json(
            _WD_API,
            params={
                "action": "wbgetclaims", "format": "json",
                "entity": qid, "property": "P18",
            },
        )
    except Exception as exc:
        LOG.warning("corpus.wikidata.claims_failed", qid=qid, error=str(exc))
        return None
    p18 = (claims.get("claims", {}) or {}).get("P18", []) if isinstance(claims, dict) else []
    for claim in p18:
        try:
            filename = claim["mainsnak"]["datavalue"]["value"]
        except (KeyError, TypeError):
            continue
        if isinstance(filename, str) and filename:
            return qid, filename, f"https://www.wikidata.org/wiki/{qid}"
    LOG.info("corpus.wikidata.no_p18", qid=qid, name=name)
    return None


def fetch_corpus_wikidata(
    settings: Settings, names: list[str], *, overwrite: bool = False
) -> int:
    """Populate the corpus with one canonical portrait per named person (Wikidata P18)."""
    dest = settings.corpus_dir
    dest.mkdir(parents=True, exist_ok=True)
    saved = 0
    with SafeFetcher(
        contact=settings.http_contact,
        timeout_s=settings.http_timeout_s,
        max_redirects=settings.http_max_redirects,
        max_bytes=settings.max_image_bytes,
    ) as fetcher:
        for name in names:
            name = name.strip()
            if not name:
                continue
            resolved = _wikidata_p18(fetcher, name)
            if resolved is None:
                continue
            qid, filename, entity_url = resolved
            slug = _slug(f"{name}-{qid}")
            existing = list(dest.glob(f"{slug}.*"))
            if existing and not overwrite:
                continue
            url = _COMMONS_FILEPATH + filename.replace(" ", "_")
            try:
                fetched = fetcher.fetch_image(url)
                loaded = load_image_bytes(fetched.content, source=url)
            except Exception as exc:
                LOG.warning("corpus.wikidata.download_failed", name=name, url=url, error=str(exc))
                continue
            ext = _ext(loaded.fingerprint.mime)
            target = dest / f"{slug}{ext}"
            target.write_bytes(fetched.content)
            target.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "post_url": entity_url,
                        "title": name,
                        "source": f"Wikidata {qid} P18 (Wikimedia Commons)",
                        "wikidata": qid,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            saved += 1
            LOG.info("corpus.wikidata.saved", name=name, qid=qid, file=target.name)
    LOG.info("corpus.wikidata.done", saved=saved, requested=len(names), dir=str(dest))
    return saved


def _find_repo_root(explicit: Path | None) -> Path:
    """Locate the repo checkout (needed for the bundled demo fixtures).

    Works for an editable install (``pip install -e .``) and when the command is
    run from anywhere inside the checkout.
    """
    if explicit is not None:
        return explicit
    candidates = [Path(__file__).resolve().parents[2], *Path.cwd().resolve().parents, Path.cwd()]
    for base in candidates:
        if (base / "tests" / "fixtures").is_dir() and (base / "samples").is_dir():
            return base
    return Path(__file__).resolve().parents[2]


def seed_demo_corpus(settings: Settings, *, repo_root: Path | None = None) -> int:
    """Populate the corpus from the repo's bundled public-domain fixtures."""
    root = _find_repo_root(repo_root)
    fixtures = root / "tests" / "fixtures"
    sources_file = root / "samples" / "SOURCES.json"
    if not fixtures.is_dir():
        raise FileNotFoundError(
            f"could not find tests/fixtures under {root}. Run `facechain fetch-corpus --seed-demo` "
            "from inside a git checkout of the repo (an editable `pip install -e .` install), "
            "or use `facechain fetch-corpus` to pull a corpus live instead."
        )
    sources: dict[str, dict[str, str]] = {}
    if sources_file.is_file():
        sources = json.loads(sources_file.read_text("utf-8"))

    dest = settings.corpus_dir
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in sorted(fixtures.glob("corpus_*.jpg")):
        target = dest / img.name
        shutil.copyfile(img, target)
        meta = sources.get(
            img.name,
            {"post_url": f"local://fixture/{img.name}", "title": img.stem, "source": "bundled fixture"},
        )
        target.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        count += 1
        LOG.info("corpus.seed", file=img.name)
    LOG.info("corpus.seed.done", count=count, dir=str(dest))
    return count
