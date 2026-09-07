"""Guess a human identity label from ranked search hit titles / URLs.

Face similarity picks the best *image*; this module looks across the top of the
ranked list for a repeated name or @handle (e.g. many Lens hits saying
"Elvish Yadav" / ``gima_ashi``). That consensus:

* gives the CLI a readable ``WHO`` line;
* can override a single high-scoring but nameless / conflicting Yandex thumbnail
  when SerpAPI Lens agrees on a person.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

from ..models import Candidate

# @handle or path segment that looks like a social username
_HANDLE_RE = re.compile(r"@([A-Za-z0-9_][A-Za-z0-9._]{2,28})")
_USER_PATH_RE = re.compile(
    r"/(?:in|user|u|@)?/?([A-Za-z][A-Za-z0-9._-]{2,40})(?:/|$|\?)",
    re.IGNORECASE,
)
# "First Last" in titles (avoid 3-word org phrases like "Elvish Yadav Foundation")
_NAME_RE = re.compile(
    r"\b([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,})\b"
)
_NOISE_HANDLES = {
    "www", "com", "http", "https", "instagram", "facebook", "twitter", "youtube",
    "reddit", "tiktok", "linkedin", "pinterest", "shorts", "reel", "reels", "post",
    "posts", "photo", "photos", "video", "videos", "status", "share", "watch",
    "channel", "profile", "explore", "popular", "discover", "hashtag", "tag",
    "amp", "index", "html", "php", "wiki", "file", "upload", "static", "cdn",
    "img", "image", "images", "media", "content", "assets",
}
# Multi-word phrases that the "First Last" regex catches but that are never a
# person: source attributions, licences, generic captions.
_NOISE_PHRASES = {
    "wikimedia commons", "creative commons", "getty images", "associated press",
    "official portrait", "official photo", "white house", "public domain",
    "local corpus", "stock photo", "royalty free", "no restrictions",
}
_PROVIDER_TRUST = {
    "serpapi": 1.0,
    "hint": 0.85,
    "multiris": 0.55,
    "wikimedia": 0.4,
    "local": 0.3,
}


@dataclass(frozen=True)
class IdentityConsensus:
    label: str
    normalized: str
    support: int
    confidence: float  # 0..1
    note: str = ""


def _provider_trust(provider: str) -> float:
    base = provider.split("/", 1)[0].lower()
    return _PROVIDER_TRUST.get(base, 0.5)


def normalize_identity(text: str) -> str:
    t = unquote(text or "").strip().lower()
    t = t.replace("@", "")
    t = re.sub(r"[_\-.]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _tokens_from_url(url: str) -> list[str]:
    out: list[str] = []
    try:
        parts = urlsplit(url)
    except Exception:
        return out
    path = unquote(parts.path or "")
    for m in _USER_PATH_RE.finditer(path):
        seg = m.group(1)
        if seg.lower() in _NOISE_HANDLES:
            continue
        if re.fullmatch(r"\d+", seg):
            continue
        out.append(seg)
    # query rarely has names; skip
    return out


def extract_identity_labels(*texts: str) -> list[str]:
    """Pull candidate identity strings from titles, snippets, and URLs."""
    found: list[str] = []
    for raw in texts:
        if not raw:
            continue
        for m in _HANDLE_RE.finditer(raw):
            h = m.group(1)
            if h.lower() not in _NOISE_HANDLES:
                found.append(h)
        if raw.startswith("http://") or raw.startswith("https://"):
            found.extend(_tokens_from_url(raw))
        else:
            for m in _NAME_RE.finditer(raw):
                if m.group(1).lower() not in _NOISE_PHRASES:
                    found.append(m.group(1))
            # snake/kebab usernames embedded in titles
            for m in re.finditer(r"\b([a-z][a-z0-9]+_[a-z0-9_]{2,})\b", raw.lower()):
                found.append(m.group(1))
    # de-dupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for label in found:
        key = normalize_identity(label)
        if len(key) < 3 or key in _NOISE_HANDLES or key in _NOISE_PHRASES or key in seen:
            continue
        seen.add(key)
        out.append(label.replace("_", " ").strip())
    return out


def identities_match(a: str, b: str) -> bool:
    """True if two normalized labels refer to the same person-ish string."""
    if not a or not b:
        return False
    if a == b:
        return True
    compact_a, compact_b = a.replace(" ", ""), b.replace(" ", "")
    if compact_a == compact_b:
        return True
    # "elvish" vs "elvish yadav"
    if a.startswith(b + " ") or b.startswith(a + " "):
        return True
    if compact_a.startswith(compact_b) or compact_b.startswith(compact_a):
        return min(len(compact_a), len(compact_b)) >= 5
    return False


def _fold_identity_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Merge short keys into longer ones (``elvish`` → ``elvish yadav``)."""
    keys = sorted(weights.keys(), key=lambda k: (-len(k), -weights[k]))
    folded: dict[str, float] = defaultdict(float)
    consumed: set[str] = set()
    for long in keys:
        if long in consumed:
            continue
        total = weights[long]
        for short in keys:
            if short == long or short in consumed:
                continue
            if identities_match(long, short) and len(long) >= len(short):
                total += weights[short]
                consumed.add(short)
        folded[long] = total
        consumed.add(long)
    return folded


def display_label(normalized: str, examples: list[str]) -> str:
    """Pick a readable label (prefer spaced Title Case over snake_case)."""
    for ex in examples:
        if normalize_identity(ex) == normalized and " " in ex:
            return ex.strip()
    for ex in examples:
        if normalize_identity(ex) == normalized:
            pretty = ex.replace("_", " ").strip()
            return pretty.title() if pretty.islower() else pretty
    return normalized.title()


def consensus_from_candidates(
    ranked: list[Candidate],
    *,
    threshold_ppm: int,
    top_k: int = 12,
    min_support: int = 2,
) -> IdentityConsensus | None:
    """Majority identity among the strongest face-matched hits."""
    if not ranked:
        return None
    threshold = threshold_ppm / 1_000_000.0
    pool = ranked[:top_k]
    # Prefer hits that cleared (or nearly cleared) the face threshold.
    strong = [c for c in pool if c.similarity_ppm / 1e6 >= max(0.35, threshold - 0.08)]
    if len(strong) >= 2:
        pool = strong

    weights: dict[str, float] = defaultdict(float)
    examples: dict[str, list[str]] = {}
    for cand in pool:
        sim = max(0.0, cand.similarity_ppm / 1e6)
        trust = _provider_trust(cand.provider)
        labels = extract_identity_labels(cand.title, cand.snippet, cand.post_url)
        if not labels:
            continue
        # One candidate contributes once per unique normalized label.
        seen_local: set[str] = set()
        for lab in labels:
            key = normalize_identity(lab)
            if key in seen_local:
                continue
            seen_local.add(key)
            weights[key] += (0.5 + sim) * trust
            examples.setdefault(key, []).append(lab)

    if not weights:
        return None

    weights = _fold_identity_weights(weights)
    # Re-attach examples onto folded keys.
    folded_examples: dict[str, list[str]] = {}
    for key, labs in examples.items():
        target = key
        for folded_key in weights:
            if identities_match(key, folded_key) and len(folded_key) >= len(key):
                target = folded_key
                break
        folded_examples.setdefault(target, []).extend(labs)
    examples = folded_examples

    best_key, best_w = max(weights.items(), key=lambda kv: kv[1])
    support = sum(
        1
        for c in pool
        if any(
            identities_match(best_key, normalize_identity(x))
            for x in extract_identity_labels(c.title, c.snippet, c.post_url)
        )
    )
    if support < min_support and best_w < 2.0:
        return None

    total = sum(weights.values()) or 1.0
    confidence = min(1.0, best_w / total + 0.15 * min(support, 5))
    label = display_label(best_key, examples.get(best_key, [best_key]))
    note = f"{support}/{len(pool)} top hits mention {label!r}"
    return IdentityConsensus(
        label=label,
        normalized=best_key,
        support=support,
        confidence=float(confidence),
        note=note,
    )


def cluster_score(cand: Candidate, consensus: IdentityConsensus | None) -> float:
    """Ranking key: face sim + Lens trust + consensus agreement."""
    sim = cand.similarity_ppm / 1e6
    trust = _provider_trust(cand.provider)
    bonus = 0.015 * trust  # small Lens preference on near-ties
    if consensus is not None:
        labels = {
            normalize_identity(x)
            for x in extract_identity_labels(cand.title, cand.snippet, cand.post_url)
        }
        agrees = any(identities_match(consensus.normalized, lab) for lab in labels)
        if agrees:
            # Strong enough to beat a lone higher-sim Yandex thumbnail.
            bonus += 0.09 * min(1.0, 0.4 + 0.6 * consensus.confidence)
        elif cand.provider.startswith("multiris") and labels:
            bonus -= 0.08
    return sim + bonus
