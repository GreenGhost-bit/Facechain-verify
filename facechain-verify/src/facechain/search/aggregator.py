"""Run providers, fetch every candidate, and rank by face-embedding similarity.

The ranking is the only thing that selects a match. Providers contribute
candidates; this module fetches each one under the SSRF policy, encodes its
dominant face with the *same* engine used for the probe, computes cosine
similarity, sorts, and applies the threshold. If nothing clears the threshold it
raises :class:`NoMatchFoundError` but still carries the ranked near-misses so the
CLI can show "closest was X at 0.79".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

from ..canonical import to_fixed
from ..errors import NoMatchFoundError, ProviderError, UnsafeURLError
from ..face import cosine, encode_candidate
from ..imaging import load_image_bytes
from ..logging import LOG
from ..models import Candidate, MatchResult, SearchSummary
from .base import ProbeContext, RawCandidate, SearchProvider


@dataclass
class ScoredCandidate:
    raw: RawCandidate
    similarity: float
    model: Candidate
    image_bytes: bytes | None = None


@dataclass
class AggregateResult:
    match: MatchResult
    summary: SearchSummary
    scored: list[ScoredCandidate] = field(default_factory=list)

    @property
    def best_bytes(self) -> bytes | None:
        return self.scored[0].image_bytes if self.scored else None


def _read_file_uri(url: str, *, max_bytes: int) -> bytes:
    parts = urlsplit(url)
    # url2pathname handles the platform quirks (Windows drive letters, %20, ...).
    local = url2pathname(parts.path)
    path = Path(f"//{parts.netloc}{local}") if parts.netloc else Path(local)
    data = path.read_bytes()
    if len(data) > max_bytes:
        raise UnsafeURLError(f"local candidate exceeds {max_bytes} bytes", detail=url)
    return data


class SearchAggregator:
    def __init__(self, providers: list[SearchProvider]) -> None:
        if not providers:
            raise ProviderError("no search providers configured")
        self.providers = providers

    def run(self, probe: ProbeContext) -> AggregateResult:
        settings = probe.settings
        raw: list[RawCandidate] = []
        providers_run: list[str] = []
        providers_ok: list[str] = []
        providers_failed: dict[str, str] = {}

        for provider in self.providers:
            providers_run.append(provider.name)
            with LOG.span("search.provider", provider=provider.name) as sp:
                try:
                    hits = list(provider.search(probe))
                except (ProviderError, UnsafeURLError) as exc:
                    providers_failed[provider.name] = str(exc)
                    sp["error"] = str(exc)
                    continue
                except Exception as exc:
                    providers_failed[provider.name] = repr(exc)
                    sp["error"] = repr(exc)
                    continue
                providers_ok.append(provider.name)
                sp["hits"] = len(hits)
                raw.extend(hits)

        # de-duplicate across providers, keeping the first (provider order = priority)
        deduped: dict[str, RawCandidate] = {}
        for cand in raw:
            deduped.setdefault(cand.key(), cand)
        unique = list(deduped.values())
        LOG.info("search.candidates", raw=len(raw), unique=len(unique))

        scored = self._score(unique, probe)
        scored.sort(key=lambda s: s.similarity, reverse=True)
        for rank, s in enumerate(scored):
            s.model.rank = rank

        summary = SearchSummary(
            providers_run=providers_run,
            providers_ok=providers_ok,
            providers_failed=providers_failed,
            candidates_seen=len(unique),
            candidates_scored=len(scored),
        )

        threshold = settings.match_threshold
        ranked_models = [s.model for s in scored]

        if not scored or scored[0].similarity < threshold:
            closest = scored[0].similarity if scored else 0.0
            raise NoMatchFoundError(
                f"no candidate reached the match threshold {threshold:.3f} "
                f"(closest similarity {closest:.3f} over {len(scored)} scored candidates)",
                detail={
                    "summary": summary.model_dump(),
                    "ranked": [m.model_dump() for m in ranked_models[:5]],
                },
            )

        best = scored[0]
        ambiguous = False
        note = ""
        runner_up_close = (
            len(scored) > 1
            and scored[1].similarity >= threshold
            and best.similarity - scored[1].similarity <= settings.ambiguous_margin
        )
        if runner_up_close:
            ambiguous = True
            note = (
                f"two candidates above threshold within {settings.ambiguous_margin:.3f}: "
                f"{best.similarity:.3f} vs {scored[1].similarity:.3f}"
            )
            LOG.warning("search.ambiguous", note=note)

        match = MatchResult(
            threshold_ppm=to_fixed(threshold),
            ambiguous=ambiguous,
            ambiguity_note=note,
            best=best.model,
            ranked=ranked_models,
        )
        LOG.info(
            "search.match",
            provider=best.raw.provider,
            similarity=round(best.similarity, 4),
            post_url=best.raw.post_url,
            ambiguous=ambiguous,
        )
        return AggregateResult(match=match, summary=summary, scored=scored)

    def _score(self, candidates: list[RawCandidate], probe: ProbeContext) -> list[ScoredCandidate]:
        settings = probe.settings
        out: list[ScoredCandidate] = []
        for cand in candidates:
            model = Candidate(
                provider=cand.provider,
                post_url=cand.post_url,
                image_url=cand.image_url,
                title=cand.title[:300],
                snippet=cand.snippet[:300],
            )
            try:
                if cand.image_url.lower().startswith("file://"):
                    data = _read_file_uri(cand.image_url, max_bytes=settings.max_image_bytes)
                    final_url = cand.image_url
                else:
                    fetched = probe.fetcher.fetch_image(cand.image_url)
                    data, final_url = fetched.content, fetched.final_url
                loaded = load_image_bytes(
                    data,
                    max_bytes=settings.max_image_bytes,
                    max_pixels=settings.max_image_pixels,
                    source=cand.image_url,
                )
            except Exception as exc:
                model.note = f"skipped: {type(exc).__name__}: {exc}"
                LOG.warning("search.candidate.skip", url=cand.image_url, error=str(exc))
                out.append(ScoredCandidate(raw=cand, similarity=-1.0, model=model))
                continue

            emb = encode_candidate(loaded.rgb, engine=probe.face_engine)
            model.fetched = True
            model.image_fingerprint = loaded.fingerprint
            if emb is None:
                model.note = "no usable face in candidate image"
                out.append(ScoredCandidate(raw=cand, similarity=-1.0, model=model))
                continue

            sim = cosine(probe.embedding, emb)
            model.similarity_ppm = to_fixed(sim)
            if final_url != cand.image_url:
                model.note = f"redirected to {final_url}"
            out.append(
                ScoredCandidate(raw=cand, similarity=sim, model=model, image_bytes=data)
            )
        return out
