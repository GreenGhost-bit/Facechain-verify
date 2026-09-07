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
from typing import Any
from urllib.parse import urlsplit
from urllib.request import url2pathname

from ..canonical import to_fixed
from ..errors import NoMatchFoundError, ProviderError, UnsafeURLError
from ..face import cosine, encode_candidate
from ..imaging import load_image_bytes
from ..logging import LOG
from ..models import Candidate, MatchResult, SearchSummary
from .base import ProbeContext, RawCandidate, SearchProvider
from .identity import (
    cluster_score,
    consensus_from_candidates,
    extract_identity_labels,
    identities_match,
    normalize_identity,
)


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
        # Accept/reject cut: operator's value if pinned, else the calibrated
        # per-engine default (LBPH rides ~0.86, SFace/ArcFace ~0.4).
        if settings.threshold_explicit:
            eff_threshold = settings.match_threshold
        else:
            from ..face.calibration import default_threshold

            eff_threshold = default_threshold(probe.face_engine.name)
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
        # Lens / multiris often return dozens of hits; keep scoring bounded.
        if any(
            c.provider.startswith("multiris") or c.provider.startswith("serpapi")
            for c in unique
        ):
            cap = max(48, settings.max_candidates_per_provider)
            if len(unique) > cap:
                LOG.info("search.candidates.capped", before=len(unique), after=cap)
                unique = unique[:cap]
        LOG.info("search.candidates", raw=len(raw), unique=len(unique))

        scored = self._score(unique, probe)
        # Raw face similarity first, then Lens/consensus-aware re-rank for near-ties.
        scored.sort(key=lambda s: s.similarity, reverse=True)
        ranked_models_pre = [s.model for s in scored]
        consensus = consensus_from_candidates(
            ranked_models_pre,
            threshold_ppm=to_fixed(eff_threshold),
        )
        scored.sort(
            key=lambda s: cluster_score(s.model, consensus),
            reverse=True,
        )
        for rank, s in enumerate(scored):
            s.model.rank = rank

        summary = SearchSummary(
            providers_run=providers_run,
            providers_ok=providers_ok,
            providers_failed=providers_failed,
            candidates_seen=len(unique),
            candidates_scored=len(scored),
        )

        threshold = eff_threshold
        ranked_models = [s.model for s in scored]

        if not scored or scored[0].similarity < threshold:
            closest = scored[0] if scored else None
            closest_sim = closest.similarity if closest else 0.0
            closest_info: dict[str, Any] | None = None
            if closest is not None:
                closest_info = {
                    "similarity": round(closest.similarity, 4),
                    "provider": closest.raw.provider,
                    "title": closest.raw.title,
                    "post_url": closest.raw.post_url,
                    "image_url": closest.raw.image_url,
                }
                LOG.info(
                    "search.closest_near_miss",
                    sim=closest_info["similarity"],
                    provider=closest_info["provider"],
                    title=(closest_info["title"] or "")[:80],
                    post_url=closest_info["post_url"],
                    identity=consensus.label if consensus else None,
                )
            raise NoMatchFoundError(
                f"no candidate reached the match threshold {threshold:.3f} "
                f"(closest similarity {closest_sim:.3f} over {len(scored)} scored candidates)",
                detail={
                    "summary": summary.model_dump(),
                    "ranked": [m.model_dump() for m in ranked_models[:5]],
                    "closest": closest_info,
                    "identity_guess": consensus.label if consensus else "",
                    "identity_note": consensus.note if consensus else "",
                    "identity_confidence": (
                        round(consensus.confidence, 4) if consensus else 0.0
                    ),
                },
            )

        best = scored[0]
        decided_by = "embedding_cosine"
        # If consensus names a person and the face-best disagrees, prefer the
        # strongest consensus-agreeing hit that still clears the threshold.
        if consensus is not None:
            for s in scored:
                if s.similarity < threshold:
                    break
                labels = {
                    normalize_identity(x)
                    for x in extract_identity_labels(
                        s.model.title, s.model.snippet, s.model.post_url
                    )
                }
                if any(identities_match(consensus.normalized, lab) for lab in labels):
                    if s is not best:
                        LOG.info(
                            "search.consensus_override",
                            from_url=best.raw.post_url,
                            to_url=s.raw.post_url,
                            identity=consensus.label,
                        )
                        best = s
                        # The winner is no longer the top embedding score -- say so.
                        decided_by = "embedding_cosine+text_consensus"
                    break

        ambiguous = False
        note = ""
        # Compare against the next distinct hit after possible consensus pick.
        others = [s for s in scored if s is not best]
        runner_up_close = (
            bool(others)
            and others[0].similarity >= threshold
            and best.similarity - others[0].similarity <= settings.ambiguous_margin
        )
        if runner_up_close:
            ambiguous = True
            note = (
                f"two candidates above threshold within {settings.ambiguous_margin:.3f}: "
                f"{best.similarity:.3f} vs {others[0].similarity:.3f}"
            )
            LOG.warning("search.ambiguous", note=note)

        # Keep scored[0] aligned with match.best for artifact writers.
        scored = [best] + [s for s in scored if s is not best]
        for rank, s in enumerate(scored):
            s.model.rank = rank
        ranked_models = [s.model for s in scored]

        match = MatchResult(
            threshold_ppm=to_fixed(threshold),
            decided_by=decided_by,
            ambiguous=ambiguous,
            ambiguity_note=note,
            identity_guess=consensus.label if consensus else "",
            identity_confidence_ppm=to_fixed(consensus.confidence) if consensus else 0,
            identity_note=consensus.note if consensus else "",
            best=best.model,
            ranked=ranked_models,
        )
        LOG.info(
            "search.match",
            provider=best.raw.provider,
            similarity=round(best.similarity, 4),
            post_url=best.raw.post_url,
            ambiguous=ambiguous,
            identity=match.identity_guess or None,
        )
        return AggregateResult(match=match, summary=summary, scored=scored)

    def _score(self, candidates: list[RawCandidate], probe: ProbeContext) -> list[ScoredCandidate]:
        settings = probe.settings
        out: list[ScoredCandidate] = []
        total = len(candidates)
        LOG.info("search.scoring.start", total=total)
        for idx, cand in enumerate(candidates):
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
                LOG.info(
                    "search.candidate.noface",
                    done=idx + 1,
                    total=total,
                    provider=cand.provider,
                    title=(cand.title or "")[:60],
                )
                continue

            sim = cosine(probe.embedding, emb)
            model.similarity_ppm = to_fixed(sim)
            if final_url != cand.image_url:
                model.note = f"redirected to {final_url}"
            out.append(
                ScoredCandidate(raw=cand, similarity=sim, model=model, image_bytes=data)
            )
            # Progress: first, last, every 6th, or anything that clears a soft bar.
            if idx == 0 or idx + 1 == total or (idx + 1) % 6 == 0 or sim >= 0.40:
                LOG.info(
                    "search.candidate.scored",
                    done=idx + 1,
                    total=total,
                    sim=round(sim, 4),
                    provider=cand.provider,
                    title=(cand.title or "")[:60],
                    post_url=cand.post_url[:120],
                )
        LOG.info(
            "search.scoring.end",
            scored=sum(1 for s in out if s.similarity >= 0),
            total=total,
            best=round(max((s.similarity for s in out), default=-1.0), 4),
        )
        return out
