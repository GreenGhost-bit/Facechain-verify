from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
from tests.conftest import FIXTURES, requires_opencv

from facechain.config import Settings
from facechain.errors import NoMatchFoundError
from facechain.face import encode_probe
from facechain.imaging import load_image_path
from facechain.netfetch import SafeFetcher
from facechain.search.aggregator import SearchAggregator
from facechain.search.base import ProbeContext, RawCandidate


class ListProvider:
    """Yields a fixed set of local fixture images as file:// candidates."""

    name = "listprovider"

    def __init__(self, files: list[tuple[Path, str]]) -> None:
        self._files = files

    @classmethod
    def available(cls, settings: Settings) -> bool:
        return True

    def search(self, probe: ProbeContext) -> Iterable[RawCandidate]:
        return [
            RawCandidate(
                provider=self.name,
                post_url=post,
                image_url=path.resolve().as_uri(),
                title=path.stem,
            )
            for path, post in self._files
        ]


def _ctx(probe_img: Path, engine, settings: Settings, providers) -> tuple[ProbeContext, SafeFetcher]:
    img = load_image_path(probe_img)
    _, emb, _ = encode_probe(img, engine=engine)
    fetcher = SafeFetcher(contact="pytest")
    ctx = ProbeContext(
        image_bytes=img.raw_bytes,
        rgb=img.rgb,
        embedding=emb,
        settings=settings,
        fetcher=fetcher,
        face_engine=engine,
    )
    return ctx, fetcher


@requires_opencv
def test_ranks_by_embedding_and_picks_the_true_match(opencv_engine, settings: Settings, tmp_path, reencoder):
    probe = FIXTURES / "corpus_kennedy.jpg"
    variant = reencoder(probe, tmp_path / "jfk_repost.jpg", scale=0.8, quality=68, rotate=2)
    provider = ListProvider(
        [
            (FIXTURES / "corpus_eisenhower.jpg", "https://example.com/ike"),
            (variant, "https://social.example/post/jfk-123"),
            (FIXTURES / "corpus_reagan.jpg", "https://example.com/rr"),
        ]
    )
    ctx, fetcher = _ctx(probe, opencv_engine, settings, [provider])
    try:
        result = SearchAggregator([provider]).run(ctx)
    finally:
        fetcher.close()

    assert result.match.best.post_url == "https://social.example/post/jfk-123"
    assert result.match.best.similarity_ppm / 1_000_000 >= settings.match_threshold
    # ranked, descending, and the impostors are below threshold
    sims = [m.similarity_ppm for m in result.match.ranked]
    assert sims == sorted(sims, reverse=True)
    assert result.match.ranked[1].similarity_ppm < result.match.threshold_ppm


@requires_opencv
def test_no_match_raises_with_ranked_nearmisses(opencv_engine, settings: Settings):
    probe = FIXTURES / "corpus_kennedy.jpg"
    provider = ListProvider(
        [
            (FIXTURES / "corpus_eisenhower.jpg", "https://example.com/ike"),
            (FIXTURES / "corpus_reagan.jpg", "https://example.com/rr"),
        ]
    )
    ctx, fetcher = _ctx(probe, opencv_engine, settings, [provider])
    try:
        with pytest.raises(NoMatchFoundError) as ei:
            SearchAggregator([provider]).run(ctx)
    finally:
        fetcher.close()
    assert isinstance(ei.value.detail, dict)
    assert ei.value.detail["summary"]["candidates_scored"] == 2


@requires_opencv
def test_unfetchable_candidate_is_skipped_not_fatal(opencv_engine, settings: Settings, tmp_path, reencoder):
    probe = FIXTURES / "corpus_kennedy.jpg"
    good = reencoder(probe, tmp_path / "jfk.jpg", scale=0.85, quality=72)
    ListProvider([(good, "https://social.example/ok")])
    bad = RawCandidate(provider="x", post_url="p", image_url="file:///no/such/file.jpg")

    class Mixed(ListProvider):
        name = "mixed"

        def search(self, probe):
            return [bad, *super().search(probe)]

    mixed = Mixed([(good, "https://social.example/ok")])
    ctx, fetcher = _ctx(probe, opencv_engine, settings, [mixed])
    try:
        result = SearchAggregator([mixed]).run(ctx)
    finally:
        fetcher.close()
    assert result.match.best.post_url == "https://social.example/ok"
    assert any("skipped" in m.note for m in result.match.ranked)


@requires_opencv
def test_dedupes_identical_candidates_across_providers(opencv_engine, settings: Settings, tmp_path, reencoder):
    probe = FIXTURES / "corpus_kennedy.jpg"
    variant = reencoder(probe, tmp_path / "v.jpg", scale=0.8, quality=70)
    p1 = ListProvider([(variant, "https://a/1")])
    p2 = ListProvider([(variant, "https://a/1")])
    ctx, fetcher = _ctx(probe, opencv_engine, settings, [p1, p2])
    try:
        result = SearchAggregator([p1, p2]).run(ctx)
    finally:
        fetcher.close()
    assert result.summary.candidates_seen == 1
