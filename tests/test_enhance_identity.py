"""Tests for candid enhance + identity consensus."""

from __future__ import annotations

import numpy as np

from facechain.enhance import enhance_rgb_for_faces
from facechain.models import Candidate
from facechain.search.identity import (
    cluster_score,
    consensus_from_candidates,
    extract_identity_labels,
    normalize_identity,
)


def test_enhance_preserves_dtype_and_can_upscale() -> None:
    rgb = np.zeros((120, 100, 3), dtype=np.uint8)
    rgb[40:80, 30:70] = 90
    out = enhance_rgb_for_faces(rgb)
    assert out.dtype == np.uint8
    assert out.ndim == 3
    assert min(out.shape[0], out.shape[1]) >= 480


def test_extract_identity_from_titles_and_urls() -> None:
    labels = extract_identity_labels(
        "Rdx bewafa Randip (@rdx_randip_chandravanshi_erz)",
        "https://www.instagram.com/reel/DVLA3e_E326/",
        "Elvish Yadav Bhai Ki Entry #elvishyadav",
    )
    norms = {normalize_identity(x) for x in labels}
    assert "rdx randip chandravanshi erz" in norms or any("randip" in n for n in norms)
    assert any("elvish" in n for n in norms)


def test_consensus_prefers_repeated_lens_name() -> None:
    ranked = [
        Candidate(
            provider="multiris/yandex",
            post_url="https://instastatistics.com/rdx_randip_chandravanshi_erz",
            image_url="https://cdn/a.jpg",
            title="Rdx Randip (@rdx_randip_chandravanshi_erz)",
            similarity_ppm=875_000,
            rank=0,
        ),
        Candidate(
            provider="serpapi",
            post_url="https://facebook.com/elvish/1",
            image_url="https://cdn/b.jpg",
            title="Systuuum Elvish Yadav",
            similarity_ppm=779_000,
            rank=1,
        ),
        Candidate(
            provider="serpapi",
            post_url="https://instagram.com/reel/1",
            image_url="https://cdn/c.jpg",
            title="Elvish Yadav Bhai Ki Entry",
            similarity_ppm=768_000,
            rank=2,
        ),
        Candidate(
            provider="serpapi",
            post_url="https://x.com/1",
            image_url="https://cdn/d.jpg",
            title="3 years of Elvish Yadav Foundation",
            similarity_ppm=763_000,
            rank=3,
        ),
        Candidate(
            provider="serpapi",
            post_url="https://facebook.com/elvish/2",
            image_url="https://cdn/e.jpg",
            title="elvishyadav #bhaicharaontop",
            similarity_ppm=702_000,
            rank=4,
        ),
    ]
    cons = consensus_from_candidates(ranked, threshold_ppm=450_000, min_support=2)
    assert cons is not None
    assert "elvish" in cons.normalized

    # Cluster score should lift a Lens Elvish hit over the lone Yandex Randip near-tie.
    yandex = ranked[0]
    lens = ranked[1]
    assert cluster_score(lens, cons) > cluster_score(yandex, cons)
    assert cons.support >= 3
