# Open-source improvements — research & plan

Notes from evaluating OSS projects that could harden Facechain’s
**candid photo → live reverse-image → ArcFace rank → identity** path.

This is a planning doc, not an implementation checklist that has already shipped.
What is already in-tree (SerpAPI Lens auto-host, multiris, hint provider, mild
enhance, identity consensus, per-run `run.log`) is assumed as the baseline.

---

## Goals (what “better” means here)

1. **More / better candidates** when Google Lens is thin or noisy.
2. **Clearer faces** from soft phone candids (without inventing a different person).
3. **Stronger local demos** when the web never indexed the face.
4. **Second-opinion verify** in the awkward 0.35–0.45 similarity band.
5. **Text → identity** when the probe itself contains a readable name.

---

## Project catalogue

| # | Project | License (check upstream) | Role for Facechain |
|---|---------|--------------------------|--------------------|
| 1 | [kitUIN/PicImageSearch](https://github.com/kitUIN/PicImageSearch) | See repo | Already used (`multiris`). Expand engines. |
| 2 | [Amr-9/Face-Search](https://github.com/Amr-9/Face-Search) | See repo | Local InsightFace + FAISS index pattern. |
| 3 | [deepinsight/insightface](https://github.com/deepinsight/insightface) | MIT | Already used (`buffalo_l`). Optional stronger packs (e.g. antelopev2). |
| 4 | [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) | NTU S-Lab (see LICENSE) | Blind face restore for blurry candids. |
| 5 | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) | Apache-2.0 | Faster face restore alternative to CodeFormer. |
| 6 | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | BSD-3-Clause | General upscale before detect; pair with face restore. |
| 7 | [serengil/deepface](https://github.com/serengil/deepface) | MIT | Multi-model verify (ArcFace / FaceNet) as second opinion. |
| 8 | [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR) | Apache-2.0 | OCR names on screenshots → auto `--hint`. |
| 9 | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Apache-2.0 | Stronger OCR alternative (heavier). |
| 10 | [pratik-shivnani/LocalLens](https://github.com/pratik-shivnani/LocalLens) | MIT | Inspiration: CLIP + InsightFace personal library (not live web). |
| 11 | [sarev/photonarium](https://github.com/sarev/photonarium) | Apache-2.0 | Inspiration: local face clustering / gallery UX. |
| 12 | [seamile/ImgSearch](https://github.com/seamile/ImgSearch) | See repo | CLIP reverse search over *your* index only. |

**Not a replacement for SerpAPI Google Lens:** no mature OSS project reproduces Google’s live social index. Keep Lens (or equivalent) for open-web identity.

---

## Plan — what to take from which repo

### Phase A — Candidate generation (web)

**From PicImageSearch**

| Action | Detail | Priority |
|--------|--------|----------|
| Enable **Copyseeker** | Extra visual match engine alongside Yandex. | P0 |
| Enable **Baidu** images | Often indexes Asian social / news copies Lens misses. | P1 |
| Soft-fail / fix **Bing** + **TinEye** | Today they often fail (`BCID`, parse errors); either harden or drop from default fan-out. | P1 |
| Keep anime engines off | SauceNAO / IQDB / ascii2d stay opt-in only. | — |

**Deliverable:** richer `multiris` provider; same `RawCandidate` → ArcFace path.

---

### Phase B — Candid clarity (probe preprocess)

**From CodeFormer (preferred) + Real-ESRGAN**

| Action | Detail | Priority |
|--------|--------|----------|
| Optional `--enhance codeformer` | Run face restore with **high fidelity** (e.g. weight ≈ 0.7–0.9) so identity is preserved. | P0 |
| Optional Real-ESRGAN ×2 | If min side &lt; ~480 after restore, light upscale. | P1 |
| GFPGAN as lighter alt | If CodeFormer deps are too heavy for demos. | P2 |
| Gate on quality | Only restore when sharpness / det score is weak (keep current mild unsharp as default). | P0 |

**Caution:** Generative restore can *change* identity if fidelity is too low. Always re-encode with InsightFace and prefer original embedding when restored sim-to-original is low.

**Deliverable:** `facechain.enhance` backends: `mild` (current) | `codeformer` | `gfpgan`.

---

### Phase C — Local face index (offline / known people)

**From Amr-9/Face-Search (architecture, not necessarily vendoring their UI)**

| Action | Detail | Priority |
|--------|--------|----------|
| FAISS (or hnswlib) store | Persist L2-normalised InsightFace vectors + metadata. | P0 |
| `facechain index` CLI | Walk a folder → detect all faces → write index. | P0 |
| Provider `faiss` / upgrade `local` | Query probe embedding against index; emit `RawCandidate` with `file://` or corpus URLs. | P0 |
| Multi-face support | Index every face in an image (their design does this). | P1 |

**Deliverable:** reliable Demo A / “people I know” path without SerpAPI quota.

---

### Phase D — Near-threshold verification

**From DeepFace and/or InsightFace second pack**

| Action | Detail | Priority |
|--------|--------|----------|
| Dual-engine check | When best sim ∈ [0.35, 0.50], also score with FaceNet or a second ArcFace pack. | P1 |
| Agreement rule | MATCH only if both engines clear soft bars, or surface `WHO (near)` with both scores. | P1 |
| Optional `antelopev2` | Stronger InsightFace pack (from Face-Search notes); larger download. | P2 |

**Deliverable:** fewer false “closest lookalike” presentations without lowering the main threshold.

---

### Phase E — OCR → hint

**From EasyOCR (default) or PaddleOCR**

| Action | Detail | Priority |
|--------|--------|----------|
| Probe OCR pass | Extract likely person names / @handles from the image. | P1 |
| Auto-enable `hint` | If OCR finds a name and user did not pass `--hint`, run hint provider. | P1 |
| Keep manual `--hint` | Still the reliable path for LinkedIn URLs. | — |

**Deliverable:** screenshots with name overlays (Thrishna-style) feed the candidate pool automatically.

---

### Phase F — Inspiration only (do not merge wholesale)

| Repo | Steal the idea | Do not |
|------|----------------|--------|
| LocalLens | CLIP text search over *your* corpus | Replace live Lens |
| Photonarium | Face clustering UX / “people” albums | Pull their full desktop app |
| ImgSearch | TinyCLIP + HNSW for local similarity | Expect open-web IDs |

---

## Suggested implementation order

```
Week 1  A0 Copyseeker + Baidu in multiris
        B0 CodeFormer optional enhance (gated)
Week 2  C0 FAISS local index + CLI
Week 3  D1 Dual-engine near-threshold verify
        E1 EasyOCR → auto hint
Week 4  Polish: Bing/TinEye, antelopev2, docs/demo script
```

Extras as optional deps so the core install stays light:

```text
pip install -e ".[ris]"          # PicImageSearch (already)
pip install -e ".[restore]"      # CodeFormer / GFPGAN / Real-ESRGAN
pip install -e ".[faiss]"        # faiss-cpu + local index
pip install -e ".[ocr]"          # EasyOCR
pip install -e ".[verify]"       # deepface (optional second opinion)
```

---

## Explicit non-goals

- Scraping login-walled LinkedIn HTML as a primary path (blocked / ToS-fragile).
- Replacing SerpAPI Lens with a fully OSS “search the whole internet” engine.
- Lowering ArcFace threshold globally to “fix” NO MATCH (increases false IDs).

---

## Decision rule for each PR

1. Does it widen **candidates**, improve **probe quality**, or improve **decision confidence**?
2. Can it soft-fail if the optional extra is not installed?
3. Does ArcFace (or dual-engine) still decide the match — never the helper alone?

If all three are yes, it belongs in Facechain.
