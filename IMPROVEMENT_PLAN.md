# facechain-verify — Improvement Plan (Task 3)

> Feed this file to Cursor as working context. It describes the current pipeline,
> where it is weak, and a front-by-front upgrade plan to make the submission the
> clear winner of **HH Goa 2026 Shortlisting Task 3 — Face Identification &
> Blockchain Verification**.

---

## 0. Strategic framing

- The **blockchain / verification half is already strong** (canonical hashing,
  Merkle-chained ledger, independent re-verification, SSRF-hardened fetch).
  Polish it, don't rebuild it.
- **Points are won on identification accuracy + provable benchmarks.** Judges will
  likely test with: a candid non-frontal photo, a reposted/cropped/filtered
  image, and a "find this person online" case. Today only the reposted case works.
- "Undoubted winner" = it visibly works on the judges' own test photos **and**
  there is a benchmark section with numbers that removes all doubt.

---

## 1. How the pipeline works today (current state)

Entry point: `src/facechain/pipeline.py::run_pipeline`.

| Stage | File(s) | What it does |
|---|---|---|
| Load / imaging | `imaging.py` | Pillow decode, EXIF transpose, RGB `uint8` array, SHA-256 + DCT pHash + dHash. No OpenCV. Perceptual hashes are provenance only — **not** used to rank candidates. |
| Face detect | `face/factory.py`, `face/opencv_backend.py`, `face/numpy_vj_backend.py`, `face/cascade.py` | Engine order `insightface → opencv → numpy`. Default in practice = **`opencv`** = `cv2.CascadeClassifier` Haar `haarcascade_frontalface_default` (`scaleFactor=1.1, minNeighbors=6, minSize=(40,40)`). OpenCV is used for **detection only**. |
| Face encode | `face/descriptor.py::lbph_embedding` | Crop bbox + 18% margin → 128×128 grayscale (Pillow LANCZOS) → histogram equalize → 8-neighbour LBP per-cell 64-bin histograms over 8×8 grid + coarse intensity map + HOG-lite → concat → L2-norm → **4736-d float32**. Pure NumPy. **No landmark alignment.** |
| Encode selection | `face/__init__.py::encode_probe`, `_select_face` | Largest face wins; flags ambiguity when runner-up ≥ 80% area. |
| Search — gather | `search/wikimedia_provider.py`, `search/local_index_provider.py`, `search/serpapi_provider.py` | `wikimedia` (default, keyless) = **text** query to MediaWiki API. `local` (default) = every file in `data/corpus/`. `serpapi` (optional, needs key) = the only true reverse-image search (Google Lens / Yandex). |
| Search — decide | `search/aggregator.py::SearchAggregator` | Fetch each candidate (SSRF policy) → run **same** face engine → cosine similarity vs probe → sort → keep top only if ≥ `match_threshold`. `_score` is **serial**. |
| Config | `config.py::Settings` | `match_threshold` default **0.86**, `ambiguous_margin` 0.04, `search_providers=("wikimedia","local")`, `face_engine="auto"`. |
| Evidence bundle | `pipeline.py`, `models.py` | Canonical JSON → SHA-256 `record_hash`. |
| Anchor | `anchor/` (`local_chain.py`, `evm_backend.py`, `merkle.py`, `factory.py`) | `local` = hash-linked Merkle ledger with inclusion proofs. `evm` = testnet via `contracts/EvidenceRegistry.sol` or calldata. |
| Verify | `verify.py::verify_run` | Re-derives every hash from raw files, re-reads chain value. Separate code path. |

### Why "certain images" fail

1. **Detection failure (Haar).** Only near-frontal, upright, evenly-lit faces
   ≥40px; `minNeighbors=6` is strict. Profile / tilt / sunglasses / shadow /
   small / low-res → `NoFaceFoundError`. NumPy fallback is worse.
2. **Matching failure (LBPH+HOG).** Descriptor is texture, **not
   identity-invariant**. No landmark alignment. Its own docstring: same-identity
   re-encode ≈ cos **0.90**, impostors **0.60–0.80**. With threshold **0.86** the
   usable margin is a sliver — a different photo of the same person routinely
   lands 0.80–0.85 → reported as "no match".
3. **Search is keyword-based** without a SerpApi key: if the person isn't on
   Wikimedia Commons or the `--hint` keywords don't surface them, there is
   nothing to rank.

Net: today's system is effectively a **near-duplicate / same-photo matcher**, not
a face recogniser.

---

## 2. Front-by-front upgrade plan

Each front lists concrete, actionable tasks with code locations. Preserve the
existing pluggable-engine / pluggable-provider architecture and the invariant
that **the face embedding alone selects the match**.

### Front 1 — Face detection

- Replace Haar as the default with **YuNet** (`cv2.FaceDetectorYN`,
  `face_detection_yunet_2023mar.onnx`, ~340 KB). Stays inside OpenCV, fast on
  CPU, **returns 5 landmarks** (needed for alignment).
- `insightface` path → **RetinaFace** (already in the `buffalo_l` pack).
- Keep pure-NumPy Viola–Jones only as a "bare install" last resort, clearly
  labeled degraded.
- In `face/__init__.py::_select_face`: add a **min detection-score gate** and a
  landmark-based **pose estimate**, so a rejected probe reports *why*.

### Front 2 — Alignment (biggest cheap win)

- There is no alignment today. Add a **5-point similarity transform** to the
  canonical ArcFace template (112×112, standard reference landmarks) via
  `cv2.estimateAffinePartial2D`.
- Implement in `face/descriptor.py::align_crop` as the path used when landmarks
  are available; fall back to the current bbox+margin crop otherwise.
- This lifts **every** downstream encoder, including the classical one.

### Front 3 — Face encoding / embedding

- **Primary: ArcFace** (`insightface` `buffalo_l`, 512-d). Make it the **real
  default**, not an optional extra. Auto-download the model pack with a **pinned
  SHA-256** (or git-lfs, or a `facechain fetch-models` command). **Fail loudly**
  if missing — do not silently drop to LBPH.
- **Middle tier, no framework: OpenCV SFace** (`cv2.FaceRecognizerSF`,
  `face_recognition_sface_2021dec.onnx`, ~37 MB, 128-d). Real recogniser, CPU-only.
- **LBPH: keep only as numpy-only emergency mode**, with a loud `degraded=true`
  flag in the report and evidence bundle.
- **Test-time augmentation:** embed the crop and its horizontal flip, average,
  re-normalize.
- Record `engine`, `engine_version`, `degraded` in `FaceRecord` (`models.py`) so
  the evidence bundle is honest about which encoder produced the match.

### Front 4 — Matching, scoring, calibration

- Replace the single hard `0.86` cutoff with **per-engine calibrated thresholds**:
  fit a logistic (Platt) or isotonic mapping cosine → P(same identity) on a
  labeled pairs set (LFW pairs or a bundled ~100-pair mini-set). Store the
  calibration curve as a versioned asset.
- **Decision bands** instead of pass/fail: `match` / `likely` / `no-match`, each
  with calibrated probability and the operating point's FAR/FRR.
- **Separate two questions** the report currently conflates: "same image"
  (perceptual-hash near-dup — pHash/dHash already computed in `imaging.py`) vs
  "same identity" (embedding). Report both.
- Optional **s-norm** against a small cohort of background faces for sharper
  separation.
- **Probe quality gate** (blur via Laplacian variance — `descriptor.py::
  sharpness_quality` exists — plus face size and pose) with an actionable
  message: "probe too blurry / too small / too oblique to identify reliably".

### Front 5 — Search (make it genuinely reverse-image)

- **SerpApi as first-class:** Google Lens **and** Yandex **and** Bing Visual
  Search. One-liner key setup in README + demo. This is the real "web / social"
  search.
- **Local vector index:** FAISS / hnswlib over ArcFace embeddings of a fetched
  corpus, so retrieval is **by face vector**, not keywords. Corpus sources:
  Wikimedia Commons + **Wikidata P18** + optionally a news-image set. Add
  `facechain build-index`.
- **Wikidata/Wikipedia resolver:** `--hint "name"` → Wikidata entity → P18
  portrait → high-precision seed candidate.
- **Parallelize** candidate fetch+encode in `aggregator.py::_score` with a
  `ThreadPoolExecutor` — currently serial and it dominates wall-clock time.
- **De-dup across providers by perceptual hash**, not just URL string
  (`search/base.py::RawCandidate.key`).
- Keep and prominently document the invariant: the embedding alone picks the
  winner; providers only widen the candidate pool.

### Front 6 — Blockchain anchoring (deepen an already-good part)

- **Real testnet in the demo:** deploy `contracts/EvidenceRegistry.sol` to Base
  Sepolia or Polygon Amoy, verify the contract, show the explorer link in the
  recording.
- **Batch Merkle anchoring:** N runs → one on-chain tx to a Merkle root; each run
  keeps its inclusion proof.
- **OpenTimestamps** as a second, free anchor (Bitcoin calendar).
- **EIP-712 signature** over `record_hash` from the operator key, stored in the
  bundle — binds *who* produced the evidence.
- Put `contractAddress`, `chainId`, `txHash`, `blockNumber`, `logIndex` in
  `receipt.json`; give `verify` a one-command **on-chain re-read**.
- RPC retry/backoff, gas + confirmation reporting.

### Front 7 — Verification / tamper-evidence

- Add to `verify.py::verify_run`: (1) **live re-fetch** of the matched image URL
  and re-score, flag if the web changed; (2) EIP-712 signature check; (3) Merkle
  proof recompute up to the on-chain root; (4) distinct **exit codes per failure
  class**; (5) a signed verification report.
- Ship a **standalone third-party verifier**: a single script with **no
  `facechain` import** that validates a bundle from just the JSON + a public RPC
  URL.

### Front 8 — Evaluation harness (the differentiator)

- **Bundled benchmark set:** LFW verification pairs, or a curated ~100-identity
  set with reposts, crops, filters, JPEG-q30, 0.5× resize, grayscale, ±30°
  rotation, brightness shifts, partial occlusion.
- **Metrics:** ROC/AUC, TAR@FAR=1e-3, accuracy at the chosen operating point, and
  a **per-engine comparison table** (Haar+LBPH vs SFace vs ArcFace).
- **Search recall:** probes with known web presence → top-1 / top-5 hit rate per
  provider.
- **End-to-end timing breakdown.**
- `make bench` produces the table + ROC plot. README leads with a **Results**
  section containing them.

### Front 9 — UX, report, demo

- **`runs/<id>/report.html`** — one self-contained artifact: probe crop, matched
  image, side-by-side, calibrated-score gauge, ranked candidates with thumbnails,
  chain receipt with explorer links, verification badge.
- **`facechain doctor`** — prints active engine/providers, model files present,
  network reachability, which keys are set.
- Clear failure messages with remedies at every stage that can fail.
- **3-minute recording script:** offline deterministic run → live web + SerpApi
  run → tamper a file → `verify` fails → restore → `verify` passes → show the
  on-chain tx on the explorer.

### Front 10 — Packaging, docs, reproducibility

- **Dockerfile** + a `docker run` one-liner that works with zero host setup
  (models baked in or fetched on first run).
- Pin every dependency; `pip install facechain-verify[all]`.
- Model assets: auto-fetch with SHA-256 pinning; never silently degrade.
- Trim README to a sharp narrative that **leads with results**; deep detail to
  `docs/`.
- Architecture diagram of the real mechanism, legible in light and dark.

### Front 11 — Security / robustness (hold the lead)

- Confirm the SSRF fetcher (`netfetch.py`) re-checks redirects against private-IP
  ranges on **every hop**; add a `HEAD` content-length pre-check, per-host
  timeout, scheme allowlist.
- Polite User-Agent + rate limiting for **every** API, not just Wikimedia.

---

## 3. Suggested sequence (highest impact first)

1. **Detection + alignment + ArcFace as the enforced default** (Fronts 1–3) —
   the visible accuracy jump.
2. **Evaluation harness + README Results section** (Front 8) — the proof.
3. **HTML report** (Front 9) — demo impact.
4. **Real testnet anchor + batch Merkle + OTS + EIP-712** (Front 6) — chain depth.
5. **SerpApi + vector index search** (Front 5) — the genuine-search story.
6. **Verification hardening + standalone verifier** (Front 7).
7. **Docker + model auto-fetch + docs polish + recording** (Fronts 10, 9).

Items 1–3 move it from "brittle near-duplicate matcher" to "recognises people,
with numbers to back it". Items 4–7 make the win uncontestable.

---

## 4. Constraints / invariants to preserve

- Keep the pluggable engine / provider / anchor factories.
- Keep: **the face embedding alone selects the match**; providers only gather.
- Keep the canonical-JSON → `record_hash` → anchor → independent re-verify spine.
- Every new model asset must be integrity-pinned (SHA-256) and must fail loudly
  if absent — no silent downgrade to a weaker engine.
- Offline deterministic demo must keep working (tests + offline screen recording
  depend on it).
