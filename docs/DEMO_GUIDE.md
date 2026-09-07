# facechain-verify — pipeline, testing & video guide

Everything below runs from the repo root. Use the project venv:

```powershell
# Windows PowerShell (Antigravity terminal)
.venv\Scripts\activate           # or prefix every command with .venv\Scripts\python.exe -m
python -m facechain --help
```

If `facechain` isn't importable: `python -m pip install -e ".[opencv,dev,sign]"`
(add `describe` for the caption model, `evm` for a real testnet).

---

## 1. How the pipeline runs

One command — `facechain run <image>` — executes four stages and writes a
self-describing `runs/<timestamp-id>/` folder. Nothing is hidden; every stage's
input and output is a file on disk.

```
your photo
   │
   ▼
┌─ STAGE 1 · detect + encode ─────────────────────────────────────────────┐
│ imaging.py   decode bytes, strip EXIF, RGB array, SHA-256 + pHash/dHash │
│ face/…       YuNet finds the face + 5 landmarks → align to 112×112 →    │
│              SFace 128-D embedding (L2-normalised)                       │
│ writes:      probe.jpg, probe_fingerprint.json, face.json,              │
│              face_crop.png, embedding.npy                               │
└────────────────────────────────────────────────────────────────────────┘
   │  embedding (128 floats)
   ▼
┌─ STAGE 2 · live / offline search ──────────────────────────────────────┐
│ providers GATHER candidate images (they never decide the match):       │
│   serpapi    Google Lens reverse-image  (needs key; social media)      │
│   multiris   Yandex/Bing/TinEye/Lens    (needs .[ris])                  │
│   wikimedia  live Commons full-text      (keyless, needs --hint)       │
│   faceindex  offline face-vector index over your corpus (deterministic)│
│ aggregator.py fetches each candidate, runs the SAME engine on it,      │
│   ranks by cosine similarity, keeps the top one iff ≥ calibrated       │
│   threshold (sface 0.40). Full ranked list → candidates.json           │
└────────────────────────────────────────────────────────────────────────┘
   │  best match (image + source URL + score)
   ▼
┌─ STAGE 3 · notarise on a blockchain ──────────────────────────────────┐
│ build a canonical EvidenceBundle (sorted keys, ints only) of:         │
│   probe hashes + face box + embedding hash + matched page/image +     │
│   similarity + provider + pipeline version                            │
│ SHA-256 of those exact bytes = record_hash                            │
│ anchor:  local  → hash-linked Merkle ledger (genesis, blocks chained  │
│                   by prev-hash, Merkle root, inclusion proof)         │
│          evm    → Sepolia / Polygon Amoy tx or EvidenceRegistry.sol   │
│ writes:  evidence.json, receipt.json                                  │
│ (--sign) Ed25519 signature over record_hash → signature.json          │
└──────────────────────────────────────────────────────────────────────┘
   │  record_hash + where it's anchored
   ▼
┌─ STAGE 4 · independent re-verification ───────────────────────────────┐
│ verify.py — a SEPARATE code path, shares no memory with the run:      │
│   • recompute the canonical hash from evidence.json                   │
│   • re-hash probe.jpg and embedding.npy                               │
│   • re-encode the stored candidate and re-check cosine ≥ threshold    │
│   • perceptual-hash the candidate vs the recorded fingerprint         │
│   • re-hash the whole chain, confirm the record is a Merkle leaf,     │
│     recompute the inclusion proof to the on-chain root               │
│   • check the Ed25519 signature                                       │
│ writes:  verification.json  → 10 checks (11 with --sign)              │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
report.html   — self-contained (open in a browser), auto-generated
```

### The `runs/<id>/` folder after a signed run

| file | what it is |
|---|---|
| `probe.jpg` | exact input bytes |
| `probe_fingerprint.json` | SHA-256 + 64-bit pHash + dHash |
| `face.json` | engine, bbox, quality, embedding hash |
| `face_crop.png` | the aligned crop that was actually encoded |
| `embedding.npy` | raw 128-D probe vector |
| `candidates.json` | every candidate, fetched + scored + ranked |
| `candidates/00_*.jpg` | the downloaded matched image |
| `evidence.json` | the notarised bundle (contains `record_hash`) |
| `receipt.json` | block index/hash, Merkle root, inclusion proof |
| `signature.json` | Ed25519 signature + public key (with `--sign`) |
| `verification.json` | the 10–11 independent checks |
| `report.html` | human-readable summary of all of the above |
| `run.log` / `telemetry.jsonl` | plain-text + JSON timeline |

---

## 2. Testing it with real pictures

### 2a. One-time setup (≈1 min, needs internet once)

```powershell
python -m facechain fetch-models            # YuNet + SFace (~37 MB, checksum-pinned)
python -m facechain doctor                  # confirm engines/providers are green
```

### 2b. Build a face corpus to match against

Pick **one** of these:

```powershell
# A. real people from Wikidata (one canonical portrait each) — best for a demo
python -m facechain fetch-corpus `
  --name "Barack Obama" --name "Angela Merkel" --name "Marie Curie" `
  --name "Cristiano Ronaldo" --name "Serena Williams" --name "Albert Einstein"
python -m facechain build-index

# B. bundled offline fixtures (4 US presidents) — zero network
python -m facechain fetch-corpus --seed-demo
python -m facechain build-index

# C. your own folder of labelled faces
#    name files  <identity>__<anything>.jpg   e.g.  alice__1.jpg  alice__2.jpg  bob__1.jpg
copy C:\path\to\your\faces\*.jpg data\corpus\
python -m facechain build-index
```

### 2c. Run the pipeline on a probe photo

```powershell
# probe = a DIFFERENT photo of someone who is in the corpus
python -m facechain run "C:\path\to\obama_candid.jpg" --providers faceindex --anchor local

# add the signature + the image caption
python -m facechain run "C:\path\to\obama_candid.jpg" --providers faceindex --sign --describe
```

Then:

```powershell
$RUN = (Get-ChildItem runs | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
python -m facechain verify $RUN --no-network      # independent re-check
python -m facechain report $RUN                   # (re)build report.html
start "$RUN\report.html"                          # open it
python -m facechain chain show                    # the ledger
```

### 2d. Live web search (real reverse-image, needs a key OR a hint)

```powershell
# keyless: steer the live Commons search with a name
python -m facechain run "C:\path\to\some_photo.jpg" --providers wikimedia --hint "Barack Obama"

# with a free SerpAPI key (real Instagram/X/LinkedIn hits via Google Lens):
setx FACECHAIN_SERPAPI_KEY "your_key"        # new terminal after this
python -m facechain run "C:\path\to\photo.jpg" --providers serpapi --allow-public-host
```

### 2e. What to try (and what each proves)

| Test | Expectation |
|---|---|
| Probe = a **different photo** of a corpus person | MATCH, cosine ~0.6–0.98, `report.html` shows the source page |
| Probe = someone **not** in the corpus / web | NO MATCH + "closest was X at 0.3x" (it doesn't guess) |
| Same probe run **twice** (offline providers) | identical `record_hash` — deterministic |
| A **filtered / cropped / rotated / greyscale** version of a known photo | still MATCH (see `facechain bench` robustness table) |
| Edit one number in `evidence.json`, then `verify` | `evidence.self_consistent` **FAIL**, OVERALL FAILED |
| `facechain chain tamper` then `facechain chain verify` | `CHAIN INTEGRITY: FAILED — Merkle root mismatch` |
| Two similarly-sized faces in the probe | run notes it as ambiguous, uses the largest |
| `facechain describe <any photo>` | caption + size/brightness/sharpness/colours/face-count |

### 2f. Quick quality gates (for your own sanity / the repo)

```powershell
python -m pytest -q -m "not slow"     # 152 fast tests
python -m ruff check .                # style
python -m mypy                        # strict types
python -m facechain bench             # ROC/AUC/EER + robustness → bench/results.md
```

---

## 3. What to record for the video (~3–4 min)

Keep it to **one continuous terminal + one browser tab**. Suggested beats:

**0:00 — Framing (10 s).**
"One command takes a face photo, finds it on a real web source, writes a
tamper-proof fingerprint of that finding onto a blockchain, then a separate
program proves the whole chain is intact."

**0:10 — Setup, fast (20 s).**
`facechain fetch-models` then `facechain doctor` — show the green rows
(engine: sface = DEFAULT, faceindex, local Merkle chain). One line: "no API keys,
runs on CPU."

**0:30 — Build the corpus live (25 s).**
`facechain fetch-corpus --name "Barack Obama" --name "Angela Merkel" --name "Marie Curie" --name "Cristiano Ronaldo"`
then `facechain build-index`. Say: "these portraits are pulled live from
Wikidata — the pipeline searches over them, it isn't given the answer."

**0:55 — The run (45 s).** The money shot:
`facechain run samples\probe_obama.jpg --providers faceindex --anchor local --sign --describe`
Let the log scroll. Call out the four lines as they appear:
`face engine: yunet-sface` … `BEST MATCH sim=0.81 https://www.wikidata.org/wiki/Q76`
… `anchored on local-merkle-chain block #1` … `OVERALL: VERIFIED (11/11)`.
Say: "the probe is a *different* photo of Obama than the one in the corpus —
0.81 cosine, well past the 0.40 threshold; impostors sat below 0.15."

**1:40 — The report (35 s).**
`start runs\<id>\report.html`. Scroll it: verdict + probability band, probe vs
matched image side by side, the `record_hash`, the Merkle block + inclusion
proof, the 11 green verification checks, the VLM caption
("president obama smiles for the camera"), the Ed25519 line.

**2:15 — Tamper-evidence (35 s).** The proof it's real:
- `facechain chain tamper` → "flipped record 0"
- `facechain chain verify` → `CHAIN INTEGRITY: FAILED — block 1 Merkle root mismatch` (red)
- (optional) open `evidence.json`, change `similarity_ppm`, save,
  `facechain verify runs\<id> --no-network` → `evidence.self_consistent FAIL`.

**2:50 — Determinism + benchmark (25 s).**
Re-run the same `facechain run …` and point at the **identical `record_hash`** —
"content-addressed: same finding, same hash, on any machine."
Flash `bench\results.md` / `roc.svg`: "sface separates identities by 0.84;
the classical descriptor by 0.19 — measured, not asserted."

**3:15 — Close (15 s).**
"Real face recognition, a genuine web source, a real Merkle ledger, and an
independent verifier that re-derives every hash — end to end, offline,
reproducible."

### Recording tips
- Pre-run `fetch-models` and the BLIP download **before** you hit record (first
  caption pulls ~1 GB). Everything else is fast.
- `del /s /q runs chaindata` before the take for a clean `runs/` folder.
- If the live Wikidata fetch is slow on the day, use `fetch-corpus --seed-demo`
  (offline, deterministic) and probe with `samples\probe_obama.jpg` — it matches
  the bundled `corpus_obama_reencode.jpg` at 0.98.
- Show `git log --oneline` once if you want to evidence the engineering: Phase 1–4.
