# facechain-verify

**A pipeline that takes a photo of a face, finds a matching image on the live web,
writes a tamper‑proof fingerprint of that finding onto a blockchain, and then
independently proves the whole chain of evidence is intact.**

Built for **HH Goa 2026 Shortlisting Task 3 — Face Identification & Blockchain
Verification**.

```
   your photo ──▶  1. detect + encode the face
                        │
                        ▼
                   2. LIVE search the web for a matching image
                      (providers fetch candidates; the face embedding alone
                       decides the winner, by cosine similarity ≥ threshold)
                        │
                        ▼
                   3. build a canonical "evidence bundle" and write its
                      SHA‑256 onto a blockchain (local Merkle chain by
                      default, or a real Ethereum testnet)
                        │
                        ▼
                   4. VERIFY: a separate command re‑derives every hash from
                      the raw files and re‑reads the value back off the chain
```

If you just want to see it work, jump to **[60‑second quick start](#60second-quick-start)**.

---

## Table of contents

1. [What this actually does](#1-what-this-actually-does)
2. [What you need installed first](#2-what-you-need-installed-first)
3. [Get the code](#3-get-the-code)
4. [Set it up (copy‑paste blocks)](#4-set-it-up-copypaste-blocks)
5. [Check the install worked](#5-check-the-install-worked)
6. [60‑second quick start](#60second-quick-start)
7. [Demo A — fully offline, deterministic](#demo-a--fully-offline-deterministic)
8. [Demo B — live web search (no API key)](#demo-b--live-web-search-no-api-key)
9. [Demo C — anchor on a real blockchain testnet](#demo-c--anchor-on-a-real-blockchain-testnet-optional)
10. [Reading the output: the `runs/<id>/` folder](#10-reading-the-output-the-runsid-folder)
11. [How re‑verification works](#11-how-reverification-works)
12. [How the blockchain part works](#12-how-the-blockchain-part-works)
13. [Every command, explained](#13-every-command-explained)
14. [Configuration & environment variables](#14-configuration--environment-variables)
15. [How the "genuine search" requirement is met](#15-how-the-genuine-search-requirement-is-met)
16. [Architecture](#16-architecture)
17. [Running the tests](#17-running-the-tests-for-developers)
18. [Known limitations (read this)](#18-known-limitations-read-this)
19. [Troubleshooting / FAQ](#19-troubleshooting--faq)
20. [What to show in the screen recording](#20-what-to-show-in-the-screen-recording)
21. [Repo layout](#21-repo-layout)
22. [Licence & credits](#22-licence--credits)

---

## 1. What this actually does

The task asks for one pipeline with three stages. Here is each stage in plain
English.

### Stage 1 — Face identification
You give it an image file. It finds the face in the picture (a bounding box) and
turns that face into a list of numbers called an **embedding** (a "fingerprint"
of the face). Two photos of the *same* face produce embeddings that are close
together; two different faces produce embeddings that are far apart. "Close" and
"far" are measured with **cosine similarity**, a number from ‑1 to 1 where 1
means identical.

Three interchangeable face engines are included:
- **`opencv`** (default) — OpenCV's classic Haar face detector + a hand‑built
  texture/shape descriptor (LBP + HOG). No model download, fully deterministic.
- **`numpy`** — the same detector re‑implemented in pure NumPy, so the pipeline
  still runs if OpenCV can't be installed. Slower, slightly less accurate.
- **`insightface`** — an optional deep‑learning face recogniser (ArcFace). Best
  accuracy; install it only if you want it.

### Stage 2 — Web / social‑media search
It uses the face embedding to look for a **matching image that really exists on
the internet right now**. This is a genuine search, not a hard‑coded answer.

How it stays genuine: "providers" go out and *collect candidate images* from a
real source. Then the pipeline downloads every candidate, runs the **same face
engine** on it, and ranks them **purely by how similar the face is** to your
input. The best one wins *only if* its similarity clears a threshold (default
`0.86`). The full ranked list with every score is saved to disk, so you can
audit exactly why a candidate won.

Providers included:
- **`wikimedia`** (default, no API key) — runs a real search against the live
  Wikimedia Commons API and pulls back real image files and their web pages.
- **`local`** (default, offline) — searches a folder of images you built earlier
  with `facechain fetch-corpus` (that folder is itself filled from the live web).
- **`serpapi`** (optional, free API key) — true reverse‑image search via Google
  Lens / Yandex; returns real **social‑media** posts (Instagram, X, etc.).

### Stage 3 — Blockchain verification
Once a match is found, the pipeline builds an **evidence bundle**: a single JSON
object containing the SHA‑256 and perceptual hashes of your input image, the face
box and embedding hash, the matched page URL, the matched image's hashes, the
similarity score, which provider found it, and the pipeline version.

That bundle is serialised in a **canonical** way (sorted keys, no whitespace,
integers only) so it always produces the **same bytes** on any computer. Its
SHA‑256 is the **`record_hash`** — and *that* single 64‑character hash is what
gets written onto a blockchain.

- **`local`** (default) — a real, purpose‑built **hash‑linked Merkle ledger** on
  disk. Genesis block, every record becomes a Merkle leaf in a new block, blocks
  are chained by the previous block's hash, optional proof‑of‑work, and every
  anchor returns a Merkle **inclusion proof**.
- **`evm`** (optional) — any Ethereum‑compatible **public testnet** (Sepolia,
  Polygon Amoy, …). Either via the included `EvidenceRegistry.sol` smart
  contract, or with zero setup by putting the hash in a transaction's calldata.

### Stage 4 — Re‑verification (the part that proves it)
`facechain verify <run_folder>` is a **separate program path** that shares no
memory with the pipeline. It re‑opens the saved files, recomputes every hash from
scratch, **re‑runs the face match**, and **reads the `record_hash` back off the
blockchain**, then prints a PASS/FAIL table. If you tamper with any file or any
block, verification fails and tells you exactly which link broke.

---

## 2. What you need installed first

You need exactly two things: **Python 3.11 or newer** and **Git**. Nothing else —
no database, no Docker, no blockchain node.

### Check what you already have

Open a terminal:
- **Windows**: press `Start`, type **PowerShell**, hit Enter.
- **macOS**: `Cmd`+`Space`, type **Terminal**, hit Enter.
- **Linux**: your usual terminal.

Then run:

```
python --version
git --version
```

You want Python to say `3.11.x`, `3.12.x`, `3.13.x`, or `3.14.x`. (On some
systems the command is `python3` instead of `python` — if `python --version`
fails or shows 2.x, use `python3` everywhere below.)

### If you don't have Python

- **Windows**: download from <https://www.python.org/downloads/>. **On the first
  installer screen, tick "Add python.exe to PATH"**, then click *Install Now*.
  Close and reopen PowerShell afterwards.
- **macOS**: `brew install python@3.12` (needs [Homebrew](https://brew.sh)), or
  the installer from python.org.
- **Linux (Debian/Ubuntu)**: `sudo apt update && sudo apt install python3 python3-venv python3-pip git`

### If you don't have Git

- **Windows**: <https://git-scm.com/download/win> — accept all defaults.
- **macOS**: `xcode-select --install` or `brew install git`.
- **Linux**: `sudo apt install git`.

---

## 3. Get the code

```
git clone https://github.com/<your-username>/facechain-verify.git
cd facechain-verify
```

*(Replace the URL with wherever you pushed it. If you downloaded a ZIP instead,
unzip it and `cd` into the folder.)*

---

## 4. Set it up (copy‑paste blocks)

We create an isolated "virtual environment" so this project's packages don't
touch the rest of your system, then install the project into it.

### Windows (PowerShell)

```powershell
# 1. create the virtual environment
python -m venv .venv

# 2. activate it (your prompt should now start with "(.venv)")
.\.venv\Scripts\Activate.ps1

# 3. install dependencies + the project
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

> If step 2 fails with *"running scripts is disabled on this system"*, run this
> once, then retry step 2:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### macOS / Linux (bash / zsh)

```bash
# 1. create the virtual environment
python3 -m venv .venv

# 2. activate it (your prompt should now start with "(.venv)")
source .venv/bin/activate

# 3. install dependencies + the project
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

That's it. Every time you open a new terminal to use this project, re‑run the
**activate** line (step 2) first.

**What just got installed:** `numpy`, `Pillow` (images), `pydantic` (data
validation), `httpx` (web requests), and `opencv-python-headless` (the face
detector). All are normal PyPI packages with prebuilt wheels for Python
3.11–3.14 on Windows, macOS and Linux.

---

## 5. Check the install worked

```
python -m facechain version
```

You should see a JSON blob starting with `"version": "1.0.0"` and your effective
settings. If you see that, you're ready.

> **`facechain: command not found`?** The project also installs a short
> `facechain` command, but it only works if your Python `Scripts`/`bin` folder is
> on your PATH. **You never need it** — just use `python -m facechain …`
> everywhere. This README uses the short form for readability; both are identical.

---

## 60‑second quick start

```bash
# build a small offline image corpus from the bundled public‑domain photos
python -m facechain fetch-corpus --seed-demo

# run the whole pipeline on a sample "reposted" photo
python -m facechain run samples/probe_repost.jpg --providers local --anchor local

# independently re‑verify the run that just happened
python -m facechain verify runs/<paste-the-run-id-it-printed> --no-network

# prove the ledger is tamper‑evident
python -m facechain chain tamper
python -m facechain chain verify        # -> CHAIN INTEGRITY: FAILED (names the block)
```

Or run the whole scripted demo in one go:

```bash
# macOS / Linux / Git Bash
bash scripts/demo.sh

# Windows PowerShell
pwsh scripts/demo.ps1          # or:  powershell -File scripts\demo.ps1
```

---

## Demo A — fully offline, deterministic

No internet needed after install. Good for a reliable screen recording.

### A1. Build the offline search corpus

```
python -m facechain fetch-corpus --seed-demo
```

This copies the bundled public‑domain portraits from `tests/fixtures/` into
`data/corpus/`, each with a small `.json` sidecar recording its real source URL
on Wikimedia Commons. Expected:

```
seeded 4 demo entries into data/corpus
```

### A2. Run the pipeline

```
python -m facechain run samples/probe_repost.jpg --providers local --anchor local
```

`samples/probe_repost.jpg` is a **cropped, rotated, re‑compressed** copy of one
of the corpus photos — it simulates "the same image was reposted somewhere and we
found it". Expected output (hashes will match; timing varies):

```
run dir       : runs/20260902T....Z-xxxxxxxx
face engine   : opencv-haar-lbph   bbox=[65, 38, 213, 213]
BEST MATCH    : sim=0.9330  [local]  https://commons.wikimedia.org/wiki/File:Dwight_D._Eisenhower,_official_photo_portrait,_May_29,_1959.jpg
record_hash   : 660a05fca106b9eb42367e7f424c16dde29b0202bc68d4bf1b677923b73264d7
anchored on   : local-merkle-chain(diff=0)
  block #1  hash=...  merkle_root ...  (idempotent=False)

verification report for run 20260902T....Z-xxxxxxxx
  [PASS] evidence.self_consistent
  [PASS] evidence.binds_receipt
  [PASS] probe.image_integrity
  [PASS] probe.embedding_integrity
  [PASS] match.candidate_integrity
  [PASS] match.face_recheck            cosine 0.9330 vs threshold 0.8600
  [PASS] local.chain_integrity
  [PASS] local.record_on_chain
  [PASS] local.receipt_block_hash
  [PASS] local.merkle_inclusion
  OVERALL: VERIFIED (10/10 checks passed)
```

The pipeline runs `verify` for you automatically at the end. Everything it
produced is in the `runs/<id>/` folder it printed.

### A3. Re‑verify independently

```
python -m facechain verify runs/<the-id-from-above> --no-network
```

Same 10/10 PASS table. This is the *"demonstrate re‑verifying the data against
the on‑chain record"* requirement: it re‑derives the `record_hash` from the raw
files and finds it, with a valid Merkle proof, inside the on‑disk chain.

### A4. Prove tamper‑evidence

```
python -m facechain chain show          # list every block in the ledger
python -m facechain chain tamper        # deliberately corrupt one block
python -m facechain chain verify        # -> CHAIN INTEGRITY: FAILED -- block 1 Merkle root mismatch
python -m facechain verify runs/<id> --no-network   # now drops to FAILED and points at local.chain_integrity
```

To reset to a clean chain, delete the `chaindata/` folder.

### A5. (Optional) show proof‑of‑work

```
python -m facechain run samples/probe_repost.jpg --providers local --anchor local --difficulty 16
```

Now each block's hash must start with 16 zero bits — you'll see a block hash like
`000005fc…`. Still runs in ~1 second.

---

## Demo B — live web search (no API key)

This does a **real, live** search against Wikimedia Commons.

```
python -m facechain run samples/probe_repost.jpg \
  --providers wikimedia \
  --hint "Dwight D. Eisenhower official photo portrait 1959" \
  --anchor local

python -m facechain verify runs/<the-id-it-printed>
```

What happens: the `--hint` is used only to *gather* candidates from the live
MediaWiki API (it does **not** decide the match). The pipeline downloads each
real candidate image, encodes its face, and ranks by similarity. Expected: the
best match is ~0.93 to a real Commons file page, a *different* Eisenhower photo
scores ~0.83 and is correctly rejected (below `0.86`), and the final verify —
which this time also **re‑downloads the matched image from the web** — prints
`OVERALL: VERIFIED (11/11 checks passed)`.

> **Windows note:** in PowerShell, use a backtick `` ` `` for line continuation
> instead of `\`, or just put the whole command on one line.

If nothing clears the threshold you'll see `NO MATCH` and a `no_match` status —
that is a **feature**: the pipeline does not invent a match. Try a more specific
`--hint`, or add `--threshold 0.80` to loosen it.

### With true social‑media results (SerpAPI, optional)

SerpAPI's reverse‑image endpoints need your probe image to be reachable at a
public URL (they can't take an upload). Get a free key at
<https://serpapi.com> (no card required), then:

```
# Windows PowerShell:  $env:FACECHAIN_SERPAPI_KEY = "your-key"
# macOS / Linux:        export FACECHAIN_SERPAPI_KEY="your-key"

python -m facechain run ./my_face.jpg \
  --providers serpapi,wikimedia \
  --probe-image-url https://<a-public-url-of-my_face.jpg> \
  --anchor local
```

---

## Demo C — anchor on a real blockchain testnet (optional)

This writes the `record_hash` to a **real public Ethereum testnet**. It costs
nothing (testnet ETH is free) but needs a funded testnet key.

```bash
python -m pip install -e ".[evm]"        # adds web3.py

# a free public RPC endpoint (no signup):
export FACECHAIN_EVM_RPC_URL="https://ethereum-sepolia-rpc.publicnode.com"
# a throwaway testnet private key funded from a faucet — NEVER a real key:
export FACECHAIN_EVM_PRIVATE_KEY="0xabc123..."

python -m facechain run samples/probe_repost.jpg \
  --providers wikimedia --hint "Dwight D. Eisenhower official photo portrait 1959" \
  --anchor evm

python -m facechain verify runs/<id>     # reads the record_hash back from the chain
```

Get free Sepolia ETH from a faucet such as <https://sepoliafaucet.com> or
<https://www.alchemy.com/faucets/ethereum-sepolia>.

**Two modes, chosen automatically:**
- **calldata mode** (default, zero setup): sends a 0‑value transaction to
  yourself whose data field is `FCV1` + the 32‑byte hash. `verify` pulls the
  transaction by its hash and re‑reads those bytes.
- **contract mode**: if you deploy `contracts/EvidenceRegistry.sol` and set
  `FACECHAIN_EVM_REGISTRY_ADDRESS=0x...`, it calls `anchor(bytes32)` and `verify`
  reads the stored block number back.

Deploy the contract with Remix (<https://remix.ethereum.org>), Foundry, or
Hardhat — it's ~40 lines, no constructor arguments.

---

## 10. Reading the output: the `runs/<id>/` folder

Every `run` creates one self‑describing folder:

| File | What it is |
|---|---|
| `manifest.json` | Index for the run: status, settings fingerprint, list of artifacts, the `record_hash`, which chain it used. |
| `probe.jpg` | The **exact bytes** of your input image (so verification can re‑hash it). |
| `probe_fingerprint.json` | SHA‑256 + 64‑bit perceptual hashes (pHash, dHash) + dimensions of the input. |
| `face.json` | The detected face: engine, bounding box, quality score, embedding hash, every box that was detected. |
| `face_crop.png` | The aligned face patch that was actually encoded (nice for the video). |
| `embedding.npy` | The raw face embedding as a NumPy array. |
| `candidates.json` | **Every** candidate the search considered, ranked, each with its provider, URLs, similarity score, and image fingerprint. This is the audit trail proving the search was genuine. |
| `candidates/00_*.jpg …` | The actual downloaded candidate images. `00_` is the winner. |
| `evidence.json` | The **evidence bundle** that was notarised. Its canonical SHA‑256 is the `record_hash` stored inside it. |
| `receipt.json` | Proof of anchoring: for `local`, the block index/hash/Merkle root + inclusion proof; for `evm`, the transaction hash, block number, chain id. |
| `verification.json` | The automatic re‑verification result (same as `facechain verify` prints). |
| `telemetry.jsonl` | One JSON line per pipeline step with millisecond timings. |

---

## 11. How re‑verification works

`facechain verify <run_dir>` runs these independent checks and prints
`[PASS]`/`[FAIL]` for each. It exits non‑zero if any fail.

| Check | What it proves |
|---|---|
| `evidence.self_consistent` | Re‑serialising `evidence.json` canonically and hashing it reproduces the stored `record_hash`. |
| `evidence.binds_receipt` | The bundle's `record_hash` equals the receipt's `record_hash`. |
| `probe.image_integrity` | SHA‑256 of `probe.jpg` matches what the bundle recorded. |
| `probe.embedding_integrity` | Hash of `embedding.npy` matches the face record. |
| `match.candidate_integrity` | The stored winning image matches the bundle's fingerprint — either byte‑identical **or** within a perceptual‑hash tolerance (survives CDN re‑compression). |
| `match.face_recheck` | **Re‑detects and re‑encodes** the probe and the winning image from scratch, recomputes cosine similarity, and asserts it still clears the threshold. |
| `match.live_refetch` | *(only without `--no-network`)* Re‑downloads the matched image from its URL and compares it perceptually. If the post is gone or you're offline, this is skipped, not failed — the chain + stored copy remain authoritative. |
| `local.chain_integrity` | Walks the entire ledger recomputing every Merkle root, block hash and back‑link. |
| `local.record_on_chain` / `local.merkle_inclusion` / `local.receipt_block_hash` | The `record_hash` is in a block, the receipt's Merkle proof validates against that block's root, and the receipt names the right block. |
| `evm.tx_mined` / `evm.confirmations` / `evm.calldata_matches_record` *(or `evm.registry_has_record`)* | *(evm backend)* The anchoring transaction is mined and confirmed, and the value on‑chain equals the `record_hash`. |

---

## 12. How the blockchain part works

### `local` — a real hash‑linked Merkle ledger (default)

Stored as plain text at `chaindata/local/blocks.jsonl`, one JSON block per line.

- **Genesis block** (index 0) is fixed and deterministic — every fresh chain
  starts identically.
- Each `anchor(record_hash)` creates a **new block** containing that hash as a
  Merkle leaf, with fields: `index`, `timestamp`, `prev_hash` (the previous
  block's hash), `merkle_root`, `difficulty`, `nonce`, and `hash`.
- `hash = SHA‑256(canonical_json(index, timestamp, prev_hash, merkle_root, difficulty, records, nonce))`.
- **Proof‑of‑work**: with `--difficulty N`, the miner searches for a `nonce` that
  makes the block hash start with `N` zero bits.
- **Merkle inclusion proof**: `anchor` returns the sibling hashes needed to
  recompute the Merkle root from just your leaf — so a verifier doesn't need the
  whole block.
- **Idempotent**: the `record_hash` covers the *finding* (probe + face + match),
  **not** the run's timestamp. Re‑running the same input yields the same hash and
  the chain refuses to add it twice. The trusted *time* comes from the block
  itself.
- `facechain chain verify` re‑checks the entire structure and pinpoints the
  first broken block.

This is exactly the *"local / simulated chain"* the task allows — implemented
properly, not faked with a dictionary.

### `evm` — a real public testnet

Uses `web3.py` against any JSON‑RPC endpoint. See [Demo C](#demo-c--anchor-on-a-real-blockchain-testnet-optional).
The contract (`contracts/EvidenceRegistry.sol`) is append‑only and idempotent and
emits an `Anchored` event. A local cache maps `record_hash → txhash` so
re‑anchoring never broadcasts twice.

---

## 13. Every command, explained

Run `python -m facechain <command> --help` for full flags.

| Command | What it does |
|---|---|
| `facechain identify <image>` | Just Stage 1: detect + encode the face, print the box, hashes and embedding preview. No search, no chain. |
| `facechain search <image>` | Stages 1–2: run the live search and print the ranked candidates. No chain. |
| `facechain run <image>` | The whole pipeline (Stages 1–4). Writes a `runs/<id>/` folder. |
| `facechain verify <run_dir>` | Independently re‑verify a finished run. `--no-network` skips the live re‑download check. |
| `facechain chain show` | Print every block in the local ledger. |
| `facechain chain verify` | Full integrity re‑check of the local ledger. |
| `facechain chain tamper` | **Demo only.** Corrupt one block so you can watch `chain verify` catch it. |
| `facechain fetch-corpus --seed-demo` | Fill `data/corpus/` from the bundled public‑domain photos (offline). |
| `facechain fetch-corpus --query "..."` | Fill `data/corpus/` by pulling real images live from Wikimedia Commons. |
| `facechain version` | Print version + effective configuration. |

**Common flags** (most commands): `--engine {opencv,numpy,insightface}`,
`--providers wikimedia,local,serpapi`, `--anchor {local,evm}`,
`--threshold 0.86`, `--difficulty 0`, `--hint "text"`,
`--probe-image-url URL`, `--runs-dir`, `--chain-dir`, `--corpus-dir`, `--json`.

---

## 14. Configuration & environment variables

Everything has a sensible default. Nothing is required for Demo A or B.

Set variables in your shell, or copy `.env.example` to `.env` (it's read
automatically, and `.env` is git‑ignored).

| Variable | Purpose |
|---|---|
| `FACECHAIN_SERPAPI_KEY` | Enables the `serpapi` provider (free key from serpapi.com). |
| `FACECHAIN_HTTP_CONTACT` | A contact string put in the `User-Agent` when calling public APIs (politeness). |
| `FACECHAIN_EVM_RPC_URL` | JSON‑RPC endpoint for `--anchor evm`. |
| `FACECHAIN_EVM_PRIVATE_KEY` | Funded **testnet** key for `--anchor evm`. Never a mainnet key. |
| `FACECHAIN_EVM_REGISTRY_ADDRESS` | Optional deployed `EvidenceRegistry` address (switches evm to contract mode). |
| `FACECHAIN_MATCH_THRESHOLD` | Override the `0.86` similarity threshold. |
| `FACECHAIN_SEARCH_PROVIDERS` | Comma‑separated default provider list. |
| `FACECHAIN_LOG_LEVEL` | `debug` \| `info` (default) \| `warning` \| `error`. |
| `FACECHAIN_LOG_JSON=1` | Force machine‑readable JSON logs. |

---

## 15. How the "genuine search" requirement is met

The task says the match must be *"a genuine search step, not a
hardcoded/pre‑picked result."* Here's the design that guarantees that:

1. **Providers only gather.** A provider's job is to return a list of *candidate*
   image URLs from a real, live source. It never says which one matches.
2. **The face decides.** The pipeline downloads every candidate, runs the **same
   face engine** used on your input, computes cosine similarity, and sorts by it.
   The winner is `ranked[0]` **only if** its score ≥ threshold.
3. **It's all on disk.** `candidates.json` lists every candidate with its
   provider, its URLs, its image fingerprint and its exact score. You can see
   why the winner won and why the others lost.
4. **Swapping providers can't change the winner** — only which images get
   considered. Change `--providers` and the *same* face still wins if it's in the
   pool.
5. A run that finds nothing above threshold reports `status: no_match` and writes
   `no_match_debug.json`. It never fabricates a match.

---

## 16. Architecture

```
your image
   │
   ▼
imaging.py ──── safe decode (size / pixel caps, format allow‑list, EXIF strip),
   │            SHA‑256 + pHash + dHash
   ▼
face/ ───────── FaceEngine protocol
   │              ├─ opencv_backend.py   (Haar + LBP/HOG descriptor)   ← default
   │              ├─ numpy_vj_backend.py (pure‑NumPy Viola‑Jones)
   │              └─ insightface_backend.py (optional ArcFace)
   │            cascade.py: gets the Haar model from OpenCV, or a bundled .gz
   ▼
search/ ─────── SearchProvider protocol → serpapi | wikimedia | local_index
   │            aggregator.py: fetch every candidate (via SSRF‑guarded
   │            netfetch.py), encode its face, rank by cosine, apply threshold
   ▼
models.py ───── build EvidenceBundle (pydantic), serialise with canonical.py,
   │            SHA‑256 = record_hash
   ▼
anchor/ ─────── AnchorBackend protocol
   │              ├─ local_chain.py  (hash‑linked Merkle ledger)   ← default
   │              │   merkle.py: tree + inclusion proofs
   │              └─ evm_backend.py  (Ethereum testnet, web3.py)
   ▼
pipeline.py ─── orchestrates all of the above, writes runs/<id>/
verify.py ───── independent re‑verification (re‑derive hashes, re‑match, walk chain)
cli.py ──────── argparse front‑end for every command
```

Design choices worth knowing:
- **Canonical JSON** (sorted keys, no whitespace, UTF‑8) with **floats banned** —
  every score is stored as an integer in parts‑per‑million, so hashing is
  byte‑stable across machines.
- **Content‑addressed & idempotent** anchoring (see §12).
- **Exact + perceptual fingerprints** so re‑verification survives a CDN
  re‑encoding the image, while the exact SHA‑256 is always reported.
- **SSRF‑hardened fetching**: candidate URLs are resolved and every hop is
  checked against a block‑list (loopback, private ranges, link‑local,
  `169.254.169.254`, reserved); response size and redirect count are capped;
  content type + magic bytes are verified.
- **Structured, timed telemetry** to `runs/<id>/telemetry.jsonl`.
- **Strictly typed** (`mypy --strict` clean) and linted (`ruff`).

---

## 17. Running the tests (for developers)

```
python -m pip install -e ".[dev]"
python -m ruff check .            # linter — should say "All checks passed!"
python -m mypy                    # type checker — "Success: no issues found"
python -m pytest -m "not network" # 98 offline tests
python -m pytest                  # + 1 test that hits the real network
```

The test suite (99 tests) covers canonical hashing, the Merkle tree, the local
chain (including every tamper mode), image loading and perceptual hashing, face
detection/encoding, the search aggregator (ranking, no‑match, dedupe, skip broken
candidates), the full offline pipeline + verify, the CLI, the SSRF guard, and the
EVM helper functions. CI (`.github/workflows/ci.yml`) runs lint + types + tests +
an end‑to‑end CLI smoke test on Python 3.11, 3.12 and 3.13.

---

## 18. Known limitations (read this)

- **The default face engine matches *the same photo*, not *the same person
  across different photos*.** OpenCV + LBP/HOG is a classical descriptor: a
  re‑encoded / cropped / rotated **copy of one photo** scores ~0.93–0.99, while
  unrelated faces score ~0.55–0.84, so the `0.86` threshold separates them
  cleanly. This is exactly the reverse‑image‑search case (someone reposted the
  picture). Matching *two different photographs* of the same person (different
  pose / lighting / year) is **not reliable** with a classical descriptor —
  install the optional deep engine for that:
  `python -m pip install -e ".[insightface]"` then add `--engine insightface`.
- **SerpAPI needs a public URL for your probe image** — it can't accept an
  upload. The `wikimedia` and `local` providers have no such limitation.
- **Wikimedia is "the web", not "social media".** It gives a keyless, fully
  reproducible search path. For literal Instagram/X/Facebook posts, use
  `--providers serpapi`.
- **Haar detection is frontal‑face only** and can occasionally return a spurious
  box; the pipeline picks the largest face and flags "ambiguous" when a
  runner‑up is a similar size.
- **The pure‑NumPy engine (`--engine numpy`)** exists for zero‑binary‑dependency
  environments; its boxes are looser than OpenCV's and it costs a few points of
  similarity. Prefer the default.
- **The EVM broadcast path is exercised manually** against a funded testnet key
  (its pure helpers are unit‑tested); it is not run in CI.
- **Local‑chain proof‑of‑work** defaults to difficulty 0 (instant). It's a
  tamper *deterrent*, not Nakamoto consensus.
- **DNS‑rebinding**: the SSRF check resolves and validates before each request; a
  determined rebinding attacker could in theory flip the DNS record between the
  check and the connect. Acceptable for a research tool that only fetches images.

---

## 19. Troubleshooting / FAQ

**`python` opens the Microsoft Store / "Python was not found" (Windows).**
Python isn't on PATH. Reinstall from python.org with *"Add python.exe to PATH"*
ticked, or use the `py` launcher: `py -m venv .venv`, `py -m pip …`.

**`facechain: command not found` / `'facechain' is not recognized`.**
Ignore it — use `python -m facechain …`. (The short command needs your Python
`Scripts`/`bin` dir on PATH; `python -m` never does.)

**`ModuleNotFoundError: No module named 'facechain'`.**
Your virtual environment isn't activated (no `(.venv)` in the prompt), or you
skipped `python -m pip install -e .`. Re‑do §4 steps 2–3.

**`AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`.**
You have OpenCV 5, which removed that API. `requirements.txt` pins
`opencv-python-headless>=4.9,<5`; run
`python -m pip install "opencv-python-headless<5" --force-reinstall`.

**A file won't open / GitHub says "we can't show files that are this big".**
That was an earlier problem with a bundled 900 KB model file. It's now shipped
gzip‑compressed (~135 KB) and loaded from OpenCV's own copy when available — a
fresh clone has no oversized files. If you cloned an old commit, `git pull`.

**`NO MATCH` on Demo B.** Nothing cleared the `0.86` threshold — that's the
pipeline refusing to guess. Use a more specific `--hint`, or lower it with
`--threshold 0.80`.

**Wikimedia / network errors.** You may be offline or rate‑limited. Use Demo A
(`--providers local`) which needs no network after `fetch-corpus --seed-demo`.

**"running scripts is disabled on this system" when activating the venv
(Windows).** Run once:
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, then
retry.

**Reset everything.** Delete the `runs/`, `chaindata/`, and `data/corpus/*`
folders (all git‑ignored). `make clean` does this on macOS/Linux.

**Install is slow / a wheel is building from source.** Make sure pip is current
(`python -m pip install --upgrade pip`); prebuilt wheels exist for all
dependencies on CPython 3.11–3.14 (Windows/macOS/Linux, 64‑bit).

---

## 20. What to show in the screen recording

The task wants the pipeline working end to end: **face scan → social/web post
found → blockchain upload/verification**. A clean 2–3 minute capture:

1. `python -m facechain version` (prove it's installed).
2. `python -m facechain fetch-corpus --seed-demo`.
3. **Live search:** `python -m facechain run samples/probe_repost.jpg --providers wikimedia --hint "Dwight D. Eisenhower official photo portrait 1959" --anchor local`
   — narrate: face detected, candidates fetched live, best match URL, `record_hash`, block written.
4. Open the printed `runs/<id>/` folder; show `candidates.json` (the audit
   trail), `evidence.json`, `receipt.json`.
5. **Re‑verify:** `python -m facechain verify runs/<id>` → `VERIFIED (11/11)`.
6. **Tamper‑evidence:** `python -m facechain chain show`, then
   `python -m facechain chain tamper`, then `python -m facechain chain verify` →
   `FAILED`, then `python -m facechain verify runs/<id> --no-network` → `FAILED`
   pointing at `local.chain_integrity`.
7. *(Optional)* repeat step 3 with `--anchor evm` to show a real Sepolia
   transaction, and `verify` reading it back.

`scripts/demo.sh` / `scripts/demo.ps1` do steps 2–6 automatically (add `--live`
for the live‑search variant).

---

## 21. Repo layout

```
facechain-verify/
├─ README.md                     ← this file
├─ LICENSE  NOTICE               ← MIT; third‑party credits
├─ pyproject.toml                ← package metadata, deps, ruff/mypy/pytest config
├─ requirements.txt              ← pinned runtime deps for a plain install
├─ requirements-dev.txt          ← + ruff, mypy, pytest, web3
├─ Makefile                      ← make install / test / demo / clean (macOS/Linux)
├─ .env.example                  ← copy to .env for optional API keys
├─ contracts/
│  └─ EvidenceRegistry.sol       ← optional Ethereum anchor contract
├─ scripts/
│  ├─ demo.sh   demo.ps1         ← one‑command end‑to‑end demos
├─ samples/
│  ├─ probe_repost.jpg           ← demo input (a "reposted" photo)
│  ├─ probe_obama.jpg
│  └─ SOURCES.json               ← provenance + licence of every bundled image
├─ src/facechain/
│  ├─ cli.py  pipeline.py  verify.py  corpus.py
│  ├─ config.py  models.py  canonical.py  imaging.py  netfetch.py  errors.py  logging.py
│  ├─ face/     base.py cascade.py descriptor.py opencv_backend.py numpy_vj_backend.py
│  │           insightface_backend.py factory.py
│  │  └─ models/haarcascade_frontalface_default.xml.gz   ← 135 KB, used only if OpenCV absent
│  ├─ search/   base.py aggregator.py wikimedia_provider.py serpapi_provider.py
│  │           local_index_provider.py factory.py
│  └─ anchor/   base.py merkle.py local_chain.py evm_backend.py factory.py
├─ tests/                        ← 99 tests + public‑domain fixture images
└─ .github/workflows/ci.yml      ← lint + types + tests on Python 3.11/3.12/3.13
```

Generated at runtime (all git‑ignored): `runs/`, `chaindata/`, `data/corpus/*`,
`.venv/`.

---

## 22. Licence & credits

- **Code**: MIT — see `LICENSE`.
- **Bundled images** (`samples/`, `tests/fixtures/`): public‑domain portraits
  from Wikimedia Commons (US federal government works / pre‑1929). Per‑file
  provenance and licence in `samples/SOURCES.json`.
- **Haar cascade model** (`src/facechain/face/models/*.xml.gz`): from OpenCV,
  BSD‑3‑Clause — see `NOTICE`. When `opencv-python[-headless]` is installed, its
  own bundled copy is used and this file is never read.
- **Dependencies**: numpy, Pillow, pydantic, httpx, opencv-python-headless;
  optional web3, insightface, onnxruntime. See each project for terms.
