#!/usr/bin/env bash
# facechain-verify — end-to-end demo for the screen recording (bash / Git Bash / WSL).
# Usage:  ./scripts/demo.sh          # offline (deterministic, no network)
#         ./scripts/demo.sh --live   # live keyless Wikimedia reverse-image search
set -euo pipefail
cd "$(dirname "$0")/.."
export FACECHAIN_LOG_LEVEL=error

step() { printf '\n=== %s ===\n\n' "$1"; }
LIVE="${1:-}"

step "1. Build the offline search corpus from bundled public-domain portraits"
python -m facechain fetch-corpus --seed-demo

if [ "$LIVE" = "--live" ]; then
  step "2. LIVE pipeline: face scan -> Wikimedia reverse-image search -> local Merkle chain"
  python -m facechain run samples/probe_repost.jpg \
    --providers wikimedia \
    --hint "Dwight D. Eisenhower official photo portrait 1959" \
    --anchor local
else
  step "2. OFFLINE pipeline: face scan -> local corpus search -> local Merkle chain"
  python -m facechain run samples/probe_repost.jpg --providers local --anchor local
fi

RUN="$(ls -dt runs/*/ | head -1)"

step "3. Independent re-verification (raw artifacts -> hashes -> chain)"
if [ "$LIVE" = "--live" ]; then python -m facechain verify "$RUN"; else python -m facechain verify "$RUN" --no-network; fi

step "4. The local ledger"
python -m facechain chain show

step "5. Tamper-evidence: corrupt one block, then re-verify (expect FAILED)"
python -m facechain chain tamper
python -m facechain chain verify || true
echo
echo "(the FAILED above is the point — tampering is detected and localised)"

step "6. Re-seal a clean chain for repeat demos"
rm -rf chaindata
echo "done."
