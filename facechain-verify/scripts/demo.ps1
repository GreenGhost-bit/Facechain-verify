# facechain-verify — end-to-end demo for the screen recording (Windows PowerShell).
# Usage:   pwsh scripts/demo.ps1            # offline (deterministic, no network)
#          pwsh scripts/demo.ps1 -Live      # live keyless Wikimedia reverse-image search
param([switch]$Live)

$ErrorActionPreference = "Stop"
$env:FACECHAIN_LOG_LEVEL = "error"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Step($msg) { Write-Host "`n=== $msg ===`n" -ForegroundColor Cyan }

Step "1. Build the offline search corpus from bundled public-domain portraits"
python -m facechain fetch-corpus --seed-demo

if ($Live) {
    Step "2. LIVE pipeline: face scan -> Wikimedia reverse-image search -> local Merkle chain"
    python -m facechain run samples/probe_repost.jpg `
        --providers wikimedia `
        --hint "Dwight D. Eisenhower official photo portrait 1959" `
        --anchor local
} else {
    Step "2. OFFLINE pipeline: face scan -> local corpus search -> local Merkle chain"
    python -m facechain run samples/probe_repost.jpg --providers local --anchor local
}

$run = (Get-ChildItem runs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName

Step "3. Independent re-verification (raw artifacts -> hashes -> chain)"
if ($Live) { python -m facechain verify $run } else { python -m facechain verify $run --no-network }

Step "4. The local ledger"
python -m facechain chain show

Step "5. Tamper-evidence: corrupt one block, then re-verify (expect FAILED)"
python -m facechain chain tamper
python -m facechain chain verify
Write-Host "`n(the FAILED above is the point — tampering is detected and localised)" -ForegroundColor Yellow

Step "6. Re-seal a clean chain for repeat demos"
Remove-Item -Recurse -Force chaindata -ErrorAction SilentlyContinue
Write-Host "done." -ForegroundColor Green
