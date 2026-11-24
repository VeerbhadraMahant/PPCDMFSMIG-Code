# One-time setup for the college PC (Windows). Safe to run again: finished steps are skipped.
# Run from the experiments folder:  powershell -ExecutionPolicy Bypass -File setup_college.ps1
# Needs internet (pip, the dataset, the FID weights). No admin rights. Everything stays in this folder.
$ErrorActionPreference = "Continue"  # native exit codes are checked by hand; "Stop" turns pip's stderr into errors in PS 5.1
Set-Location $PSScriptRoot
$here = $PSScriptRoot
$cache = Join-Path $here ".cache"
foreach ($d in "pip", "torch", "hf", "tmp") { New-Item -ItemType Directory -Force -Path (Join-Path $cache $d) | Out-Null }
New-Item -ItemType Directory -Force -Path (Join-Path $here "runs") | Out-Null
$env:PIP_CACHE_DIR = "$cache\pip"; $env:TORCH_HOME = "$cache\torch"; $env:HF_HOME = "$cache\hf"
$env:TMP = "$cache\tmp"; $env:TEMP = "$cache\tmp"
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
$verCode = "import sys; print(sys.version_info[0] * 100 + sys.version_info[1])"

function Say($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Rest([string[]]$cmd) { if ($cmd.Length -gt 1) { return $cmd[1..($cmd.Length - 1)] } else { return @() } }
function PyOk([string[]]$cmd) {  # true if this command is Python 3.11 to 3.13
    try {
        $rest = @(Rest $cmd)
        $v = & $cmd[0] @rest -c $verCode 2>$null
        return ($LASTEXITCODE -eq 0 -and [int]$v -ge 311 -and [int]$v -le 313)
    } catch { return $false }
}

$venvPy = Join-Path $here ".venv\Scripts\python.exe"

Say "1/6 Python virtual environment"
if ((Test-Path $venvPy) -and (PyOk @($venvPy))) {
    Write-Host "already there, skipping"
} else {
    $cands = @(@("py", "-3.13"), @("py", "-3.12"), @("py", "-3.11"), @("python"), @("python3"))
    $found = $null
    foreach ($c in $cands) {
        if ((Get-Command $c[0] -ErrorAction SilentlyContinue) -and (PyOk $c)) { $found = $c; break }
    }
    if (-not $found) {
        throw "No Python 3.11, 3.12 or 3.13 found. Install Python 3.13 from python.org (tick 'Add to PATH', no admin needed with 'Install just for me'), then run this again."
    }
    Write-Host ("using " + ($found -join " "))
    $rest = @(Rest $found)
    & $found[0] @rest -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "could not create .venv" }
}

Say "2/6 Python packages (PyTorch CUDA 12.8 build, about 3 GB the first time)"
$reqHash = (Get-FileHash (Join-Path $here "requirements.txt") -Algorithm SHA256).Hash
$marker = Join-Path $here ".venv\requirements.installed"
if ((Test-Path $marker) -and ((Get-Content $marker -Raw).Trim() -eq $reqHash)) {
    Write-Host "already installed, skipping"
} else {
    & $venvPy -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw "pip install failed (check the internet connection and run this script again)" }
    Set-Content -Path $marker -Value $reqHash
}
& $venvPy -c "import opacus" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "installing opacus (needed for the DP experiment E5)"
    & $venvPy -m pip install opacus
    if ($LASTEXITCODE -ne 0) { Write-Host "WARNING: opacus did not install; E5 will not work" -ForegroundColor Yellow }
}

Say "3/6 Dataset (PneumoniaMNIST, 64x64)"
if (Test-Path (Join-Path $here "data\pneumoniamnist_64.npz")) {
    Write-Host "already downloaded, skipping"
} else {
    & $venvPy -c "from common import load_data; load_data()"
    if ($LASTEXITCODE -ne 0) { throw "dataset download failed" }
}

Say "4/6 FID Inception weights"
$fid = Get-ChildItem -Path (Join-Path $cache "torch\hub\checkpoints") -Filter "*inception*" -ErrorAction SilentlyContinue
if ($fid) {
    Write-Host "already cached, skipping"
} else {
    & $venvPy -c "from torchmetrics.image.fid import FrechetInceptionDistance; FrechetInceptionDistance(feature=2048, normalize=True); print('FID weights ready')"
    if ($LASTEXITCODE -ne 0) { throw "FID weight download failed" }
}

Say "5/6 GPU check"
& $venvPy run_college.py check
$rc = $LASTEXITCODE

Say "6/6 Done"
if ($rc -eq 0) { Write-Host "Setup finished. Next step: .venv\Scripts\python.exe run_college.py bench" -ForegroundColor Green }
else { Write-Host "Setup finished but the check above has FAIL lines. Send the output to Claude." -ForegroundColor Yellow }
