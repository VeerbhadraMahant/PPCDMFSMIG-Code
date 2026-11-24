# Pipeline: per N, train then evaluate (256, 1024, 4096), then figures. Every stage is resumable or
# idempotent, so a failed stage is retried (3 attempts) and, if it still fails, the script moves on.
$STEPS = 30000

if ($env:STEPS) { $STEPS = [int]$env:STEPS }
if ($STEPS -le 0) { throw "Set STEPS at the top of run_all.ps1 first." }
$here = $PSScriptRoot
Set-Location $here
$py = Join-Path $here ".venv\Scripts\python.exe"
$cache = Join-Path $here ".cache"
foreach ($d in "pip", "torch", "hf", "tmp") { New-Item -ItemType Directory -Force -Path (Join-Path $cache $d) | Out-Null }
$env:PIP_CACHE_DIR = "$cache\pip"; $env:TORCH_HOME = "$cache\torch"; $env:HF_HOME = "$cache\hf"
$env:TMP = "$cache\tmp"; $env:TEMP = "$cache\tmp"
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
New-Item -ItemType Directory -Force -Path "runs" | Out-Null
$status = "runs\run_all_status.log"

function Say($log, $line) {
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}

function Stage($log, [string[]]$a) {
    for ($i = 1; $i -le 3; $i++) {
        Say $log ("[{0}] attempt {1}: {2}" -f (Get-Date -Format s), $i, ($a -join " "))
        & $py -u @a 2>&1 | ForEach-Object { Say $log "$_" }
        if ($LASTEXITCODE -eq 0) { return $true }
    }
    Say $status ("[{0}] FAILED after 3 attempts: {1} (see {2})" -f (Get-Date -Format s), ($a -join " "), $log)
    return $false
}

foreach ($n in 256, 1024, 4096) {
    if (Stage "runs\n${n}_train.log" @("train.py", "--n", $n, "--steps", $STEPS, "--out", "runs/n$n")) {
        [void](Stage "runs\n${n}_eval.log" @("evaluate.py", "--run", "runs/n$n", "--ckpt", "all", "--n-samples", "1000"))
    } else {
        Say $status "skipping evaluation of N=$n because training failed"
    }
}
[void](Stage "runs\figures.log" @("make_figures.py"))
Say $status ("[{0}] run_all finished" -f (Get-Date -Format s))
