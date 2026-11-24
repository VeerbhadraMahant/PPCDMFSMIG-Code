#!/usr/bin/env bash
# Pipeline: per N, train then evaluate (256, 1024, 4096), then figures. Every stage is resumable or
# idempotent, so a failed stage is retried (3 attempts) and, if it still fails, the script moves on.
STEPS="${STEPS:-30000}"

set -o pipefail
[ "$STEPS" -gt 0 ] || { echo "Set STEPS at the top of run_all.sh first." >&2; exit 1; }
cd "$(dirname "$0")"
HERE="$(pwd)"
PY=".venv/Scripts/python.exe"
mkdir -p .cache/pip .cache/torch .cache/hf .cache/tmp runs
export PIP_CACHE_DIR="$HERE/.cache/pip" TORCH_HOME="$HERE/.cache/torch" HF_HOME="$HERE/.cache/hf"
export TMP="$HERE/.cache/tmp" TEMP="$HERE/.cache/tmp" PYTHONUTF8=1 PYTHONIOENCODING=utf-8
STATUS="runs/run_all_status.log"

stage() {
  local log="$1"; shift
  for i in 1 2 3; do
    echo "[$(date +%FT%T)] attempt $i: $*" | tee -a "$log"
    "$PY" -u "$@" 2>&1 | tee -a "$log"
    [ "${PIPESTATUS[0]}" -eq 0 ] && return 0
  done
  echo "[$(date +%FT%T)] FAILED after 3 attempts: $* (see $log)" | tee -a "$STATUS"
  return 1
}

for n in 256 1024 4096; do
  if stage "runs/n${n}_train.log" train.py --n "$n" --steps "$STEPS" --out "runs/n$n"; then
    stage "runs/n${n}_eval.log" evaluate.py --run "runs/n$n" --ckpt all --n-samples 1000 || true
  else
    echo "skipping evaluation of N=$n because training failed" | tee -a "$STATUS"
  fi
done
stage runs/figures.log make_figures.py || true
echo "[$(date +%FT%T)] run_all finished" | tee -a "$STATUS"
