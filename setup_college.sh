#!/usr/bin/env bash
# One-time setup for the college PC (Linux). Safe to run again: finished steps are skipped.
# Run from anywhere:  bash setup_college.sh
# Needs internet (pip, the dataset, the FID weights). No admin rights. Everything stays in this folder.
set -e
cd "$(dirname "$0")"
HERE="$(pwd)"
mkdir -p .cache/pip .cache/torch .cache/hf .cache/tmp runs
export PIP_CACHE_DIR="$HERE/.cache/pip" TORCH_HOME="$HERE/.cache/torch" HF_HOME="$HERE/.cache/hf"
export TMP="$HERE/.cache/tmp" TEMP="$HERE/.cache/tmp" TMPDIR="$HERE/.cache/tmp"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
VERCODE='import sys; print(sys.version_info[0] * 100 + sys.version_info[1])'

say() { printf '\n== %s\n' "$1"; }
py_ok() {  # 0 if "$1" is Python 3.11 to 3.13
  command -v "$1" >/dev/null 2>&1 || return 1
  local v
  v="$("$1" -c "$VERCODE" 2>/dev/null)" || return 1
  [ "$v" -ge 311 ] && [ "$v" -le 313 ]
}

VENV_PY=".venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY=".venv/Scripts/python.exe"  # Git Bash on Windows

say "1/6 Python virtual environment"
if [ -x "$VENV_PY" ] && py_ok "$VENV_PY"; then
  echo "already there, skipping"
else
  PY=""
  for c in python3.13 python3.12 python3.11 python3 python; do
    if py_ok "$c"; then PY="$c"; break; fi
  done
  [ -n "$PY" ] || { echo "No Python 3.11, 3.12 or 3.13 found. Ask for one to be installed (or use conda/pyenv in your home folder), then run this again."; exit 1; }
  echo "using $PY"
  rm -rf .venv
  "$PY" -m venv .venv || { echo "Could not create the venv. On Debian/Ubuntu this needs the python3-venv package (ask the admin), or use a Python from conda/pyenv."; exit 1; }
  VENV_PY=".venv/bin/python"
  [ -x "$VENV_PY" ] || VENV_PY=".venv/Scripts/python.exe"
fi

say "2/6 Python packages (PyTorch CUDA 12.8 build, about 3 GB the first time)"
REQ_HASH="$("$VENV_PY" -c "import hashlib; print(hashlib.sha256(open('requirements.txt','rb').read()).hexdigest())")"
if [ -f .venv/requirements.installed ] && [ "$(cat .venv/requirements.installed)" = "$REQ_HASH" ]; then
  echo "already installed, skipping"
else
  "$VENV_PY" -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
  echo "$REQ_HASH" > .venv/requirements.installed
fi
if ! "$VENV_PY" -c "import opacus" 2>/dev/null; then
  echo "installing opacus (needed for the DP experiment E5)"
  "$VENV_PY" -m pip install opacus || echo "WARNING: opacus did not install; E5 will not work"
fi

say "3/6 Dataset (PneumoniaMNIST, 64x64)"
if [ -f data/pneumoniamnist_64.npz ]; then
  echo "already downloaded, skipping"
else
  "$VENV_PY" -c "from common import load_data; load_data()"
fi

say "4/6 FID Inception weights"
if ls .cache/torch/hub/checkpoints/*inception* >/dev/null 2>&1; then
  echo "already cached, skipping"
else
  "$VENV_PY" -c "from torchmetrics.image.fid import FrechetInceptionDistance; FrechetInceptionDistance(feature=2048, normalize=True); print('FID weights ready')"
fi

say "5/6 GPU check"
rc=0
"$VENV_PY" run_college.py check || rc=$?

say "6/6 Done"
if [ "$rc" -eq 0 ]; then
  echo "Setup finished. Next step: $VENV_PY run_college.py bench"
else
  echo "Setup finished but the check above has FAIL lines. Send the output to Claude."
fi
