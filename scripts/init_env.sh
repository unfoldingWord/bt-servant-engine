#!/usr/bin/env bash
set -euo pipefail

# Initialize a Python venv and install runtime + dev tools.
# Safe to re-run. Detects/works around non-UTF8 requirements.txt.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$repo_root"

PY=${PY:-python3}

if [[ ! -d .venv ]]; then
  echo "[init_env] Creating venv in .venv" >&2
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

REQ_FILE="requirements.txt"
REQ_TO_USE="$REQ_FILE"

if file -I "$REQ_FILE" | grep -qi 'charset=utf-16'; then
  echo "[init_env] Detected UTF-16 requirements.txt; converting to UTF-8 for pip" >&2
  REQ_TO_USE=.requirements.utf8.txt
  iconv -f UTF-16 -t UTF-8 "$REQ_FILE" > "$REQ_TO_USE"
fi

echo "[init_env] Installing runtime requirements from $REQ_TO_USE" >&2
pip install -r "$REQ_TO_USE"

echo "[init_env] Installing dev tools (pytest, ruff, pylint, mypy, pyright)" >&2
pip install pytest ruff pylint mypy pyright

echo "[init_env] Done. Activate with: source .venv/bin/activate" >&2

