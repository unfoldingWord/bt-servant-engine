#!/usr/bin/env bash
set -euo pipefail

# Run incremental, enforced checks for brain.py and the test suite.
# Usage: scripts/check_brain.sh

RED=\033[31m
GRN=\033[32m
YLW=\033[33m
RST=\033[0m

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$repo_root"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo -e "${YLW}Running ruff on brain.py...${RST}"
ruff check brain.py

echo -e "${YLW}Running pylint on brain.py...${RST}"
pylint -rn -sn brain.py

echo -e "${YLW}Running mypy on brain.py...${RST}"
mypy brain.py

echo -e "${YLW}Running tests (pytest -q)...${RST}"
pytest -q

echo -e "${GRN}brain.py checks and tests passed.${RST}"

