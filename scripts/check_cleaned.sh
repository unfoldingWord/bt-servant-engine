#!/usr/bin/env bash
set -euo pipefail

# Run incremental, enforced checks for the current set of cleaned files.
# Usage: scripts/check_cleaned.sh

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

CHECK_FILES=(brain.py user_message.py)

echo -e "${YLW}Running ruff on cleaned files: ${CHECK_FILES[*]}...${RST}"
ruff check "${CHECK_FILES[@]}"

echo -e "${YLW}Running pylint on cleaned files: ${CHECK_FILES[*]}...${RST}"
pylint -rn -sn "${CHECK_FILES[@]}"

echo -e "${YLW}Running mypy on cleaned files: ${CHECK_FILES[*]}...${RST}"
mypy "${CHECK_FILES[@]}"

echo -e "${YLW}Running tests (pytest -q)...${RST}"
pytest -q

echo -e "${GRN}Cleaned-files checks and tests passed.${RST}"
