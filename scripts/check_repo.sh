#!/usr/bin/env bash
set -euo pipefail

# Run full repo checks: ruff, pylint, mypy, and test suite.
# Usage: scripts/check_repo.sh

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

echo -e "${YLW}Running ruff on repo...${RST}"
ruff check .

echo -e "${YLW}Running pylint on repo...${RST}"
pylint -rn -sn $(git ls-files '*.py')

echo -e "${YLW}Running mypy on repo...${RST}"
mypy .

echo -e "${YLW}Running pyright on repo...${RST}"
if command -v pyright >/dev/null 2>&1; then
  pyright
else
  echo -e "${RED}pyright not found in PATH. Install with: pip install pyright${RST}"
  exit 1
fi

echo -e "${YLW}Running tests (pytest -q -m 'not openai')...${RST}"
# Exclude OpenAI-costly tests by default; run them on-demand only.
pytest -q -m "not openai"

echo -e "${GRN}Repo checks and tests passed.${RST}"
