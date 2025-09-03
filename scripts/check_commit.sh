#!/usr/bin/env bash
set -euo pipefail

# Pre-commit checks for this repo.
# Default: run on the current set of cleaned files.
# Set CHECK_ALL=1 to run repo-wide (equivalent to scripts/check_repo.sh).
# Usage: scripts/check_commit.sh

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

if [[ "${CHECK_ALL:-}" == "1" ]]; then
  echo -e "${YLW}CHECK_ALL=1 detected; running repo-wide checks...${RST}"
  exec "$repo_root/scripts/check_repo.sh"
fi

CHECK_FILES=(
  brain.py
  user_message.py
  db/user.py
  messaging.py
  db/chroma_db.py
  bt_servant.py
  config.py
  logger.py
  db/user_db.py
)

echo -e "${YLW}Running ruff on cleaned files: ${CHECK_FILES[*]}...${RST}"
ruff check "${CHECK_FILES[@]}"

echo -e "${YLW}Running pylint on cleaned files: ${CHECK_FILES[*]}...${RST}"
pylint -rn -sn "${CHECK_FILES[@]}"

echo -e "${YLW}Running mypy on cleaned files: ${CHECK_FILES[*]}...${RST}"
mypy "${CHECK_FILES[@]}"

echo -e "${YLW}Running pyright on repo...${RST}"
if command -v pyright >/dev/null 2>&1; then
  pyright
else
  echo -e "${RED}pyright not found in PATH. Install with: pip install pyright${RST}"
  exit 1
fi

echo -e "${YLW}Running tests (pytest -q)...${RST}"
pytest -q

echo -e "${GRN}Cleaned-files checks and tests passed.${RST}"
