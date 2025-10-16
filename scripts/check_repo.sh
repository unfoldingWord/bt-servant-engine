#!/usr/bin/env bash
set -euo pipefail

# Run full repo checks: format, lint, type, security, deps, and tests.
# Usage: scripts/check_repo.sh

RED=\033[31m
GRN=\033[32m
YLW=\033[33m
RST=\033[0m

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$repo_root"

export PYTHONPATH="$repo_root:${PYTHONPATH:-}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo -e "${YLW}Running ruff format (check only)...${RST}"
ruff format --check bt_servant_engine

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

echo -e "${YLW}Running import linter (lint-imports)...${RST}"
if command -v lint-imports >/dev/null 2>&1; then
  lint-imports
else
  echo -e "${RED}lint-imports not found in PATH. Install with: pip install import-linter${RST}"
  exit 1
fi

echo -e "${YLW}Running bandit (security scan)...${RST}"
bandit -q -r bt_servant_engine

echo -e "${YLW}Running pip-audit (supply chain)...${RST}"
if command -v pip-audit >/dev/null 2>&1; then
  pip-audit -r requirements.txt
else
  echo -e "${RED}pip-audit not found in PATH. Install with: pip install pip-audit${RST}"
  exit 1
fi

echo -e "${YLW}Running deptry (dependency hygiene)...${RST}"
if command -v deptry >/dev/null 2>&1; then
  deptry .
else
  echo -e "${RED}deptry not found in PATH. Install with: pip install deptry${RST}"
  exit 1
fi

echo -e "${YLW}Running tests (pytest with coverage)...${RST}"
# Provide required environment defaults for settings validation.
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-test}" \
  META_VERIFY_TOKEN="${META_VERIFY_TOKEN:-verifytoken}" \
  META_WHATSAPP_TOKEN="${META_WHATSAPP_TOKEN:-whatsapptoken}" \
  META_PHONE_NUMBER_ID="${META_PHONE_NUMBER_ID:-1234567890}" \
  META_APP_SECRET="${META_APP_SECRET:-metasecret}" \
  LOG_PSEUDONYM_SECRET="${LOG_PSEUDONYM_SECRET:-test-secret}" \
  FACEBOOK_USER_AGENT="${FACEBOOK_USER_AGENT:-facebookexternalua}" \
  BASE_URL="${BASE_URL:-https://example.invalid}" \
  META_SANDBOX_PHONE_NUMBER="${META_SANDBOX_PHONE_NUMBER:-15555555555}" \
  DATA_DIR="${DATA_DIR:-./data}" \
  ENABLE_ADMIN_AUTH="${ENABLE_ADMIN_AUTH:-False}" \
  ADMIN_API_TOKEN="${ADMIN_API_TOKEN:-admin-token}" \
  HEALTHCHECK_API_TOKEN="${HEALTHCHECK_API_TOKEN:-health-token}"
mkdir -p "${DATA_DIR}"
# Exclude OpenAI-costly tests by default; run them on-demand only.
pytest --maxfail=1 --disable-warnings -q -m "not openai" \
  --cov=bt_servant_engine --cov-report=term-missing --cov-fail-under=65

echo -e "${GRN}Repo checks and tests passed.${RST}"
