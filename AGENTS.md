# Repository Guidelines

## Project Structure & Module Organization
- `bt_servant.py`: FastAPI entrypoint (webhook + routing).
- `brain.py`: Decision graph and message-processing pipeline.
- `messaging.py`, `user_message.py`, `utils/`: Helper modules.
- `db/`: Persistence and vector DB helpers.
- `db_loaders/`: One-off scripts for ingest (e.g., `load_bsb.py`).
- `data/`: Local data artifacts (created at runtime).
- `tests/`: Unit tests (Python files named `test_*.py`).
- `docs/`: ADRs, diagrams, and reference material.

## Build, Test, and Development Commands
- Create venv and install: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Run API locally: `uvicorn bt_servant:app --reload`
- Lint (fast): `ruff check .`
- Lint (strict, all files): `pylint $(git ls-files '*.py')`
- Type check (all files): `mypy .`
- Tests: `pytest -q` (tests live under `tests/`).

## Coding Style & Naming Conventions
- Python 3.12+, 4-space indentation, UTF-8 files.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Keep functions small and single-purpose; prefer explicit returns.
- Docstrings for public functions; keep comments minimal and useful.
- Tools: `ruff` for style, `pylint` for code hygiene, `mypy` for typing.

### Linting & Type-Checking Policy
- Always run linters and type-checkers on the entire project, not a subset:
  - `ruff check .`
  - `pylint $(git ls-files '*.py')`
  - `mypy .`
- Do not stop until all findings introduced by your change are resolved. If pre‑existing issues remain, surface them in the PR and get explicit sign‑off before merging.
- Treat a clean `pylint` and `ruff` run as a pre‑PR requirement. Aim for a clean `mypy` run; add precise annotations or adjust types where practical.
- Rationale: issues in un-touched files can be missed when running tools on a subset (e.g., a wrong return type annotation in `db/chroma_db.py` wasn’t flagged because only a few files were linted). Running repo‑wide prevents misses.

Recommended workflow
- Before committing: run `ruff`, `pylint`, and `mypy` repo‑wide. Fix or explicitly document remaining issues.
- Prefer local casts or minimal docstrings to satisfy static analysis when frameworks (e.g., Pydantic) confuse linters.

## Testing Guidelines
- Place tests in `tests/` as `test_*.py`.
- Arrange/Act/Assert structure; mock network/LLM calls.
- Run locally with `pytest -q`; aim to cover new logic paths.
- When adding loaders, test parsing and chunking separately.

## Commit & Pull Request Guidelines
- Commits: clear, imperative subject (e.g., "Add loader test", "Fix model name").
- Scope changes narrowly; keep diffs focused and self-contained.
- PRs: include description, rationale, screenshots/logs when useful, and a test plan.
- Link related issues; note any follow-ups or known limitations.

## Security & Configuration Tips
- Secrets via `.env` (see `env.example`); never commit secrets.
- Required vars include `OPENAI_API_KEY` and Meta tokens (see README).
- Prefer dependency versions from `requirements.txt` and review upgrades.
- Avoid network calls in tests; stub external clients (OpenAI, Meta).

## Agent Handoff + Persistence Policy
- Update this AGENTS.md only when there are material findings, caveats,
  decisions, or fixes that future Codex sessions need to avoid re-solving the
  same problems. If there’s nothing noteworthy, do not update this file.
- Treat this as the single source of truth for persistent handoff notes across
  sessions, but keep it lean—avoid unnecessary growth or repetition.
- Record root causes and resolutions (not just symptoms) when they would change
  how the next session should proceed.
- Prefer concise bullets with concrete file paths, commands, and error messages.
- Keep a running “Latest Session Notes” section below. Add a new dated block for
  each session only when there is something worth persisting.

## Testing Policy (Non-Negotiable)
- Tests must never fail during local runs. If any test fails or errors during
  collection, the agent must stop new feature work and fix the failure before
  proceeding. It is not acceptable to leave failing tests.
- If a test targets functionality moved to another repo or is obsolete, delete
  or rewrite the test so the suite remains green. Document the rationale here
  in AGENTS.md under the session notes.

### Latest Session Notes (2025-08-21)
- Pytest failures: root cause was a missing `db_loaders` package referenced by
  `tests/test_load_bsb.py`, causing `ModuleNotFoundError: db_loaders` during
  collection. Fixed by adding `db_loaders/__init__.py` and `db_loaders/load_bsb.py`
  with minimal implementations for `fetch_verses()` and
  `group_semantic_chunks()` used by the test.
- Test status: `pytest -q` now passes locally (`2 passed, 26 warnings`). The
  warnings include Pydantic v2 deprecations and Chroma embedding-config
  deprecations. These are pre-existing and not addressed here.
- New API endpoints added:
  - `POST /chroma/collections` to create a collection (201/409/400).
  - `DELETE /chroma/collections/{name}` to delete a collection (204/404/400).
  Backed by new helpers in `db/chroma_db.py` and tests in
  `tests/test_chroma_endpoints.py`.
