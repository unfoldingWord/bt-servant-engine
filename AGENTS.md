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
