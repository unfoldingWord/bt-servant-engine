# Repository Guidelines

## General
When working on a task, do not stop to ask if changes should be applied. Please simply apply the changes. There is no
need to ask for permission when editing files in the repo.

## Project Structure & Module Organization
- `bt_servant.py`: FastAPI app (Meta WhatsApp webhook) and request flow.
- `brain.py`: LangGraph pipeline: intent detection, RAG, translation, chunking.
- `db/`: TinyDB helpers and ChromaDB access (`user.py`, `chroma_db.py`).
- `db_loaders/`: Data ingestion utilities (e.g., BSB loader).
- `utils/`, `logger.py`, `config.py`: Utilities, logging, configuration.
- `knowledgebase/`: Seed content and metadata.
- `tests/`: Pytest suite (e.g., `test_load_bsb.py`).

## Build, Test, and Development Commands
- Create env and install: `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
- Run API locally: `uvicorn bt_servant:app --reload --port 8080`
- Run tests: `pytest -q`
- Docker (prod-like): `docker build -t bt-servant . && docker run -p 8080:8080 --env-file .env bt-servant`

## Coding Style & Naming Conventions
- Python 3.12, follow PEP 8 and use type hints (see `brain.py` for patterns).
- Functions: `snake_case`; classes/enums: `PascalCase`; constants: `UPPER_SNAKE_CASE`.
- Keep modules focused; prefer small, pure helpers in `utils/`.
- Logging via `logger.get_logger(__name__)`; prefer structured, informative messages.

## Testing Guidelines
- Framework: Pytest. Name tests `test_*.py` and functions `test_*`.
- Keep tests deterministic; use fixtures/monkeypatch for network/OpenAI calls (see example in `tests/test_load_bsb.py`).
- Run `pytest -q`; aim to cover new logic paths and error handling.

## Commit & Pull Request Guidelines
- Commits: concise imperative subject (max ~72 chars), explain why in body when non-trivial.
- Reference issues with `#<id>` when applicable. Group related changes; avoid noisy unrelated edits.
- PRs: clear description, rationale, testing notes, and screenshots/log excerpts when behavior changes. Link issues.

## Security & Configuration Tips
- Required env vars in `.env` (see `env.example` and `config.py`): OpenAI/Groq keys, Meta tokens, `BASE_URL`.
- Do not commit secrets. Use sandbox/test keys in CI and tests (mock external calls).
- Default data dir: `/data`; ensure writable when running in Docker or locally.
