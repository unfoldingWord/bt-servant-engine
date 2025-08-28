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
- Commits: subject must be succinct and prefixed as `(CODEX) <SUCINCT SUBJECT>`.
  - Use a clear, imperative, and short subject (<= 72 chars when possible).
  - Always include a non-empty commit body that describes:
    - What changed, why, and any alternatives considered.
    - Key files or areas touched and notable design decisions.
    - Test plan (commands run, tests added/updated, results) and any manual verification.
    - Backwards-compatibility notes, migrations, or operational considerations.
    - Follow-ups or known limitations.
  - Do not leave the description blank.
- Commit body formatting (newlines must render properly):
  - Do not embed literal "\n" into commit messages with normal quotes; these show up as text.
  - Use one of the following methods to ensure real newlines:
    - Here-doc body:
      - `git commit -m "(CODEX) <SUCINCT SUBJECT>" -F- <<'MSG'`
      - `Detailed body line 1`
      - `- bullet 1`
      - `- bullet 2`
      - `MSG`
    - ANSI-C quoting for `-m`:
      - `git commit -m "(CODEX) <SUCINCT SUBJECT>" -m $'Line 1\n\n- bullet 1\n- bullet 2'`
    - Or write to a temp file and use `-F`:
      - `printf '%s\n' "Line 1" "" "- bullet 1" "- bullet 2" > /tmp/msg.txt`
      - `git commit -m "(CODEX) <SUCINCT SUBJECT>" -F /tmp/msg.txt`
- Scope changes narrowly; keep diffs focused and self-contained.
- PRs: include description, rationale, screenshots/logs when useful, and a test plan.
- Link related issues; note any follow-ups or known limitations.

## Security & Configuration Tips
- Secrets via `.env` (see `env.example`); never commit secrets.
- Required vars include `OPENAI_API_KEY` and Meta tokens (see README).
- Prefer dependency versions from `requirements.txt` and review upgrades.
- Avoid network calls in tests; stub external clients (OpenAI, Meta).

## Agent Handoff + Persistence Policy
- Maintain a concise "Things To Remember" section (below) and update it only
  when there are material findings, caveats, decisions, or fixes that future
  sessions need to avoid re-solving problems. If nothing noteworthy, do not
  edit this file.
- Treat AGENTS.md as the single source of truth for persistent handoff notes,
  but keep it lean—avoid unnecessary growth or repetition.
- Capture root causes and resolutions (not just symptoms) only when they change
  how the next session should proceed.
- Prefer concise bullets with concrete file paths, commands, and error messages.

## Testing Policy (Non-Negotiable)
- Tests must never fail during local runs. If any test fails or errors during
  collection, the agent must stop new feature work and fix the failure before
  proceeding. It is not acceptable to leave failing tests.
- If a test targets functionality moved to another repo or is obsolete, delete
  or rewrite the test so the suite remains green. Document the rationale here
  in AGENTS.md under the session notes.

## Things To Remember
- The `db_loaders` module was intentionally removed from this repo (logic lives
  in a separate repository). Do not re-add it. Any tests referencing it should
  be deleted or rewritten here.
- Admin endpoints are protected behind a simple token guard when
  `ENABLE_ADMIN_AUTH=True`. Tokens are read from `ADMIN_API_TOKEN` and accepted
  via `Authorization: Bearer <token>` or `X-Admin-Token: <token>`. When
  disabled (default), no auth is required for these endpoints.
- Commit message convention:
  - Always prefix the commit subject with `(CODEX)` and keep it succinct:
    `(CODEX) <SUCINCT SUBJECT>`.
  - Always provide a meaty, non-empty commit body detailing what changed,
    rationale, test plan, and any risks/limitations. Never leave the
    description blank.

### Typing With LangGraph + OpenAI SDK
- LangGraph `StateNode` is contravariant in the state type. PyCharm may expect
  `StateNode[Any]` at `add_node(...)` call sites. To satisfy IDE type checks,
  define node functions as `def node(state: Any) -> dict` and cast to
  `BrainState` at the top: `s = cast(BrainState, state)`. This preserves runtime
  behavior and internal typing, and avoids per-call casts or wrappers.
- OpenAI SDK typed inputs:
  - Responses API inputs should be built/annotated as
    `list[EasyInputMessageParam]`.
  - Chat Completions messages should be built/annotated as
    `list[ChatCompletionMessageParam]`.
  - Avoid using raw `list[dict[str, str]]` for these payloads; PyCharm will
    warn since it expects richer typed dicts from the SDK stubs.
  - When inference is stubborn, wrap list literals with `cast(List[...Param], [...])`
    to signal to the IDE that the union of TypedDict variants is intended.
- LangGraph schema type for `StateGraph`:
  - When using `TypedDict` for state (e.g., `BrainState`), PyCharm may warn that
    the constructor expects a `type[StateT]`. Use `StateGraph(cast(type, BrainState))`
    and annotate the builder as `StateGraph[BrainState]` to satisfy the generic.
- Enum to primitive conversions:
  - When SDK models expose enums (e.g., `resp_lang.language.value`), PyCharm
    can sometimes lose the concrete type and infer a callable union. Use
    `cast(str, ...)` or `str(...)` at the call site to make the argument type
    explicit, e.g., `set_user_response_language(cast(str, user_id), cast(str, code))`.
