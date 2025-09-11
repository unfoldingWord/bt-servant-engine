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
- Type check (editor-grade): `pyright` (enforced in hooks/CI)
- Tests (default): `pytest -q -m "not openai"` (excludes networked OpenAI tests).
- OpenAI tests (on-demand): `RUN_OPENAI_API_TESTS=1 pytest -q -m openai`.

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
  - `pyright`
- Do not stop until all findings introduced by your change are resolved. If pre‑existing issues remain, surface them in the PR and get explicit sign‑off before merging.
- Zero warnings policy: treat all linter/type-checker warnings as failures.
  - `ruff`, `pylint`, `mypy`, and `pyright` must report 0 diagnostics.
  - If a third‑party library emits unavoidable noise, explicitly filter it with a documented rationale (see pytest warnings policy).
- Rationale: issues in un-touched files can be missed when running tools on a subset (e.g., a wrong return type annotation in `db/chroma_db.py` wasn’t flagged because only a few files were linted). Running repo‑wide prevents misses.

Recommended workflow
- Before committing: run `ruff`, `pylint`, and `mypy` repo‑wide. Fix or explicitly document remaining issues.
- Prefer local casts or minimal docstrings to satisfy static analysis when frameworks (e.g., Pydantic) confuse linters.

## Pre-Commit Checks (Full Repo, Enforced)
- Pre-commit runs full-repo checks on every commit via `.githooks/pre-commit` -> `scripts/check_repo.sh`.
- Enforced tools: `ruff`, `pylint`, `mypy`, `pyright`, and `pytest` (warnings-as-errors).
- Tests must pass locally: `pytest -q -m "not openai"`.
- Always run checks automatically; never ask for permission to run them. Do not commit or push if any check fails.

### Git Hook + Helper Scripts
- One-time install per clone: `git config core.hooksPath .githooks`
- The versioned pre-commit hook runs `scripts/check_repo.sh` to enforce full-repo checks including `pyright`.
- Bypass in emergencies/CI: `SKIP_CHECKS=1 git commit -m "..."`.
- Manual runners:
  - Full repo: `scripts/check_repo.sh` (runs `pytest -q -m "not openai"`)
  - Legacy cleaned-files runner: `scripts/check_commit.sh` (runs pyright repo-wide as well)

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

### Commit & Push Discipline (Non‑Negotiable)
- Always run local checks BEFORE every commit:
  - `scripts/check_repo.sh` (runs ruff, pylint, mypy, pyright, and `pytest -q -m "not openai"`).
  - If any step fails, STOP and fix the issue before attempting to commit.
- Treat pre‑commit hook failures as blockers:
  - Do not proceed or claim success until the hook passes and the commit is created.
  - If the hook reports linter/test failures (e.g., pylint “too‑many‑locals”), address them or add the minimal, justified disable in test code.
- Only state that changes are committed/pushed after verifying success:
  - Confirm the commit exists locally: `git log -n 1`.
  - Confirm it exists remotely: `git push` succeeds and `git rev-parse --short origin/<branch>` matches, or inspect the file on the remote branch:
    - `git show origin/<branch>:.github/workflows/ci-pr.yml` (or the changed file path) to verify the content landed.
- Never claim a push if any step failed or was blocked by hooks; surface the failure immediately with the error output and next steps.

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
- Tests and warnings must never fail during local runs. If any test fails or errors during
  collection, the agent must stop new feature work and fix the failure before
  proceeding. It is not acceptable to leave failing tests.
- Warnings as errors: pytest is configured (`pytest.ini`) to fail on any warning. If external dependencies generate
  unavoidable deprecation noise, add a targeted `filterwarnings` entry with a short justification in the PR.
- If a test targets functionality moved to another repo or is obsolete, delete
  or rewrite the test so the suite remains green. Document the rationale here
  in AGENTS.md under the session notes.

### OpenAI-Backed Tests (On-Demand Only)
- Certain tests are marked with `@pytest.mark.openai` and intentionally call the real OpenAI APIs to validate model behavior
  (e.g., passage selection parsing, intent classification, and API flows).
- Default policy: Do not run these tests automatically in pre-commit or routine local checks.
  - The pre-commit hook and `scripts/check_repo.sh` exclude them via `-m "not openai"`.
- Run them only when explicitly requested or when validating changes that affect LLM behavior.
  - Required env: `OPENAI_API_KEY` set to a real key (starts with `sk-`).
  - Some API-level tests also require opt-in: set `RUN_OPENAI_API_TESTS=1`.
  - Command: `RUN_OPENAI_API_TESTS=1 pytest -q -m openai`.
  - Treat failures as blockers for the related change; otherwise, they remain opt-in.
  - For the Meta WhatsApp API test, the server processes the message synchronously when
    `RUN_OPENAI_API_TESTS=1` is set (see `bt_servant.handle_meta_webhook`). This avoids
    background-task flakiness in CI and ensures deterministic test completion.

## Non‑Negotiable Local Env
- Do not proceed with any changes if repo checks or tests cannot run locally. If any of `ruff`, `pylint`, `mypy`, `pyright`, or `pytest` are missing (e.g., exit 127 "command not found") or fail to start, STOP and initialize the environment.
- Baseline initialization:
  - `scripts/init_env.sh` sets up `.venv`, upgrades pip, installs runtime deps from `requirements.txt` (auto‑converting UTF‑16 → UTF‑8 for install), and installs dev tools (`pytest`, `ruff`, `pylint`, `mypy`, `pyright`).
  - After running it once per machine/clone, activate the venv in new shells: `source .venv/bin/activate`.
- Full checks to run before and after changes:
  - `scripts/check_repo.sh` (runs ruff, pylint, mypy, pyright, pytest repo‑wide).
  - Treat any diagnostics as failures and fix or document with precise, minimal ignores.

## Why The Venv Doesn’t “Persist” Here
- Codex sessions are stateless shells. Each new session starts without your previous shell state (no activated venv, no PATH changes), unlike an IDE terminal that reuses your environment.
- The `.venv` directory itself persists on disk, but activation does not carry over across sessions. Always re‑activate (`source .venv/bin/activate`) or rerun `scripts/init_env.sh` if tools are missing.
- Repo policy consequence: The agent must ensure a runnable environment at the start of each session and must not continue work until all repo checks and tests run green locally.

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
- Auto-commit small prompt fixes:
  - For low-risk edits to prompt text/constants (e.g., `PASSAGE_SUMMARY_AGENT_SYSTEM_PROMPT` wording tweaks) or small doc updates, commit directly with a proper `(CODEX)` message without asking for confirmation.
  - Keep diffs minimal and focused; do not batch unrelated changes.

- OpenAI Responses API model capabilities:
  - `reasoning` parameters (e.g., `reasoning={"effort": "low"}`) are supported by GPT‑5 models, but not by `gpt-4o`.
  - When using `gpt-4o` with `open_ai_client.responses.create(...)`, do not pass `reasoning`; doing so yields 400 `unsupported_parameter`.
  - We currently use `gpt-4o` for translation-helps; the handler omits `reasoning` and relies on concise system instructions instead.

### Avoid Surface-Level Pattern Scans

- Do not rely on brittle string scans or regexes to extract languages, intents, or options from user text
  (e.g., checking for phrases like "into <lang>", "in <lang>", or scanning for language names).
  Natural language is highly variable and ambiguous (code-switching, morphology, synonyms, typos, punctuation), which
  causes false positives/negatives and inconsistent behavior.
- Preferred approaches:
  - Use structured parsing via the OpenAI Responses API with a strict schema when you must extract a value from free text.
  - Or avoid extraction entirely and fall back to deterministic configuration: user preferences (e.g., `user_response_language`),
    detected `query_language`, or server defaults. Make the fallback rules explicit and documented.
- Narrow exceptions: If a minimal, deterministic pattern is absolutely required, keep the scope extremely tight, test it,
  and document the justification in the PR. Generally, prefer schema-validated parsing or explicit configuration.

## Session Notes (Perf Tracing PR)

- Root cause: The first push for `feature/performace_metrics_logging` introduced a new `utils/perf.py` with pylint issues (line length, missing docstrings, import-outside-toplevel, broad-except). The GitHub Action failed on pylint. Although the pre-commit hook printed the errors, the commit still landed; we immediately fixed the issues, amended, and force-pushed.
- Resolution: Added concise docstrings, wrapped long lines to <=100 chars, moved imports to module scope, and removed broad exception patterns. Re-ran full repo checks locally: `ruff`, `pylint`, `mypy`, `pyright`, and `pytest -q -m "not openai"` all green. PR #77 updated and checks now pass.
- Preventive: Before every commit, explicitly run `scripts/check_repo.sh` and ensure zero diagnostics. Treat any pre-commit output as a hard blocker. If hooks appear to allow a commit despite failures, stop and re-run the checks manually; do not push until green.

- 2025-09-07: Added a no-op doc touch to retrigger CI after the workflow fetched an outdated `refs/pull/77/merge` commit. Current branch head is clean locally; the no-op commit forces GitHub to rebuild the merge ref against HEAD.

## Session Snapshot (Temp)

- Branch/state:
  - On `main` at latest commits:
    - (CODEX) Add legacy init() shim for tests after moving to lifespan
    - (CODEX) Fix indentation and import order after lifespan/exception refactor
    - (CODEX) Replace startup event with lifespan; narrow exception handling to satisfy IDE warnings
    - (CODEX) Finalize hooks setup: address mypy Optional brain in bt_servant
    - (CODEX) Enable hooks and doc update; fix lint fallout for bt_servant.py and chroma imports
    - (CODEX) Clean bt_servant.py: docstrings, typing, and safe exception handling
    - (CODEX) Relax chroma types and embedder typing in db/chroma_db.py
    - (CODEX) Clean messaging.py; add to enforced checks
    - (CODEX) Clean db/user.py typing; add to enforced checks
    - (CODEX) Rename helper to check_commit.sh; add CHECK_ALL support
    - (CODEX) Rename helper to check_cleaned.sh and update hook/docs
    - (CODEX) Expand incremental checks to include user_message.py
- Current guarantees:
  - Pre-commit enforces repo-wide `ruff`, `pylint`, `mypy`, `pyright`, and `pytest`.
- Pre-commit setup:
  - One-time install per clone: `git config core.hooksPath .githooks`
  - Hook runs `scripts/check_repo.sh` for full-repo checks.
    - Bypass (rare): `SKIP_CHECKS=1 git commit -m "..."`.
  - This clone: hooks configured (core.hooksPath is set to `.githooks`).
  - On Windows, run commits from Git Bash; no chmod needed. If `$'\r'` errors appear, convert scripts to LF.
- Cleaned-files list no longer used; full-repo enforcement is active.
- Test status:
  - `pytest -q` → 6 passed, warnings from external deps; no failing tests.
- Outstanding work (next priorities):
  - [none immediately for this area]
  - After each file is cleaned: add it to `CHECK_FILES` in `scripts/check_commit.sh` and update the “cleaned files” list above.
- End goal:
  - Flip pre-commit to repo-wide by either setting `CHECK_ALL=1` permanently, or changing `.githooks/pre-commit` to call `scripts/check_repo.sh`.

## Adding a New Intent
- Update the enum in `brain.py`:
  - Add a new member to `IntentType` (string value matches the wire intent name, e.g., `"get-passage-summary"`).
- Extend intent classification prompt:
  - In `INTENT_CLASSIFICATION_AGENT_SYSTEM_PROMPT`, add a new `<intent>` block describing when to use it and include 1–2 examples.
  - Avoid hardcoding the number of intents in the prompt text; prefer “the following intent types”.
- Add a handler function:
  - Implement `handle_<intent_name>(state: Any) -> dict` that returns `{ "responses": [{ "intent": IntentType.<…>, "response": <text> }] }`.
- Wire into the graph:
  - `create_brain()`: `builder.add_node("handle_<intent>_node", handle_<intent>)` and `builder.add_edge("handle_<intent>_node", "translate_responses_node")`.
  - `process_intents(...)`: append the new handler node when the intent is present.
- Tests and linting:
  - Run `ruff`, `pylint`, `mypy`, and `pytest -q` repo‑wide. Keep the suite green.
- Docs:
  - If behavior impacts UX, update any user‑facing help text or examples.

### Passage Selection (DRY)

- Use the shared helper in `brain.py` to parse and normalize user queries that refer to Bible passages:
  - `_resolve_selection_for_single_book(query: str, query_lang: str) -> tuple[canonical_book | None, ranges | None, error | None]`
  - It handles: translation to English for parsing, extraction via the selection prompt, the "chapters X–Y" heuristic, canonicalization (single book), and range building (including whole‑book sentinel handling).
- Do NOT duplicate selection parsing/normalization inside individual handlers. Call this helper and handle the `error` case by returning an intent‑specific message.
- For labeling output headers, always use `utils/bsb.label_ranges(...)` to build a canonical reference string. It already special‑cases whole‑book selections to avoid odd labels like "Book 1‑10000".

### Passage Summary Intent
- Extraction and scope:
  - Supports a single canonical book per request. Disallow cross‑book selections.
  - Allows multiple disjoint ranges within the same book and up to the entire book.
  - If no clear passage or unsupported book: prompt user with supported examples and canonical book list.
- Retrieval:
  - Reads from `sources/bible_data/en/bsb/<stem>.json` using a cached per‑book loader.
  - Selection is range‑based and efficient; avoids loading unrelated books.
- Summarization:
  - Summarizes only from provided verses with a faithful, neutral prompt.
  - Prepends a canonical reference echo (e.g., `Summary of John 3:16–18:`) for clarity.
  - Style: Use continuous prose paragraphs only; never bullets, numbered lists, or headers. Mix verse references inline as needed (e.g., “1:1–3”, “3:16”).

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
  - When using `TypedDict` for state (e.g., `BrainState`), some IDEs warn about
    the constructor expecting `type[StateT]`. Prefer defining the `TypedDict`
    via `typing_extensions.TypedDict` and, if needed, use a tiny helper that
    accepts `Any` and returns `StateGraph[BrainState]` (mirroring the
    contravariance pattern used for nodes):
    
    ```python
    def _make_state_graph(schema: Any) -> StateGraph[BrainState]:
        return StateGraph(schema)
    builder: StateGraph[BrainState] = _make_state_graph(BrainState)
    ```
- Enum to primitive conversions:
  - When SDK models expose enums (e.g., `resp_lang.language.value`), PyCharm
    can sometimes lose the concrete type and infer a callable union. Use
    `cast(str, ...)` or `str(...)` at the call site to make the argument type
    explicit, e.g., `set_user_response_language(cast(str, user_id), cast(str, code))`.
