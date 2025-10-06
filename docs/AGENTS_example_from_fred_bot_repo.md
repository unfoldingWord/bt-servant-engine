# AGENTS.md — Codex CLI Handoff & Engineering Guide (Python 3.13)

This document is the playbook for refactoring and extending **unfoldingWord/fred-zulip-bot**. It’s written for Codex CLI agents and human reviewers. Treat it as your “first day at the shop” guide.

---


## Doctrine (Non-Negotiables & Defaults)

**Principles**
- Small, reviewable PRs; keep `main` releasable; prefer clarity over cleverness.
- Boundaries: `api → domain → adapters` (no back edges).

**Policies (must)**
- Pre-commit must pass: `ruff --fix`, `ruff format`, `mypy`.
- Tests required for behavior changes; CI coverage ≥ project floor; no new flaky tests.
- Security: no secrets in repo; Bandit/pip-audit clean or justified.
- Architecture: import-linter contracts pass (or exception PR with rationale).

**Defaults (should)**
- Precise typing at boundaries (`TypedDict`, `Protocol`, `Annotated`); avoid `Any`.
- Prefer composition over inheritance; keep functions small & single-purpose.
- Log meaningful context on error paths (no PII; include correlation IDs if available).

**Escape hatches**
- If a rule blocks delivery: narrow the ignore, add an inline rationale, open a follow-up issue, and explain in the PR body.

**Escalate to human review when**
- Changing boundaries, adding external I/O, handling auth/security, or touching perf-critical code.

---

## Global Constraints & Conventions

- **Python:** 3.13 only (`requires-python >= 3.13`).
- **Formatting & Lint:** `ruff` (also as the formatter) with line length **100** and double quotes.
- **Typing:** `mypy --strict` for all code under `fred_zulip_bot/`.
- **Tests:** `pytest` + coverage gate (start at **70%**, ratchet up over time).
- **Pre-commit:** required locally and in CI (`ruff`, `mypy`, `pytest`).
- **Secrets:** No secrets in code. Use env / `.env` with `pydantic-settings`.
- **Commit Author:** Use **Codex Assistant** for bot-authored commits — **non-negotiable**.
- **Local Checks Before Every Commit (non-negotiable):** run `ruff check fred_zulip_bot tests`,
  `ruff format --check fred_zulip_bot tests`, `mypy fred_zulip_bot`, and `pytest`. Pre-commit hooks
  will fail if you forget, but you must run them manually first.
- **Messages:** **Never** commit with an empty description—subject **and** body are required.

---

## Architecture Rules (DRY, SRP, small pieces)

> Golden rule: **routes are thin, services are testable, adapters touch the outside world.**

- **Layering**
  - **apps/api/routes/** — FastAPI endpoints: validation, serialization, call services.
  - **services/** — Business logic; NO HTTP or DB specifics inside.
  - **adapters/** — Integrations (Zulip, MySQL, TinyDB, LLM). Side effects live here.
  - **core/** — Config, logging, shared Pydantic models.
  - **orchestration/** — LangGraph graph definition & node glue.

- **Dependency Injection**
  - Use `app.state.services` as a tiny service container initialized in `create_app()`.
  - Pass dependencies explicitly; **no module-level singletons**.

- **SRP & Size Limits**
  - **Class length ≤ 300 lines**; **module length ≤ 500 lines**.
  - **Cyclomatic complexity:** keep functions ≤ 10 (prefer ≤ 7). Refactor early.
  - **One reason to change:** If a function/class has more than one responsibility, split it.

- **DRY & Reuse**
  - Extract repeated logic into helpers in the correct layer (not utils dumping-ground).
  - Prefer composition over inheritance. Avoid deep class hierarchies.

- **Naming**
  - Be specific and boring: `submit_query()` → `run_readonly_select()`; `process_user_message()` → split into `determine_intent()`, `handle_chatbot()`, etc.
  - Test names describe behavior: `test_sql_guard_rejects_multi_statement()`.

---

## Code Style (Python)

- **Imports**
  - `from __future__ import annotations` where helpful.
  - Stdlib → third-party → local, separated by blank lines. No wildcard imports.
  - Prefer `pathlib.Path` to `os.path`.

- **Strings & f-strings**
  - Use **double quotes**. Use f-strings; avoid `+` concatenation.

- **Types & Docstrings**
  - Type every function signature. Avoid `Any` unless justified.
  - Docstrings explain **why**, not just **what**. Follow short Google-style docstrings.

- **Errors & Exceptions**
  - Don’t catch broad `Exception` unless re-raising with context.
  - Define small custom exceptions at boundaries (e.g., `SqlSafetyError`).

- **Logging**
  - No `print()`. Use `core.logging.get_logger(name)`.
  - Log key=value; never log secrets or full SQL statements. Redact user emails if needed.

- **Concurrency**
  - If endpoint is `async`, use async clients (`httpx.AsyncClient`, `aiomysql`); otherwise keep sync consistently. Don’t mix without reason.
  - Never block the event loop with long CPU work; consider background tasks if needed.

---

## FastAPI Rules (Ingress)

- `apps/api/app.py` exposes `create_app()`; routers register via `register_*_routes(app)`.
- Route handlers:
  - Validate with Pydantic request models.
  - Call a service method.
  - Map domain errors to `HTTPException(status_code, detail)`.
- Health routes: `/healthz` and `/ready` return simple JSON payloads.
- No business logic in routes; keep handlers ≤ ~25 lines.

---

## Adapters (Zulip, MySQL, History, LLM)

- **Zulip (`ZulipClient`)**
  - Wrap outbound HTTP; set timeouts and retries (e.g., 2–3 attempts, jittered backoff).
  - Log request intent and status only; never log tokens.
- **MySQL (`MySqlClient`)**
  - **Read-only** only. Enforce at the adapter (reject anything not `SELECT`).
  - Parameterize queries; block multi-statements; deny dangerous tokens (see SQL Safety).
  - Prefer connection pooling; set sane timeouts.
- **History (`TinyDBHistoryRepository`)**
  - Persist chat histories in TinyDB keyed by user email; the filesystem adapter has been removed.
- **LLM (Gemini adapter)**
  - Single place for prompts, model name, timeouts, retries.
  - No raw SDK calls from services; all via adapter methods.

---

## Orchestration (LangGraph)

- Minimal graph: `classify_intent → {chatbot|other|database}`.
- Nodes are **thin** and call services; **no** heavy logic in node functions.
- Feature flag allows bypassing LangGraph if needed during debugging.

---

## SQL Safety (hard rules)

- Allow only **single `SELECT`** statements.
- Reject queries containing any of: `;`, `--`, `/* */`, `INTO OUTFILE`, `LOAD DATA`,
  `DROP`, `ALTER`, `INSERT`, `UPDATE`, `DELETE`, `TRUNCATE`, `CALL`, `CREATE`, `GRANT`, `REVOKE`.
- Unit test every rule (positive & negative cases).

---

## Testing Strategy

- **Unit tests** for services and SQL guards (mock adapters).
- **Route tests** using FastAPI `TestClient` (or async client if async stack).
- **Golden tests** (optional) for prompt assembly if present.
- **Fakes over mocks** when easier (e.g., in-memory history repo).
- **Coverage** starts at 70% on touched files; raise by 5% after stability.

**Fixtures & Structure**
```
tests/
  conftest.py            # app factory/fixtures
  test_routes_chat.py
  test_chat_flow.py
  test_sql_safety.py
```
- Name tests by behavior, not method names.

---

## Configuration & Secrets

- Centralize in `core/config.py` using `pydantic-settings`.
- Load from env / `.env`. Never read secrets from code defaults.
- All external clients (`ZulipClient`, `MySqlClient`, LLM) receive config via DI.
- History storage uses TinyDB only. Configure the path via `HISTORY_DB_PATH` and adjust
  `HISTORY_MAX_LENGTH` to cap retained turns.

---

## Observability

- Structured logging with request IDs where possible.
- Log intent classification and timings (LLM latency, DB latency).
- Keep logs concise; no PII/secrets.

---

## Commit & PR Policy (sourced from BT Servant & Zion)

- **Author:** Use **Codex Assistant** for all bot-created commits.
- **Subject:** Short, imperative, meaningful. Example: `refactor: split chat flow into services`.
- **Body (non-empty required):** Explain **what changed**, **why**, and the **test plan**.
- **Formatting tips:** Use a commit message file (`-F`) or here-doc so multiline bodies render.
- **Discipline:** Don’t commit while local format/lint/type/tests are failing.
- **PR size:** Aim for ≤ ~400 changed lines. Keep PRs focused and reversible.
- **Linkage:** Reference issues and include migration notes when applicable.
- **Merge:** Prefer squash-merge with a clean message.

**Template**
```
refactor: extract chat_service and intent_service (no behavior change)

- move route logic into services; keep endpoints thin
- add tinydb history repo interface (not yet default)
- add tests for sql safety and intent branching

test plan:
- pytest -q (passes)
- curl POST /chat (same responses on sample inputs)
```

---

## Local Dev & CI Commands

```bash
# One-time
pre-commit install

> **Non-negotiable:** Every new Codex session starts with a fresh `.git/hooks` directory. Run `pre-commit install` (or `.venv/bin/pre-commit install` if you prefer the explicit venv path) as the very first setup step so formatting, lint, and mypy hooks always fire before commits.

# Full checks
ruff check . && ruff format --check . && mypy fred_zulip_bot && pytest -q
bandit -q -r fred_zulip_bot
pip-audit --ignore-vuln GHSA-f96h-pmfr-66vw --ignore-vuln GHSA-2c2j-9gv5-cj73 --ignore-vuln GHSA-4xh5-x5gv-qwph

# Run dev server
uvicorn fred_zulip_bot.apps.api.app:create_app --factory --reload --port 8000
```

---

## Agent Roles & Tasks

### Planner
Own the phased roadmap and keep PRs small with rollbacks.

### CodeMod
Apply structural changes without altering runtime behavior. Reduce large functions until they
pass Ruff's `PLR0915` check (≤ 50 executable statements) while keeping behavior unchanged.

### TestBuilder
Create/maintain tests; prefer fakes/mocks at boundaries; enforce coverage.

### CI/Toolsmith
Own `pyproject.toml`, pre-commit, and GitHub Actions; keep gates green.

### Migrationist
Migrate JSON history → TinyDB; keep fallback; provide idempotent script.

### Orchestrator
Wire the minimal LangGraph; ensure nodes remain thin and call services.

---

## Definition of Done (per PR)

- Code follows style rules (DRY, SRP, size limits).
- `ruff`, `mypy`, `pytest` pass locally and in CI.
- Coverage on changed files ≥ threshold.
- No secrets in diffs; config via `core/config.py` only.
- Commit message with **non-empty body**; PR description includes “what/why/test plan”.

---

## Required Reading (Modern Classics & How to Apply Here)

These titles encode the “why” behind our guardrails. Agents should lift concrete practices into
plans, tests, and PR notes. Human reviewers can use the bullets as review heuristics.

**Python craft (daily impact)**
- *Effective Python (3e, 2024) — Slatkin* — Prefer built-ins and idioms Ruff already nudges
  (iterators, context managers, dataclass/slots when appropriate). Make small, composable
  functions; use precise types (TypedDict, Protocol, Annotated) at boundaries. When refactoring,
  consult relevant “Items” and note which ones guided the change in the PR body.
- *Fluent Python (2e, 2022) — Ramalho* — Embrace the data model (dunder methods) for clarity; use
  iterators/generators over index loops; lean on protocols and structural typing to decouple
  services from adapters. Prefer collections.abc interfaces in signatures.
- *Architecture Patterns with Python (“Cosmic Python”, 2020) — Percival & Gregory* — Keep routes
  thin, put business rules in services/domain, talk to the world via adapters (ports & adapters).
  Model workflows explicitly (e.g., LangGraph nodes call services, not vice-versa). Capture
  decisions in tests.
- *Python Testing with pytest (2e, 2022) — Okken* — Use fixtures/parametrization to express
  behavior, not mechanics. For each change: one happy path, one edge path, one error path. Prefer
  fakes over mocks for adapters. Keep `--cov-fail-under` green and raise slowly.
- *High Performance Python (2e, 2020) — Gorelick & Ozsvald* — Profile before optimizing. Prefer
  vectorized/streaming approaches for data transforms; avoid accidental N+1 I/O in adapters;
  measure with timers in observability.
- *Python Concurrency with asyncio (2022) — M. Fowler* — If an endpoint is async, keep the chain
  async (http/db clients). Don’t block the loop; offload CPU work; set timeouts and backoffs in
  adapters.

**Architecture, delivery, and team flow**
- *Accelerate — Forsgren, Humble, Kim* — Treat our CI gates as fitness functions for lead time and
  change fail rate. Keep PRs small; keep the main branch always releasable. If a change increases
  MTTR risk, add observability and rollback notes.
- *Software Engineering at Google — Winters, Manshreck, Wright* — Bias to readability and
  long-term maintainability. Enforce style mechanically (Ruff/formatter), keep code review focused
  on design and clarity. Prefer clear interfaces over cleverness.
- *Team Topologies — Skelton & Pais* — Reflect team boundaries in code boundaries. Use import-linter
  contracts to keep api → domain → adapters flow clean; raise a ticket if work crosses multiple
  “streams.”
- *Building Evolutionary Architectures (2e) — Ford, Parsons, Kua* — Promote non-functional goals to
  first-class tests: lint/type/test/security gates, basic latency budgets, error-rate SLOs. When
  adding a new quality goal, encode it in CI (a new check) rather than tribal knowledge.
- *Software Architecture: The Hard Parts — Ford et al.* — Be explicit about trade-offs (consistency
  vs. autonomy, sync vs. async). In PRs that change boundaries, include a short “trade-off” note
  and why this repo tilts that way.
- *Fundamentals of Software Architecture — Richards & Ford* — Make coupling/cohesion visible.
  Prefer stable interfaces and unstable implementations; when churn grows, extract modules/services
  rather than adding feature flags everywhere.
- *A Philosophy of Software Design (2e) — Ousterhout* — Create deep modules that eliminate
  complexity for callers. If a function/class leaks decisions outward, refactor until the interface
  is simpler than the implementation.

**Design, refactoring, and craft**
- *Refactoring (2e) — Fowler* — Small, behavior-preserving steps with tests. When reducing
  complexity (e.g., failing PLR0915 limits), list the named refactorings used (Extract Function,
  Introduce Parameter Object, etc.) in the PR body.
- *The Pragmatic Programmer (20th Anniversary) — Hunt & Thomas* — Automate the boring
  (pre-commit, formatters), orthogonalize concerns (services vs. adapters), and leave a “tracer
  bullet” (minimal vertical slice) before scaling up a feature).
