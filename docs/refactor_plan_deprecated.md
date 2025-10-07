# Refactor Plan — `unfoldingWord/bt-servant-engine` (Final Blueprint, Strict Onion/Hexagonal)

**Objective:** Bring `bt-servant-engine` into alignment with the refined “onion/hexagonal/clean” architecture used in **fred-zulip-bot**, strengthen fitness functions (quality gates), and break up intent handling into small, testable units. This version adopts **strict** dependency direction: **infrastructure (adapters) depends inward on ports/domain; application/services never import adapters directly**.

> **Scope of deliverables (checked into git):**
> - `.importlinter` — dependency-direction contracts (strict onion).
> - `.pre-commit-config.yaml` — local fast checks on **commit**; heavier checks on **push**.
> - `pyproject.toml` — Ruff/Lint/Mypy/Pylint/McCabe/Deptry config.
> - CI workflows — invoke the same checks (Ruff/Mypy/Pyright/Import Linter/Bandit/Pip-audit/Deptry/Pytest+Coverage).
> - `docs/refactor_plan.md` — this document.
> - `AGENTS.md` — updated doctrine + fitness functions (separate patch).

---

## 0) Architectural Tenets (strict)

- **Layering (outer → inner):**

  1. `apps/api` — delivery: FastAPI app + routers (webhooks, admin, health). **Thin only.**
  2. `adapters` — infrastructure: DB (TinyDB/Chroma), HTTP/SDK clients (OpenAI/Meta). **Implements ports.**
  3. `services/intents` — application use cases (one file per intent).
  4. `services` — shared application services, orchestration, auth guards.
  5. `core` — domain DTOs, config, logging, **ports (interfaces/Protocols)**, small utilities.

- **Dependency rule:** arrows point **inward only**. Routes → services; services → **ports**; adapters → **ports/core**. **Services must NOT import adapters.**
- **Ingress discipline:** routes normalize input, call an intent router, which dispatches to small intent handlers (≤ 50 statements). IO is behind adapters via ports.
- **Observability:** structured JSON logs with correlation IDs; no secrets/PII in logs.
- **Small, safe PRs:** behavior-preserving refactors, staged below.

---

## 1) Target Layout (post-refactor; folders may be created incrementally)

```
bt_servant_engine/
  apps/api/app.py                   # create_app(), register_routes()
  apps/api/routes/health.py
  apps/api/routes/admin.py
  apps/api/routes/webhooks.py

  services/intent_router.py
  services/intents/merge_docs.py
  services/intents/status.py
  services/auth_service.py          # token guards, etc.
  services/chroma_service.py        # app-level orchestration (uses ports)

  adapters/chroma_client.py         # implements ports defined in core/ports.py
  adapters/tinydb_client.py
  adapters/openai_client.py
  adapters/meta_client.py

  core/config.py                    # pydantic-settings
  core/logging.py                   # logger factory (JSON; corr_id helpers)
  core/models.py                    # DTOs used across layers
  core/ports.py                     # Protocol interfaces for adapters

  tests/api/
  tests/unit/

  docs/refactor_plan.md             # THIS file
  .importlinter                     # contracts (strict onion)  <-- referenced in §2
  .pre-commit-config.yaml           # hooks (commit/push)       <-- referenced in §3
  pyproject.toml                    # lint/type/dep config      <-- referenced in §4 & §5
```

`bt_servant.py` becomes a **thin shim** that exposes the app factory and contains **no routes**.

---

## 2) Import Linter (strict onion/hexagonal contracts) — **`.importlinter`**

**Commit this file at the repo root** (referenced by pre-commit & CI).

```ini
[importlinter]
root_package = bt_servant_engine

[contract:layers]
name = Onion layering (strict)
type = layers
layers =
    bt_servant_engine.apps.api
    bt_servant_engine.adapters
    bt_servant_engine.services.intents
    bt_servant_engine.services
    bt_servant_engine.core

[contract:no_api_to_adapters]
name = Routes must not import adapters directly
type = forbidden
source_modules =
    bt_servant_engine.apps.api
forbidden_modules =
    bt_servant_engine.adapters

[contract:no_services_to_adapters]
name = Services must not import adapters
type = forbidden
source_modules =
    bt_servant_engine.services
    bt_servant_engine.services.intents
forbidden_modules =
    bt_servant_engine.adapters
```

**Effect:** DB/HTTP/etc. lives in `adapters/` (outer ring) and depends inward on `core` ports; services never import adapters; routes can’t call adapters directly.

---

## 3) Pre-commit hooks — **`.pre-commit-config.yaml`** (mirrors fred-bot + adds Deptry)

```yaml
repos:
  - repo: local
    hooks:
      # Fast editors: run on every commit
      - id: ruff-lint
        name: ruff (lint)
        entry: ruff check .
        language: system
        pass_filenames: false
        stages: [commit]

      - id: ruff-format
        name: ruff (format)
        entry: ruff format .
        language: system
        pass_filenames: false
        stages: [commit]

      - id: mypy
        name: mypy (types)
        entry: mypy bt_servant_engine
        language: system
        pass_filenames: false
        stages: [commit]

      - id: pyright
        name: pyright (types)
        entry: pyright
        language: system
        pass_filenames: false
        stages: [commit]

      - id: import-linter
        name: import-linter (onion architecture)
        entry: lint-imports
        language: system
        pass_filenames: false
        stages: [commit]

      # Security, dependency hygiene, tests: run on push
      - id: bandit
        name: bandit (security)
        entry: bandit -q -r bt_servant_engine
        language: system
        pass_filenames: false
        stages: [push]

      - id: pip-audit
        name: pip-audit (supply chain)
        entry: pip-audit
        language: system
        pass_filenames: false
        stages: [push]

      - id: deptry
        name: deptry (dependency hygiene)
        entry: deptry .
        language: system
        pass_filenames: false
        stages: [push]

      - id: pytest
        name: pytest (with coverage threshold)
        entry: pytest --maxfail=1 --disable-warnings -q --cov=bt_servant_engine --cov-report=term-missing --cov-fail-under=70
        language: system
        pass_filenames: false
        stages: [push]
```

> Dev quickstart: `pip install pre-commit && pre-commit install && pre-commit install --hook-type pre-push`

---

## 4) Lint/Type/Complexity caps — **`pyproject.toml`** (Ruff + Pylint + McCabe)

```toml
[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E","F","W","I","UP","B","PL","C90","RUF"]

[tool.ruff.lint.pylint]
max-statements = 50   # PLR0915
max-branches  = 12    # PLR0912
max-returns   = 6     # PLR0911
max-args      = 5     # PLR0913

[tool.ruff.lint.mccabe]
max-complexity = 10   # C901
```

(If also using Pylint proper, mirror caps under `[tool.pylint.design]`.)

---

## 5) Dependency hygiene — **`pyproject.toml`** (Deptry config)

Deptry prevents **unused**, **missing**, **transitive misuse**, and **mismatched section** dependencies. Start strict and add ignores only as needed.

```toml
[tool.deptry]
# Fail on key classes of errors
ignore_notebooks = true
exclude = ["venv", ".venv", "build", "dist"]
pep621_metadata = true  # read project deps from pyproject

# Adjust per your package layout
known_first_party = ["bt_servant_engine"]

# Rule toggles (default all true; keep strict)
scan_ignored = false

# Fine-grained ignores example (uncomment when needed)
# [tool.deptry.per_rule_ignores]
# # e.g., allow 'typing-extensions' unused on py312 (if used only for type checking)
# DEP001 = ["typing-extensions"]  # unused
# DEP002 = []                     # missing
# DEP003 = []                     # transitive
# DEP004 = []                     # miscategorized (dev vs prod)
```

---

## 6) CI upgrades (PR & post-merge) — **references to files above**

- Uses **`.importlinter`** via `lint-imports`.
- Uses **`.pre-commit-config.yaml`** locally; CI re-invokes the *same tools* explicitly.
- Reads ruff/mccabe/pylint caps and deptry settings from **`pyproject.toml`**.

Example CI block:
```yaml
- name: Lint (ruff)
  run: ruff check .
- name: Format (ruff)
  run: ruff format --check .
- name: Types (mypy)
  run: mypy bt_servant_engine
- name: Types (pyright)
  run: pyright
- name: Architecture (Import Linter)
  run: lint-imports
- name: Security (bandit)
  run: bandit -q -r bt_servant_engine
- name: Supply chain (pip-audit)
  run: pip-audit
- name: Dependency hygiene (deptry)
  run: deptry .
- name: Tests (pytest + coverage)
  run: pytest --maxfail=1 --disable-warnings -q --cov=bt_servant_engine --cov-report=term-missing --cov-fail-under=70
```

---

## 7) Intent handling (break up `brain.py`)

- **Router (thin):** `apps/api/routes/webhooks.py` reads request, normalizes payload, calls intent router.
- **Router core:** `services/intent_router.py` owns `parse(event) -> intent` and `dispatch(intent, event, services)`.
- **Handlers:** `services/intents/<intent>.py` exposes `async def handle(event, services)`. Keep each ≤ 50 statements; push IO to adapters via **ports**.
- **Shared services:** auth guards, orchestration helpers in `services/` (no adapter imports; only **ports**).

---

## 8) Ports (interfaces) in `core/ports.py`

`typing.Protocol` interfaces define seams; adapters implement them; services accept them via DI.

```py
from typing import Protocol, Any, Iterable

class ChromaPort(Protocol):
    async def upsert(self, items: Iterable[dict]) -> None: ...
    async def query(self, text: str, k: int = 5) -> list[dict]: ...
    async def delete_collection(self, name: str) -> None: ...

class TinyDBPort(Protocol):
    def insert(self, item: dict) -> int: ...
    def search(self, query: Any) -> list[dict]: ...
    def remove(self, cond: Any) -> int: ...

class OpenAIPort(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
    async def chat(self, messages: list[dict]) -> dict: ...

class MetaPort(Protocol):
    async def send_message(self, to: str, body: str) -> None: ...
```

---

## 9) Logging & correlation (observability)

- `core/logging.py` exposes JSON logger.
- Middleware injects `corr_id` + latency metrics.
- Scrub secrets/PII in logs.

---

## 10) Tests (additive, then tighten coverage)

- **Unit tests:** services (auth guards, merge orchestration), intent handlers (ports mocked), logging helpers.
- **API tests:** `/alive`, `/ready`, admin, webhook happy/sad.
- Start at **70%** coverage; ratchet upward.

---

## 11) PR Choreography (staged, small, safe)

1. **`chore/precommit+arch-linter`**  
   - Add **`.pre-commit-config.yaml`**, **`.importlinter`**, **`pyproject.toml`** updates (Ruff caps, Deptry section).
   - CI: add steps for `lint-imports`, `deptry`, `bandit`, `pip-audit`, `ruff`, `mypy`, `pyright`, `pytest --cov`.

2. **`refactor/app-factory`**  
   - Create `apps/api/app.py`, split `health.py`, make `bt_servant.py` a thin shim.

3. **`refactor/intent-router-skeleton`**  
   - Add `services/intent_router.py`, sample intents + tests.

4. **`refactor/ports+adapters`**  
   - Add `core/ports.py`; migrate adapters to implement ports; services accept ports (no adapter imports).

5. **`refactor/models+config+logging`**  
   - Centralize DTOs (`core/models.py`), env (`core/config.py`), logger (`core/logging.py`); add corr_id middleware.

6. **`tests/expand+coverage`**  
   - Expand tests; raise coverage threshold when green stabilizes.

**Definition of Done per PR:** No behavioral changes unless explicitly stated; CI green; `lint-imports` and `deptry` pass; handlers small/testable; no `services → adapters` imports.

---

## 12) AGENTS.md updates (applied separately)

- **Doctrine:** layers, routing rules, ports/adapters discipline; `.importlinter` must pass.
- **Intent Handling Standard:** router + handlers (≤ 50 statements).
- **Fitness Functions:** ruff/pylint caps, mccabe, mypy, pyright, import-linter, bandit, pip-audit, **deptry**, coverage.
- **Definition of Done:** local commands + PR checklist.

---

## Appendix A — Minimal DI wiring example

```py
# apps/api/app.py
from fastapi import FastAPI
from bt_servant_engine.core.config import Settings
from bt_servant_engine.adapters.chroma_client import ChromaClient
from bt_servant_engine.services.intent_router import IntentRouter
from bt_servant_engine.services import chroma_service
from bt_servant_engine.apps.api.routes import health, admin, webhooks

def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI()

    # Bind ports to adapter implementations
    chroma_port = ChromaClient(settings.CHROMA_URL)

    # Build services with ports
    services = type("Services", (), {})()
    services.chroma = chroma_service.ChromaService(chroma_port)
    services.intent_router = IntentRouter({
        "merge_docs": lambda event, s=services: chroma_service.handle_merge_docs(event, s),
        "status":     lambda event, s=services: chroma_service.handle_status(event, s),
    })

    app.state.services = services

    app.include_router(health.router)
    app.include_router(admin.router)
    app.include_router(webhooks.router)
    return app
```

---

**End of plan.**


---

## 13) **File creation checklist (exact contents Codex should write)**

> Codex: create these files verbatim, then run the install commands shown below.

### A) `.importlinter` (strict onion) — **create at repo root**
```
[importlinter]
root_package = bt_servant_engine

[contract:layers]
name = Onion layering (strict)
type = layers
layers =
    bt_servant_engine.apps.api
    bt_servant_engine.adapters
    bt_servant_engine.services.intents
    bt_servant_engine.services
    bt_servant_engine.core

[contract:no_api_to_adapters]
name = Routes must not import adapters directly
type = forbidden
source_modules =
    bt_servant_engine.apps.api
forbidden_modules =
    bt_servant_engine.adapters

[contract:no_services_to_adapters]
name = Services must not import adapters
type = forbidden
source_modules =
    bt_servant_engine.services
    bt_servant_engine.services.intents
forbidden_modules =
    bt_servant_engine.adapters
```

### B) `.pre-commit-config.yaml` — **create at repo root**
```
repos:
  - repo: local
    hooks:
      - id: ruff-lint
        name: ruff (lint)
        entry: ruff check .
        language: system
        pass_filenames: false
        stages: [commit]

      - id: ruff-format
        name: ruff (format)
        entry: ruff format .
        language: system
        pass_filenames: false
        stages: [commit]

      - id: mypy
        name: mypy (types)
        entry: mypy bt_servant_engine
        language: system
        pass_filenames: false
        stages: [commit]

      - id: pyright
        name: pyright (types)
        entry: pyright
        language: system
        pass_filenames: false
        stages: [commit]

      - id: import-linter
        name: import-linter (onion architecture)
        entry: lint-imports
        language: system
        pass_filenames: false
        stages: [commit]

      - id: bandit
        name: bandit (security)
        entry: bandit -q -r bt_servant_engine
        language: system
        pass_filenames: false
        stages: [push]

      - id: pip-audit
        name: pip-audit (supply chain)
        entry: pip-audit
        language: system
        pass_filenames: false
        stages: [push]

      - id: deptry
        name: deptry (dependency hygiene)
        entry: deptry .
        language: system
        pass_filenames: false
        stages: [push]

      - id: pytest
        name: pytest (with coverage threshold)
        entry: pytest --maxfail=1 --disable-warnings -q --cov=bt_servant_engine --cov-report=term-missing --cov-fail-under=70
        language: system
        pass_filenames: false
        stages: [push]
```

### C) `pyproject.toml` — **full minimal scaffold including tool configs**
```
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "bt-servant-engine"
version = "0.1.0"
description = "BT Servant Engine — strict onion/hexagonal refactor baseline"
readme = "README.md"
requires-python = ">=3.12"
authors = [{ name = "unfoldingWord", email = "info@unfoldingword.org" }]
license = { text = "MIT" }
dependencies = [
  # Runtime deps (trim/update to match the repo as needed)
  "fastapi>=0.112",
  "uvicorn>=0.30",
  "pydantic>=2.7",
  "pydantic-settings>=2.3",
]

[project.optional-dependencies]
dev = [
  "ruff>=0.4",
  "mypy>=1.10",
  "pyright>=1.1",
  "pytest>=8.2",
  "pytest-cov>=5.0",
  "bandit>=1.7",
  "pip-audit>=2.7",
  "deptry>=0.16",
  "import-linter>=2.0",
  "pre-commit>=3.8",
  "typing-extensions>=4.12",
]

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E","F","W","I","UP","B","PL","C90","RUF"]

[tool.ruff.lint.pylint]
max-statements = 50   # PLR0915
max-branches  = 12    # PLR0912
max-returns   = 6     # PLR0911
max-args      = 5     # PLR0913

[tool.ruff.lint.mccabe]
max-complexity = 10   # C901

[tool.mypy]
python_version = "3.12"
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_return_any = true
pretty = true
strict_equality = true

[tool.pyright]
pythonVersion = "3.12"
typeCheckingMode = "strict"
reportMissingTypeStubs = false

[tool.pytest.ini_options]
addopts = "-q --maxfail=1 --disable-warnings"
testpaths = ["tests"]

[tool.deptry]
ignore_notebooks = true
exclude = ["venv", ".venv", "build", "dist"]
pep621_metadata = true
known_first_party = ["bt_servant_engine"]
scan_ignored = false
# Uncomment and tune when necessary:
# [tool.deptry.per_rule_ignores]
# DEP001 = []  # unused
# DEP002 = []  # missing
# DEP003 = []  # transitive
# DEP004 = []  # miscategorized
```

### D) `requirements-dev.txt` (optional, if you prefer pinned dev installs)
```
ruff
mypy
pyright
pytest
pytest-cov
bandit
pip-audit
deptry
import-linter
pre-commit
```

### E) Developer bootstrap commands (Codex can run)
```
python -m pip install -U pip
python -m pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type pre-push
```

---

With this section, **all file contents and commands are unambiguous** for Codex to create and wire up the configuration.
