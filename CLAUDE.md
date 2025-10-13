# Claude Instructions - bt-servant-engine

## Project Overview

BT Servant Engine is a WhatsApp chatbot that provides Bible translation assistance and scripture resources. It uses:
- **FastAPI** for the web framework
- **LangGraph** for orchestrating the decision pipeline
- **OpenAI** for language models
- **ChromaDB** for vector storage
- **Meta WhatsApp API** for messaging

## Quick Start (First Time)

When you first encounter this repository:

1. **Check environment**: Ensure Python 3.12+ and virtual environment are set up
   ```bash
   python --version  # Should be 3.12+
   source .venv/bin/activate || scripts/init_env.sh
   ```

2. **Understand current state**: Check refactor status
   ```bash
   git status  # See which branch and uncommitted changes
   git log --oneline -5  # Recent commits
   lint-imports || true  # See architecture violations (expected during refactor)
   ```

3. **Run tests**: Verify working state
   ```bash
   pytest -q -m "not openai"  # Should pass (or note existing failures)
   ```

4. **Read the plans**:
   - Start with this file (CLAUDE.md)
   - Then read `docs/refactor_plan_revised.md` for current tasks
   - Reference `AGENTS.md` for general guidelines

## General Guidelines

For general repository guidelines not specific to the refactor, see `AGENTS.md`, which covers:
- Detailed testing policies and OpenAI test handling
- Security and configuration management
- Agent handoff and persistence policies
- Session notes and Things To Remember

## Python Environment

- **Python 3.12+** required
- **Virtual environment**: `.venv/` (activate with `source .venv/bin/activate`)
- **Dependencies**: `requirements.txt` (runtime) + dev tools via pip
- **Initial setup**: `scripts/init_env.sh` sets up everything

## Coding Standards

### Style & Naming
- **Python 3.12+**, 4-space indentation, UTF-8 encoding
- **Naming conventions**:
  - `snake_case` for functions and variables
  - `PascalCase` for classes
  - `UPPER_SNAKE_CASE` for constants
- **Functions**: Keep small (≤50-60 lines), single-purpose, explicit returns
- **Docstrings**: Required for public functions; keep comments minimal

### Linting & Type Checking (Zero Warnings Policy)
Always run on the ENTIRE project before committing:
```bash
scripts/check_repo.sh  # Runs everything below plus tests
# Or individually:
ruff check .          # Fast linting
ruff format .         # Auto-formatting
pylint $(git ls-files '*.py')  # Strict linting
mypy .                # Type checking
pyright               # Strict type checking
lint-imports          # Architecture compliance
```

### Testing Requirements

**⚠️ CRITICAL: Test Coverage Must Stay Above 65% ⚠️**

- **Coverage threshold**: ≥65% (`--cov-fail-under=65`)
  - **Note**: Temporarily lowered from 70% - working to bring it back up
  - If your changes drop coverage below 65%, you MUST add tests before committing
  - Adding new untested code penalizes overall coverage - test as you go
  - Never bypass pre-commit hooks due to coverage failures
- **Warnings as errors**: Tests fail on any warning
- **Never commit failing tests** - Fix immediately
- **OpenAI tests**: Marked with `@pytest.mark.openai`, run only when needed:
  ```bash
  RUN_OPENAI_API_TESTS=1 pytest -q -m openai
  ```

### Commit Conventions
- **Subject format**: `(Claude) <concise subject>` (keep under 72 chars)
  - The `(Claude)` prefix indicates AI-generated commits
- **Always include a body** describing:
  - What changed and why
  - Key files touched
  - Test plan and results
  - Any risks or follow-ups
  - **NEVER leave the commit body empty** - always provide context
- **Use proper newlines** in commit messages (not literal `\n`):
  ```bash
  git commit -m "(Claude) Subject" -m $'Body line 1\n\nBody line 2'
  ```
- **Author identity**: Set to "Claude Assistant" for AI commits:
  ```bash
  export GIT_AUTHOR_NAME="Claude Assistant"
  export GIT_COMMITTER_NAME="Claude Assistant"
  ```
- **No co-author**: Do not add Co-Authored-By lines to commits

## Development Workflows

### Running Locally
```bash
# Activate virtual environment
source .venv/bin/activate

# Run the API server
uvicorn bt_servant_engine.apps.api.app:create_app --factory --reload

# Or with specific host/port
uvicorn bt_servant_engine.apps.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000
```

### Common Tasks
```bash
# Before ANY commits - run full validation
scripts/check_repo.sh

# Quick test run (excludes OpenAI tests)
pytest -q -m "not openai"

# Check architecture compliance
lint-imports

# Auto-fix formatting
ruff format .

# See what changed
git status
git diff
```

### Pre-commit Hooks

**⚠️ CRITICAL: NEVER BYPASS PRE-COMMIT HOOKS ⚠️**

Pre-commit hooks are quality gates that must pass before ANY commit. These hooks enforce:
- Code formatting (ruff format)
- Linting (ruff check, pylint)
- Type checking (mypy, pyright)
- Test coverage ≥70%
- Architecture compliance (lint-imports)

**ABSOLUTELY FORBIDDEN:**
- ❌ **DO NOT use `SKIP_CHECKS=1`** - This bypasses all quality gates
- ❌ **DO NOT commit with failing tests**
- ❌ **DO NOT commit code that drops coverage below 70%**
- ❌ **DO NOT use any git commit flags that bypass hooks**

**If pre-commit hooks fail, you MUST:**
1. **Fix the underlying issue** - Add tests, fix linting errors, etc.
2. **Never bypass the hooks** - The failure indicates real problems
3. **Ask the user if unclear** - Don't guess or take shortcuts

**One-time setup:**
```bash
git config core.hooksPath .githooks
pre-commit install
pre-commit install --hook-type pre-push
```

**The ONLY acceptable reason to bypass** would be a critical production outage where the repository owner explicitly instructs you to bypass checks. Even then, document it thoroughly and fix it immediately after.

## Configuration

### Required Environment Variables
```bash
OPENAI_API_KEY=sk-...           # OpenAI API key
META_VERIFY_TOKEN=...            # Meta webhook verification
META_WHATSAPP_TOKEN=...          # WhatsApp API token
META_PHONE_NUMBER_ID=...        # WhatsApp phone number ID
META_APP_SECRET=...              # Meta app secret
LOG_PSEUDONYM_SECRET=...        # For PII scrubbing in logs
DATA_DIR=./data                  # Local data directory
```

See `env.example` for full list.

## ⚠️ Active Refactor in Progress

**CRITICAL**: This codebase is undergoing a major architectural refactor to achieve strict onion/hexagonal architecture.

### Which Plan to Use
- **✅ USE THIS**: `docs/refactor_plan_revised.md` - The CURRENT action plan
- **❌ DO NOT USE**: `docs/refactor_plan_deprecated.md` - Original plan (historical reference only)

## Current State (October 2025)

The refactor is partially complete. Major accomplishments and remaining work:

### Completed ✅
- Directory structure (core, services, adapters, apps)
- Fitness functions (.importlinter, .pre-commit-config.yaml)
- Most intent logic extracted to services/intents/
- FastAPI app properly packaged
- bt_servant.py is now a thin shim

### Still Needed ❌
- **brain.py is 1533 lines** - needs decomposition
- **Legacy modules at root** - config.py, logger.py, messaging.py, user_message.py
- **db/ directory exists** - should be fully migrated to adapters
- **IntentRouter not wired** - created but unused
- **Mixed imports** - some from old locations, some from new

## Implementation Guidelines

### When Working on the Refactor

1. **Always reference** `docs/refactor_plan_revised.md` for the current state and next steps
2. **Follow phases in order** - Don't skip ahead (Phases 1-5 must be done sequentially)
3. **Test continuously** - Run after each file migration:
   ```bash
   pytest -q -m "not openai"  # Quick tests
   lint-imports                # Architecture compliance
   ruff check .                # Linting
   ```
4. **Preserve behavior** - Each change must maintain functionality
5. **Update imports atomically** - Change all references in the same commit

### What NOT to Do

- **DON'T create new features** during the refactor - only restructure existing code
- **DON'T skip import linter checks** - Architecture compliance is mandatory
- **DON'T leave tests failing** - Fix immediately or revert changes
- **DON'T mix refactor commits with feature changes** - Keep them separate
- **DON'T use the deprecated refactor plan** - Only use refactor_plan_revised.md

### Key Commands

```bash
# Architecture validation
lint-imports                    # Must pass after each phase

# Code quality
ruff check .                    # Linting
ruff format .                   # Formatting
mypy .                          # Type checking
pyright                         # Strict type checking

# Tests
pytest -q -m "not openai"       # Unit tests (fast)
pytest --cov=bt_servant_engine  # With coverage

# Pre-commit hooks
pre-commit run --all-files      # Run all checks locally
```

### Import Path Migration

When migrating modules, update ALL imports:

| Old Import | New Import |
|------------|------------|
| `from config import config` | `from bt_servant_engine.core.config import config` |
| `from logger import get_logger` | `from bt_servant_engine.core.logging import get_logger` |
| `from user_message import UserMessage` | `from bt_servant_engine.core.models import UserMessage` |
| `from db import ...` | `from bt_servant_engine.adapters import ...` |
| `from messaging import ...` | `from bt_servant_engine.adapters.messaging import ...` |

## Architecture Rules

The codebase follows strict onion/hexagonal architecture:

```
apps/api/ → services/ → core/
    ↓           ↓         ↑
         adapters/ -------↑
```

**Dependency Rules:**
- **apps/api** can import from services and core (NOT adapters directly)
- **services** can import from core (NOT adapters)
- **adapters** can import from core (implement ports)
- **core** imports from nowhere (it's the innermost layer)
- **Non-negotiable**: The clean onion/hexagonal architecture already established must never be violated—every change must respect these dependency boundaries

## Files to Reference

- `docs/refactor_plan_revised.md` - Current refactor plan
- `.importlinter` - Architecture enforcement configuration
- `.pre-commit-config.yaml` - Automated quality gates
- `AGENTS.md` - General repository guidelines
- `bt_servant_engine/services/intent_router.py` - Needs to be wired up
- `brain.py` - Needs to be decomposed and removed

## Post-Refactor TODO

Once the refactor is complete:
1. Delete this section from CLAUDE.md
2. Update AGENTS.md with the new architecture
3. Archive both refactor plan documents
4. Update README with new project structure

## Important Notes & Gotchas

### Project-Specific Considerations
- **LangGraph state**: Uses `TypedDict` for `BrainState` - may need type casts for IDE satisfaction
- **OpenAI SDK**: Use typed message params (`EasyInputMessageParam`, `ChatCompletionMessageParam`)
- **Agentic strength**: Values are `normal`, `low`, `very_low` - use helper functions, not string comparisons
- **Bible data**: Located in `sources/bible_data/en/bsb/` as JSON files
- **Vector collections**: Stack-ranked in ChromaDB, queried with cosine similarity cutoff

### Common Issues
- **Import errors after moving files**: Update ALL imports atomically, including tests
- **Type checker complaints**: Use `cast()` judiciously for LangGraph/OpenAI SDK compatibility
- **Tests can't find modules**: Ensure `__init__.py` files exist in all package directories
- **Architecture violations**: Run `lint-imports` to identify and fix dependency issues

## Questions?

If unclear about next steps:
1. Check `docs/refactor_plan_revised.md` for the phase-by-phase plan
2. Run `lint-imports` to see if architecture rules are violated
3. Check git status to see which phase is in progress
4. Look for TODO/FIXME comments in the code
5. Review AGENTS.md "Things To Remember" section for historical context
