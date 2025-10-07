# Revised Refactor Plan - bt-servant-engine

**Created:** October 2025
**Purpose:** Complete the unfinished refactor work to achieve strict onion/hexagonal architecture

## Current State Assessment

### ✅ Completed Work (Keep As-Is)

1. **Directory Structure**: Proper onion/hexagonal layout established
   - `bt_servant_engine/core/` - domain layer with ports, models, config
   - `bt_servant_engine/services/` - application services and intent handlers
   - `bt_servant_engine/adapters/` - infrastructure implementations
   - `bt_servant_engine/apps/api/` - delivery layer with FastAPI routes

2. **Fitness Functions**: Architecture enforcement configured
   - `.importlinter` - strict onion dependency rules
   - `.pre-commit-config.yaml` - automated quality gates
   - `pyproject.toml` - linting and complexity limits

3. **Intent Extraction**: Most intent logic moved to service modules
   - Individual intent handlers in `services/intents/*.py`
   - Response pipeline extracted to `services/response_pipeline.py`
   - Graph pipeline helpers in `services/graph_pipeline.py`

4. **API Structure**: FastAPI app properly packaged
   - `bt_servant.py` is now a thin shim
   - Routes organized under `apps/api/routes/`
   - Middleware and dependencies properly separated

### ❌ Critical Issues to Fix

1. **brain.py is still 1533 lines** - Contains all LangGraph orchestration
2. **Top-level module pollution** - config.py, logger.py, messaging.py, user_message.py at root
3. **Legacy db/ directory exists** - Should be fully migrated to adapters
4. **Import chaos** - Mixed imports from old and new locations
5. **IntentRouter not utilized** - Created but not wired; brain still called directly
6. **Adapters depend on legacy** - Import from db module instead of being self-contained

## Implementation Phases

### Phase 1: Complete Core Module Migration
**Goal:** Consolidate all foundational modules into core layer

1. **Migrate config.py**
   - Merge root `config.py` into `bt_servant_engine/core/config.py`
   - Update all imports: `from config import` → `from bt_servant_engine.core.config import`

2. **Migrate logger.py**
   - Merge root `logger.py` into `bt_servant_engine/core/logging.py`
   - Update all imports: `from logger import` → `from bt_servant_engine.core.logging import`

3. **Migrate messaging.py**
   - Move messaging logic into `bt_servant_engine/adapters/messaging.py`
   - Update imports accordingly

4. **Migrate user_message.py**
   - Move UserMessage class to `bt_servant_engine/core/models.py`
   - Update all imports: `from user_message import` → `from bt_servant_engine.core.models import`

5. **Fix utils imports**
   - Ensure `utils/` modules use proper imports if they remain
   - Consider moving utils into appropriate service or core modules

**Validation:** No Python files at root except `bt_servant.py`

### Phase 2: Complete Adapter Migration
**Goal:** Eliminate legacy db/ directory

1. **Consolidate ChromaDB adapter**
   - Move all `db/chroma_db.py` functionality → `bt_servant_engine/adapters/chroma.py`
   - Ensure adapter fully implements ChromaPort protocol

2. **Consolidate UserDB adapter**
   - Move all `db/user_db.py` functionality → `bt_servant_engine/adapters/user_state.py`
   - Move `db/user.py` models → `bt_servant_engine/core/models.py`
   - Ensure adapter fully implements UserStatePort protocol

3. **Update adapter imports**
   - Remove all `from db import` statements
   - Adapters should only import from `bt_servant_engine.core`

4. **Delete db/ directory**
   - Remove entire directory once migration confirmed
   - Update any test fixtures or data paths

**Validation:** `db/` directory no longer exists; adapters pass unit tests

### Phase 3: Extract LangGraph Orchestration
**Goal:** Move brain.py orchestration into services layer

1. **Create brain_orchestrator.py**
   ```
   bt_servant_engine/services/brain_orchestrator.py
   ```
   - Move `create_brain()` function
   - Move `BrainState` TypedDict
   - Move graph assembly logic
   - Move node wrapper functions

2. **Create preprocessing service**
   ```
   bt_servant_engine/services/preprocessing.py
   ```
   - Move: `start`, `determine_query_language`, `determine_intents`
   - Move: `preprocess_user_query`, `PreprocessorResult`
   - Move associated prompts and helpers

3. **Create query processor service**
   ```
   bt_servant_engine/services/query_processor.py
   ```
   - Move: `query_vector_db`, `query_open_ai`
   - Move: `consult_fia_resources`
   - Consolidate query orchestration logic

4. **Distribute remaining functions**
   - Move response processing to `response_pipeline.py`
   - Move thin intent wrappers to respective intent modules
   - Move utility functions to appropriate services

**Validation:** brain.py < 100 lines

### Phase 4: Wire IntentRouter
**Goal:** Replace direct brain invocation with proper routing

1. **Update brain_orchestrator**
   - Integrate IntentRouter for intent dispatch
   - Replace conditional edges with router.dispatch()

2. **Register intent handlers**
   - Create handler registration in service initialization
   - Map IntentType enum values to handler functions

3. **Update webhook route**
   - Replace `brain.invoke()` with `services.intent_router.dispatch()`
   - Pass proper IntentRequest objects

4. **Remove brain state dependency**
   - Ensure intent handlers work with IntentRequest/Response
   - Remove direct BrainState passing where possible

**Validation:** IntentRouter handles all intent dispatching

### Phase 5: Final Cleanup
**Goal:** Achieve target architecture

1. **Delete brain.py** (or reduce to minimal backward compatibility shim)
2. **Delete all top-level Python files** except bt_servant.py
3. **Update all test imports** to use new paths
4. **Run full validation suite**:
   ```bash
   lint-imports  # Architecture compliance
   ruff check .  # Linting
   mypy .        # Type checking
   pytest        # Tests with coverage
   ```

## Success Metrics

- [ ] brain.py deleted or < 50 lines
- [ ] No top-level Python files except bt_servant.py
- [ ] db/ directory removed completely
- [ ] All imports use `bt_servant_engine.*` paths
- [ ] Import linter passes all contracts
- [ ] Tests maintain 70%+ coverage
- [ ] IntentRouter handles all intent dispatch
- [ ] Pre-commit hooks pass on all changes

## Migration Rules

1. **Preserve behavior** - Each phase must maintain functionality
2. **Update imports atomically** - Change all references in same commit
3. **Test continuously** - Run tests after each file migration
4. **Commit frequently** - Small, reviewable changes
5. **Document decisions** - Note any architectural choices in comments

## Risk Mitigation

- **Backup current state** before starting
- **Create feature branch** for entire refactor
- **Run tests frequently** to catch breaks early
- **Use git bisect** if issues arise to identify breaking changes
- **Keep original files** temporarily with `.backup` extension if needed

## Estimated Timeline

- Phase 1: 2-3 hours (mechanical import updates)
- Phase 2: 2-3 hours (adapter consolidation)
- Phase 3: 3-4 hours (brain decomposition)
- Phase 4: 2-3 hours (router wiring)
- Phase 5: 1-2 hours (cleanup and validation)

**Total: 10-15 hours of focused work**