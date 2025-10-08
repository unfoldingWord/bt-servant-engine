# Revised Refactor Plan - bt-servant-engine

**Created:** October 2025
**Updated:** October 2025 (adjusted to match actual main branch state)
**Purpose:** Complete the refactor work to achieve strict onion/hexagonal architecture

## Key Changes from Original Plan

This revision accounts for the actual state of the main branch, which is less complete than originally assumed:
- brain.py is 3403 lines (not 1533) with ALL intent handlers still embedded
- No intent extraction has been done (only 1 of 14 intents extracted)
- Helper services don't exist yet
- Added **Phase 0** for the intent extraction work

## Current State Assessment (main branch)

### ✅ Completed Work (Keep As-Is)

1. **Directory Structure**: Basic onion/hexagonal layout established
   - `bt_servant_engine/core/` - has config, logging, models, ports
   - `bt_servant_engine/services/` - has intent_router skeleton
   - `bt_servant_engine/adapters/` - has basic chroma, messaging, user_state
   - `bt_servant_engine/apps/api/` - FastAPI routes structure

2. **Fitness Functions**: Architecture enforcement configured
   - `.importlinter` - strict onion dependency rules
   - `.pre-commit-config.yaml` - automated quality gates
   - `pyproject.toml` - linting and complexity limits

3. **API Structure**: FastAPI app properly packaged
   - `bt_servant.py` is now a thin shim
   - Routes organized under `apps/api/routes/`
   - Middleware and dependencies properly separated

### ❌ Critical Issues to Fix

1. **brain.py is 3403 lines** - Contains ALL intent handlers and LangGraph orchestration
2. **Top-level module pollution** - config.py, logger.py, messaging.py, user_message.py at root
3. **Legacy db/ directory exists** - Should be fully migrated to adapters
4. **No intent extraction done** - Only 1 intent (converse) extracted; 13 remain in brain.py
5. **Helper services missing** - No graph_pipeline, response_pipeline, passage_selection, openai_utils
6. **Core modules incomplete** - Missing intents, language, agentic modules
7. **IntentRouter not utilized** - Created but not wired; brain still called directly
8. **Adapters depend on legacy** - Import from db module instead of being self-contained

## Branch Strategy

Completed and planned branch names for each phase:
- ✅ `refactor/phase-0-intent-extraction` - Extract core domain models (COMPLETE)
- ✅ `refactor/phase-1-core-migration` - Move top-level modules to core (COMPLETE)
- → `refactor/phase-2-adapter-migration` - Consolidate db/ into adapters (NEXT)
- `refactor/phase-3-brain-orchestration` - Extract LangGraph orchestration + intent handlers
- `refactor/phase-4-intent-router` - Wire IntentRouter
- `refactor/phase-5-cleanup` - Final cleanup and deletion

Note: Phase 0 branch name references "intent-extraction" but was scoped to core domain models only.

## Implementation Phases

### Phase 0: Extract Core Domain Models ✅ COMPLETE
**Goal:** Establish core domain layer with pure business logic

This phase extracts fundamental domain concepts from brain.py into the core layer.

**Completed:**
1. **Created core domain modules**
   - `bt_servant_engine/core/intents.py` - IntentType enum and UserIntents model
   - `bt_servant_engine/core/language.py` - Language models and constants (Language, ResponseLanguage, MessageLanguage, TranslatedPassage, SUPPORTED_LANGUAGE_MAP)
   - `bt_servant_engine/core/agentic.py` - Agentic strength models and constants (AgenticStrengthChoice, AgenticStrengthSetting, ALLOWED_AGENTIC_STRENGTH)

2. **Created initial helper service**
   - `bt_servant_engine/services/openai_utils.py` - extract_cached_input_tokens utility

3. **Updated brain.py**
   - Import from new core modules
   - Remove duplicate definitions
   - Reduced from 3403 to 3311 lines

**Why intent extraction was deferred:**
Intent handlers are heavily coupled to brain.py infrastructure (open_ai_client, BrainState, helper functions). Extracting them now would just move code around without architectural benefit. They will be properly refactored in Phase 3 alongside orchestration extraction, with proper dependency injection.

**Validation:**
- ✅ Core domain modules created with no infrastructure dependencies
- ✅ All linters pass
- ✅ Tests pass (80.20% coverage)
- ✅ Import linter passes (no architectural violations)

### Phase 1: Complete Core Module Migration ✅ COMPLETE
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

### Phase 3: Extract LangGraph Orchestration and Intent Handlers
**Goal:** Decompose brain.py into services layer with proper dependency injection

This phase extracts both orchestration and intent handlers, allowing us to refactor them together with clean abstractions.

1. **Create shared infrastructure**
   ```
   bt_servant_engine/services/graph_pipeline.py
   ```
   - Vector query and OpenAI response generation helpers
   - Shared by multiple intent handlers

   ```
   bt_servant_engine/services/response_pipeline.py
   ```
   - Translation and chunking logic
   - Response formatting utilities

   ```
   bt_servant_engine/services/passage_selection.py
   ```
   - Bible passage parsing and validation
   - PassageRef, PassageSelection models

2. **Extract intent handlers** (13 intents with proper DI)
   - `set_response_language` → `services/intents/set_response_language.py`
   - `set_agentic_strength` → `services/intents/set_agentic_strength.py`
   - `handle_unsupported_function` → `services/intents/unsupported_function.py`
   - `handle_system_information_request` → `services/intents/system_information.py`
   - `handle_get_passage_summary` → `services/intents/passage_summary.py`
   - `handle_get_passage_keywords` → `services/intents/passage_keywords.py`
   - `handle_get_translation_helps` → `services/intents/translation_helps.py`
   - `handle_retrieve_scripture` → `services/intents/retrieve_scripture.py`
   - `handle_listen_to_scripture` → `services/intents/listen_to_scripture.py`
   - `handle_translate_scripture` → `services/intents/translate_scripture.py`
   - `consult_fia_resources` → `services/intents/consult_fia_resources.py`
   - Update existing `converse_with_bt_servant` → `services/intents/converse.py`
   - Bible translation assistance handled by query helpers

3. **Create brain_orchestrator.py**
   ```
   bt_servant_engine/services/brain_orchestrator.py
   ```
   - Move `create_brain()` function
   - Move `BrainState` TypedDict
   - Move graph assembly logic
   - Move node wrapper functions
   - Inject dependencies (OpenAI client, config, logger)

4. **Create preprocessing service**
   ```
   bt_servant_engine/services/preprocessing.py
   ```
   - Move: `start`, `determine_query_language`, `determine_intents`
   - Move: `preprocess_user_query`, `PreprocessorResult`
   - Move associated prompts and helpers

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

**Phase 0 (Complete):**
- ✅ 3 core domain modules created (intents, language, agentic)
- ✅ openai_utils helper service created
- ✅ brain.py reduced to 3311 lines (from 3403)
- ✅ Import linter passes
- ✅ Tests pass with 80%+ coverage

**Overall (Remaining):**
- [ ] brain.py deleted or < 50 lines (currently 3311 lines)
- [ ] No top-level Python files except bt_servant.py (currently 6 files)
- [ ] db/ directory removed completely
- [ ] All imports use `bt_servant_engine.*` paths
- [ ] Import linter passes all contracts
- [ ] Tests maintain 70%+ coverage
- [ ] IntentRouter handles all intent dispatch
- [ ] Pre-commit hooks pass on all changes
- [ ] All 14 intent handlers extracted to separate modules
- [ ] Helper services created (graph_pipeline, response_pipeline, passage_selection)

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

- ✅ Phase 0: 2 hours (extract core domain models) - COMPLETE
- Phase 1: 2-3 hours (mechanical import updates)
- Phase 2: 2-3 hours (adapter consolidation)
- Phase 3: 6-8 hours (orchestration + intent handlers with DI)
- Phase 4: 2-3 hours (router wiring)
- Phase 5: 1-2 hours (cleanup and validation)

**Total: 15-21 hours of focused work** (2 hours complete, 13-19 remaining)

Note: Phase 3 is the most substantial because it involves extracting orchestration, 13 intent handlers, and creating helper services with proper dependency injection.