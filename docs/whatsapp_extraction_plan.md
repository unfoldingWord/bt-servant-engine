# Plan: Extract WhatsApp Logic to Gateway

## Goal
Transform `bt-servant-engine` into a client-agnostic REST API by extracting all WhatsApp-specific code into `bt-servant-whatsapp-gateway`.

## Architecture Decisions
- **Communication**: Gateway → Engine via REST API
- **Auth**: Use existing `ADMIN_API_TOKEN` pattern (proper key management in Phase 1b)
- **Streaming**: Design API to support it later, implement in Phase 2
- **OpenAI**: All AI calls (transcription, brain, TTS) stay in engine - gateway has zero OpenAI dependency

---

## Phase 1a: Add Engine REST API (bt-servant-engine)

### New Files to Create

**`bt_servant_engine/apps/api/routes/chat.py`** - Core chat endpoint (handles everything)
```
POST /api/v1/chat
- Auth: Bearer token (ADMIN_API_TOKEN for now)
- Handles text AND audio input
- Returns text responses AND audio output (when voice requested)
```

**`bt_servant_engine/apps/api/routes/users.py`** - User preferences
```
GET/PUT /api/v1/users/{user_id}/preferences
```

**`bt_servant_engine/core/api_models.py`** - Pydantic models
```python
class ChatRequest(BaseModel):
    client_id: str                              # "whatsapp", "web", "telegram"
    user_id: str                                # User identifier from client
    message: str = ""                           # Text message (empty if audio)
    message_type: Literal["text", "audio"] = "text"
    audio_base64: str | None = None             # Base64 audio (if message_type="audio")
    audio_format: str = "ogg"                   # Audio format hint

class ChatResponse(BaseModel):
    responses: list[str]                        # Response texts (includes continuation prompts if queued intents)
    response_language: str                      # Language of response
    voice_audio_base64: str | None = None       # TTS audio (if generated) - client decides how to use
    intent_processed: str                       # Which intent was handled
    has_queued_intents: bool = False            # True if more intents waiting for next request
```

### Files to Modify
- `bt_servant_engine/apps/api/app.py` - Include new routers
- `bt_servant_engine/apps/api/routes/__init__.py` - Export new modules

---

## Phase 1a: Build Gateway (bt-servant-whatsapp-gateway)

### Directory Structure
```
bt-servant-whatsapp-gateway/
├── pyproject.toml
├── requirements.txt
├── .env.example
├── whatsapp_gateway/
│   ├── __init__.py
│   ├── config.py              # META_* settings + ENGINE_BASE_URL
│   ├── main.py                # FastAPI app
│   ├── routes/
│   │   ├── webhooks.py        # /meta-whatsapp handler
│   │   └── health.py
│   ├── services/
│   │   ├── engine_client.py   # HTTP client for engine API
│   │   ├── chunking.py        # 1500-char WhatsApp chunking
│   │   └── response_formatter.py
│   └── meta_api/
│       ├── client.py          # send_text_message, send_voice_message
│       ├── signature.py       # verify_facebook_signature
│       └── media.py           # upload/download media
└── tests/
```

### Code Migration Map

| From (Engine) | To (Gateway) |
|---------------|--------------|
| `adapters/messaging.py` (all) | `meta_api/client.py` |
| `webhooks.py` `/meta-whatsapp` routes | `routes/webhooks.py` |
| `webhooks.py` `verify_facebook_signature` | `meta_api/signature.py` |
| `core/models.py` `UserMessage.from_data` | `meta_api/models.py` |
| `response_pipeline.py` chunking logic | `services/chunking.py` |
| `config.py` `META_*` settings | `config.py` |

### Gateway Config (New)
```python
class Settings(BaseSettings):
    # Meta API
    META_VERIFY_TOKEN: str
    META_WHATSAPP_TOKEN: str
    META_PHONE_NUMBER_ID: str
    META_APP_SECRET: str
    FACEBOOK_USER_AGENT: str
    IN_META_SANDBOX_MODE: bool = False
    META_SANDBOX_PHONE_NUMBER: str = "11111111"
    MESSAGE_AGE_CUTOFF_IN_SECONDS: int = 3600

    # Engine connection
    ENGINE_BASE_URL: str  # e.g., "http://engine:8000"
    ENGINE_API_KEY: str   # ADMIN_API_TOKEN from engine

    # WhatsApp limits
    MAX_MESSAGE_LENGTH: int = 4096
    CHUNK_SIZE: int = 1500
```

### Gateway Fitness Functions & CI/CD

**`.importlinter`** - Onion architecture enforcement
```ini
[importlinter]
root_package = whatsapp_gateway

[importlinter:layering]
name = Onion layering (strict)
type = layers
layers =
    whatsapp_gateway.routes
    whatsapp_gateway.services
    whatsapp_gateway.meta_api
    whatsapp_gateway.core

[importlinter:no_routes_to_meta_api]
name = Routes must not import meta_api directly
type = forbidden
source_modules =
    whatsapp_gateway.routes
forbidden_modules =
    whatsapp_gateway.meta_api

[importlinter:no_services_to_routes]
name = Services must not import routes
type = forbidden
source_modules =
    whatsapp_gateway.services
forbidden_modules =
    whatsapp_gateway.routes
```

**`.pre-commit-config.yaml`** - Same hooks as engine
```yaml
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
        entry: ruff format --check whatsapp_gateway
        language: system
        pass_filenames: false
        stages: [commit]

      - id: mypy
        name: mypy (types)
        entry: mypy .
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
        entry: bandit -q -r whatsapp_gateway
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
        entry: pytest --cov=whatsapp_gateway --cov-fail-under=65
        language: system
        pass_filenames: false
        stages: [push]
```

**`.github/workflows/ci-pr.yml`** - CI on PRs (mirror engine's setup)
- Ruff format + lint
- Pylint
- Mypy + Pyright
- Import Linter
- Bandit (security)
- Pip Audit (supply chain)
- Deptry (dependency hygiene)
- Pytest with 65% coverage threshold

**`pyproject.toml`** - Tool configs (ruff, mypy, pyright, pytest)

**`CLAUDE.md`** - Same coding standards, commit conventions, pre-commit policies

---

## Phase 1a: Parallel Operation

1. Deploy gateway alongside engine
2. Configure Meta webhook to point to gateway
3. Gateway calls engine's new `/api/v1/chat` endpoint
4. Engine's `/meta-whatsapp` endpoint remains (fallback)

---

## Phase 1b: Remove WhatsApp from Engine (after gateway proven)

### Files to Delete
- `bt_servant_engine/adapters/messaging.py` (entire file)
- WhatsApp routes from `webhooks.py` (may delete entire file if only WhatsApp)

### Files to Modify
- `bt_servant_engine/core/config.py` - Remove `META_*` settings
- `bt_servant_engine/core/ports.py` - Remove `MessagingPort`
- `bt_servant_engine/bootstrap.py` - Remove `MessagingAdapter`
- `bt_servant_engine/services/__init__.py` - Remove `messaging` from ServiceContainer
- `bt_servant_engine/core/models.py` - Remove Meta-specific parsing

### Config Settings to Remove from Engine
```
META_VERIFY_TOKEN
META_WHATSAPP_TOKEN
META_PHONE_NUMBER_ID
META_APP_SECRET
FACEBOOK_USER_AGENT
IN_META_SANDBOX_MODE
META_SANDBOX_PHONE_NUMBER
MAX_META_TEXT_LENGTH
```

---

## Phase 1b: Proper API Key Management (Deferred)
- Database-backed API keys with CLI management
- Per-client revocation capability
- Key format: `bts_<env>_<random>`

---

## Multi-Intent Handling (Preserved)

The engine supports detecting multiple intents from a single message and processing them across subsequent requests. This functionality is **fully preserved** with the new architecture.

**How it works:**
1. User sends: "Show me Romans 8 and summarize Genesis 4:2"
2. Engine detects 2 intents, processes highest priority (RETRIEVE_SCRIPTURE)
3. Remaining intent queued in TinyDB via `UserStatePort` (keyed by `user_id`)
4. Response includes continuation prompt: "Would you like me to summarize Genesis 4:2?"
5. User responds: "yes"
6. Engine checks `has_queued_intents(user_id)`, pops and processes queued intent

**Why it works with REST API:**
- All queue logic is inside `brain_orchestrator.py` - no changes needed
- `user_id` from `ChatRequest` is used to load/save queue
- `UserStatePort` handles persistence (TinyDB → JSON)
- Continuation prompts are appended to `responses` automatically
- `has_queued_intents` in `ChatResponse` lets clients know if more intents are pending

**Key files (no changes needed):**
- `services/intent_queue.py` - Queue operations (pop, peek, clear)
- `services/brain_orchestrator.py` - Multi-intent detection & queueing
- `adapters/user_state.py` - TinyDB persistence
- `core/intents.py` - `IntentQueue`, `IntentQueueItem` models

**TTL:** Queued intents expire after 10 minutes. Max 5 items per queue.

---

## Phase 2: Streaming Support (Future)

Design consideration: Engine API should support two modes:
1. **Sync** (current): `POST /api/v1/chat` → full response
2. **Stream** (future): `POST /api/v1/chat/stream` → SSE events

WhatsApp gateway will always use sync (messages are atomic).
Web clients will use stream for real-time UX.

---

## Implementation Order

**Pre-requisite: Create feature branch for engine work**
```bash
git checkout -b feature/client-agnostic-api
```

1. **Engine: Create `/api/v1/chat` endpoint** (handles text, audio input, and voice output)
2. **Engine: Create `/api/v1/users` endpoints** (preferences)
3. **Engine: Add tests for new endpoints**
4. **Gateway: Scaffold project structure**
5. **Gateway: Implement Meta API client** (copy from engine)
6. **Gateway: Implement Engine client**
7. **Gateway: Implement webhook handler**
8. **Gateway: Implement chunking**
9. **Integration test: Gateway → Engine**
10. **Deploy and validate**
11. **Engine: Remove WhatsApp code** (Phase 1b)

---

## Critical Files Reference

### Engine (to read/modify)
- `bt_servant_engine/apps/api/routes/webhooks.py` (698 lines) - webhook logic to extract
- `bt_servant_engine/adapters/messaging.py` (285 lines) - Meta API client to extract
- `bt_servant_engine/core/config.py` - settings to split
- `bt_servant_engine/services/brain_orchestrator.py` - understand how to invoke brain
- `bt_servant_engine/services/response_pipeline.py` - chunking logic to extract

### Gateway (to create)
- `../bt-servant-whatsapp-gateway/` - empty repo ready for implementation
