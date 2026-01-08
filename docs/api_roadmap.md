# API Roadmap & Future Considerations

This document tracks potential enhancements to the bt-servant-engine REST API.

## Current State

The API provides:
- **Authentication:** Single shared bearer token (`ADMIN_API_TOKEN`)
- **Documentation:** Auto-generated OpenAPI at `/docs`, `/redoc`, `/openapi.json`
- **Versioning:** `/api/v1/` prefix
- **Validation:** Pydantic request/response models
- **Health checks:** `/alive` endpoint

## Planned Enhancements

| Feature | Description | Priority | Status |
|---------|-------------|----------|--------|
| **Rate Limiting** | Prevent abuse via middleware or API gateway (e.g., slowapi, nginx) | Medium | Not started |
| **Per-Client API Keys** | Database-backed keys with revocation, replacing shared `ADMIN_API_TOKEN` | Medium | Not started |
| **CORS Configuration** | Required if browser clients call the API directly | Low | Not started |
| **Streaming Responses** | SSE endpoint (`/api/v1/chat/stream`) for real-time output | Low | Not started |
| **Request Logging** | Structured logs with request/response bodies for debugging | Low | Not started |
| **API Versioning Strategy** | Document upgrade path when `/api/v2/` is introduced | Future | Not started |

## Per-Client API Keys Design

Current authentication uses a single shared bearer token. For production multi-tenant use, consider:

### Key Format
```
bts_<env>_<random>
```
Examples:
- `bts_prod_a1b2c3d4e5f6`
- `bts_dev_x9y8z7w6v5u4`

### CLI Management
```bash
# Create a new API key
bt-servant keys create --name "WhatsApp Gateway" --env prod

# List all keys
bt-servant keys list

# Revoke a key
bt-servant keys revoke bts_prod_a1b2c3d4e5f6
```

### Database Schema (future)
```sql
CREATE TABLE api_keys (
    id UUID PRIMARY KEY,
    key_hash TEXT NOT NULL,        -- bcrypt hash of the key
    name TEXT NOT NULL,            -- human-readable name
    environment TEXT NOT NULL,     -- prod, staging, dev
    created_at TIMESTAMP,
    revoked_at TIMESTAMP,
    last_used_at TIMESTAMP,
    rate_limit_per_minute INT DEFAULT 60
);
```

### Per-Key Features
- Usage tracking (requests per day/month)
- Individual rate limits
- Scoped permissions (read-only, admin, etc.)
- Audit logging

## Streaming Responses

For real-time UX (web clients), add an SSE endpoint:

```
POST /api/v1/chat/stream
```

Response: Server-Sent Events stream
```
event: token
data: {"text": "Here"}

event: token
data: {"text": " is"}

event: token
data: {"text": " John 3:16..."}

event: done
data: {"response_language": "en", "intent_processed": "retrieve-scripture"}
```

WhatsApp gateway will always use the sync endpoint (messages are atomic).

## Rate Limiting Options

### Middleware (slowapi)
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_api_key)

@app.post("/api/v1/chat")
@limiter.limit("60/minute")
async def chat(...):
    ...
```

### API Gateway (nginx/kong)
Offload to infrastructure layer for better performance and centralized config.

## CORS Configuration

If browser clients call the API directly (not through a backend):

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com"],
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Authorization", "Content-Type"],
)
```

Currently not needed since the WhatsApp gateway is a server-side client.
