# BT-Servant Log API Specification

**Version:** 1.0
**Updated:** 2025-10-17
**Purpose:** Document the minimal FastAPI endpoints required for the BT Servant Log Viewer to discover and download server log files.

---

## 1. Scope and Constraints

BT-Servant already writes structured application logs via `bt_servant_engine.core.logging`. Logs are appended to `LOG_FILE_PATH`, which resolves to `LOGS_DIR / "bt_servant.log"`. `LOGS_DIR` is derived as follows (first writable directory wins):

1. `BT_SERVANT_LOG_DIR` environment override (optional)
2. `<repo-root>/logs`
3. `<DATA_DIR>/logs` (defaults to `/data/logs`)
4. `bt_servant_engine/core/logs`

The new API surfaces the files located in that resolved log directory. We only expose regular files with a `.log` suffix. Compressed archives and other extensions remain out of scope for v1.

The Log Viewer runs server-side (same origin) or through an internal proxy. We therefore do **not** enable CORS. Requests must originate from trusted infrastructure that can attach the admin token described below.


## 2. Authentication

All endpoints reuse the existing admin token guard that protects the Chroma administration APIs:

- Requests must include `Authorization: Bearer <ADMIN_API_TOKEN>` **or** `X-Admin-Token: <ADMIN_API_TOKEN>`
- Enforcement is toggled by `ENABLE_ADMIN_AUTH` (defaults to `True`)
- Configuration lives in `bt_servant_engine.apps.api.dependencies.require_admin_token`

No new secrets, OAuth flows, or middleware are introduced. The same token gates both the current admin endpoints and the new log APIs.


## 3. Endpoint Reference

The routes live alongside the existing admin endpoints in `bt_servant_engine/apps/api/routes/admin_logs.py` and are included by the API factory with the same admin-token dependency. To avoid conflicts, we namespace them under `/admin/logs`.

### 3.1 List Available Log Files

- **Method & Path:** `GET /admin/logs/files`
- **Auth:** Admin token required (see §2)
- **Purpose:** Enumerate `.log` files in `LOGS_DIR`, newest first
- **Response:**

```json
{
  "files": [
    {
      "name": "bt_servant_2025-10-16.log",
      "size_bytes": 8598456,
      "modified_at": "2025-10-16T23:59:59Z",
      "created_at": "2025-10-16T00:00:01Z"
    }
  ],
  "total_files": 1,
  "total_size_bytes": 8598456
}
```

- **Error Codes:**
  - `404` if the log directory cannot be resolved or read
  - `500` for unexpected filesystem errors

### 3.2 Download a Specific Log File

- **Method & Path:** `GET /admin/logs/files/{filename}`
- **Auth:** Admin token required
- **Purpose:** Stream a single `.log` file to the caller
- **Path Parameter:** `filename` – exact basename listed by `GET /admin/logs/files`
- **Response:** Streaming `text/plain; charset=utf-8`
- **Headers:**
  - `Content-Disposition: attachment; filename="{filename}"`
  - `Content-Length: <size>` (if known)

- **Error Codes:**
  - `400` if the filename fails validation (path traversal or missing `.log` suffix)
  - `404` if the file is absent
  - `500` for unexpected read errors

### 3.3 Filter Recent Log Files

- **Method & Path:** `GET /admin/logs/recent`
- **Auth:** Admin token required
- **Query Parameters:**
  - `days` (integer, optional, default `7`, min `1`, max `90`) – include files modified within the last *n* days
  - `limit` (integer, optional, default `100`, min `1`, max `500`) – cap the number of files returned

- **Response:** Same structure as `/admin/logs/files`, but filtered and truncated by the parameters above.

- **Error Codes:** Same as `/admin/logs/files`.

### 3.4 Shared Behaviour

- Directory enumeration uses `Path.iterdir()` / `glob` constrained to `.log`
- Sorting is performed on modification time (`stat().st_mtime`, descending)
- All datetimes are serialized as UTC ISO-8601 strings
- Non-regular files (directories, symlinks) are ignored


## 4. Validation and Security Guards

1. **Filename validation** – Reject any value containing path separators, `..`, or lacking the `.log` suffix. Validation occurs before hitting the filesystem.
2. **Directory scope** – All paths are resolved relative to the canonical `LOGS_DIR`; no additional directories are inspected.
3. **File size limits** – We stream directly from disk using `StreamingResponse` to avoid loading entire files into memory. No explicit size ceiling is enforced; callers must download responsibly.
4. **Permission errors** – Surface as `404` to avoid leaking directory structure details.


## 5. Implementation Notes

1. **Router placement** – Extend `bt_servant_engine/apps/api/routes/admin.py` with three new handlers. Each should include `Depends(require_admin_token)` to enforce the guard when `ENABLE_ADMIN_AUTH` is true.
2. **Filesystem helpers** – Reuse `bt_servant_engine.core.logging.LOGS_DIR` to determine the directory. Define a small utility (e.g., `_iter_log_files()`) to consolidate validation and metadata collection.
3. **Streaming** – Use `StreamingResponse` with a chunked file iterator (`open(...)` in binary mode, `read` 64KB chunks). Set `media_type="text/plain; charset=utf-8"`.
4. **Serialization** – For responses, return plain dicts annotated with Pydantic models or `TypedDict` if desired; keep fields limited to `name`, `size_bytes`, `modified_at`, `created_at`.
5. **Error handling** – Wrap filesystem access in `try/except OSError` blocks. Map `FileNotFoundError` to `404`; everything else to `500` (with a concise `detail`).
6. **OpenAPI schema** – FastAPI automatically documents the endpoints. Ensure docstrings clearly describe the authentication requirement so the Swagger UI aligns with expectations.


## 6. Testing Checklist

- Unit tests under `tests/apps/api/test_admin_logs.py` (new file) covering:
  - Listing files with temporary directories and monkeypatched `LOGS_DIR`
  - Filename validation rejects traversal attempts
  - Download endpoint streams file contents and sets `Content-Disposition`
  - `days` and `limit` filters behave as documented
- Re-run `scripts/check_repo.sh` to ensure lint, type, and test suites remain green.


## 7. Operational Notes

- No new environment variables are required.
- Ensure `ENABLE_ADMIN_AUTH=True` (default) in environments where the API is exposed.
- Infrastructure that serves the Log Viewer must include the admin token header; no browser-facing CORS flow is provided.
- Log rotation should continue to write `.log` files into the existing directory; rotated archives (`.log.1`, `.gz`) remain hidden until explicitly added in a future revision.

---

This condensed specification delivers the essentials requested by the Log Viewer bot while aligning with the current BT-Servant architecture and security model. EOF
