# Log API Pagination Plan and Contract

## Goal and Scope
- Add paginated log entry retrieval to avoid loading entire log histories while keeping admin export routes unchanged.
- Maintain stable ordering, honor existing filters, and avoid impacting WhatsApp/message handling throughput.

## API Contract

### Route
- `GET /admin/logs/entries`
- Admin-token protected (same dependency as existing log routes).

### FastAPI Signature (draft)
```python
@router.get("/admin/logs/entries")
async def list_log_entries(
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    cursor: str | None = Query(default=None),  # future; ignored for first cut
    sort: Literal["timestamp_desc", "timestamp_asc"] = "timestamp_desc",
    since: datetime | None = Query(default=None),
    until: datetime | None = Query(default=None),
    level: str | None = Query(default=None),
    logger: str | None = Query(default=None),
    cid: str | None = Query(default=None),
    user: str | None = Query(default=None),
    client_ip: str | None = Query(default=None),
    contains: str | None = Query(default=None),
    filename: str | None = Query(default=None),
    _: None = Depends(require_admin_token),
) -> LogEntriesPage:
    ...
```
- Sorting is deterministic: primary timestamp, then filename, then byte_offset.
- Filters apply **before** pagination.

### Response Types
```python
class LogEntry(TypedDict, total=False):
    timestamp: str  # ISO-8601 UTC
    level: str
    logger: str
    message: str
    cid: str | None
    user: str | None
    client_ip: str | None
    extras: dict[str, Any] | None
    filename: str
    byte_offset: int  # starting offset in file


class LogEntriesPage(TypedDict):
    items: list[LogEntry]
    limit: int
    offset: int
    has_next: bool
    next_offset: int | None
    total: int | None  # null when counting would be expensive
    applied_filters: dict[str, Any]  # echo normalized filters
    # future: cursor: str | None
```

### Semantics
- Default order: newest first (`timestamp_desc`). `offset` is relative to this order.
- `limit` is capped; reject over-cap with HTTP 400.
- `has_next` drives pagination; `total` may be `null` if counting is skipped.
- `contains` matches substrings in message and extras; most expensive filter—still bounded by `limit`.
- `filename` optionally constrains to a single log file (must pass existing filename validator).
- Existing routes stay unchanged: `/admin/logs/files`, `/admin/logs/files/{filename}`, `/admin/logs/recent`, and file download flows.

## Performance and Throughput Safeguards
- **Indexing:** Build a lightweight per-file index of `(timestamp, byte_offset)` on first access; cache keyed by `(filepath, mtime, size)`. Invalidate when files rotate or grow.
- **Page reads:** Use the index to seek directly to candidate offsets; parse only the needed rows. Iterate files newest→oldest for `timestamp_desc`.
- **Threadpool I/O:** Run file reads in a threadpool (`run_in_threadpool`) to keep the event loop free for WhatsApp/message handling.
- **Locking:** Guard the index cache with a lock to avoid stampedes; keep the cache small (rotation is already bounded).
- **Counting:** Populate `total` only when the index makes it cheap; otherwise return `null`. Avoid full scans on hot paths.
- **Caps:** Keep `limit` ≤ 500 to bound per-request work; consider lowering if load tests show pressure.
- **Error handling:** Reject invalid filenames, out-of-range params, and unsupported sort keys with 400s; return 500s only on unexpected I/O errors.

## Testing Plan
- Unit: parameter validation (limits, ranges), filter normalization, deterministic ordering (timestamp + filename + offset), `has_next` correctness, and `contains` behavior.
- Integration: fetch pages across multiple rotated files, verify `since/until` windows, verify that filter changes reset pagination, and ensure `total` is `null` when counting is skipped.
- Concurrency: threadpool execution does not block the event loop; measure typical page latency (<200 ms on 3–4 weeks of logs).
- Security: admin token still required; path validation prevents traversal.

## Implementation Steps (backend)
1) Define `LogEntry`/`LogEntriesPage` TypedDicts and the new router handler in `bt_servant_engine/apps/api/routes/admin_logs.py`.
2) Add filename validation reuse; wire query params with caps and defaults.
3) Implement the per-file index cache (mtime/size keyed) with thread-safe access.
4) Implement paged retrieval using the index; apply filters → sort → paginate; compute `has_next` and optional `total`.
5) Add tests for contract, sorting, filters, caps, and `has_next`.
6) Document the endpoint in README or OpenAPI description; align the viewer to use `limit/offset` and `has_next` (cursor reserved for later).
