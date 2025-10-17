"""Administrative routes for serving BT-Servant log files."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, TypedDict, cast

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from bt_servant_engine.apps.api.dependencies import require_admin_token
from bt_servant_engine.core.logging import LOGS_DIR

router = APIRouter()

LOG_SUFFIX = ".log"
LOG_CHUNK_SIZE = 64 * 1024
RECENT_DAYS_MIN = 1
RECENT_DAYS_MAX = 90
RECENT_LIMIT_MIN = 1
RECENT_LIMIT_MAX = 500


class LogFileEntry(TypedDict):
    """Serialized metadata for a single log file."""

    name: str
    size_bytes: int
    modified_at: str
    created_at: str


class LogFilesPayload(TypedDict):
    """Response payload returned by log listing endpoints."""

    files: list[LogFileEntry]
    total_files: int
    total_size_bytes: int


def _isoformat_utc(timestamp: float) -> str:
    """Encode a POSIX timestamp as an ISO-8601 UTC string."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _iter_log_files() -> Iterator[Path]:
    """Yield .log files located in the configured logs directory."""
    try:
        yield from (
            entry for entry in LOGS_DIR.iterdir() if entry.is_file() and entry.suffix == LOG_SUFFIX
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Log directory not found"
        ) from exc
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Log directory not accessible"
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to read log directory",
        ) from exc


def _build_log_entry(file_path: Path, stat_result: os.stat_result) -> LogFileEntry:
    """Return serialized metadata for a log file."""
    return cast(
        LogFileEntry,
        {
            "name": file_path.name,
            "size_bytes": stat_result.st_size,
            "modified_at": _isoformat_utc(stat_result.st_mtime),
            "created_at": _isoformat_utc(stat_result.st_ctime),
        },
    )


def _validated_log_path(filename: str) -> Path:
    """Validate the requested filename and return its resolved Path."""
    if not filename or filename.endswith("/") or filename.endswith("\\"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid log filename")
    if any(sep in filename for sep in ("/", "\\")) or ".." in filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid log filename")
    if not filename.endswith(LOG_SUFFIX):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid log filename")

    candidate = LOGS_DIR / filename
    try:
        candidate_resolved = candidate.resolve()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Log file not found"
        ) from exc
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Log file not accessible"
        ) from exc

    if candidate_resolved.parent != LOGS_DIR.resolve():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid log filename")
    if not candidate_resolved.exists() or not candidate_resolved.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log file not found")
    return candidate_resolved


def _stream_file(path: Path) -> Iterator[bytes]:
    """Yield file contents in fixed-size chunks."""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(LOG_CHUNK_SIZE)
            if not chunk:
                break
            yield chunk


def _collect_log_entries() -> LogFilesPayload:
    """Return serialized metadata for all log files."""
    files: list[LogFileEntry] = []
    total_size = 0

    try:
        ordered_entries = sorted(
            _iter_log_files(), key=lambda entry: entry.stat().st_mtime, reverse=True
        )
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to read log directory",
        ) from exc

    for file_path in ordered_entries:
        try:
            stat_result = file_path.stat()
        except FileNotFoundError:
            # File was removed between listing and stat; skip gracefully.
            continue
        except PermissionError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Log file not accessible"
            ) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to read log file metadata",
            ) from exc

        files.append(_build_log_entry(file_path, stat_result))
        total_size += stat_result.st_size

    return cast(
        LogFilesPayload,
        {"files": files, "total_files": len(files), "total_size_bytes": total_size},
    )


@router.get("/admin/logs/files")
async def list_log_files(_: None = Depends(require_admin_token)) -> LogFilesPayload:
    """Return metadata for available log files."""
    return _collect_log_entries()


@router.get("/admin/logs/files/{filename}")
async def download_log_file(filename: str, _: None = Depends(require_admin_token)):
    """Stream the requested log file to the caller."""
    path = _validated_log_path(filename)
    try:
        stat_result = path.stat()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Log file not found"
        ) from exc
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Log file not accessible"
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to read log file"
        ) from exc

    headers = {
        "Content-Disposition": f'attachment; filename="{path.name}"',
        "Content-Length": str(stat_result.st_size),
    }
    return StreamingResponse(
        _stream_file(path),
        media_type="text/plain; charset=utf-8",
        headers=headers,
    )


@router.get("/admin/logs/recent")
async def list_recent_logs(
    days: int = 7,
    limit: int = 100,
    _: None = Depends(require_admin_token),
) -> LogFilesPayload:
    """Return log files modified within the requested timeframe."""
    if days < RECENT_DAYS_MIN or days > RECENT_DAYS_MAX:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(f"Parameter 'days' must be between {RECENT_DAYS_MIN} and {RECENT_DAYS_MAX}"),
        )
    if limit < RECENT_LIMIT_MIN or limit > RECENT_LIMIT_MAX:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(f"Parameter 'limit' must be between {RECENT_LIMIT_MIN} and {RECENT_LIMIT_MAX}"),
        )

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    base_payload = _collect_log_entries()
    filtered = [
        entry
        for entry in base_payload["files"]
        if datetime.fromisoformat(entry["modified_at"].replace("Z", "+00:00")) >= cutoff
    ][:limit]

    total_size = sum(entry["size_bytes"] for entry in filtered)
    return cast(
        LogFilesPayload,
        {"files": filtered, "total_files": len(filtered), "total_size_bytes": total_size},
    )


__all__ = ["router"]
