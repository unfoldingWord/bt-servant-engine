# BT-Servant Log API Specification

**Version:** 1.0
**Date:** 2025-10-16
**Purpose:** Add log-serving API endpoints to BT-Servant to enable the BT Servant Log Viewer application

---

## 1. Overview

The BT Servant Log Viewer requires access to BT-Servant's log files via HTTP API. This specification defines three new endpoints to be added to the BT-Servant application that will allow the log viewer to discover and download log files.

### Requirements
- BT-Servant must expose endpoints to list and serve log files
- Only `.log` files should be accessible (security)
- Support for streaming large files
- Proper error handling and security measures

### Integration Flow
1. Log Viewer calls `/api/logs/files` to discover available logs
2. Log Viewer calls `/api/logs/recent?days=21` to get last 21 days of logs
3. Log Viewer downloads each file via `/api/logs/files/{filename}`
4. Log Viewer parses and displays the unified log data

---

## 2. Endpoint Specifications

### 2.1 List Log Files
**Endpoint:** `GET /api/logs/files`
**Purpose:** List all available log files with metadata
**Authentication:** None (v1) - prepare for future auth
**Rate Limit:** 60 requests per minute

#### Response
```json
{
  "files": [
    {
      "name": "bt_servant_2025-01-15.log",
      "size_mb": 8.2,
      "size_bytes": 8598456,
      "modified": "2025-01-15T23:59:59Z",
      "created": "2025-01-15T00:00:01Z",
      "line_count": 45000,  // Optional, can be null if expensive to compute
      "readable": true
    },
    {
      "name": "bt_servant_2025-01-14.log",
      "size_mb": 10.1,
      "size_bytes": 10590234,
      "modified": "2025-01-14T23:59:59Z",
      "created": "2025-01-14T00:00:01Z",
      "line_count": 52000,
      "readable": true
    }
  ],
  "total_files": 2,
  "total_size_mb": 18.3,
  "log_directory": "/var/log/bt-servant"  // For debugging, optional
}
```

#### Error Responses
- `500 Internal Server Error` - Log directory not accessible
- `503 Service Unavailable` - Temporarily cannot read logs

---

### 2.2 Download Log File
**Endpoint:** `GET /api/logs/files/{filename}`
**Purpose:** Download/stream a specific log file
**Authentication:** None (v1) - prepare for future auth
**Rate Limit:** 100 requests per minute
**Response Type:** `text/plain` or `application/octet-stream`

#### Parameters
- `filename` (path parameter): Name of the log file (e.g., `bt_servant_2025-01-15.log`)
- `compress` (query parameter, optional): If `true`, return gzipped content

#### Response Headers
```
Content-Type: text/plain; charset=utf-8
Content-Disposition: attachment; filename="bt_servant_2025-01-15.log"
Content-Length: 8598456
X-Line-Count: 45000  // Optional
```

#### Streaming Response
For files > 1MB, use streaming response to avoid memory issues.

#### Error Responses
- `404 Not Found` - File does not exist
- `403 Forbidden` - File exists but is not a .log file (security)
- `500 Internal Server Error` - Cannot read file

---

### 2.3 Get Recent Log Files
**Endpoint:** `GET /api/logs/recent`
**Purpose:** Get list of log files from the last N days
**Authentication:** None (v1) - prepare for future auth
**Rate Limit:** 60 requests per minute

#### Query Parameters
- `days` (optional, default=7): Number of days to look back
- `max_files` (optional, default=100): Maximum number of files to return

#### Response
Same format as `/api/logs/files`, but filtered to recent files.

#### Example
```
GET /api/logs/recent?days=21
```

Returns all log files modified within the last 21 days, sorted by date descending.

---

## 3. Implementation Guide (Python/FastAPI)

### 3.1 Configuration
Add to your application configuration:

```python
# config.py or settings.py
import os
from pathlib import Path

class LogAPIConfig:
    LOG_DIRECTORY = os.getenv("BT_SERVANT_LOG_DIR", "/var/log/bt-servant")
    MAX_FILE_SIZE_MB = 500  # Maximum file size to serve
    ALLOWED_EXTENSIONS = [".log"]
    ENABLE_COMPRESSION = True
    CACHE_TTL_SECONDS = 60  # Cache file list for 1 minute
```

### 3.2 FastAPI Route Implementation

```python
# routes/log_api.py
import os
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

router = APIRouter(prefix="/api/logs", tags=["logs"])

class LogFileInfo(BaseModel):
    name: str
    size_mb: float
    size_bytes: int
    modified: datetime
    created: datetime
    line_count: Optional[int] = None
    readable: bool

class LogFilesResponse(BaseModel):
    files: List[LogFileInfo]
    total_files: int
    total_size_mb: float
    log_directory: Optional[str] = None

def get_line_count(file_path: Path) -> Optional[int]:
    """Get line count for files < 50MB, otherwise return None."""
    if file_path.stat().st_size > 50 * 1024 * 1024:
        return None
    try:
        with open(file_path, 'rb') as f:
            return sum(1 for _ in f)
    except:
        return None

def is_safe_filename(filename: str) -> bool:
    """Prevent path traversal attacks."""
    return (
        ".." not in filename and
        "/" not in filename and
        "\\" not in filename and
        filename.endswith(tuple(LogAPIConfig.ALLOWED_EXTENSIONS))
    )

@router.get("/files", response_model=LogFilesResponse)
async def list_log_files():
    """List all available log files."""
    try:
        log_dir = Path(LogAPIConfig.LOG_DIRECTORY)
        if not log_dir.exists():
            raise HTTPException(status_code=500, detail="Log directory not found")

        files = []
        total_size = 0

        for file_path in sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True):
            stat = file_path.stat()
            size_bytes = stat.st_size

            files.append(LogFileInfo(
                name=file_path.name,
                size_mb=round(size_bytes / (1024 * 1024), 2),
                size_bytes=size_bytes,
                modified=datetime.fromtimestamp(stat.st_mtime),
                created=datetime.fromtimestamp(stat.st_ctime),
                line_count=get_line_count(file_path),
                readable=os.access(file_path, os.R_OK)
            ))
            total_size += size_bytes

        return LogFilesResponse(
            files=files,
            total_files=len(files),
            total_size_mb=round(total_size / (1024 * 1024), 2),
            log_directory=str(log_dir)  # Remove in production
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing log files: {str(e)}")

@router.get("/files/{filename}")
async def download_log_file(filename: str, compress: bool = False):
    """Download a specific log file."""

    # Security: Validate filename
    if not is_safe_filename(filename):
        raise HTTPException(status_code=403, detail="Invalid filename")

    file_path = Path(LogAPIConfig.LOG_DIRECTORY) / filename

    # Check if file exists
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Check file size limit
    file_size = file_path.stat().st_size
    if file_size > LogAPIConfig.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        # Streaming response for large files
        def iterfile():
            with open(file_path, 'rb') as f:
                chunk_size = 64 * 1024  # 64KB chunks
                while chunk := f.read(chunk_size):
                    yield chunk

        if compress and LogAPIConfig.ENABLE_COMPRESSION:
            # Compress on-the-fly
            def iterfile_compressed():
                compressor = gzip.GzipFile(mode='wb', fileobj=None)
                for chunk in iterfile():
                    yield compressor.compress(chunk)
                yield compressor.flush()

            return StreamingResponse(
                iterfile_compressed(),
                media_type="application/gzip",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}.gz"',
                    "X-Original-Size": str(file_size)
                }
            )
        else:
            # Get line count for small files
            line_count = get_line_count(file_path) if file_size < 50 * 1024 * 1024 else None

            headers = {
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(file_size)
            }
            if line_count:
                headers["X-Line-Count"] = str(line_count)

            return StreamingResponse(
                iterfile(),
                media_type="text/plain; charset=utf-8",
                headers=headers
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

@router.get("/recent", response_model=LogFilesResponse)
async def get_recent_logs(
    days: int = Query(7, ge=1, le=90, description="Number of days to look back"),
    max_files: int = Query(100, ge=1, le=500, description="Maximum files to return")
):
    """Get log files from the last N days."""
    try:
        log_dir = Path(LogAPIConfig.LOG_DIRECTORY)
        if not log_dir.exists():
            raise HTTPException(status_code=500, detail="Log directory not found")

        cutoff_date = datetime.now() - timedelta(days=days)
        files = []
        total_size = 0

        for file_path in sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True):
            stat = file_path.stat()

            # Check if file is within date range
            if datetime.fromtimestamp(stat.st_mtime) < cutoff_date:
                continue

            # Limit number of files
            if len(files) >= max_files:
                break

            size_bytes = stat.st_size

            files.append(LogFileInfo(
                name=file_path.name,
                size_mb=round(size_bytes / (1024 * 1024), 2),
                size_bytes=size_bytes,
                modified=datetime.fromtimestamp(stat.st_mtime),
                created=datetime.fromtimestamp(stat.st_ctime),
                line_count=get_line_count(file_path),
                readable=os.access(file_path, os.R_OK)
            ))
            total_size += size_bytes

        return LogFilesResponse(
            files=files,
            total_files=len(files),
            total_size_mb=round(total_size / (1024 * 1024), 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing recent files: {str(e)}")
```

### 3.3 Register Routes
Add to your main application file:

```python
# main.py or app.py
from fastapi import FastAPI
from routes import log_api

app = FastAPI()

# ... other route registrations ...

app.include_router(log_api.router)
```

---

## 4. Security Considerations

### 4.1 Path Traversal Prevention
- **CRITICAL**: Validate all filenames to prevent directory traversal attacks
- Only serve files with `.log` extension
- Reject any filename containing `..`, `/`, or `\`

### 4.2 File Access Control
- Only serve files from the configured log directory
- Check file readability before attempting to serve
- Implement file size limits to prevent DoS

### 4.3 Rate Limiting
Implement rate limiting to prevent abuse:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.get("/files")
@limiter.limit("60/minute")
async def list_log_files():
    # ... implementation ...
```

### 4.4 CORS Configuration
Configure CORS to allow access from the log viewer application:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://log-viewer.yourdomain.com"],  # Configure appropriately
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)
```

### 4.5 Future Authentication
Design endpoints to easily add authentication later:
```python
# Future: Add dependency injection for auth
async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Verify token
    pass

@router.get("/files")
async def list_log_files(current_user: User = Depends(get_current_user)):
    # ... implementation ...
```

---

## 5. Testing Guidelines

### 5.1 Unit Tests
```python
# test_log_api.py
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile

def test_list_log_files(client: TestClient, tmp_path: Path):
    """Test listing log files."""
    # Create test log files
    (tmp_path / "test1.log").write_text("log content")
    (tmp_path / "test2.log").write_text("more log content")

    response = client.get("/api/logs/files")
    assert response.status_code == 200
    data = response.json()
    assert data["total_files"] == 2
    assert len(data["files"]) == 2

def test_download_file(client: TestClient, tmp_path: Path):
    """Test downloading a specific file."""
    test_content = "test log content\nline 2\nline 3"
    (tmp_path / "test.log").write_text(test_content)

    response = client.get("/api/logs/files/test.log")
    assert response.status_code == 200
    assert response.content.decode() == test_content

def test_path_traversal_protection(client: TestClient):
    """Test that path traversal attempts are blocked."""
    response = client.get("/api/logs/files/../etc/passwd")
    assert response.status_code == 403

    response = client.get("/api/logs/files/../../secret.log")
    assert response.status_code == 403

def test_recent_logs(client: TestClient, tmp_path: Path):
    """Test getting recent log files."""
    # Create log files with different timestamps
    from datetime import datetime, timedelta
    import os

    old_file = tmp_path / "old.log"
    old_file.write_text("old")
    # Set modification time to 30 days ago
    old_time = (datetime.now() - timedelta(days=30)).timestamp()
    os.utime(old_file, (old_time, old_time))

    recent_file = tmp_path / "recent.log"
    recent_file.write_text("recent")

    response = client.get("/api/logs/recent?days=7")
    assert response.status_code == 200
    data = response.json()
    assert data["total_files"] == 1
    assert data["files"][0]["name"] == "recent.log"
```

### 5.2 Integration Tests
1. Test with actual log files in staging environment
2. Test streaming large files (>10MB)
3. Test concurrent requests
4. Test with malformed filenames
5. Test CORS headers

### 5.3 Manual Testing Checklist
- [ ] List all log files via `/api/logs/files`
- [ ] Download a small log file (<1MB)
- [ ] Download a large log file (>10MB)
- [ ] Request with compression (`?compress=true`)
- [ ] Request recent logs for various day ranges
- [ ] Try path traversal attacks (should fail)
- [ ] Try downloading non-.log files (should fail)
- [ ] Check response headers (Content-Type, Content-Disposition)
- [ ] Verify CORS headers for log viewer domain
- [ ] Test rate limiting (make 100+ requests rapidly)

---

## 6. Deployment Considerations

### 6.1 Environment Variables
```bash
# .env or environment configuration
BT_SERVANT_LOG_DIR=/var/log/bt-servant
LOG_API_MAX_FILE_SIZE_MB=500
LOG_API_ENABLE_COMPRESSION=true
LOG_API_CACHE_TTL=60
```

### 6.2 Log Rotation Integration
Ensure the API works correctly with your log rotation strategy:
- File naming pattern: `bt_servant_YYYY-MM-DD_N.log`
- Handle compressed archived logs (`.log.gz`)
- Consider excluding very old archived logs

### 6.3 Monitoring
Add metrics for:
- Request count per endpoint
- File download sizes
- Error rates
- Response times

```python
from prometheus_client import Counter, Histogram

log_api_requests = Counter('log_api_requests_total', 'Total log API requests', ['endpoint'])
log_api_download_size = Histogram('log_api_download_bytes', 'Size of downloaded log files')
```

### 6.4 Performance Optimization
- Cache file listings for 1 minute
- Use async file operations where possible
- Stream large files instead of loading into memory
- Consider CDN for frequently accessed logs

---

## 7. Rollout Plan

### Phase 1: Development
1. Implement endpoints in development branch
2. Add unit tests
3. Test with sample log files

### Phase 2: Staging
1. Deploy to staging environment
2. Test with real log files
3. Performance testing with large files
4. Security audit

### Phase 3: Production
1. Deploy during low-traffic window
2. Monitor error rates and performance
3. Gradual rollout if using feature flags

---

## 8. Example Usage

### From Log Viewer Application (JavaScript/TypeScript)
```typescript
// Fetch list of log files
const response = await fetch('https://bt-servant.api/api/logs/files');
const { files } = await response.json();

// Get last 21 days of logs
const recentResponse = await fetch('https://bt-servant.api/api/logs/recent?days=21');
const { files: recentFiles } = await recentResponse.json();

// Download each file
for (const file of recentFiles) {
    const fileResponse = await fetch(`https://bt-servant.api/api/logs/files/${file.name}`);
    const content = await fileResponse.text();
    // Process log content...
}
```

### cURL Examples
```bash
# List all log files
curl https://bt-servant.api/api/logs/files

# Download specific file
curl -O https://bt-servant.api/api/logs/files/bt_servant_2025-01-15.log

# Get compressed version
curl https://bt-servant.api/api/logs/files/bt_servant_2025-01-15.log?compress=true -o bt_servant_2025-01-15.log.gz

# Get recent logs
curl https://bt-servant.api/api/logs/recent?days=21
```

---

## 9. Troubleshooting Guide

### Common Issues

**Issue:** 500 Error - "Log directory not found"
- Check `BT_SERVANT_LOG_DIR` environment variable
- Verify directory exists and has read permissions

**Issue:** 404 Error when downloading file
- Check exact filename (case-sensitive)
- Verify file has .log extension
- Ensure file exists in log directory

**Issue:** CORS errors from log viewer
- Check CORS middleware configuration
- Verify allowed origins includes log viewer URL

**Issue:** Slow performance with large files
- Ensure streaming is working (not loading entire file into memory)
- Check network bandwidth
- Consider enabling compression

---

## 10. Version History

- **v1.0** (2025-10-16): Initial specification
  - Basic endpoints for listing and downloading logs
  - Security measures for path traversal
  - Streaming support for large files

---

## Notes for Implementer

1. Start with the basic endpoints without compression
2. Add compression support once basic functionality works
3. Implement comprehensive logging for debugging
4. Consider adding OpenAPI/Swagger documentation
5. Test thoroughly with actual production log files
6. Monitor memory usage when serving large files
7. Consider implementing ETag support for caching

This specification provides all necessary details to implement the log API endpoints in BT-Servant. The implementation should take approximately 4-8 hours including testing.