"""Health and readiness routes."""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ..dependencies import require_healthcheck_token

router = APIRouter()


@router.get("/")
def read_root() -> dict[str, str]:
    """Health/info endpoint with a short usage message."""
    return {"message": "Welcome to the API. Refer to /docs for available endpoints."}


@router.get("/alive")
async def alive_check(_: None = Depends(require_healthcheck_token)) -> JSONResponse:
    """Authenticated health check endpoint for infrastructure probes."""
    return JSONResponse({"status": "ok", "message": "BT Servant is alive and healthy."})


__all__ = ["router"]
