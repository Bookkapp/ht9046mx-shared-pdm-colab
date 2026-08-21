from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .catalog import catalog_payload
from .handler_store import (
    create_handler,
    delete_handler,
    list_handlers,
    sync_all_handler_sources,
    update_handler,
)
from .model_store import store
from .settings import settings


class HandlerCreate(BaseModel):
    machine_code: str
    ip: str


class HandlerUpdate(BaseModel):
    ip: str | None = None
    enabled: bool | None = None


class ComparisonSeries(BaseModel):
    machine_id: str
    module_id: int = Field(ge=1, le=8)
    metric: str


class ComparisonRequest(BaseModel):
    selected_date: date | None = None
    series: list[ComparisonSeries] = Field(min_length=2, max_length=6)


class ApprovalRequest(BaseModel):
    actor: str = Field(min_length=2, max_length=120)
    reason: str | None = Field(default=None, max_length=500)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        sync_all_handler_sources()
    except (FileNotFoundError, ValueError, OSError):
        # Readiness reports unavailable paths. The API still starts so an
        # operator can inspect and repair configuration from the server.
        pass
    yield


app = FastAPI(
    title="HT9046MX Controlled Hybrid Model Monitor API",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)


@app.middleware("http")
async def response_headers(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith(settings.api_prefix):
        response.headers["Cache-Control"] = "no-store"
    elif request.url.path.startswith("/assets/"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid X-API-Key")


def _value_error(error: Exception) -> HTTPException:
    text = str(error)
    status = 409 if "already exists" in text or "Duplicate" in text else 422
    return HTTPException(status_code=status, detail=text)


API = settings.api_prefix


@app.get(f"{API}/health/live")
def liveness() -> dict[str, str]:
    return {"status": "ok"}


@app.get(f"{API}/health")
def health(response: Response) -> dict[str, Any]:
    checks = {
        "handlers": settings.handlers_file.exists(),
        "controlled_config": settings.controlled_system_config.exists(),
        "controlled_policy": settings.controlled_policy_file.exists(),
        "controlled_runtime": settings.controlled_runtime_dir.exists(),
        "shared_model": (
            settings.shared_model_artifact / "shared_model.keras"
        ).exists(),
        "frontend_built": (settings.frontend_dist / "index.html").exists(),
    }
    ready = all(checks.values())
    if not ready:
        response.status_code = 503
    return {
        "status": "ready" if ready else "not_ready",
        "checked_at": datetime.now(timezone.utc),
        "checks": checks,
        "policy_version": store.pipeline().get("policy", {}).get("policy_version"),
    }


@app.get(f"{API}/config")
def config() -> dict[str, Any]:
    return {
        "refresh_seconds": settings.refresh_seconds,
        "timezone": settings.timezone,
        "api_prefix": settings.api_prefix,
        "writes_require_api_key": bool(settings.api_key),
        "model_monitor_mode": "controlled_hybrid_v1",
    }


@app.get(f"{API}/catalog")
def catalog() -> dict[str, Any]:
    return catalog_payload()


@app.get(f"{API}/handlers")
def handlers() -> list[dict[str, Any]]:
    try:
        return list_handlers()
    except (FileNotFoundError, ValueError) as error:
        raise _value_error(error) from error


@app.post(
    f"{API}/handlers",
    status_code=201,
    dependencies=[Depends(require_api_key)],
)
def add_handler(payload: HandlerCreate) -> dict[str, Any]:
    try:
        return create_handler(payload.machine_code, payload.ip)
    except (FileNotFoundError, ValueError) as error:
        raise _value_error(error) from error


@app.patch(
    f"{API}/handlers/{{machine_id}}",
    dependencies=[Depends(require_api_key)],
)
def patch_handler(machine_id: str, payload: HandlerUpdate) -> dict[str, Any]:
    try:
        return update_handler(
            machine_id, ip=payload.ip, enabled=payload.enabled
        )
    except (FileNotFoundError, ValueError) as error:
        raise _value_error(error) from error


@app.delete(
    f"{API}/handlers/{{machine_id}}",
    dependencies=[Depends(require_api_key)],
)
def remove_handler(machine_id: str) -> dict[str, Any]:
    try:
        removed = delete_handler(machine_id)
        return {
            "removed": removed,
            "data_deleted": False,
            "message": "Future sync/model onboarding disabled; copied logs and profiles were retained.",
        }
    except (FileNotFoundError, ValueError) as error:
        raise _value_error(error) from error


@app.get(f"{API}/model/artifact")
def model_artifact() -> dict[str, Any]:
    return store.artifact()


@app.get(f"{API}/model/pipeline")
def model_pipeline() -> dict[str, Any]:
    return store.pipeline()


@app.get(f"{API}/model/fleet")
def model_fleet() -> dict[str, Any]:
    return store.fleet()


@app.get(f"{API}/model/machines/{{machine_id}}/monitor")
def machine_monitor(
    machine_id: str,
    module_id: int = Query(default=1, ge=1, le=8),
    selected_date: date | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return store.machine_monitor(machine_id, module_id, selected_date)
    except (FileNotFoundError, ValueError) as error:
        raise _value_error(error) from error


@app.get(f"{API}/model/machines/{{machine_id}}/profiles")
def machine_profiles(machine_id: str) -> dict[str, Any]:
    try:
        return store.profiles(machine_id)
    except (FileNotFoundError, ValueError) as error:
        raise _value_error(error) from error


@app.post(f"{API}/model/comparison")
def comparison(payload: ComparisonRequest) -> dict[str, Any]:
    try:
        return store.comparison(
            payload.selected_date,
            [item.model_dump() for item in payload.series],
        )
    except (FileNotFoundError, ValueError) as error:
        raise _value_error(error) from error


@app.post(
    f"{API}/model/machines/{{machine_id}}/approve",
    dependencies=[Depends(require_api_key)],
)
def approve_profile(machine_id: str, payload: ApprovalRequest) -> dict[str, Any]:
    try:
        return store.approve(machine_id, payload.actor, payload.reason)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        raise _value_error(error) from error


@app.post(
    f"{API}/model/machines/{{machine_id}}/reject",
    dependencies=[Depends(require_api_key)],
)
def reject_profile(machine_id: str, payload: ApprovalRequest) -> dict[str, Any]:
    if not payload.reason:
        raise HTTPException(status_code=422, detail="reason is required when rejecting")
    try:
        return store.reject(machine_id, payload.actor, payload.reason)
    except (FileNotFoundError, ValueError) as error:
        raise _value_error(error) from error


@app.post(
    f"{API}/model/machines/{{machine_id}}/continue-learning",
    dependencies=[Depends(require_api_key)],
)
def continue_learning(machine_id: str, payload: ApprovalRequest) -> dict[str, Any]:
    try:
        return store.continue_learning(machine_id, payload.actor, payload.reason)
    except (FileNotFoundError, ValueError) as error:
        raise _value_error(error) from error


assets = settings.frontend_dist / "assets"
if assets.exists():
    app.mount("/assets", StaticFiles(directory=assets), name="assets")


@app.get("/", include_in_schema=False)
def frontend_index():
    index = settings.frontend_dist / "index.html"
    if index.exists():
        return FileResponse(index)
    raise HTTPException(
        status_code=503,
        detail="React frontend is not built. Run frontend\\npm run build.",
    )


@app.get("/{full_path:path}", include_in_schema=False)
def frontend_fallback(full_path: str):
    requested = (settings.frontend_dist / full_path).resolve()
    dist = settings.frontend_dist.resolve()
    if requested.is_file() and dist in requested.parents:
        return FileResponse(requested)
    index = settings.frontend_dist / "index.html"
    if index.exists():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="Not found")
