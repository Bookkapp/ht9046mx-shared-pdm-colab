from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


DASHBOARD_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = DASHBOARD_ROOT / "backend"
load_dotenv(BACKEND_ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _integer(name: str, default: int, minimum: int = 1) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _path(name: str, default: Path, *, base: Path | None = None) -> Path:
    configured = os.getenv(name)
    if not configured:
        return default.resolve()
    candidate = Path(configured)
    return (candidate if candidate.is_absolute() else (base or BACKEND_ROOT) / candidate).resolve()


@dataclass(frozen=True)
class Settings:
    dashboard_root: Path = DASHBOARD_ROOT
    backend_root: Path = BACKEND_ROOT
    model_project_root: Path = _path(
        "MODEL_PROJECT_ROOT", DASHBOARD_ROOT.parent, base=DASHBOARD_ROOT
    )
    handlers_file: Path = _path(
        "HANDLERS_FILE", BACKEND_ROOT / "config" / "handlers.json"
    )
    controlled_system_config: Path = _path(
        "CONTROLLED_SYSTEM_CONFIG",
        DASHBOARD_ROOT.parent / "configs" / "controlled_condition_monitoring.json",
    )
    controlled_policy_file: Path = _path(
        "CONTROLLED_POLICY_FILE",
        DASHBOARD_ROOT.parent / "configs" / "controlled_condition_monitoring_policy.json",
    )
    controlled_runtime_dir: Path = _path(
        "CONTROLLED_RUNTIME_DIR", DASHBOARD_ROOT.parent / "controlled_runtime"
    )
    shared_model_artifact: Path = _path(
        "SHARED_MODEL_ARTIFACT",
        DASHBOARD_ROOT.parent / "artifacts" / "shared_lstm_colab_full",
    )
    frontend_dist: Path = _path(
        "FRONTEND_DIST", DASHBOARD_ROOT / "frontend" / "dist"
    )
    handler_destination_root: Path = _path(
        "HANDLER_DESTINATION_ROOT", Path(r"C:\HT9046MX")
    )
    handler_share_template: str = os.getenv(
        "HANDLER_SHARE_TEMPLATE", "Comp_log_data_{machine_code}"
    )
    timezone: str = os.getenv("DASHBOARD_TIMEZONE", "Asia/Bangkok")
    api_prefix: str = os.getenv("API_PREFIX", "/api/v1").rstrip("/")
    api_key: str = os.getenv("API_KEY", "")
    cors_origins: tuple[str, ...] = tuple(
        item.strip()
        for item in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if item.strip()
    )
    refresh_seconds: int = _integer("DASHBOARD_REFRESH_SECONDS", 15, 5)
    chart_point_limit: int = _integer("DASHBOARD_CHART_POINT_LIMIT", 576, 48)
    prediction_read_limit: int = _integer(
        "DASHBOARD_PREDICTION_READ_LIMIT", 20_000, 1_000
    )
    sync_controlled_sources: bool = _bool("SYNC_CONTROLLED_SOURCES", True)
    host: str = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    port: int = _integer("DASHBOARD_PORT", 8000, 1)


settings = Settings()
