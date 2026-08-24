"""Incremental SMB-to-local-file synchronization for HT-9046MX handlers.

The worker deliberately uses Windows UNC paths and the persistent handler
registry. It never writes model configuration or deletes copied log files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


DEFAULT_EXTENSIONS = {".csv", ".log", ".txt"}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _resolve(value: str | Path, *, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SyncOptions:
    state_dir: Path
    extensions: set[str]
    max_files_per_machine: int
    max_bytes_per_file: int
    copy_buffer_bytes: int
    connection_mode: str
    guest_username: str
    connect_timeout_seconds: int

    @classmethod
    def from_system(cls, payload: dict[str, Any], *, config_dir: Path) -> "SyncOptions":
        raw = payload.get("sync", {})
        if not isinstance(raw, dict):
            raise ValueError("sync must be a JSON object")
        extensions = {
            str(item).strip().lower()
            for item in raw.get("extensions", sorted(DEFAULT_EXTENSIONS))
            if str(item).strip()
        }
        if not extensions or any(not value.startswith(".") for value in extensions):
            raise ValueError("sync.extensions must contain file extensions such as .csv")
        mode = str(raw.get("connection_mode", "direct")).strip().lower()
        if mode not in {"direct", "guest"}:
            raise ValueError("sync.connection_mode must be direct or guest")
        return cls(
            state_dir=_resolve(raw.get("state_dir", "state/sync_state"), base=config_dir),
            extensions=extensions,
            max_files_per_machine=max(1, int(raw.get("max_files_per_machine", 64))),
            max_bytes_per_file=max(1, int(raw.get("max_bytes_per_file", 134_217_728))),
            copy_buffer_bytes=max(65_536, int(raw.get("copy_buffer_bytes", 4_194_304))),
            connection_mode=mode,
            guest_username=str(raw.get("guest_username", "Guest")).strip() or "Guest",
            connect_timeout_seconds=max(1, int(raw.get("connect_timeout_seconds", 10))),
        )


@dataclass(frozen=True)
class HandlerSource:
    machine_id: str
    share_path: str
    source_subfolder: str
    destination: Path
    connection_mode: str | None = None
    username: str | None = None

    @property
    def source_root(self) -> Path:
        return Path(self.share_path) / self.source_subfolder


class SyncState:
    def __init__(self, state_dir: Path):
        self.path = state_dir / "sync_state.json"
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.files = dict(payload.get("files", {})) if isinstance(payload, dict) else {}
        else:
            self.files: dict[str, dict[str, Any]] = {}

    def needs_copy(self, key: str, source: Path, destination: Path) -> bool:
        stat = source.stat()
        previous = self.files.get(key, {})
        if (
            previous.get("size") != stat.st_size
            or previous.get("mtime_ns") != stat.st_mtime_ns
            or not destination.is_file()
        ):
            return True
        return destination.stat().st_size != stat.st_size

    def mark(self, key: str, source: Path) -> None:
        stat = source.stat()
        self.files[key] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "copied_at_utc": _utc_now(),
        }

    def save(self) -> None:
        _atomic_json(self.path, {"version": 1, "files": self.files})


def _load_handlers(path: Path) -> list[HandlerSource]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("handlers_file must contain a JSON array")
    handlers: list[HandlerSource] = []
    seen: set[str] = set()
    for raw in payload:
        if not isinstance(raw, dict) or not bool(raw.get("enabled", True)):
            continue
        machine_id = str(raw.get("name", "")).strip().upper()
        share_path = str(raw.get("share_path", "")).strip().rstrip("\\")
        destination = str(raw.get("destination", "")).strip()
        if not machine_id or not share_path.startswith("\\\\") or not destination:
            raise ValueError("Each enabled handler requires name, UNC share_path, and destination")
        if machine_id in seen:
            raise ValueError(f"Duplicate handler: {machine_id}")
        seen.add(machine_id)
        subfolder = str(raw.get("source_subfolder", "")).strip("\\/ ")
        if Path(subfolder).is_absolute() or ".." in Path(subfolder).parts:
            raise ValueError(f"{machine_id}: source_subfolder must stay within share_path")
        mode = str(raw.get("connection_mode", "")).strip().lower() or None
        if mode is not None and mode not in {"direct", "guest"}:
            raise ValueError(f"{machine_id}: connection_mode must be direct or guest")
        handlers.append(
            HandlerSource(
                machine_id=machine_id,
                share_path=share_path,
                source_subfolder=subfolder,
                destination=Path(destination),
                connection_mode=mode,
                username=str(raw.get("username", "")).strip() or None,
            )
        )
    return handlers


def _connect_guest(share_path: str, username: str, timeout_seconds: int) -> None:
    if os.name != "nt":
        raise RuntimeError("Guest SMB connection is supported only on Windows")
    result = subprocess.run(
        ["net", "use", share_path, "", f"/user:{username}", "/persistent:no"],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode:
        message = (result.stderr or result.stdout or "net use failed").strip()
        raise RuntimeError(f"Guest SMB connection failed for {share_path}: {message}")


def _destination_for(root: Path, relative: Path) -> Path:
    candidate = (root / relative).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError("Refusing a path outside the configured destination") from error
    return candidate


def _copy_stable_file(source: Path, destination: Path, buffer_bytes: int) -> None:
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=buffer_bytes)
        after = source.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise RuntimeError("source changed while copying")
        shutil.copystat(source, temporary)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


class SMBSyncRunner:
    def __init__(self, system_config_path: str | Path):
        self.system_config_path = Path(system_config_path).resolve()
        self.system = json.loads(self.system_config_path.read_text(encoding="utf-8"))
        if not isinstance(self.system, dict):
            raise ValueError("system config must be a JSON object")
        self.options = SyncOptions.from_system(
            self.system, config_dir=self.system_config_path.parent
        )
        handler_path = self.system.get("handlers_file")
        if not handler_path:
            raise ValueError("system config requires handlers_file for SMB synchronization")
        self.handlers_file = _resolve(handler_path, base=self.system_config_path.parent)
        self.state = SyncState(self.options.state_dir)

    def _source_files(self, handler: HandlerSource) -> list[tuple[Path, Path]]:
        source_root = handler.source_root
        if not source_root.is_dir():
            raise FileNotFoundError(f"SMB source is unavailable: {source_root}")
        files: list[tuple[Path, Path]] = []
        for source in source_root.rglob("*"):
            if not source.is_file() or source.is_symlink():
                continue
            if source.suffix.lower() not in self.options.extensions:
                continue
            relative = source.relative_to(source_root)
            files.append((source, relative))
        return sorted(files, key=lambda item: (item[0].stat().st_mtime_ns, str(item[1]).lower()))

    def _sync_handler(self, handler: HandlerSource) -> dict[str, Any]:
        result: dict[str, Any] = {
            "machine_id": handler.machine_id,
            "share_path": handler.share_path,
            "destination": str(handler.destination),
            "status": "ok",
            "copied_files": 0,
            "skipped_oversize": 0,
            "unchanged_files": 0,
            "errors": [],
        }
        mode = handler.connection_mode or self.options.connection_mode
        if mode == "guest":
            _connect_guest(
                handler.share_path,
                handler.username or self.options.guest_username,
                self.options.connect_timeout_seconds,
            )
        try:
            for source, relative in self._source_files(handler):
                if result["copied_files"] >= self.options.max_files_per_machine:
                    result["status"] = "limited"
                    break
                destination = _destination_for(handler.destination, relative)
                key = f"{handler.machine_id}/{relative.as_posix()}"
                if not self.state.needs_copy(key, source, destination):
                    result["unchanged_files"] += 1
                    continue
                if source.stat().st_size > self.options.max_bytes_per_file:
                    result["skipped_oversize"] += 1
                    continue
                try:
                    _copy_stable_file(source, destination, self.options.copy_buffer_bytes)
                    self.state.mark(key, source)
                    result["copied_files"] += 1
                except (OSError, RuntimeError) as error:
                    result["errors"].append({"file": str(relative), "error": str(error)})
        except (OSError, ValueError) as error:
            result["status"] = "error"
            result["errors"].append({"file": None, "error": str(error)})
        if result["errors"] and result["status"] != "error":
            result["status"] = "degraded"
        return result

    def cycle(self) -> dict[str, Any]:
        started = _utc_now()
        handlers = _load_handlers(self.handlers_file)
        results: list[dict[str, Any]] = []
        for handler in handlers:
            try:
                results.append(self._sync_handler(handler))
            except (OSError, RuntimeError, ValueError) as error:
                results.append(
                    {
                        "machine_id": handler.machine_id,
                        "share_path": handler.share_path,
                        "destination": str(handler.destination),
                        "status": "error",
                        "copied_files": 0,
                        "skipped_oversize": 0,
                        "unchanged_files": 0,
                        "errors": [{"file": None, "error": str(error)}],
                    }
                )
        self.state.save()
        summary = {
            "started_at_utc": started,
            "completed_at_utc": _utc_now(),
            "handlers_file": str(self.handlers_file),
            "handlers": results,
            "status": "ready" if all(item["status"] in {"ok", "limited"} for item in results) else "degraded",
        }
        _atomic_json(self.options.state_dir / "latest_sync.json", summary)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally synchronize handler SMB log shares")
    parser.add_argument("--system-config", default="configs/controlled_condition_monitoring.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any handler is degraded")
    args = parser.parse_args()
    result = SMBSyncRunner(args.system_config).cycle()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.strict and result["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
