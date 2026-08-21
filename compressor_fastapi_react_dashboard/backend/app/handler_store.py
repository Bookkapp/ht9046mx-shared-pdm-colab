from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import threading
from typing import Any

from .settings import settings


HANDLER_RE = re.compile(r"^(?:MX)?(?P<number>[0-9]{1,20})$", re.IGNORECASE)
_LOCK = threading.RLock()


def normalize_machine(value: str) -> str:
    match = HANDLER_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError("machine_code must look like MX012")
    return f"MX{match.group('number').zfill(3)}"


def normalize_ip(value: str) -> str:
    try:
        address = ipaddress.ip_address(str(value).strip())
    except ValueError as error:
        raise ValueError("ip must be a valid IPv4 address") from error
    if address.version != 4:
        raise ValueError("Only IPv4 handler addresses are supported")
    return str(address)


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Handlers config not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise ValueError("handlers.json must contain an array of objects")
    return [dict(item) for item in payload]


def _validated(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names: set[str] = set()
    ips: set[str] = set()
    output: list[dict[str, Any]] = []
    for index, source in enumerate(records, start=1):
        name = normalize_machine(str(source.get("name", "")))
        ip = normalize_ip(str(source.get("ip", "")))
        share = str(source.get("share_path", "")).strip().rstrip("\\")
        destination = str(source.get("destination", "")).strip()
        if not share.startswith("\\\\"):
            raise ValueError(f"{name}: share_path must be a Windows UNC path")
        if not destination:
            raise ValueError(f"Handler #{index} requires destination")
        if name in names:
            raise ValueError(f"Duplicate handler name: {name}")
        if ip in ips:
            raise ValueError(f"Duplicate handler IP: {ip}")
        names.add(name)
        ips.add(ip)
        clean = dict(source)
        clean.update(
            {
                "name": name,
                "enabled": bool(source.get("enabled", True)),
                "ip": ip,
                "share_path": share,
                "source_subfolder": str(source.get("source_subfolder", "")).strip("\\/ "),
                "destination": destination,
                "timezone": str(source.get("timezone", settings.timezone)),
                "username": str(source.get("username", "")),
                "password_env": str(source.get("password_env", "")),
                "notes": str(source.get("notes", "")),
            }
        )
        output.append(clean)
    return sorted(output, key=lambda item: int(item["name"][2:]))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    temporary = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _public(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"password", "password_env"}
    }


def list_handlers(path: Path | None = None) -> list[dict[str, Any]]:
    with _LOCK:
        return [_public(item) for item in _validated(_read(path or settings.handlers_file))]


def _new_record(machine_code: str, ip: str) -> dict[str, Any]:
    share_name = settings.handler_share_template.format(
        machine_code=machine_code,
        machine_number=machine_code[2:],
        ip=ip,
    ).strip("\\/ ")
    if not share_name or "\\" in share_name or "/" in share_name:
        raise ValueError("HANDLER_SHARE_TEMPLATE must produce one SMB share name")
    return {
        "name": machine_code,
        "enabled": True,
        "ip": ip,
        "share_path": f"\\\\{ip}\\{share_name}",
        "source_subfolder": "",
        "destination": str(
            settings.handler_destination_root / f"Comp_log_data_{machine_code}"
        ),
        "timezone": settings.timezone,
        "username": "",
        "password_env": "",
        "notes": "Managed from the Controlled Hybrid model monitor",
    }


def _sync_source(machine_code: str, destination: str | None) -> None:
    if not settings.sync_controlled_sources or not settings.controlled_system_config.exists():
        return
    payload = json.loads(settings.controlled_system_config.read_text(encoding="utf-8"))
    sources = payload.setdefault("machine_sources", {})
    if destination is None:
        current = str(sources.get(machine_code, ""))
        expected = str(settings.handler_destination_root / f"Comp_log_data_{machine_code}")
        if current.casefold() == expected.casefold():
            sources.pop(machine_code, None)
    else:
        # Preserve an explicit existing source (useful for replay workstations),
        # but automatically onboard machines not yet known to the model runner.
        sources.setdefault(machine_code, destination)
    _atomic_json(settings.controlled_system_config, payload)


def sync_all_handler_sources() -> dict[str, str]:
    if not settings.sync_controlled_sources or not settings.controlled_system_config.exists():
        return {}
    records = [item for item in _validated(_read(settings.handlers_file)) if item["enabled"]]
    with _LOCK:
        payload = json.loads(settings.controlled_system_config.read_text(encoding="utf-8"))
        sources = payload.setdefault("machine_sources", {})
        for item in records:
            sources.setdefault(item["name"], item["destination"])
        _atomic_json(settings.controlled_system_config, payload)
        return dict(sources)


def create_handler(machine_code: str, ip: str, path: Path | None = None) -> dict[str, Any]:
    name = normalize_machine(machine_code)
    address = normalize_ip(ip)
    config_path = path or settings.handlers_file
    with _LOCK:
        records = _validated(_read(config_path))
        if any(item["name"] == name for item in records):
            raise ValueError(f"Handler already exists: {name}")
        if any(item["ip"] == address for item in records):
            raise ValueError(f"Handler IP already exists: {address}")
        record = _new_record(name, address)
        records.append(record)
        _atomic_json(config_path, _validated(records))
        if path is None:
            _sync_source(name, record["destination"])
        return _public(record)


def update_handler(
    machine_code: str,
    *,
    ip: str | None = None,
    enabled: bool | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    name = normalize_machine(machine_code)
    address = normalize_ip(ip) if ip is not None else None
    if address is None and enabled is None:
        raise ValueError("Provide ip or enabled to update a handler")
    config_path = path or settings.handlers_file
    with _LOCK:
        records = _validated(_read(config_path))
        target = next((item for item in records if item["name"] == name), None)
        if target is None:
            raise ValueError(f"Unknown handler: {name}")
        if address is not None:
            if any(item["ip"] == address and item["name"] != name for item in records):
                raise ValueError(f"Handler IP already exists: {address}")
            share_name = PureWindowsPath(target["share_path"]).name
            target["ip"] = address
            target["share_path"] = f"\\\\{address}\\{share_name}"
        if enabled is not None:
            target["enabled"] = bool(enabled)
        _atomic_json(config_path, _validated(records))
        if path is None and target["enabled"]:
            _sync_source(name, target["destination"])
        return _public(target)


def delete_handler(machine_code: str, path: Path | None = None) -> dict[str, Any]:
    name = normalize_machine(machine_code)
    config_path = path or settings.handlers_file
    with _LOCK:
        records = _validated(_read(config_path))
        target = next((item for item in records if item["name"] == name), None)
        if target is None:
            raise ValueError(f"Unknown handler: {name}")
        _atomic_json(config_path, [item for item in records if item["name"] != name])
        if path is None:
            _sync_source(name, None)
        return _public(target)
