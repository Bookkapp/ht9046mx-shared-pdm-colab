"""Five-minute Controlled Hybrid runner backed exclusively by MySQL telemetry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..mysql_source import MySQLReadingsSource
from .config import ControlledMonitoringConfig
from .engine import ControlledMonitoringEngine
from .fusion import PersistenceTracker
from .lifecycle import BootstrapLifecycle, LifecycleState, ProfileRepository
from .shadow import SharedLSTMShadow


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class ControlledSystemConfig:
    policy_file: str
    shared_model_artifact: str
    runtime_dir: str
    modules: list[int]
    mysql: dict[str, Any]
    env_file: str | None = None
    bootstrap_history_days: int = 120
    cycle_lookback_minutes: int = 15

    @classmethod
    def load(cls, path: str | Path) -> "ControlledSystemConfig":
        config_path = Path(path).resolve()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("mysql"), dict):
            raise ValueError("controlled system config requires a mysql object")
        system = cls(**payload)
        if not system.modules or any(int(module) < 1 or int(module) > 8 for module in system.modules):
            raise ValueError("modules must contain values from 1 through 8")
        if system.bootstrap_history_days < 1:
            raise ValueError("bootstrap_history_days must be at least 1")
        if system.cycle_lookback_minutes < 5:
            raise ValueError("cycle_lookback_minutes must be at least 5")
        if system.env_file:
            env_path = Path(system.env_file)
            system.env_file = str(
                env_path if env_path.is_absolute() else (config_path.parent / env_path)
            )
        return system


class MySQLCursorRegistry:
    """Stores only the last consumed telemetry time per machine."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.entries = _read_json(self.path, {})

    def get(self, machine_id: str) -> pd.Timestamp | None:
        value = self.entries.get(machine_id)
        if not value:
            return None
        stamp = pd.Timestamp(value)
        return stamp if stamp.tzinfo else stamp.tz_localize("Asia/Bangkok")

    def set(self, machine_id: str, timestamp: pd.Timestamp) -> None:
        self.entries[machine_id] = pd.Timestamp(timestamp).isoformat()

    def save(self) -> None:
        _atomic_json(self.path, self.entries)


class ControlledRuntime:
    def __init__(self, system_config_path: str | Path) -> None:
        self.system_config_path = Path(system_config_path).resolve()
        self.system = ControlledSystemConfig.load(self.system_config_path)
        self.policy = ControlledMonitoringConfig.load(self.system.policy_file)
        self.runtime_dir = Path(self.system.runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.source = MySQLReadingsSource.from_mapping(
            self.system.mysql, env_file=self.system.env_file
        )
        self.repository = ProfileRepository(self.runtime_dir / "profiles")
        self.shadow = SharedLSTMShadow(self.system.shared_model_artifact, config=self.policy)
        self.lifecycle = BootstrapLifecycle(self.repository, self.policy, self.shadow)
        self.persistence = PersistenceTracker(
            self.policy, self.runtime_dir / "state" / "persistence.json"
        )
        self.engine = ControlledMonitoringEngine(
            self.repository,
            self.policy,
            self.shadow,
            persistence=self.persistence,
        )
        self.cursors = MySQLCursorRegistry(self.runtime_dir / "state" / "mysql_cursors.json")

    def machine_ids(self) -> list[str]:
        return self.source.machines()

    def _load(self, machine_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        frame = self.source.read_machine(machine_id, start, end)
        if frame.empty:
            return frame
        return frame.loc[frame["module_id"].isin(self.system.modules)].copy()

    @staticmethod
    def _latest_row_timestamp(frame: pd.DataFrame) -> pd.Timestamp | None:
        if frame.empty:
            return None
        value = pd.Timestamp(frame["timestamp"].max())
        return None if pd.isna(value) else value

    def bootstrap_machine(self, machine_id: str) -> dict[str, Any]:
        latest = self.source.latest_timestamp(machine_id)
        if latest is None:
            return self.repository.transition(
                machine_id,
                LifecycleState.COLLECTING_DATA,
                reason="no_mysql_telemetry",
            )
        start = latest - pd.Timedelta(days=self.system.bootstrap_history_days)
        raw = self._load(machine_id, start, latest + pd.Timedelta(seconds=1))
        if raw.empty:
            return self.repository.transition(
                machine_id,
                LifecycleState.COLLECTING_DATA,
                reason="no_readable_mysql_telemetry",
            )
        result = self.lifecycle.bootstrap(machine_id, raw)
        if result.get("state") == LifecycleState.CANDIDATE_PROFILE_READY.value:
            result = self.lifecycle.begin_shadow(machine_id)
            consumed = self._latest_row_timestamp(raw)
            if consumed is not None:
                self.cursors.set(machine_id, consumed)
                self.cursors.save()
        result["rows"] = int(len(raw))
        result["history_start"] = start.isoformat()
        result["history_end"] = latest.isoformat()
        return result

    def _append_predictions(self, machine_id: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        destination = self.runtime_dir / "predictions" / f"{machine_id}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as handle:
            for record in frame.to_dict(orient="records"):
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def cycle_machine(self, machine_id: str) -> dict[str, Any]:
        lifecycle = self.repository.read_lifecycle(machine_id)
        state = lifecycle.get("state")
        if state == LifecycleState.REJECTED.value:
            return {
                "machine_id": machine_id,
                "state": state,
                "rows": 0,
                "windows": 0,
                "reason": "candidate_rejected_requires_new_bootstrap",
            }
        if state in {LifecycleState.COLLECTING_DATA.value, LifecycleState.LEARNING.value}:
            return self.bootstrap_machine(machine_id)

        latest = self.source.latest_timestamp(machine_id)
        if latest is None:
            return {"machine_id": machine_id, "state": state, "rows": 0, "windows": 0}
        cursor = self.cursors.get(machine_id)
        if cursor is None:
            cursor = latest - pd.Timedelta(minutes=self.system.cycle_lookback_minutes)
        # Include one extra window before the cursor so new five-minute windows
        # retain context, then only persist decisions that are actually new.
        start = cursor - pd.Timedelta(seconds=self.policy.window_size_sec)
        raw = self._load(machine_id, start, latest + pd.Timedelta(seconds=1))
        if raw.empty:
            return {"machine_id": machine_id, "state": state, "rows": 0, "windows": 0}
        profile_source = "active" if state == LifecycleState.ACTIVE.value else "candidate"
        scored = self.engine.score_frame(raw, profile_source=profile_source)
        if not scored.empty:
            scored = scored.loc[pd.to_datetime(scored["event_time"]) > cursor].copy()
        self._append_predictions(machine_id, scored)
        consumed = self._latest_row_timestamp(raw)
        if consumed is not None:
            self.cursors.set(machine_id, consumed)
            self.cursors.save()
        if state == LifecycleState.SHADOW_VALIDATION.value and not scored.empty:
            lifecycle = self.lifecycle.record_shadow(machine_id, scored)
        return {
            "machine_id": machine_id,
            "state": lifecycle.get("state"),
            "rows": int(len(raw)),
            "windows": int(len(scored)),
            "review_levels": (
                scored["review_level"].value_counts().to_dict() if not scored.empty else {}
            ),
            "cursor": consumed.isoformat() if consumed is not None else None,
        }

    def cycle(self) -> dict[str, Any]:
        started = datetime.now(timezone.utc).isoformat()
        results: list[dict[str, Any]] = []
        for machine_id in self.machine_ids():
            try:
                results.append(self.cycle_machine(machine_id))
            except Exception as error:
                results.append({"machine_id": machine_id, "error": f"{type(error).__name__}: {error}"})
        summary = {
            "started_at_utc": started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "policy_version": self.policy.policy_version,
            "shared_model_artifact": self.system.shared_model_artifact,
            "source": "mysql",
            "machines": results,
        }
        _atomic_json(self.runtime_dir / "latest_cycle.json", summary)
        return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Controlled COM2 + Shared-LSTM MySQL monitoring")
    parser.add_argument("--system-config", default="configs/controlled_condition_monitoring.json")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("cycle")
    bootstrap = commands.add_parser("bootstrap")
    bootstrap.add_argument("--machine-id")
    status = commands.add_parser("status")
    status.add_argument("--machine-id")
    commands.add_parser("source-check")
    approve = commands.add_parser("approve")
    approve.add_argument("--machine-id", required=True)
    approve.add_argument("--approved-by", required=True)
    reject = commands.add_parser("reject")
    reject.add_argument("--machine-id", required=True)
    reject.add_argument("--rejected-by", required=True)
    reject.add_argument("--reason", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    system = ControlledSystemConfig.load(args.system_config)
    repository = ProfileRepository(Path(system.runtime_dir) / "profiles")
    source = MySQLReadingsSource.from_mapping(system.mysql, env_file=system.env_file)
    if args.command == "source-check":
        health = source.health()
        result = {"mysql": health, "machines": [], "columns": []}
        if health.get("connected"):
            result["machines"] = source.machines()
            result["columns"] = source.table_columns()
    elif args.command == "status":
        machines = [args.machine_id] if args.machine_id else source.machines()
        result = {machine: repository.read_lifecycle(machine) for machine in machines}
    elif args.command == "approve":
        result = repository.approve(args.machine_id, approved_by=args.approved_by)
    elif args.command == "reject":
        result = repository.reject(args.machine_id, rejected_by=args.rejected_by, reason=args.reason)
    else:
        runtime = ControlledRuntime(args.system_config)
        if args.command == "cycle":
            result = runtime.cycle()
        elif args.command == "bootstrap":
            machines = [args.machine_id] if args.machine_id else runtime.machine_ids()
            result = {machine: runtime.bootstrap_machine(machine) for machine in machines}
        else:
            raise RuntimeError(f"Unknown command: {args.command}")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
