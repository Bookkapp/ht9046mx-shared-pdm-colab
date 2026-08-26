"""Read HT9046MX telemetry from MySQL and normalize it for the model pipeline.

The production reader is deliberately read-only.  It supports the two layouts
encountered in handler imports: one row per machine/time with suffixed module
columns (wide), or one row per machine/module/time (long).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd
from dotenv import load_dotenv

from .machine import database_machine_candidates, normalize_machine


IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
CANONICAL_COLUMNS = (
    "timestamp",
    "machine_id",
    "module_id",
    "global_status",
    "module_status",
    "busy",
    "sv",
    "hp1",
    "lp1",
    "hp2",
    "lp2",
    "valve",
    "temphi",
    "templo",
)
SENSOR_ALIASES = {
    "hp1": ("hp1", "hp1st", "highpressure1st"),
    "lp1": ("lp1", "lp1st", "lowpressure1st"),
    "hp2": ("hp2", "hp2nd", "highpressure2nd"),
    "lp2": ("lp2", "lp2nd", "lowpressure2nd"),
    "valve": ("valve", "valveposition"),
    "temphi": ("temphi", "temperaturehi", "hightemperature"),
    "templo": ("templo", "temperaturelo", "lowtemperature"),
}


def _identifier(value: str, label: str) -> str:
    text = str(value).strip()
    if not IDENTIFIER_RE.fullmatch(text):
        raise ValueError(f"{label} must be a simple MySQL identifier")
    return f"`{text}`"


def _normal(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _first_column(columns: Iterable[str], *aliases: str) -> str | None:
    normalized = {_normal(column): str(column) for column in columns}
    for alias in aliases:
        result = normalized.get(_normal(alias))
        if result is not None:
            return result
    return None


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns, Asia/Bangkok]"),
            "machine_id": pd.Series(dtype=str),
            "module_id": pd.Series(dtype=int),
            "global_status": pd.Series(dtype=str),
            "module_status": pd.Series(dtype=str),
            "busy": pd.Series(dtype=float),
            "sv": pd.Series(dtype=str),
            "hp1": pd.Series(dtype=float),
            "lp1": pd.Series(dtype=float),
            "hp2": pd.Series(dtype=float),
            "lp2": pd.Series(dtype=float),
            "valve": pd.Series(dtype=float),
            "temphi": pd.Series(dtype=float),
            "templo": pd.Series(dtype=float),
        }
    )


@dataclass(frozen=True)
class MySQLSourceConfig:
    host: str
    port: int
    database: str
    user_env: str
    password_env: str
    readings_table: str
    machine_column: str
    timestamp_column: str
    module_column: str | None
    timezone: str
    sensor_scale: str
    connect_timeout_seconds: int
    read_timeout_seconds: int
    max_rows_per_query: int

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any],
        *,
        env_file: str | Path | None = None,
    ) -> "MySQLSourceConfig":
        if env_file:
            load_dotenv(Path(env_file), override=False)

        def configured(name: str, default: Any) -> Any:
            env_name = str(payload.get(f"{name}_env", "")).strip()
            if env_name and os.getenv(env_name, "").strip():
                return os.getenv(env_name, "").strip()
            return payload.get(name, default)

        module_column = str(configured("module_column", "")).strip() or None
        return cls(
            host=str(configured("host", "10.195.17.73")).strip(),
            port=int(configured("port", 3306)),
            database=str(configured("database", "ht9046mx_iot")).strip(),
            user_env=str(payload.get("user_env", "MYSQL_USER")),
            password_env=str(payload.get("password_env", "MYSQL_PASSWORD")),
            readings_table=str(configured("readings_table", "ht9046mx_readings")).strip(),
            machine_column=str(configured("machine_column", "machine_number")).strip(),
            timestamp_column=str(configured("timestamp_column", "recorded_at")).strip(),
            module_column=module_column,
            timezone=str(payload.get("timezone", "Asia/Bangkok")).strip(),
            sensor_scale=str(payload.get("sensor_scale", "auto")).strip().lower(),
            connect_timeout_seconds=int(payload.get("connect_timeout_seconds", 10)),
            read_timeout_seconds=int(payload.get("read_timeout_seconds", 300)),
            max_rows_per_query=int(payload.get("max_rows_per_query", 2_000_000)),
        )


class MySQLReadingsSource:
    """A read-only MySQL telemetry source with schema-safe SQL identifiers."""

    def __init__(self, config: MySQLSourceConfig) -> None:
        self.config = config

    @classmethod
    def from_mapping(
        cls, payload: dict[str, Any], *, env_file: str | Path | None = None
    ) -> "MySQLReadingsSource":
        return cls(MySQLSourceConfig.from_mapping(payload, env_file=env_file))

    def _connect(self):
        try:
            import mysql.connector
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "mysql-connector-python is required for the MySQL data source"
            ) from error
        user = os.getenv(self.config.user_env, "").strip()
        password = os.getenv(self.config.password_env, "")
        if not user or not password:
            raise RuntimeError(
                f"Missing required MySQL credentials: {self.config.user_env}, {self.config.password_env}"
            )
        return mysql.connector.connect(
            host=self.config.host,
            port=self.config.port,
            user=user,
            password=password,
            database=self.config.database,
            connection_timeout=self.config.connect_timeout_seconds,
            read_timeout=self.config.read_timeout_seconds,
            autocommit=True,
        )

    def _query(self, query: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
        connection = self._connect()
        try:
            cursor = connection.cursor(dictionary=True)
            try:
                cursor.execute(query, params)
                return pd.DataFrame(cursor.fetchall())
            finally:
                cursor.close()
        finally:
            connection.close()

    def table_columns(self) -> list[str]:
        table = _identifier(self.config.readings_table, "readings_table")
        frame = self._query(f"SHOW COLUMNS FROM {table}")
        return [str(value) for value in frame.get("Field", pd.Series(dtype=str)).tolist()]

    def health(self) -> dict[str, Any]:
        try:
            frame = self._query("SELECT 1 AS ok")
            columns = self.table_columns()
            return {
                "connected": bool(not frame.empty),
                "host": self.config.host,
                "port": self.config.port,
                "database": self.config.database,
                "readings_table": self.config.readings_table,
                "column_count": len(columns),
            }
        except Exception as error:
            return {
                "connected": False,
                "host": self.config.host,
                "port": self.config.port,
                "database": self.config.database,
                "readings_table": self.config.readings_table,
                "error": f"{type(error).__name__}: {error}",
            }

    def machines(self) -> list[str]:
        table = _identifier(self.config.readings_table, "readings_table")
        machine = _identifier(self.config.machine_column, "machine_column")
        frame = self._query(
            f"SELECT DISTINCT {machine} AS machine_value FROM {table} "
            f"WHERE {machine} IS NOT NULL ORDER BY {machine}"
        )
        machines: list[str] = []
        for value in frame.get("machine_value", pd.Series(dtype=object)).tolist():
            try:
                machines.append(normalize_machine(str(value)))
            except ValueError:
                continue
        return sorted(set(machines))

    def latest_by_machine(self) -> dict[str, pd.Timestamp]:
        table = _identifier(self.config.readings_table, "readings_table")
        machine = _identifier(self.config.machine_column, "machine_column")
        timestamp = _identifier(self.config.timestamp_column, "timestamp_column")
        frame = self._query(
            f"SELECT {machine} AS machine_value, MAX({timestamp}) AS latest_at "
            f"FROM {table} WHERE {machine} IS NOT NULL GROUP BY {machine}"
        )
        result: dict[str, pd.Timestamp] = {}
        for row in frame.to_dict(orient="records"):
            try:
                machine_id = normalize_machine(str(row.get("machine_value")))
                stamp = self._timestamp_series(pd.Series([row.get("latest_at")])).iloc[0]
            except (TypeError, ValueError, IndexError):
                continue
            if not pd.isna(stamp):
                result[machine_id] = stamp
        return result

    def latest_timestamp(self, machine_id: str) -> pd.Timestamp | None:
        table = _identifier(self.config.readings_table, "readings_table")
        machine = _identifier(self.config.machine_column, "machine_column")
        timestamp = _identifier(self.config.timestamp_column, "timestamp_column")
        candidates = database_machine_candidates(machine_id)
        placeholders = ", ".join("%s" for _ in candidates)
        frame = self._query(
            f"SELECT MAX({timestamp}) AS latest_at FROM {table} "
            f"WHERE {machine} IN ({placeholders})",
            tuple(candidates),
        )
        if frame.empty:
            return None
        stamp = self._timestamp_series(pd.Series([frame.iloc[0].get("latest_at")])).iloc[0]
        return None if pd.isna(stamp) else stamp

    def read_machine(
        self,
        machine_id: str,
        start: pd.Timestamp | datetime | None = None,
        end: pd.Timestamp | datetime | None = None,
    ) -> pd.DataFrame:
        table = _identifier(self.config.readings_table, "readings_table")
        machine = _identifier(self.config.machine_column, "machine_column")
        timestamp = _identifier(self.config.timestamp_column, "timestamp_column")
        candidates = database_machine_candidates(machine_id)
        clauses = [f"{machine} IN ({', '.join('%s' for _ in candidates)})"]
        params: list[Any] = list(candidates)
        if start is not None:
            clauses.append(f"{timestamp} >= %s")
            params.append(self._database_time(start))
        if end is not None:
            clauses.append(f"{timestamp} < %s")
            params.append(self._database_time(end))
        params.append(self.config.max_rows_per_query)
        frame = self._query(
            f"SELECT * FROM {table} WHERE {' AND '.join(clauses)} "
            f"ORDER BY {timestamp} ASC LIMIT %s",
            tuple(params),
        )
        return self.canonicalize(frame, machine_id)

    def canonicalize(self, frame: pd.DataFrame, machine_id: str) -> pd.DataFrame:
        """Convert a MySQL result into the canonical model input columns."""
        if frame.empty:
            return _empty_frame()
        columns = [str(column) for column in frame.columns]
        timestamp_column = self.config.timestamp_column if self.config.timestamp_column in frame else _first_column(
            columns, "recorded_at", "timestamp", "logged_at", "created_at", "datetime"
        )
        if timestamp_column is None:
            raise ValueError("MySQL readings table has no configured timestamp column")
        long_module = self.config.module_column or _first_column(
            columns, "module_id", "module_number", "module", "module_no"
        )
        if long_module and any(
            _first_column(columns, *aliases) is not None
            for aliases in SENSOR_ALIASES.values()
        ):
            output = self._canonical_long(frame, machine_id, timestamp_column, long_module)
        else:
            output = self._canonical_wide(frame, machine_id, timestamp_column)
        if output.empty:
            return _empty_frame()
        output["timestamp"] = self._timestamp_series(output["timestamp"])
        output = output.dropna(subset=["timestamp", "module_id"]).copy()
        output["module_id"] = pd.to_numeric(output["module_id"], errors="coerce")
        output = output.dropna(subset=["module_id"])
        output["module_id"] = output["module_id"].astype(int)
        output = output.loc[output["module_id"].between(1, 8)].sort_values(
            ["machine_id", "module_id", "timestamp"]
        )
        output = self._apply_sensor_scale(output).reset_index(drop=True)
        return output.loc[:, CANONICAL_COLUMNS]

    def _canonical_long(
        self, frame: pd.DataFrame, machine_id: str, timestamp_column: str, module_column: str
    ) -> pd.DataFrame:
        columns = [str(column) for column in frame.columns]
        values: dict[str, Any] = {
            "timestamp": frame[timestamp_column],
            "machine_id": normalize_machine(machine_id),
            "module_id": pd.to_numeric(frame[module_column], errors="coerce"),
            "global_status": self._text_column(frame, _first_column(columns, "status", "global_status")),
            "module_status": self._text_column(frame, _first_column(columns, "module_status", "status_module")),
            "busy": self._numeric_column(frame, _first_column(columns, "busy", "is_busy")),
            "sv": self._text_column(frame, _first_column(columns, "sv", "set_value", "setvalue")),
        }
        for target, aliases in SENSOR_ALIASES.items():
            values[target] = self._numeric_column(frame, _first_column(columns, *aliases))
        return pd.DataFrame(values).dropna(subset=["module_id"])

    def _canonical_wide(
        self, frame: pd.DataFrame, machine_id: str, timestamp_column: str
    ) -> pd.DataFrame:
        columns = [str(column) for column in frame.columns]
        parts: list[pd.DataFrame] = []
        global_status = self._text_column(frame, _first_column(columns, "status", "global_status"))
        for module_id in range(1, 9):
            values: dict[str, Any] = {
                "timestamp": frame[timestamp_column],
                "machine_id": normalize_machine(machine_id),
                "module_id": module_id,
                "global_status": global_status,
                "module_status": self._text_column(frame, self._module_column(columns, ("status", "module_status"), module_id)),
                "busy": self._numeric_column(frame, self._module_column(columns, ("busy", "is_busy"), module_id)),
                "sv": self._text_column(frame, self._module_column(columns, ("sv", "set_value", "setvalue"), module_id)),
            }
            found_sensors = 0
            for target, aliases in SENSOR_ALIASES.items():
                column = self._module_column(columns, aliases, module_id)
                if column is not None:
                    found_sensors += 1
                values[target] = self._numeric_column(frame, column)
            if found_sensors:
                parts.append(pd.DataFrame(values))
        if not parts:
            raise ValueError(
                "MySQL readings columns do not match supported wide or long telemetry names"
            )
        return pd.concat(parts, ignore_index=True)

    @staticmethod
    def _module_column(columns: list[str], aliases: Iterable[str], module_id: int) -> str | None:
        normalized = {_normal(column): column for column in columns}
        for alias in aliases:
            base = _normal(alias)
            for candidate in (f"{base}{module_id}", f"{base}m{module_id}", f"m{module_id}{base}"):
                if candidate in normalized:
                    return normalized[candidate]
        return None

    @staticmethod
    def _numeric_column(frame: pd.DataFrame, column: str | None) -> pd.Series:
        return pd.to_numeric(frame[column], errors="coerce") if column else pd.Series(float("nan"), index=frame.index)

    @staticmethod
    def _text_column(frame: pd.DataFrame, column: str | None) -> pd.Series:
        return frame[column].astype(str).str.strip() if column else pd.Series("<blank>", index=frame.index)

    def _timestamp_series(self, values: pd.Series) -> pd.Series:
        parsed = pd.to_datetime(values, errors="coerce")
        if getattr(parsed.dt, "tz", None) is None:
            return parsed.dt.tz_localize(self.config.timezone, ambiguous="NaT", nonexistent="NaT")
        return parsed.dt.tz_convert(self.config.timezone)

    def _database_time(self, value: pd.Timestamp | datetime) -> datetime:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert(self.config.timezone).tz_localize(None)
        return stamp.to_pydatetime()

    def _apply_sensor_scale(self, frame: pd.DataFrame) -> pd.DataFrame:
        scale = self.config.sensor_scale
        if scale not in {"auto", "1", "0.1"}:
            raise ValueError("sensor_scale must be auto, 1, or 0.1")
        factor = float(scale) if scale != "auto" else 1.0
        if scale == "auto":
            pressure = frame[["hp1", "lp1", "hp2", "lp2"]].stack().abs().median()
            temperature = frame[["temphi", "templo"]].stack().abs().median()
            if (pd.notna(pressure) and pressure > 1_000) or (pd.notna(temperature) and temperature > 250):
                factor = 0.1
        if factor != 1.0:
            for column in ("hp1", "lp1", "hp2", "lp2", "temphi", "templo"):
                frame[column] = frame[column] * factor
        return frame
