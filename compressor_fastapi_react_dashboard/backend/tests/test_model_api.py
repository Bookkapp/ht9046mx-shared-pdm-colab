from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pandas as pd

from app.api import app
from app.model_store import store


class _Source:
    config = SimpleNamespace(
        host="10.195.17.73",
        readings_table="ht9046mx_readings",
        timezone="Asia/Bangkok",
    )

    def health(self):
        return {"connected": True, "host": self.config.host, "readings_table": self.config.readings_table}

    def machines(self):
        return ["MX017", "MX057", "MX070", "MX083"]

    def latest_by_machine(self):
        now = pd.Timestamp.now(tz=self.config.timezone)
        return {
            "MX017": now - pd.Timedelta(minutes=5),
            "MX057": now - pd.Timedelta(minutes=31),
        }


store.source = _Source()


def get(path: str) -> httpx.Response:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path)

    return asyncio.run(request())


def test_artifact_contract_points_to_full_immutable_shared_model() -> None:
    payload = get("/api/v1/model/artifact").json()
    assert payload["available"] is True
    assert payload["model_version"] == "shared_lstm_full_v1"
    assert payload["epochs_completed"] == 30
    assert payload["group_count"] == 42
    assert payload["input_shape"] == [60, 24]
    assert payload["weights_mutable"] is False


def test_pipeline_contract_exposes_gates_equations_and_human_activation() -> None:
    response = get("/api/v1/model/pipeline")
    assert response.status_code == 200
    payload = response.json()
    keys = {stage["key"] for stage in payload["stages"]}
    assert {"quality", "gmm", "robust", "ridge", "iforest", "lstm", "fusion"} <= keys
    assert payload["policy"]["activation_policy"] == "human_approval"
    assert payload["policy"]["shared_lstm_weights_mutable"] is False


def test_fleet_contract_uses_mysql_machine_directory() -> None:
    payload = get("/api/v1/model/fleet").json()
    assert payload["summary"]["mysql_machines"] == 4
    assert payload["source"]["status"] == "ONLINE"
    assert payload["source"]["stale_after_minutes"] == 30
    assert payload["summary"]["mysql_online"] == 1
    assert payload["summary"]["mysql_stale"] == 1
    assert {item["machine_id"] for item in payload["machines"]} >= {"MX017", "MX057", "MX070", "MX083"}
    assert all("lifecycle_state" in item for item in payload["machines"])
    states = {item["machine_id"]: item["telemetry_status"] for item in payload["machines"]}
    assert states == {"MX017": "ONLINE", "MX057": "STALE", "MX070": "NO_DATA", "MX083": "NO_DATA"}


def test_source_status_reports_mysql_connectivity() -> None:
    response = get("/api/v1/source/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["connected"] is True
    assert payload["host"] == "10.195.17.73"


def test_flattened_evidence_keeps_com2_and_lstm_separate() -> None:
    point = store._flatten_prediction(
        {
            "event_time": "2026-08-01T00:00:00+07:00",
            "review_level": "SHADOW",
            "com2": {"active": True, "score": 4.2, "details": {"z_hp2": 4.2}},
            "lstm": {"active": False, "score": 0.2, "threshold": 0.4, "details": {}},
        }
    )
    assert point["com2_active"] is True
    assert point["z_hp2"] == 4.2
    assert point["lstm_active"] is False
    assert point["lstm_ratio"] == 0.5
    assert "combined_risk" not in point
