from __future__ import annotations

import asyncio

import httpx

from app.api import app
from app.model_store import store


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


def test_fleet_contract_uses_supplied_handler_configuration() -> None:
    payload = get("/api/v1/model/fleet").json()
    assert payload["summary"]["configured_handlers"] == 14
    assert {item["machine_id"] for item in payload["machines"]} >= {"MX017", "MX057", "MX070", "MX083"}
    assert all("lifecycle_state" in item for item in payload["machines"])


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
