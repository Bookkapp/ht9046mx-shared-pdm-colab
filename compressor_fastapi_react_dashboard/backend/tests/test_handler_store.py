from __future__ import annotations

import json

import pytest

from app.handler_store import (
    create_handler,
    delete_handler,
    list_handlers,
    normalize_machine,
    update_handler,
)


def test_handler_crud_derives_paths_and_keeps_no_password_in_api(tmp_path) -> None:
    path = tmp_path / "handlers.json"
    path.write_text("[]\n", encoding="utf-8")
    created = create_handler("mx12", "10.196.132.12", path)
    assert created["name"] == "MX012"
    assert created["share_path"] == r"\\10.196.132.12\Comp_log_data_MX012"
    assert created["destination"].endswith("Comp_log_data_MX012")
    assert "password_env" not in created
    assert path.with_suffix(".json.bak").is_file()

    updated = update_handler("MX012", ip="10.196.132.13", enabled=False, path=path)
    assert updated["enabled"] is False
    assert updated["share_path"].startswith("\\\\10.196.132.13\\")
    removed = delete_handler("MX012", path)
    assert removed["name"] == "MX012"
    assert list_handlers(path) == []
    assert json.loads(path.read_text(encoding="utf-8")) == []


def test_handler_validation_rejects_duplicate_ip_and_bad_machine(tmp_path) -> None:
    path = tmp_path / "handlers.json"
    path.write_text("[]\n", encoding="utf-8")
    create_handler("MX057", "10.196.132.182", path)
    with pytest.raises(ValueError, match="IP already exists"):
        create_handler("MX058", "10.196.132.182", path)
    with pytest.raises(ValueError, match="must look like"):
        normalize_machine("MX057; DROP TABLE")
