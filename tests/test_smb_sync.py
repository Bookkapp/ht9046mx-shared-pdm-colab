from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from compressor_ml.controlled_monitoring.runner import ControlledSystemConfig
from compressor_ml.smb_sync import HandlerSource, SMBSyncRunner


class SMBSyncTests(unittest.TestCase):
    def test_persistent_handler_registry_overrides_and_disables_legacy_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            handlers = root / "handlers.json"
            handlers.write_text(
                json.dumps(
                    [
                        {"name": "MX001", "enabled": True, "destination": "D:/incoming/MX001"},
                        {"name": "MX002", "enabled": False, "destination": "D:/incoming/MX002"},
                    ]
                ),
                encoding="utf-8",
            )
            config = root / "monitoring.json"
            config.write_text(
                json.dumps(
                    {
                        "policy_file": "policy.json",
                        "shared_model_artifact": "artifact",
                        "runtime_dir": "runtime",
                        "handlers_file": str(handlers),
                        "machine_sources": {"MX001": "old", "MX002": "old", "MX003": "replay"},
                        "modules": [1],
                    }
                ),
                encoding="utf-8",
            )
            system = ControlledSystemConfig.load(config)
            self.assertEqual(system.machine_sources["MX001"], "D:/incoming/MX001")
            self.assertNotIn("MX002", system.machine_sources)
            self.assertEqual(system.machine_sources["MX003"], "replay")

    def _runner(self, root: Path) -> SMBSyncRunner:
        handlers = root / "handlers.json"
        handlers.write_text("[]\n", encoding="utf-8")
        config = root / "monitoring.json"
        config.write_text(
            json.dumps(
                {
                    "handlers_file": str(handlers),
                    "sync": {
                        "state_dir": str(root / "sync_state"),
                        "extensions": [".txt"],
                        "max_files_per_machine": 10,
                        "max_bytes_per_file": 1024 * 1024,
                    },
                }
            ),
            encoding="utf-8",
        )
        return SMBSyncRunner(config)

    def test_incremental_copy_preserves_relative_paths_and_skips_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            nested = source / "2026_08"
            nested.mkdir(parents=True)
            log = nested / "2026_08_01.txt"
            log.write_text("first", encoding="utf-8")
            runner = self._runner(root)
            handler = HandlerSource("MX001", str(source), "", destination)

            first = runner._sync_handler(handler)
            runner.state.save()
            self.assertEqual(first["copied_files"], 1)
            self.assertEqual((destination / "2026_08" / log.name).read_text(encoding="utf-8"), "first")

            second = runner._sync_handler(handler)
            self.assertEqual(second["copied_files"], 0)
            self.assertEqual(second["unchanged_files"], 1)

            log.write_text("changed", encoding="utf-8")
            third = runner._sync_handler(handler)
            self.assertEqual(third["copied_files"], 1)
            self.assertEqual((destination / "2026_08" / log.name).read_text(encoding="utf-8"), "changed")

    def test_oversize_file_is_left_for_a_later_policy_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "large.txt").write_bytes(b"x" * 128)
            runner = self._runner(root)
            runner.options = runner.options.__class__(
                state_dir=runner.options.state_dir,
                extensions=runner.options.extensions,
                max_files_per_machine=runner.options.max_files_per_machine,
                max_bytes_per_file=64,
                copy_buffer_bytes=runner.options.copy_buffer_bytes,
                connection_mode=runner.options.connection_mode,
                guest_username=runner.options.guest_username,
                connect_timeout_seconds=runner.options.connect_timeout_seconds,
            )
            result = runner._sync_handler(HandlerSource("MX001", str(source), "", root / "destination"))
            self.assertEqual(result["copied_files"], 0)
            self.assertEqual(result["skipped_oversize"], 1)
            self.assertFalse((root / "destination" / "large.txt").exists())


if __name__ == "__main__":
    unittest.main()
