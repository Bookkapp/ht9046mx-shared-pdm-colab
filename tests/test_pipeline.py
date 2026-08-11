import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from compressor_ml.anomaly import StandardScaler3D, anomaly_score, health_score, pseudo_label
from compressor_ml.config import PipelineConfig
from compressor_ml.features import engineer_features
from compressor_ml.preprocessing import validate_and_filter
from compressor_ml.windowing import make_windows
from compressor_ml.prepare_dataset import (
    MachineSource,
    discover_daily_files,
    even_take,
    prepare_dataset,
    safe_group_name,
)


class PipelineTests(unittest.TestCase):
    def test_prepared_dataset_helpers_are_deterministic(self):
        windows = np.arange(20 * 2 * 3, dtype=np.float32).reshape(20, 2, 3)
        metadata = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=20, freq="s")})
        selected, selected_metadata = even_take(windows, metadata, 5)
        self.assertEqual(selected.shape, (5, 2, 3))
        self.assertEqual(selected_metadata.iloc[0]["timestamp"], metadata.iloc[0]["timestamp"])
        self.assertEqual(selected_metadata.iloc[-1]["timestamp"], metadata.iloc[-1]["timestamp"])
        self.assertEqual(safe_group_name("MX 007", 1), "MX_007__M01")

    def test_daily_file_discovery_accepts_an_explicit_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "2026_07_12_cleaned.csv"
            source.touch()
            self.assertEqual(discover_daily_files(source), [source])

    def test_prepared_dataset_skips_unreadable_source_with_audit_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "MXA"
            second = root / "MXB"
            first.mkdir()
            second.mkdir()
            bad = first / "2026_08_01.txt"
            good_a = first / "2026_08_02.txt"
            good_b = second / "2026_08_02.txt"
            for path in (bad, good_a, good_b):
                path.touch()

            def canonical(machine_id: str) -> pd.DataFrame:
                stamps = pd.date_range("2026-08-02", periods=180, freq="s")
                return pd.DataFrame(
                    {
                        "timestamp": stamps,
                        "machine_id": machine_id,
                        "module_id": 1,
                        "global_status": "Run",
                        "module_status": "On",
                        "busy": 0,
                        "sv": "Off",
                        "hp1": 145.0,
                        "lp1": 28.0,
                        "hp2": 195.0,
                        "lp2": 17.0,
                        "valve": 80.0,
                        "temphi": 90.0,
                        "templo": -30.0,
                    }
                )

            def fake_read(path: Path, machine_id: str) -> pd.DataFrame:
                if Path(path) == bad:
                    raise ValueError("No tabular header found")
                return canonical(machine_id)

            output = root / "prepared"
            with patch("compressor_ml.prepare_dataset.read_handler_log", side_effect=fake_read):
                manifest = prepare_dataset(
                    [MachineSource("MXA", first), MachineSource("MXB", second)],
                    [1],
                    output,
                    PipelineConfig(),
                    "unit_full_v1",
                )

            skipped = [source for source in manifest["sources"] if source["ingest_status"] == "skipped"]
            self.assertEqual(len(skipped), 1)
            self.assertIn("No tabular header found", skipped[0]["ingest_error"])
            quality = pd.read_csv(output / "data_quality_summary.csv")
            skipped_quality = quality.loc[quality["source_status"].eq("skipped")]
            self.assertEqual(len(skipped_quality), 1)
            self.assertIn("No tabular header found", skipped_quality.iloc[0]["read_error"])

    def test_transition_and_sentinel_rows_are_excluded(self):
        stamps = pd.date_range("2026-01-01", periods=70, freq="s")
        frame = pd.DataFrame({
            "timestamp": stamps, "machine_id": "MXTEST", "module_id": 1, "global_status": "Run",
            "module_status": "On", "busy": 0, "sv": "Off",
            "hp1": 145.0, "lp1": 28.0, "hp2": 195.0, "lp2": 17.0, "valve": 80.0, "temphi": 90.0, "templo": -30.0,
        })
        frame.loc[5, "module_status"] = "ChangeValve"
        frame.loc[6, "busy"] = 1
        frame.loc[7, "valve"] = -200
        frame.loc[8, "templo"] = -200
        valid, rejected = validate_and_filter(frame, PipelineConfig())
        self.assertEqual(len(rejected), 4)
        self.assertFalse(valid[["is_transition", "is_homing", "is_sentinel"]].any().any())

    def test_windows_are_not_made_across_transition_gap(self):
        cfg = PipelineConfig(window_size_sec=10, step_size_sec=5)
        stamps = pd.date_range("2026-01-01", periods=80, freq="s")
        frame = pd.DataFrame({
            "timestamp": stamps, "machine_id": "MXTEST", "module_id": 1, "global_status": "Run",
            "module_status": "On", "busy": 0, "sv": "Off",
            "hp1": 145.0, "lp1": 28.0, "hp2": 195.0, "lp2": 17.0, "valve": 80.0, "temphi": 90.0, "templo": -30.0,
        })
        frame.loc[30:35, "module_status"] = "ChangeValve"
        valid, _ = validate_and_filter(frame, cfg)
        featured = engineer_features(valid, cfg)
        windows, meta = make_windows(featured, cfg)
        self.assertGreater(len(windows), 0)
        self.assertTrue((pd.to_datetime(meta["timestamp"]) - pd.to_datetime(meta["window_start"]) == pd.Timedelta(seconds=9)).all())

    def test_scores_preserve_severity_and_health_labels(self):
        scores = anomaly_score(np.array([0.1, 1.0, 5.0]), threshold=1.0)
        self.assertTrue(scores[0] < scores[1] < scores[2] < 1)
        health = health_score(scores, smoothing_windows=1)
        self.assertEqual(pseudo_label(health[0]), "Normal")
        self.assertEqual(pseudo_label(20), "Critical")
        scaler = StandardScaler3D().fit(np.ones((2, 3, 2), dtype=np.float32))
        self.assertTrue(np.isfinite(scaler.transform(np.ones((1, 3, 2), dtype=np.float32))).all())


if __name__ == "__main__":
    unittest.main()
