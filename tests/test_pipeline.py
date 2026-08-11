import unittest

import numpy as np
import pandas as pd

from compressor_ml.anomaly import StandardScaler3D, anomaly_score, health_score, pseudo_label
from compressor_ml.config import PipelineConfig
from compressor_ml.features import engineer_features
from compressor_ml.preprocessing import validate_and_filter
from compressor_ml.windowing import make_windows


class PipelineTests(unittest.TestCase):
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
