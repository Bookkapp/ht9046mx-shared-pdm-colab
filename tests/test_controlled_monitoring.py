import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from compressor_ml.controlled_monitoring.config import ControlledMonitoringConfig
from compressor_ml.controlled_monitoring.context import (
    fit_regime_model,
    operating_mode,
)
from compressor_ml.controlled_monitoring.engine import ControlledMonitoringEngine
from compressor_ml.controlled_monitoring.fusion import (
    PersistenceTracker,
    fuse_review_level,
)
from compressor_ml.controlled_monitoring.lifecycle import (
    BootstrapLifecycle,
    LifecycleState,
    ProfileRepository,
)
from compressor_ml.controlled_monitoring.profiles import (
    fit_context_profile,
    fit_frozen_profile_bundle,
    score_context_profile,
)
from compressor_ml.controlled_monitoring.types import Evidence, ReviewLevel, WindowStatus
from compressor_ml.controlled_monitoring.windowing import build_event_windows


def raw_points(machine_id="MXTEST", module_id=1, periods=300, start="2026-08-01 00:00:00"):
    index = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=periods, freq="s"),
            "machine_id": machine_id,
            "module_id": module_id,
            "global_status": "Run",
            "module_status": "On",
            "busy": 0,
            "sv": "On",
            "hp1": 120.0 + np.sin(index / 30.0),
            "lp1": 30.0 + np.sin(index / 35.0),
            "hp2": 210.0 + np.sin(index / 25.0),
            "lp2": 52.0 + np.sin(index / 20.0),
            "valve": 60.0 + np.sin(index / 40.0),
            "temphi": 70.0 + np.sin(index / 50.0),
            "templo": 20.0 + np.sin(index / 45.0),
        }
    )


def context_windows(count=120):
    rng = np.random.default_rng(12)
    hp2 = rng.normal(210, 3, count)
    valve = rng.normal(60, 2, count)
    temphi = rng.normal(70, 1, count)
    templo = rng.normal(20, 1, count)
    lp2 = 10 + 0.18 * hp2 + 0.05 * valve + 0.03 * temphi - 0.02 * templo
    lp2 += rng.normal(0, 0.3, count)
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-07-01", periods=count, freq="5min", tz="Asia/Bangkok"),
            "machine_id": "MXTEST",
            "module_id": 1,
            "sv": "On",
            "hp1": rng.normal(120, 2, count),
            "lp1": rng.normal(30, 1, count),
            "hp2": hp2,
            "lp2": lp2,
            "valve": valve,
            "temphi": temphi,
            "templo": templo,
            "pressure_gap": hp2 - lp2,
            "pressure_ratio": hp2 / np.maximum(np.abs(lp2), 0.1),
            "temperature_span": temphi - templo,
            "window_status": WindowStatus.ELIGIBLE.value,
        }
    )
    return frame


class ControlledWindowTests(unittest.TestCase):
    def setUp(self):
        self.config = ControlledMonitoringConfig()

    def test_event_time_window_uses_median_and_passes_quality_gate(self):
        raw = raw_points()
        result = build_event_windows(raw, self.config)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["window_status"], WindowStatus.ELIGIBLE.value)
        self.assertEqual(int(result.iloc[0]["point_count"]), 300)
        self.assertAlmostEqual(float(result.iloc[0]["coverage"]), 1.0)
        self.assertAlmostEqual(float(result.iloc[0]["hp2"]), float(raw["hp2"].median()))
        self.assertEqual(str(result.iloc[0]["event_time"].tz), "Asia/Bangkok")

    def test_sentinel_window_is_review_not_normal(self):
        raw = raw_points()
        raw.loc[100, "templo"] = -200
        result = build_event_windows(raw, self.config)
        self.assertEqual(result.iloc[0]["window_status"], WindowStatus.DATA_QUALITY_REVIEW.value)
        self.assertIn("SENTINEL_VALUE", result.iloc[0]["quality_reasons"])

    def test_mode_is_deterministic(self):
        self.assertEqual(operating_mode("On", 10, self.config), "SV_ON_VALVE_B0")
        self.assertEqual(operating_mode("Off", 85, self.config), "SV_OFF_VALVE_B3")


class ControlledProfileTests(unittest.TestCase):
    def setUp(self):
        self.config = ControlledMonitoringConfig()
        self.frame = context_windows()

    def test_gmm_resolves_training_context_and_abstains_far_outside(self):
        model = fit_regime_model(self.frame, "SV_ON_VALVE_B2", self.config)
        normal = model.resolve(self.frame.iloc[0], apply_policy_likelihood_floor=False)
        far = self.frame.iloc[0].copy()
        far[["hp2", "lp2", "valve", "temphi", "templo"]] = [900, 900, 900, 200, 200]
        unknown = model.resolve(far)
        self.assertTrue(normal.regime.startswith("R"))
        self.assertEqual(unknown.regime, "UNKNOWN_REGIME")

    def test_lp2_conditional_residual_detects_large_negative_shift(self):
        profile = fit_context_profile(
            self.frame, "SV_ON_VALVE_B2", "R0", self.config
        )
        normal_row = self.frame.iloc[len(self.frame) // 2].copy()
        normal = score_context_profile(profile, normal_row, self.config)
        shifted_row = normal_row.copy()
        shifted_row["lp2"] = float(shifted_row["lp2"]) - 20.0
        shifted_row["pressure_gap"] = float(shifted_row["hp2"] - shifted_row["lp2"])
        shifted = score_context_profile(profile, shifted_row, self.config)
        self.assertLess(abs(float(normal.details["z_lp2_residual"])), 3.5)
        self.assertIn("LP2_NEGATIVE_RESIDUAL", shifted.reason_codes)

    def test_human_approval_is_required_before_active_profile(self):
        bundle = fit_frozen_profile_bundle(
            self.frame,
            "MXTEST",
            1,
            self.config,
            profile_version="MXTEST_auto_v1_M01",
        )
        with tempfile.TemporaryDirectory() as temporary:
            repository = ProfileRepository(temporary)
            repository.save_candidate("MXTEST", "MXTEST_auto_v1", {1: bundle}, {})
            repository.transition(
                "MXTEST",
                LifecycleState.CANDIDATE_PROFILE_READY,
                reason="test_candidate",
                updates={"candidate_version": "MXTEST_auto_v1"},
            )
            with self.assertRaises(ValueError):
                repository.approve("MXTEST", approved_by="engineer")
            repository.transition(
                "MXTEST", LifecycleState.APPROVAL_REQUIRED, reason="shadow_passed"
            )
            approved = repository.approve(
                "MXTEST", approved_by="engineer", reason="baseline graph reviewed"
            )
            self.assertEqual(approved["state"], LifecycleState.ACTIVE.value)
            self.assertEqual(approved["approval_reason"], "baseline graph reviewed")
            self.assertEqual(repository.load_active("MXTEST")[1].status, "ACTIVE_FROZEN")

    def test_shadow_validation_counts_unknown_regime_as_abstention(self):
        class _UnusedShadow:
            pass

        with tempfile.TemporaryDirectory() as temporary:
            repository = ProfileRepository(temporary)
            repository.transition(
                "MXTEST",
                LifecycleState.SHADOW_VALIDATION,
                reason="test_shadow",
            )
            lifecycle = BootstrapLifecycle(repository, self.config, _UnusedShadow())
            decisions = pd.DataFrame(
                {
                    "event_time": pd.to_datetime(
                        ["2026-08-01 00:00", "2026-08-01 00:05"]
                    ),
                    "window_status": [
                        WindowStatus.ELIGIBLE.value,
                        WindowStatus.UNKNOWN_REGIME.value,
                    ],
                    "unknown_regime": [False, True],
                    "com2_flag": [False, False],
                    "lstm_flag": [False, False],
                }
            )
            result = lifecycle.record_shadow("MXTEST", decisions)
            self.assertEqual(result["shadow"]["windows"], 2)
            self.assertEqual(result["shadow"]["known_windows"], 1)
            self.assertEqual(result["shadow"]["unknown_regime_rate"], 0.5)


class PersistenceFusionTests(unittest.TestCase):
    def test_dual_evidence_reaches_p1_after_fifteen_minutes(self):
        config = ControlledMonitoringConfig()
        tracker = PersistenceTracker(config)
        key = "MXTEST__M01::MODE::R0"
        result = None
        com2 = Evidence("COM2", True, reason_codes=["LP2_NEGATIVE_RESIDUAL"])
        lstm = Evidence("LSTM", True, reason_codes=["LSTM_RECONSTRUCTION_ANOMALY"])
        for index in range(3):
            result = tracker.update(
                key,
                pd.Timestamp("2026-08-01", tz="Asia/Bangkok") + pd.Timedelta(minutes=5 * index),
                com2,
                lstm,
                z_lp2=-4.0,
            )
        self.assertIsNotNone(result)
        level = fuse_review_level(WindowStatus.ELIGIBLE.value, com2, lstm, result, config)
        self.assertEqual(result.com2_seconds, 900)
        self.assertEqual(result.lstm_seconds, 900)
        self.assertEqual(level, ReviewLevel.P1_REVIEW.value)


class _NeverCalledShadow:
    def score_bucket(self, *args, **kwargs):
        raise AssertionError("LSTM must not score when no profile is active")


class EngineAbstentionTests(unittest.TestCase):
    def test_missing_profile_is_not_reported_as_normal(self):
        config = ControlledMonitoringConfig()
        with tempfile.TemporaryDirectory() as temporary:
            repository = ProfileRepository(temporary)
            engine = ControlledMonitoringEngine(repository, config, _NeverCalledShadow())
            result = engine.score_frame(raw_points())
            self.assertEqual(result.iloc[0]["window_status"], WindowStatus.PROFILE_NOT_ACTIVE.value)
            self.assertEqual(result.iloc[0]["review_level"], WindowStatus.PROFILE_NOT_ACTIVE.value)


if __name__ == "__main__":
    unittest.main()
