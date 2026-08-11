import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from compressor_ml.adaptive import (
    AdaptiveConfig,
    AdaptiveRuntime,
    CalibrationProfile,
    build_candidate_profile,
    build_reference_profile,
    score_profile,
    window_summaries,
)


class AdaptiveCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.random = np.random.default_rng(42)
        self.windows = self.random.normal(size=(300, 12, 4)).astype(np.float32)
        self.config = AdaptiveConfig(
            min_candidate_windows=20,
            min_candidate_days=2,
            max_buffer_windows=100,
            shadow_min_observations=2,
            max_reference_alert_rate=0.20,
            min_synthetic_detection_rate=0.0,
            max_synthetic_detection_drop=1.0,
        )
        self.profile = build_reference_profile(
            group_name="MXTEST__M01",
            machine_id="MXTEST",
            module_id=1,
            feature_columns=["a", "b", "c", "d"],
            model_version="shared_test_v1",
            train_windows=self.windows,
            reconstruction_threshold=1.0,
            config=self.config,
        )

    def test_dual_profile_scoring_detects_large_shift(self):
        summaries = window_summaries(self.windows[:20])
        errors = np.full(20, 0.2, dtype=np.float32)
        normal = score_profile(self.profile, summaries, errors, self.config)
        shifted = score_profile(self.profile, summaries + 10.0, errors, self.config)
        self.assertLess(float(np.median(normal["golden_risk"])), 1.0)
        self.assertGreater(float(np.median(shifted["golden_risk"])), 1.0)
        self.assertTrue(shifted["baseline_drift"].all())

    def test_candidate_update_is_bounded(self):
        summaries = window_summaries(self.windows[:80]) + 20.0
        errors = np.full(80, 5.0, dtype=np.float32)
        candidate = build_candidate_profile(
            self.profile, summaries, errors, 80, self.config
        )
        center_step = np.max(
            np.abs(np.asarray(candidate.adaptive_center) - np.asarray(self.profile.adaptive_center))
            / np.asarray(self.profile.golden_scale)
        )
        threshold_change = abs(
            candidate.adaptive_reconstruction_threshold
            / self.profile.adaptive_reconstruction_threshold
            - 1.0
        )
        self.assertLessEqual(center_step, self.config.max_center_step_mad + 1e-8)
        self.assertLessEqual(
            threshold_change, self.config.max_threshold_change_fraction + 1e-8
        )

    def test_runtime_promotes_only_after_new_shadow_observation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.config.save(root / "adaptive_config.json")
            group_dir = root / "profiles" / self.profile.group_name
            self.profile.save(group_dir / "golden.json")
            champion_payload = asdict(self.profile)
            champion_payload.update(
                {
                    "profile_version": "MXTEST__M01_champion_v1",
                    "status": "CHAMPION",
                    "parent_version": self.profile.profile_version,
                }
            )
            champion = CalibrationProfile(**champion_payload)
            champion.save(group_dir / "champion.json")
            frozen = root / "frozen" / "MXTEST__M01.npz"
            frozen.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                frozen,
                summaries=window_summaries(self.windows[100:160]),
                errors=np.full(60, 0.2, dtype=np.float32),
            )

            runtime = AdaptiveRuntime(root)
            first_summaries = window_summaries(self.windows[:20])
            first_errors = np.full(20, 0.2, dtype=np.float32)
            first_timestamps = [
                "2026-01-01T00:00:00" if index < 10 else "2026-01-02T00:00:00"
                for index in range(20)
            ]
            runtime.append_eligible(
                self.profile.group_name,
                first_summaries,
                first_errors,
                first_timestamps,
            )
            first_decision = runtime.propose_or_advance(self.profile.group_name)
            self.assertEqual(first_decision.outcome, "SHADOW")
            self.assertTrue(runtime.candidate_path(self.profile.group_name).exists())

            waiting = runtime.propose_or_advance(self.profile.group_name)
            self.assertEqual(waiting.outcome, "SHADOW_WAIT")

            runtime.append_eligible(
                self.profile.group_name,
                window_summaries(self.windows[20:21]),
                np.asarray([0.2], dtype=np.float32),
                ["2026-01-03T00:00:00"],
            )
            approved = runtime.propose_or_advance(self.profile.group_name)
            self.assertEqual(approved.outcome, "AUTO_APPROVED")
            self.assertFalse(runtime.candidate_path(self.profile.group_name).exists())
            self.assertEqual(
                runtime.load_champion(self.profile.group_name).status,
                "AUTO_APPROVED",
            )


if __name__ == "__main__":
    unittest.main()
