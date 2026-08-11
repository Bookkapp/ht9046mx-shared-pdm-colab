import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from compressor_ml.adaptive import AdaptiveConfig, CalibrationProfile
from compressor_ml.adaptive_runner import bootstrap_from_prepared
from compressor_ml.anomaly import StandardScaler3D
from compressor_ml.config import PipelineConfig


class IdentityLikeModel:
    def predict(self, values, batch_size=256, verbose=0):
        return np.asarray(values) * 0.95


class AdaptiveBootstrapTests(unittest.TestCase):
    def test_bootstrap_contract_creates_seed_for_every_group(self):
        random = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            artifact = root / "artifact"
            seed = artifact / "adaptive_seed"
            (prepared / "groups").mkdir(parents=True)
            (artifact / "scalers").mkdir(parents=True)
            pipeline = PipelineConfig()
            pipeline.save(prepared / "config.json")
            pipeline.save(artifact / "config.json")

            groups = []
            thresholds = {}
            for machine_id in ("MXA", "MXB"):
                group_name = f"{machine_id}__M01"
                train = random.normal(size=(20, 60, 24)).astype(np.float32)
                validation = random.normal(size=(10, 60, 24)).astype(np.float32)
                test = random.normal(size=(10, 60, 24)).astype(np.float32)
                np.savez_compressed(
                    prepared / "groups" / f"{group_name}.npz",
                    train=train,
                    validation=validation,
                    test=test,
                )
                StandardScaler3D().fit(train).save(
                    str(artifact / "scalers" / f"{group_name}.npz")
                )
                groups.append(
                    {
                        "group_name": group_name,
                        "machine_id": machine_id,
                        "module_id": 1,
                        "windows_file": f"groups/{group_name}.npz",
                    }
                )
                thresholds[group_name] = {
                    "machine_id": machine_id,
                    "module_id": 1,
                    "percentile": 99.0,
                    "value": 1.0,
                }

            (prepared / "manifest.json").write_text(
                json.dumps(
                    {
                        "dataset_version": "unit_test_v1",
                        "feature_columns": list(pipeline.feature_columns),
                        "groups": groups,
                    }
                ),
                encoding="utf-8",
            )
            (artifact / "manifest.json").write_text(
                json.dumps({"model_version": "shared_unit_test_v1"}), encoding="utf-8"
            )
            (artifact / "thresholds.json").write_text(
                json.dumps(thresholds), encoding="utf-8"
            )

            result = bootstrap_from_prepared(
                prepared,
                artifact,
                seed,
                adaptive_config=AdaptiveConfig(
                    min_candidate_windows=10,
                    max_buffer_windows=20,
                ),
                model=IdentityLikeModel(),
            )
            self.assertEqual(result["groups"], 2)
            seed_manifest = json.loads(
                (seed / "seed_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(seed_manifest["groups"]), 2)
            for group in groups:
                group_name = group["group_name"]
                champion = CalibrationProfile.load(
                    seed / "profiles" / group_name / "champion.json"
                )
                self.assertEqual(champion.status, "CHAMPION")
                self.assertIn("deployment_baseline", champion.approval)
                self.assertTrue((seed / "frozen" / f"{group_name}.npz").exists())
if __name__ == "__main__":
    unittest.main()
