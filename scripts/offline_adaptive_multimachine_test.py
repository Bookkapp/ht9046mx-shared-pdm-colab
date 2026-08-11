from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compressor_ml.adaptive import AdaptiveRuntime, score_profile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _eligible_reference_rows(
    runtime: AdaptiveRuntime,
    group_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    champion = runtime.load_champion(group_name)
    with np.load(runtime.frozen_path(group_name), allow_pickle=False) as frozen:
        summaries = frozen["summaries"].astype(np.float32)
        errors = frozen["errors"].astype(np.float32)
    scored = score_profile(champion, summaries, errors, runtime.config)
    eligible = np.asarray(scored["eligible_for_calibration"], dtype=bool)
    if not eligible.any():
        raise RuntimeError(f"{group_name} has no eligible frozen reference rows")
    return summaries[eligible], errors[eligible]


def _repeat_rows(
    summaries: np.ndarray,
    errors: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(count) % len(summaries)
    return summaries[indices].copy(), errors[indices].copy()


def _timestamps(count: int, days: int) -> list[str]:
    return [
        f"2026-08-{1 + (index % days):02d}T{(index // days) % 24:02d}:00:00+00:00"
        for index in range(count)
    ]


def run_test(seed_dir: Path, target_group: str | None, control_group: str | None) -> dict:
    profile_dirs = sorted(path for path in (seed_dir / "profiles").iterdir() if path.is_dir())
    group_names = [path.name for path in profile_dirs]
    if len(group_names) < 2:
        raise RuntimeError("The adaptive seed needs at least two machine/module groups")
    if target_group is not None and target_group not in group_names:
        raise ValueError("target group must exist in the adaptive seed")
    if control_group is not None and control_group not in group_names:
        raise ValueError("control group must exist in the adaptive seed")
    if target_group is not None and target_group == control_group:
        raise ValueError("target and control groups must be different")

    with tempfile.TemporaryDirectory(prefix="ht9046mx_adaptive_offline_") as temporary:
        runtime_dir = Path(temporary) / "runtime"
        runtime = AdaptiveRuntime(runtime_dir)
        runtime.initialize_from_seed(seed_dir)

        if target_group is None:
            for name in group_names:
                try:
                    eligible_summaries, eligible_errors = _eligible_reference_rows(runtime, name)
                except RuntimeError:
                    continue
                target_group = name
                break
            else:
                raise RuntimeError("No seeded group has eligible frozen reference rows")
        else:
            eligible_summaries, eligible_errors = _eligible_reference_rows(runtime, target_group)
        control_group = control_group or next(name for name in group_names if name != target_group)
        if control_group not in group_names:
            raise ValueError("target/control group must exist in the adaptive seed")

        target_before = runtime.load_champion(target_group)
        target_golden_hash_before = _sha256(runtime.golden_path(target_group))
        control_hash_before = _sha256(runtime.champion_path(control_group))

        required = runtime.config.min_candidate_windows
        initial_summaries, initial_errors = _repeat_rows(
            eligible_summaries,
            eligible_errors,
            required,
        )
        runtime.append_eligible(
            target_group,
            initial_summaries,
            initial_errors,
            _timestamps(required, runtime.config.min_candidate_days),
        )
        decisions = [runtime.propose_or_advance(target_group)]

        for shadow_index in range(1, runtime.config.shadow_min_observations):
            next_summary, next_error = _repeat_rows(
                eligible_summaries[shadow_index % len(eligible_summaries) :],
                eligible_errors[shadow_index % len(eligible_errors) :],
                1,
            )
            runtime.append_eligible(
                target_group,
                next_summary,
                next_error,
                [f"2026-08-{runtime.config.min_candidate_days + shadow_index:02d}T00:00:00+00:00"],
            )
            decisions.append(runtime.propose_or_advance(target_group))

        target_after = runtime.load_champion(target_group)
        target_golden_hash_after = _sha256(runtime.golden_path(target_group))
        control_hash_after = _sha256(runtime.champion_path(control_group))
        outcomes = [decision.outcome for decision in decisions]

        checks = {
            "candidate_entered_shadow": bool(outcomes and outcomes[0] == "SHADOW"),
            "target_auto_approved": bool(outcomes and outcomes[-1] == "AUTO_APPROVED"),
            "target_profile_version_changed": target_after.profile_version != target_before.profile_version,
            "target_golden_profile_immutable": target_golden_hash_after == target_golden_hash_before,
            "target_golden_fields_immutable": (
                target_after.golden_center == target_before.golden_center
                and target_after.golden_scale == target_before.golden_scale
                and target_after.golden_reconstruction_threshold
                == target_before.golden_reconstruction_threshold
            ),
            "control_machine_profile_isolated": control_hash_after == control_hash_before,
            "shared_model_not_required_for_calibration_test": True,
            "mysql_not_required": True,
        }
        passed = all(checks.values())
        report = {
            "passed": passed,
            "seed_dir": str(seed_dir.resolve()),
            "target_group": target_group,
            "control_group": control_group,
            "eligible_reference_rows": int(len(eligible_summaries)),
            "calibration_rows_appended": int(required + runtime.config.shadow_min_observations - 1),
            "decision_outcomes": outcomes,
            "target_profile_before": target_before.profile_version,
            "target_profile_after": target_after.profile_version,
            "checks": checks,
            "scope": (
                "This validates per-group calibration isolation and guarded automatic approval. "
                "It does not measure fault-detection accuracy or retrain shared model weights."
            ),
        }
        if not passed:
            raise AssertionError(json.dumps(report, indent=2, ensure_ascii=False))
        return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline multi-machine adaptive calibration test; no MySQL connection is used"
    )
    parser.add_argument(
        "--seed-dir",
        default="artifacts/shared_lstm_colab_smoke/adaptive_seed",
        help="Adaptive seed directory produced by the Colab notebook",
    )
    parser.add_argument("--target-group")
    parser.add_argument("--control-group")
    parser.add_argument("--output", default="analysis_output/offline_adaptive_multimachine_test.json")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args()
    report = run_test(Path(args.seed_dir), args.target_group, args.control_group)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
