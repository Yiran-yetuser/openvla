"""Read-only CPU evidence checks for the bounded 1e-4/5e-4 comparison.

No training or inference. Missing final reports are errors, never invented metrics.
--verify-local additionally verifies immutable snapshots, original files and logs.
--output uses exclusive creation and never overwrites existing evidence.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import fmean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.fit_small_sample import (RESULT, CONTROL_RESULT, matched_control_spec,
                                          matched_initial_evaluation, verify_snapshot)
from reproduction.audit_training_runtime import file_hash


def require_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite evidence")
    if isinstance(value, dict):
        for item in value.values():
            require_finite(item)
    elif isinstance(value, list):
        for item in value:
            require_finite(item)


def validate_report(report, expected_rows, stats=None):
    require_finite(report)
    assert report["optimizer_updates"] == 50 and not report["original_checkpoint_modified"]
    assert report["status"] in ("completed", "completed_loader_difference_requires_diagnosis")
    assert [e["step"] for e in report["evaluations"]] == [0, 10, 25, 50]
    assert all(e["loader"] == "training_BF16_autocast" for e in report["evaluations"])
    assert report["production_evaluation"]["step"] == 50
    assert report["production_evaluation"]["loader"] == "production_no_autocast"
    identities = lambda rows: [[r[k] for k in ("episode", "t", "image_sha256", "instruction")] for r in rows]
    for evaluation in report["evaluations"] + [report["production_evaluation"]]:
        assert evaluation["center_crop"] is False
        assert len(evaluation["rows"]) == 24
        assert identities(evaluation["rows"]) == identities(expected_rows)
        for i, row in enumerate(evaluation["rows"]):
            assert row["split"] == ("fit" if i < 16 else "probe_previously_seen")
            assert len(row["autoregressive"]["action"]) == len(row["raw_target"]) == 7
            assert 0 <= row["teacher_token_accuracy"] <= 1
            if stats is not None:
                import numpy as np
                from reproduction.action_metrics import metrics, training_target
                assert np.allclose(training_target(row["raw_target"], stats), expected_rows[i]["target"], atol=2e-6, rtol=0)
                for key, value in metrics(row["autoregressive"]["action"], row["raw_target"], stats).items():
                    assert np.allclose(value, row["autoregressive"][key], atol=1e-12, rtol=0)
        for split, count in (("fit", 16), ("probe_previously_seen", 8)):
            rows = [r for r in evaluation["rows"] if r["split"] == split]
            summary = evaluation["summary"][split]
            assert summary["samples"] == count == len(rows)
            values = {
                "teacher_token_accuracy": fmean(r["teacher_token_accuracy"] for r in rows),
                "teacher_token_l1": fmean(r["teacher_token_l1"] for r in rows),
                "autoregressive_motion_mae": fmean(r["autoregressive"]["motion_mae_6d"] for r in rows),
                "autoregressive_normalized_l1": fmean(r["autoregressive"]["normalized_l1_7d"] for r in rows),
                "gripper_correct": sum(r["autoregressive"]["gripper_correct"] for r in rows),
                "small_translation_count": sum(r["autoregressive"]["translation_norm"] < .02 for r in rows),
            }
            assert all(abs(summary[k] - v) < 1e-12 for k, v in values.items())
    actual_equal = ([r["autoregressive"]["action"] for r in report["evaluations"][-1]["rows"]] ==
                    [r["autoregressive"]["action"] for r in report["production_evaluation"]["rows"]])
    assert actual_equal == report["production_actions_equal"]
    assert report["status"] == ("completed" if actual_equal else "completed_loader_difference_requires_diagnosis")


def verify_local(report):
    source = Path(report["spec"]["source_checkpoint"])
    for name, digest in report["spec"]["original_sha256"].items():
        assert file_hash(source / name) == digest
    run = Path(report["final_checkpoint"]).parent
    for evaluation in report["evaluations"]:
        path = run / f"step_{evaluation['step']:03d}"
        record = verify_snapshot(path, report["spec"])
        assert record["step"] == evaluation["step"]
        assert json.loads((path / "evaluation.json").read_text()) == evaluation
    # Latest attempt wins only for updates replayed after a verified resume;
    # earlier attempts remain immutable evidence rather than being erased.
    latest, attempts = {}, []
    for path in sorted(run.glob("*-losses.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        require_finite(records)
        assert records and all(1 <= r["optimizer_step"] <= 50 for r in records)
        assert [r["optimizer_step"] for r in records] == sorted(set(r["optimizer_step"] for r in records))
        attempts.append({"path": str(path), "sha256": file_hash(path), "records": len(records)})
        for row in records:
            assert len(row["losses"]) == 8 and abs(fmean(row["losses"]) - row["mean_loss"]) < 1e-12
            latest[row["optimizer_step"]] = row
    assert sorted(latest) == list(range(1, 51))
    return {"snapshot_and_source_hashes": "pass", "loss_attempts": attempts,
            "loss_record_selection": "latest attempt per update; prior attempts retained",
            "mean_loss_before_update": {str(i): latest[i]["mean_loss"] for i in (1, 10, 25, 50)}}


def compare(verify_local_files=False):
    reference, control = [json.loads(p.read_text()) for p in (RESULT, CONTROL_RESULT)]
    verification = json.loads((ROOT / "reproduction/results/small_fit_verification_v1.json").read_text())
    assert file_hash(RESULT) == verification["result_sha256"]
    assert control["control_reference"]["sha256"] == file_hash(RESULT)
    matched_control_spec(control["spec"], reference["spec"])
    matched_initial_evaluation(control["evaluations"][0], reference)
    cpu = json.loads((ROOT / "reproduction/results/training_chain_cpu_audit_v1.json").read_text())
    stats = None
    if verify_local_files:
        stats = json.loads((Path(reference["spec"]["source_checkpoint"]) / "dataset_statistics.json").read_text())["libero_spatial_no_noops"]["action"]
    for report in (reference, control):
        validate_report(report, cpu["rows"], stats)
    # Raw targets must stay identical at all nodes/loaders in both runs.
    targets = [r["raw_target"] for r in reference["evaluations"][0]["rows"]]
    for report in (reference, control):
        for e in report["evaluations"] + [report["production_evaluation"]]:
            assert [r["raw_target"] for r in e["rows"]] == targets
    result = {"schema_version": 1, "status": "verified_comparison",
              "only_spec_difference": "learning_rate", "step0_exactly_equal": True,
              "reports_sha256": {"lr1e-4": file_hash(RESULT), "lr5e-4": file_hash(CONTROL_RESULT)},
              "local_snapshot_verification": verify_local_files,
              "milestones": [{"step": a["step"], "lr1e-4": a["summary"], "lr5e-4": b["summary"]}
                             for a, b in zip(reference["evaluations"], control["evaluations"])],
              "production": {name: {"actions_equal_training": r["production_actions_equal"],
                                    "summary": r["production_evaluation"]["summary"]}
                             for name, r in (("lr1e-4", reference), ("lr5e-4", control))},
              "limits": ["fixed-frame fitting, not held-out generalization or rollout success",
                         "one seed and one run per learning rate, not an uncertainty estimate",
                         "cannot uniquely explain historical training failure",
                         "pre-update loss and post-update action metrics have different timing"]}
    if verify_local_files:
        result["local_checks"] = {"lr1e-4": verify_local(reference), "lr5e-4": verify_local(control)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-local", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError("Comparison exists; verify it without overwriting")
    result = compare(args.verify_local)
    if args.output is not None:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
