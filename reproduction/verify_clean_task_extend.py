"""Read-only verifier for the bounded clean-task 50-to-200 update extension."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction import extend_clean_task
from reproduction.audit_episode_split import sha256
from reproduction.fit_small_sample import verify_snapshot
from reproduction.verify_clean_task import verify_evaluation


RUN = ROOT / "runs/clean_task_extend_v1"
RESULT = ROOT / "reproduction/results/clean_task_extend_v1.json"


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def verify_loss_rows(losses, schedule):
    if [row.get("optimizer_step") for row in losses] != list(range(51, 201)):
        raise ValueError("Loss steps must be exactly 51..200; resumed histories require explicit diagnosis")
    for row in losses:
        step = row["optimizer_step"]
        if row["sample_indices"] != schedule[(step - 1) * 16:step * 16]:
            raise ValueError("Continuation sampling schedule changed")
        values = row.get("microbatch_losses", [])
        if len(values) != 8 or not finite(row.get("mean_loss")) or not all(finite(v) for v in values):
            raise ValueError("Invalid continuation loss record")
        if not np.isclose(row["mean_loss"], np.mean(values), atol=1e-12, rtol=0):
            raise ValueError("Stored mean loss disagrees with the eight microbatch losses")


def verify_parent_predictions(evaluation, parent):
    if evaluation.get("step") != 50 or parent.get("step") != 50:
        raise ValueError("Continuation must begin with step-50 predictions")
    for split in ("train", "validation"):
        for key in ("teacher_rows", "autoregressive_rows"):
            if evaluation["splits"][split][key] != parent["splits"][split][key]:
                raise ValueError("Step-50 predictions differ from the parent")


def production_equality(evaluations, production):
    if production.get("loader") != "production_no_autocast" or production.get("step") != 200:
        raise ValueError("Production loader/stage differs from the declared comparison")
    if set(production.get("splits", {})) != {"train", "validation"}:
        raise ValueError("Missing production evaluation split")
    return all([r["action"] for r in evaluations[-1]["splits"][split]["autoregressive_rows"]] ==
               [r["action"] for r in production["splits"][split]["autoregressive_rows"]]
               for split in ("train", "validation"))


def verify():
    if not RESULT.exists():
        raise FileNotFoundError("200-update extension result does not exist; do not fabricate verification")
    audit = json.loads(extend_clean_task.core.AUDIT.read_text())
    expected = extend_clean_task.specification(audit)
    report = json.loads(RESULT.read_text())
    if report["spec"] != expected:
        raise ValueError("Extension result spec differs from the immutable planned spec")
    if report.get("parent_updates") != 50 or report.get("new_updates") != 150:
        raise ValueError("Missing 50+150 continuation accounting")
    if not report.get("parent_predictions_verified"):
        raise ValueError("Parent predictions were not verified before continuation")
    if report.get("base_checkpoint_modified") is not False:
        raise ValueError("Base checkpoint modification guard failed")
    if type(report.get("production_actions_equal")) is not bool:
        raise ValueError("Production loader result is missing")

    parent = Path(expected["parent_checkpoint"])
    if sha256(parent / "snapshot.json") != expected["parent_snapshot_sha256"]:
        raise ValueError("Parent snapshot changed")
    parent_record = verify_snapshot(parent, expected["parent_spec"])
    if parent_record["step"] != 50:
        raise ValueError("Parent checkpoint is not step 50")

    evaluations = report.get("evaluations", [])
    if [row.get("step") for row in evaluations] != [50, 100, 150, 200]:
        raise ValueError("Expected exactly four extension milestones")
    for row in evaluations:
        if row.get("loader") != "training_BF16_autocast" or set(row.get("splits", {})) != {"train", "validation"}:
            raise ValueError("Missing train/validation evaluation split")
        for split, expected_teacher, expected_ar in (("train", 24, 24), ("validation", 246, 6)):
            current = row["splits"][split]
            if current["teacher_frames"] != expected_teacher or current["autoregressive_frames"] != expected_ar:
                raise ValueError("Evaluation frame coverage changed")
            for key in ("teacher_loss", "teacher_token_accuracy", "autoregressive_motion_mae",
                        "autoregressive_normalized_l1"):
                if not finite(current[key]):
                    raise ValueError(f"Non-finite evaluation metric: {key}")

    parent_evaluation = json.loads((parent / "evaluation.json").read_text())
    verify_parent_predictions(evaluations[0], parent_evaluation)
    snapshots = {}
    for step in (50, 100, 150, 200):
        checkpoint = RUN / f"step_{step:03d}"
        if not checkpoint.is_dir() or any(RUN.glob(f"step_{step:03d}.partial")):
            raise ValueError(f"Incomplete checkpoint at step {step}")
        record = verify_snapshot(checkpoint, expected)
        required = {"adapter_model.safetensors", "adapter_config.json", "dataset_statistics.json",
                    "optimizer_rng.pt", "evaluation.json"}
        if record["step"] != step or set(record["sha256"]) != required:
            raise ValueError("Checkpoint step does not match directory")
        stored_evaluation = json.loads((checkpoint / "evaluation.json").read_text())
        if stored_evaluation != evaluations[(step - 50) // 50]:
            raise ValueError("Report evaluation differs from its hash-verified checkpoint")
        snapshots[str(step)] = record

    # CPU reload of the actual episodes checks identities, supervision tokens and
    # independently recomputed AR summaries, rather than trusting stored counts.
    for name, record in audit["base"]["files"].items():
        if sha256(Path(expected["base_path"]) / name) != record["sha256"]:
            raise ValueError(f"Base checkpoint changed: {name}")
    frames = extend_clean_task.core.load_frames(audit)
    from transformers import AutoProcessor
    from prismatic.vla.action_tokenizer import ActionTokenizer
    processor = AutoProcessor.from_pretrained(expected["base_path"], trust_remote_code=True)
    tokenizer = ActionTokenizer(processor.tokenizer)
    for evaluation in evaluations:
        verify_evaluation(evaluation, audit, frames, tokenizer, evaluation["step"])
    production = report["production_evaluation"]
    equal = production_equality(evaluations, production)
    verify_evaluation(production, audit, frames, tokenizer, 200)
    status = "completed" if equal else "completed_loader_difference_requires_diagnosis"
    if report["production_actions_equal"] != equal or report.get("status") != status:
        raise ValueError("Production loader comparison/status disagrees with the recorded actions")

    loss_files = sorted(RUN.glob("losses-*.jsonl"))
    if not loss_files:
        raise ValueError("No continuation loss log")
    losses = [json.loads(line) for path in loss_files for line in path.read_text().splitlines() if line.strip()]
    schedule = extend_clean_task.core.sample_schedule(expected["training_frames"], 200, maximum=200)
    verify_loss_rows(losses, schedule)

    return {"status": "verified_clean_task_extension", "milestones": [50, 100, 150, 200],
            "new_updates_verified": 150, "loss_records_verified": len(losses),
            "loss_files_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in loss_files},
            "parent_snapshot_unchanged": True, "partial_checkpoints": 0,
            "production_actions_equal": equal, "result_sha256": sha256(RESULT),
            "snapshots": snapshots, "base_files_unchanged": True,
            "frame_identity_and_targets_verified": True, "row_action_metrics_recomputed": True,
            "evaluations": evaluations, "production_evaluation": production}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify()
    if args.output:
        if args.output.exists():
            if json.loads(args.output.read_text()) != result:
                raise ValueError("Existing verifier output differs; refusing overwrite")
        else:
            with args.output.open("x") as handle:
                json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
