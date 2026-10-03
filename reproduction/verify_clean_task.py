"""Read-only integrity/metric verification of clean pilot snapshots.

--start verifies only step0, never presents an initial snapshot as final training.
Optional evidence output is exclusively created, not overwritten.
"""
import argparse
import json
from pathlib import Path
import math
import numpy as np
from reproduction.train_clean_task import AUDIT, RUN, RESULT, specification, sample_schedule, load_frames
from reproduction.fit_small_sample import verify_snapshot
from reproduction.audit_episode_split import sha256
from reproduction.action_metrics import metrics, training_target


def verify_evaluation(evaluation, audit, frames, tokenizer, expected_step):
    assert evaluation["step"] == expected_step and evaluation["center_crop"] is False
    for split, result in evaluation["splits"].items():
        fixed = {(r["episode_index"], r["timestep"]) for r in audit["fixed_frames"][split]}
        actual_frames = {(r["episode_index"], r["timestep"]): r for r in frames[split]}
        expected_teacher = set(actual_frames) if split == "validation" else fixed
        teacher_ids = []
        for row in result["teacher_rows"]:
            identity = (row["episode_index"], row["timestep"])
            frame = actual_frames[identity]
            assert row["image_sha256"] == frame["image_sha256"]
            target = training_target(frame["raw_action"], audit["training_statistics"]["action"])
            ids = tokenizer.tokenizer.vocab_size - np.digitize(target, tokenizer.bins)
            assert row["target_tokens"] == ids.tolist()
            assert len(row["predicted_tokens"]) == 7
            accuracy = float((np.asarray(row["predicted_tokens"]) == ids).mean())
            assert row["token_accuracy"] == accuracy
            teacher_ids.append(identity)
        assert set(teacher_ids) == expected_teacher and len(teacher_ids) == len(expected_teacher)
        assert result["teacher_frames"] == len(teacher_ids)
        assert math.isfinite(result["teacher_loss"])
        assert np.isclose(result["teacher_token_accuracy"], np.mean([r["token_accuracy"] for r in result["teacher_rows"]]), atol=1e-12, rtol=0)
        ar_ids = []
        for row in result["autoregressive_rows"]:
            identity = (row["episode_index"], row["timestep"])
            frame = actual_frames[identity]
            assert row["raw_action"] == frame["raw_action"] and row["image_sha256"] == frame["image_sha256"]
            calculated = metrics(row["action"], frame["raw_action"], audit["training_statistics"]["action"])
            for key, value in calculated.items():
                assert np.allclose(row[key], value, atol=1e-12, rtol=0), key
            ar_ids.append(identity)
        assert set(ar_ids) == fixed and len(ar_ids) == len(fixed)
        assert result["autoregressive_frames"] == len(fixed)
        for stored, raw in (("autoregressive_motion_mae", "motion_mae_6d"),
                            ("autoregressive_normalized_l1", "normalized_l1_7d")):
            assert np.isclose(result[stored], np.mean([r[raw] for r in result["autoregressive_rows"]]), atol=1e-12, rtol=0)
        assert result["gripper_correct"] == sum(r["gripper_correct"] for r in result["autoregressive_rows"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("Do not overwrite verified evidence")
    audit = json.loads(AUDIT.read_text())
    spec = specification(audit)
    steps = [0] if args.start else spec["milestones"]
    snapshots = {}
    for step in steps:
        snapshots[str(step)] = verify_snapshot(RUN / f"step_{step:03d}", spec)
    for name, record in audit["base"]["files"].items():
        assert sha256(Path(spec["base_path"]) / name) == record["sha256"]
    frames = load_frames(audit)
    from transformers import AutoProcessor
    from prismatic.vla.action_tokenizer import ActionTokenizer
    processor = AutoProcessor.from_pretrained(spec["base_path"], trust_remote_code=True)
    tokenizer = ActionTokenizer(processor.tokenizer)
    evaluations = []
    for step in steps:
        evaluation = json.loads((RUN / f"step_{step:03d}/evaluation.json").read_text())
        assert evaluation["loader"] == "training_BF16_autocast"
        verify_evaluation(evaluation, audit, frames, tokenizer, step)
        evaluations.append(evaluation)
    evidence = {"status": "verified_start_not_final" if args.start else "verified_complete",
                "spec": spec, "snapshots": snapshots, "base_files_unchanged": True,
                "frame_identity_and_targets_verified": True, "row_action_metrics_recomputed": True,
                "evaluations": evaluations}
    if not args.start:
        final = json.loads(RESULT.read_text())
        assert final["spec"] == spec and final["evaluations"] == evaluations
        production = final["production_evaluation"]
        assert production["loader"] == "production_no_autocast"
        verify_evaluation(production, audit, frames, tokenizer, 50)
        equal = all([r["action"] for r in evaluations[-1]["splits"][s]["autoregressive_rows"]] ==
                    [r["action"] for r in production["splits"][s]["autoregressive_rows"]] for s in frames)
        assert final["production_actions_equal"] == equal
        assert final["status"] == ("completed" if equal else "completed_loader_difference_requires_diagnosis")
        losses = []
        for path in sorted(RUN.glob("losses-*.jsonl")):
            losses.extend(json.loads(line) for line in path.read_text().splitlines())
        assert [r["optimizer_step"] for r in losses] == list(range(1, 51)), "Resume/duplicate loss histories need explicit diagnosis"
        schedule = sample_schedule(len(frames["train"]))
        for row in losses:
            step = row["optimizer_step"]
            assert row["sample_indices"] == schedule[(step - 1) * 16:step * 16]
            assert len(row["microbatch_losses"]) == 8 and all(math.isfinite(x) for x in row["microbatch_losses"])
            assert np.isclose(row["mean_loss"], np.mean(row["microbatch_losses"]), atol=1e-12, rtol=0)
        evidence.update(final_report_sha256=sha256(RESULT), production_actions_equal=equal,
                        loss_records_verified=50, production_evaluation=production)
    if args.output:
        with args.output.open("x") as handle:
            json.dump(evidence, handle, indent=2, allow_nan=False)
    print(json.dumps({"status": evidence["status"], "snapshots": steps, "output": str(args.output)}))


if __name__ == "__main__":
    main()
