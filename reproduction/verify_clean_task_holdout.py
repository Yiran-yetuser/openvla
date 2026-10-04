"""CPU verifier for the frozen step-200 unused-episode evaluation."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction import extend_clean_task
from reproduction.action_metrics import metrics
from reproduction.audit_episode_split import sha256
from reproduction.fit_small_sample import verify_snapshot

AUDIT = ROOT / "reproduction/results/clean_task_split_audit_v1.json"
RESULT = ROOT / "reproduction/results/clean_task_holdout_eval_v1.json"
SUMMARY = ROOT / "reproduction/results/clean_task_holdout_summary_v1.json"
CHECKPOINT = ROOT / "runs/clean_task_extend_v1/step_200"


def close(a, b, tolerance=1e-12):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tolerance


def verify():
    audit = json.loads(AUDIT.read_text())
    report = json.loads(RESULT.read_text())
    summary = json.loads(SUMMARY.read_text())
    spec = extend_clean_task.specification(audit)
    source_episodes = [row for row in audit["episode_inventory"]
                       if row["instruction"] == audit["task_instruction"]]
    selected = audit["selection"]["train"] + audit["selection"]["validation"]
    selected_ids = {row["episode_index"] for row in selected}
    heldout = [row for row in source_episodes if row["episode_index"] not in selected_ids]
    heldout.sort(key=lambda row: row["episode_index"])
    ids = [row["episode_index"] for row in heldout]
    if len(source_episodes) != 46 or len({row["content_sha256"] for row in source_episodes}) != 46:
        raise ValueError("Source inventory no longer contains the expected 46 unique episodes")
    if len(selected_ids) != 10 or len(ids) != 36 or set(ids) & selected_ids:
        raise ValueError("Held-out episode split is not 36 episodes disjoint from train/validation")
    if report.get("status") != "completed_offline_holdout_only":
        raise ValueError("Evaluation did not complete")
    if report.get("audit_sha256") != sha256(AUDIT):
        raise ValueError("Source split audit hash differs")
    if report.get("heldout_episode_indices") != ids:
        raise ValueError("Held-out episode ids differ from the audited remainder")
    expected_hashes = {str(row["episode_index"]): row["content_sha256"] for row in heldout}
    if report.get("heldout_episode_sha256") != expected_hashes:
        raise ValueError("Held-out content hashes differ from the source inventory")
    if report.get("checkpoint_step") != 200 or report.get("optimizer_updates") != 0 or report.get("simulator_rollouts") != 0:
        raise ValueError("Evaluation exceeded its declared frozen offline scope")
    snapshot = verify_snapshot(CHECKPOINT, spec)
    if snapshot.get("step") != 200 or sha256(CHECKPOINT / "snapshot.json") != report.get("checkpoint_snapshot_sha256"):
        raise ValueError("Frozen checkpoint manifest changed")
    for name, record in audit["base"]["files"].items():
        if sha256(Path(spec["base_path"]) / name) != record["sha256"]:
            raise ValueError(f"Base weight file changed: {name}")

    episodes = report.get("episodes", [])
    if [row.get("episode_index") for row in episodes] != ids:
        raise ValueError("Episode result order or coverage differs")
    all_teacher, all_ar = [], []
    episode_accuracy, episode_motion_mae = [], []
    token_correct = token_total = 0
    for result, expected in zip(episodes, heldout):
        episode = expected["episode_index"]
        teacher = result.get("teacher_rows", [])
        ar = result.get("autoregressive_rows", [])
        if result.get("eligible_frames") != expected["eligible_frames"] or len(teacher) != expected["eligible_frames"]:
            raise ValueError(f"Teacher frame count differs for episode {episode}")
        if len(ar) != 3 or result.get("autoregressive_frames") != 3:
            raise ValueError(f"Expected three autoregressive stages for episode {episode}")
        teacher_keys = [(row.get("episode_index"), row.get("timestep")) for row in teacher]
        if len(teacher_keys) != len(set(teacher_keys)) or any(key[0] != episode for key in teacher_keys):
            raise ValueError(f"Teacher row identity is invalid for episode {episode}")
        if [row.get("timestep") for row in teacher] != sorted(row.get("timestep") for row in teacher):
            raise ValueError(f"Teacher rows are out of temporal order for episode {episode}")
        if [row.get("stage") for row in ar] != ["early", "middle", "late"]:
            raise ValueError(f"Autoregressive stage labels differ for episode {episode}")
        if any(row.get("episode_index") != episode for row in ar):
            raise ValueError(f"Autoregressive row identity is invalid for episode {episode}")
        if result.get("stage_timesteps") != [row.get("timestep") for row in ar]:
            raise ValueError(f"Stage timestep list differs for episode {episode}")
        local_correct = 0
        for row in teacher:
            predicted, target = row.get("predicted_tokens"), row.get("target_tokens")
            if not isinstance(predicted, list) or not isinstance(target, list) or len(predicted) != 7 or len(target) != 7:
                raise ValueError(f"Invalid seven-token row in episode {episode}")
            if not close(row.get("token_accuracy"), sum(a == b for a, b in zip(predicted, target)) / 7):
                raise ValueError(f"Stored token accuracy differs for episode {episode}")
            if len(row.get("image_sha256", "")) != 64:
                raise ValueError(f"Missing frame hash for episode {episode}")
            local_correct += sum(a == b for a, b in zip(predicted, target))
            token_total += 7
            all_teacher.append(row)
        token_correct += local_correct
        episode_accuracy.append(local_correct / (7 * len(teacher)))
        for row in ar:
            for key, value in metrics(row["action"], row["raw_action"], audit["training_statistics"]["action"]).items():
                stored = row.get(key)
                if isinstance(value, list):
                    if not np.allclose(np.asarray(value), np.asarray(stored), atol=1e-12, rtol=0):
                        raise ValueError(f"Stored {key} differs for episode {episode}")
                elif not close(stored, value):
                    raise ValueError(f"Stored {key} differs for episode {episode}")
            all_ar.append(row)
        expected_loss = float(result["teacher_loss"])
        if not math.isfinite(expected_loss):
            raise ValueError(f"Non-finite teacher loss for episode {episode}")
        if not close(result.get("teacher_token_accuracy"), episode_accuracy[-1]):
            raise ValueError(f"Per-episode token accuracy differs for episode {episode}")
        episode_motion_mae.append(float(np.mean([row["motion_mae_6d"] for row in ar])))
        if not close(result.get("autoregressive_motion_mae"), episode_motion_mae[-1]):
            raise ValueError(f"Per-episode motion MAE differs for episode {episode}")

    if len(all_teacher) != 4491 or len(all_ar) != 108 or token_total != 31437:
        raise ValueError("Aggregate frame/token counts differ from the planned evaluation")
    calc = {
        "episodes": len(episodes), "teacher_frames": len(all_teacher),
        "teacher_action_tokens": token_total, "teacher_correct_tokens": token_correct,
        "teacher_token_accuracy": token_correct / token_total,
        "teacher_loss_mean_episode": float(np.mean([row["teacher_loss"] for row in episodes])),
        "teacher_accuracy_mean_episode": float(np.mean(episode_accuracy)),
        "teacher_accuracy_median_episode": float(np.median(episode_accuracy)),
        "teacher_accuracy_sd_episode": float(np.std(episode_accuracy, ddof=1)),
        "autoregressive_frames": len(all_ar),
        "autoregressive_motion_mae": float(np.mean([row["motion_mae_6d"] for row in all_ar])),
        "autoregressive_motion_mae_mean_episode": float(np.mean(episode_motion_mae)),
        "autoregressive_motion_mae_sd_episode": float(np.std(episode_motion_mae, ddof=1)),
        "autoregressive_normalized_l1": float(np.mean([row["normalized_l1_7d"] for row in all_ar])),
        "autoregressive_gripper_correct": sum(row["gripper_correct"] for row in all_ar),
    }
    if calc["teacher_correct_tokens"] != 9210:
        raise ValueError("Unexpected held-out token correct count")
    for key, value in calc.items():
        if key not in report.get("summary", {}) or not close(report["summary"][key], value):
            raise ValueError(f"Stored summary differs for {key}")
    digest = sha256(RESULT)
    if summary.get("result_sha256") != digest or summary.get("summary") != report["summary"]:
        raise ValueError("Compact result summary does not match the raw report")
    return {"status": "verified_clean_task_holdout_eval", **calc,
            "heldout_episode_hashes_match_inventory": True,
            "teacher_token_metrics_recomputed": True,
            "autoregressive_action_metrics_recomputed": True,
            "checkpoint_snapshot_sha256": report["checkpoint_snapshot_sha256"],
            "checkpoint_step": 200, "base_files_unchanged": True,
            "optimizer_updates": 0, "simulator_rollouts": 0,
            "result_sha256": digest, "summary_sha256": sha256(SUMMARY)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify()
    if args.output:
        if args.output.exists():
            if json.loads(args.output.read_text()) != result:
                raise FileExistsError("Existing verification differs; refusing overwrite")
        else:
            with args.output.open("x") as handle:
                json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
