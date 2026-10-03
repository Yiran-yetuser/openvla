"""CPU verification of existing video and expert-replay evidence; no simulator."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.analyze_failure_videos import command_events, decode_video_frames
from reproduction.audit_episode_split import sha256
from reproduction.audit_expert_correspondence import array_hash, keep_action_indices
from reproduction.fetch_expert_demo import DESTINATION, DIGEST, SIZE

OUTPUT = ROOT / "reproduction/results/failure_diagnosis_verification_v1.json"


def verify_expert_trace(trace, actions, kept):
    rows = trace["actions"]
    if [r["step"] for r in rows] != list(range(len(actions))):
        raise ValueError("Expert trace action coverage mismatch")
    if [r["original_action_index"] for r in rows] != kept:
        raise ValueError("Original indices mismatch")
    actual = np.asarray([r["simulator_action"] for r in rows])
    if not np.array_equal(actual, actions):
        raise ValueError("Executed expert actions differ from original filtered actions")
    if not all(r["goal_after_action"] == r["done_after_action"] for r in rows):
        raise ValueError("Recorded goal/done disagree")
    if rows[-1]["goal_after_action"] != trace["success"]:
        raise ValueError("Final expert goal disagrees with report")
    for row in rows:
        for key, shape in (("eef_pos", (3,)), ("gripper_qpos", (2,)),
                           ("bowl_pos_m", (3,)), ("plate_pos_m", (3,))):
            value = np.asarray(row[key])
            assert value.shape == shape and np.isfinite(value).all()
    return next((r["step"] for r in rows if r["goal_after_action"]), None)


def verify():
    assert DESTINATION.stat().st_size == SIZE and sha256(DESTINATION) == DIGEST
    audit_file = ROOT / "reproduction/results/clean_task_video_audit_v1.json"
    audit = json.loads(audit_file.read_text())
    assert audit["status"] == "decoded_aligned_existing_videos" and len(audit["entries"]) == 4
    assert audit["source_report_sha256"] == sha256(ROOT / "reproduction/results/clean_task_closed_loop_v1.json")
    video_rows = []
    for row in audit["entries"]:
        for file_key, hash_key in (("trace_file", "trace_sha256"), ("log_file", "log_sha256"),
                                   ("video_file", "video_sha256"), ("contact_sheet", "contact_sheet_sha256")):
            assert sha256(ROOT / row[file_key]) == row[hash_key]
        trace = json.loads((ROOT / row["trace_file"]).read_text())
        assert command_events(trace["actions"]) == row["gripper_command_events"]
        with imageio.get_reader(ROOT / row["video_file"]) as reader:
            frames = decode_video_frames(reader, len(trace["actions"]))
        assert len(frames) == row["video_frames"] == row["trace_actions"] == 220
        for keyframe in row["keyframes"]:
            assert hashlib.sha256(frames[keyframe["policy_step"]].tobytes()).hexdigest() == keyframe["decoded_rgb_sha256"]
        video_rows.append({"path": row["path"], "trial": row["trial"], "frames": len(frames),
                           "command_switches": len(row["gripper_command_events"]) - 1})
    replay_file = ROOT / "reproduction/results/expert_replay_v1.json"
    replay = json.loads(replay_file.read_text())
    correspondence_file = ROOT / "reproduction/results/expert_correspondence_v1.json"
    correspondence = json.loads(correspondence_file.read_text())
    assert replay["correspondence_sha256"] == sha256(correspondence_file)
    assert replay["hdf5_sha256"] == DIGEST and replay["hdf5_hash_unchanged"]
    assert replay["expert_replays"] == len(replay["results"]) == 2
    assert replay["model_loads"] == replay["optimizer_updates"] == replay["policy_rollouts"] == 0
    expert_rows = []
    with h5py.File(DESTINATION, "r") as original:
        for row in replay["results"]:
            match = next(m for m in correspondence["matches"] if m["episode_index"] == row["episode_index"])
            assert match["demo_name"] == row["demo_name"] and match["partition"] == row["partition"]
            demo = original["data"][row["demo_name"]]
            initial = np.asarray(demo["states"][0])
            assert hashlib.sha256(initial.tobytes()).hexdigest() == row["initial_state_sha256"]
            actions = demo["actions"][()]
            kept = keep_action_indices(actions)
            filtered = actions[kept]
            assert array_hash(filtered) == row["action_sha256_f32"] == match["action_sha256_f32"]
            for file_key, hash_key in (("trace_file", "trace_sha256"), ("contact_sheet", "contact_sheet_sha256"),
                                       ("final_post_action_image", "final_image_sha256")):
                assert sha256(ROOT / row[file_key]) == row[hash_key]
            trace = json.loads((ROOT / row["trace_file"]).read_text())
            first = verify_expert_trace(trace, filtered, kept)
            assert first == row["first_goal_after_action_step"]
            assert len(filtered) == row["action_count"] and trace["success"] == row["success_at_end"]
            expert_rows.append({"episode_index": row["episode_index"], "demo_name": row["demo_name"],
                                "action_count": len(filtered), "success_at_end": trace["success"],
                                "first_goal_after_action_step": first})
    assert sum(r["success_at_end"] for r in expert_rows) == replay["successes"]
    return {"schema_version": 1, "status": "verified_existing_failure_diagnosis_evidence",
            "video_audit_sha256": sha256(audit_file), "expert_replay_sha256": sha256(replay_file),
            "decoded_frames_verified": sum(r["frames"] for r in video_rows), "video_rows": video_rows,
            "original_expert_actions_verified": sum(r["action_count"] for r in expert_rows),
            "expert_rows": expert_rows, "expert_successes": replay["successes"],
            "new_rollouts": 0, "optimizer_updates": 0,
            "limits": ["This verifier checks recorded simulator outcomes and source identity; it does not rerun physics.",
                       "Manual visual stage annotations are interpretation, not contact-state ground truth.",
                       "Expert replay successes are not OpenVLA policy successes or matched-state performance comparisons."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify()
    if args.output:
        if args.output.exists():
            assert json.loads(args.output.read_text()) == result
        else:
            with args.output.open("x") as handle:
                json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
