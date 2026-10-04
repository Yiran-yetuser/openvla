"""CPU-only verification of the four bounded rollout traces and source hashes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduction.audit_episode_split import sha256
from reproduction.libero_eval_results import parse_eval_text
from reproduction.run_clean_task_closed_loop import CHECKPOINT, OUTPUT, ROOT, verify_sources


def verify():
    report = json.loads(OUTPUT.read_text())
    assert report["status"] == "bounded_comparison_complete"
    assert report["rollouts"] == 4 and report["optimizer_updates"] == 0
    audit = json.loads((ROOT / "reproduction/results/clean_task_split_audit_v1.json").read_text())
    final = json.loads((ROOT / "reproduction/results/clean_task_v1.json").read_text())
    snapshot = json.loads((CHECKPOINT / "snapshot.json").read_text())
    verify_sources(audit, final, snapshot)
    assert report["snapshot_sha256"] == snapshot["sha256"]
    raw = {}
    summaries = {}
    for name, prepare in (("default", False), ("compatible", True)):
        entries = report["paths"][name]["rows"]
        assert [e["trial"] for e in entries] == [0, 1]
        rows = []
        for entry in entries:
            path = ROOT / entry["file"]
            assert path.resolve().is_relative_to(ROOT) and sha256(path) == entry["sha256"]
            trace = json.loads(path.read_text())
            assert trace["trial"] == entry["trial"] and trace["success"] == entry["success"]
            assert trace["instruction"] == audit["task_instruction"] and trace["task_id"] == report["task_id"]
            assert trace["error"] is None and trace["prepare_for_kbit_inference"] == prepare
            assert trace["bf16_autocast"] == prepare and trace["seed"] == 7
            assert not trace["center_crop"] and not trace["double_quant"]
            actions = np.asarray([a["simulator_action"] for a in trace["actions"]])
            policy = np.asarray([a["policy_action"] for a in trace["actions"]])
            eef = np.asarray([a["eef_pos"] for a in trace["actions"]])
            assert len(actions) == entry["steps"] and 0 < len(actions) <= 220
            assert np.isfinite(actions).all() and np.isfinite(policy).all() and np.isfinite(eef).all()
            assert np.array_equal(actions[:, :6], policy[:, :6])
            assert np.all((policy[:, -1] >= 0) & (policy[:, -1] <= 1))
            assert np.array_equal(actions[:, -1], -np.sign(2 * policy[:, -1] - 1))
            metrics = {"translation_below_0p02_fraction": float((np.linalg.norm(actions[:, :3], axis=1) < .02).mean()),
                       "eef_max_distance_from_start_m": float(np.linalg.norm(eef - eef[0], axis=1).max()),
                       "open_gripper_command_fraction": float((actions[:, -1] < 0).mean())}
            assert all(entry[k] == v for k, v in metrics.items())
            rows.append({"trial": entry["trial"], "success": trace["success"], "steps": len(actions), **metrics})
            raw[name, entry["trial"]] = trace
        successes = sum(r["success"] for r in rows)
        assert report["paths"][name]["successes"] == successes and report["paths"][name]["episodes"] == 2
        text_logs = list((ROOT / entries[0]["file"]).parent.glob("*.txt"))
        assert len(text_logs) == 1
        log = parse_eval_text(text_logs[0].read_text(), expected_tasks=1, trials_per_task=2)
        assert log["status"] == "complete" and log["episodes"] == 2 and log["successes"] == successes
        assert log["tasks"][0]["task"] == audit["task_instruction"]
        summaries[name] = {"episodes": 2, "successes": successes, "rows": rows}
    pairs = []
    for trial in (0, 1):
        left, right = raw["default", trial], raw["compatible", trial]
        for key in ("init_state_sha256", "init_state_shape", "init_state_dtype"):
            assert left[key] == right[key]
        for key in ("image_sha256", "eef_pos", "gripper_qpos"):
            assert left["actions"][0][key] == right["actions"][0][key]
        actions_equal = [a["simulator_action"] == b["simulator_action"]
                         for a, b in zip(left["actions"], right["actions"])]
        first_difference = next((i for i, same in enumerate(actions_equal) if not same), None)
        if first_difference is None and len(left["actions"]) != len(right["actions"]):
            first_difference = len(actions_equal)
        pairs.append({"trial": trial, "matched_full_initial_state_and_first_image": True,
                      "first_policy_action_equal": left["actions"][0]["policy_action"] == right["actions"][0]["policy_action"],
                      "first_simulator_action_difference_step": first_difference,
                      "complete_simulator_action_sequence_equal": first_difference is None,
                      "scope": "After actions diverge, subsequent observations can diverge too."})
    return {"schema_version": 1, "status": "verified_bounded_closed_loop_comparison",
            "report_sha256": sha256(OUTPUT), "source_hashes_verified": True,
            "paths": summaries, "paired_states": pairs,
            "limits": report["limits"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify()
    if args.output:
        if args.output.exists():
            assert json.loads(args.output.read_text()) == result, "Existing verification disagrees"
        else:
            with args.output.open("x") as handle:
                json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
