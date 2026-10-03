"""Summarize real LIBERO action/state traces; does not infer task success."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = []
    for path in sorted(args.trace_dir.glob("*--task*-trial*.json")):
        data = json.loads(path.read_text())
        assert data["error"] is None, f"Simulator error is not a policy failure: {path}"
        commands = np.array([s["simulator_action"] for s in data["actions"]])
        eef = np.array([s["eef_pos"] for s in data["actions"]])
        assert len(commands) > 0
        records.append({"file": path.name, "task_id": data["task_id"], "trial": data["trial"],
                        "success": data["success"], "steps": len(commands),
                        "mean_translation_command_norm": float(np.linalg.norm(commands[:, :3], axis=1).mean()),
                        "translation_below_0p02_fraction": float((np.linalg.norm(commands[:, :3], axis=1) < .02).mean()),
                        "eef_axis_span_m": np.ptp(eef, axis=0).tolist(),
                        "eef_max_distance_from_start_m": float(np.linalg.norm(eef - eef[0], axis=1).max()),
                        "open_gripper_command_fraction": float((commands[:, -1] < 0).mean()),
                        "gripper_command_switches": int((commands[1:, -1] != commands[:-1, -1]).sum())})
    assert records
    report = {"schema_version": 1, "episodes": len(records),
              "successes": sum(r["success"] for r in records), "records": records,
              "scope": "Fixed task/initial-state diagnostic, NOT full-suite success estimate",
              "threshold_definition": "Translation command Euclidean norm <0.02; not physical displacement"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
