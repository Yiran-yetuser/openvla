"""Two fixed simulator states on each of two inference paths; no training.

Default prints the bounded plan. --execute runs two 2-rollout CLI processes,
preserves their logs/traces, and refuses to repeat an existing run directory.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.audit_episode_split import sha256

RUN = ROOT / "runs/clean_task_closed_loop_v1"
OUTPUT = ROOT / "reproduction/results/clean_task_closed_loop_v1.json"
CHECKPOINT = ROOT / "runs/clean_task_v1/step_050"


def gpu_preflight():
    commands = subprocess.check_output(["ps", "-eo", "args"], text=True).splitlines()
    if any("/envs/fast3r/bin/python" in c or "co3d_continuous_prepare_v4.lock" in c for c in commands):
        raise RuntimeError("Fast3R project process still running; wait without preemption")
    occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                       "--format=csv,noheader"], text=True).strip()
    if occupied:
        raise RuntimeError(f"GPU occupied; no preemption: {occupied}")
    if shutil.disk_usage(ROOT).free < 3 * 2**30:
        raise RuntimeError("Less than 3 GiB disk reserve; do not start")


def verify_sources(audit, final, snapshot):
    if snapshot["step"] != 50 or snapshot["spec"] != final["spec"]:
        raise ValueError("Final clean-task snapshot/spec mismatch")
    if sha256(ROOT / "reproduction/results/clean_task_split_audit_v1.json") != snapshot["spec"]["audit_sha256"]:
        raise ValueError("Split audit changed")
    for name, digest in snapshot["sha256"].items():
        if sha256(CHECKPOINT / name) != digest:
            raise ValueError(f"Snapshot hash mismatch: {name}")
    base = Path(snapshot["spec"]["base_path"])
    for name, record in audit["base"]["files"].items():
        if sha256(base / name) != record["sha256"]:
            raise ValueError(f"Base hash mismatch: {name}")
    return base


def command(base, task_id, prepare, trace_dir):
    return [sys.executable, "experiments/robot/libero/run_libero_eval.py",
            "--pretrained_checkpoint", str(CHECKPOINT), "--base_model_path", str(base),
            "--load_in_4bit", "True", "--bnb_double_quant", "False", "--center_crop", "False",
            "--prepare_for_kbit_inference", str(prepare), "--task_ids", json.dumps([task_id]),
            "--num_trials_per_task", "2", "--trace_actions", "True", "--fail_fast", "True",
            "--seed", "7", "--local_log_dir", str(trace_dir),
            "--run_id_note", "clean_task_compatible_v1" if prepare else "clean_task_default_v1"]


def read_traces(trace_dir, task_id, instruction, prepare):
    files = sorted(trace_dir.glob("*--task*-trial*.json"))
    if len(files) != 2:
        raise ValueError("Exactly two recorded rollouts required per path")
    result = []
    for path in files:
        row = json.loads(path.read_text())
        if (row["task_id"] != task_id or row["instruction"] != instruction or row["error"] is not None
                or row["prepare_for_kbit_inference"] != prepare or row["bf16_autocast"] != prepare
                or row["seed"] != 7 or row["center_crop"] or row["double_quant"]):
            raise ValueError(f"Unexpected rollout metadata/error: {path}")
        commands = np.asarray([a["simulator_action"] for a in row["actions"]])
        eef = np.asarray([a["eef_pos"] for a in row["actions"]])
        if commands.ndim != 2 or commands.shape[1] != 7 or not len(commands) or not np.isfinite(commands).all():
            raise ValueError("Empty/nonfinite action trace")
        result.append({"file": str(path.relative_to(ROOT)), "sha256": sha256(path),
                       "trial": row["trial"], "success": row["success"], "steps": len(commands),
                       "init_state_sha256": row["init_state_sha256"],
                       "init_state_shape": row["init_state_shape"], "init_state_dtype": row["init_state_dtype"],
                       "first_image_sha256": row["actions"][0]["image_sha256"],
                       "first_eef_pos": row["actions"][0]["eef_pos"],
                       "first_gripper_qpos": row["actions"][0]["gripper_qpos"],
                       "first_policy_action": row["actions"][0]["policy_action"],
                       "translation_below_0p02_fraction": float((np.linalg.norm(commands[:, :3], axis=1) < .02).mean()),
                       "eef_max_distance_from_start_m": float(np.linalg.norm(eef - eef[0], axis=1).max()),
                       "open_gripper_command_fraction": float((commands[:, -1] < 0).mean())})
    if sorted(r["trial"] for r in result) != [0, 1]:
        raise ValueError("Trial identities must be 0 and 1")
    return sorted(result, key=lambda r: r["trial"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"experiment": "clean_task_closed_loop_v1", "checkpoint": str(CHECKPOINT),
                          "states_per_path": 2, "paths": ["default", "compatible"],
                          "optimizer_updates": 0, "maximum_rollouts": 4}, indent=2))
        return
    if OUTPUT.exists() or RUN.exists():
        raise FileExistsError("Prior result/run must be inspected; do not repeat rollouts")
    gpu_preflight()
    audit = json.loads((ROOT / "reproduction/results/clean_task_split_audit_v1.json").read_text())
    final = json.loads((ROOT / "reproduction/results/clean_task_v1.json").read_text())
    snapshot = json.loads((CHECKPOINT / "snapshot.json").read_text())
    base = verify_sources(audit, final, snapshot)
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task_ids = [i for i in range(suite.n_tasks) if suite.get_task(i).language == audit["task_instruction"]]
    if len(task_ids) != 1:
        raise ValueError("Training instruction must match exactly one suite task")
    task_id = task_ids[0]
    RUN.mkdir()
    results = {}
    progress_path = RUN / "progress.json"

    def progress(value):
        value["worker_pid"] = os.getpid()
        temporary = RUN / "progress.tmp.json"
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(progress_path)

    try:
        for name, prepare in (("default", False), ("compatible", True)):
            gpu_preflight()
            trace_dir = ROOT / "reproduction/results/sim_clean_task_v1" / name
            if trace_dir.exists():
                raise FileExistsError("Trace destination already exists")
            trace_dir.mkdir(parents=True)
            argv = command(base, task_id, prepare, trace_dir)
            progress({"status": "running", "path": name, "task_id": task_id,
                      "completed_paths": list(results), "command": argv})
            environment = dict(os.environ, NUMBA_CACHE_DIR="/tmp", HF_HUB_OFFLINE="1",
                               TRANSFORMERS_OFFLINE="1", PYTHONPATH=str(ROOT), MUJOCO_GL="egl")
            with (RUN / f"{name}.log").open("x") as log:
                child = subprocess.run(argv, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if child.returncode != 0:
                raise RuntimeError(f"{name} eval exited {child.returncode}; log preserved, no automatic retry")
            rows = read_traces(trace_dir, task_id, audit["task_instruction"], prepare)
            results[name] = {"successes": sum(r["success"] for r in rows), "episodes": 2, "rows": rows}
        match_fields = ("init_state_sha256", "init_state_shape", "init_state_dtype",
                        "first_image_sha256", "first_eef_pos", "first_gripper_qpos")
        for left, right in zip(results["default"]["rows"], results["compatible"]["rows"]):
            if any(left[k] != right[k] for k in match_fields):
                raise ValueError("Initial state/image mismatch; no inference-path causal comparison")
        verify_sources(audit, final, snapshot)
        report = {"schema_version": 1, "status": "bounded_comparison_complete", "task_id": task_id,
                  "instruction": audit["task_instruction"], "checkpoint": str(CHECKPOINT),
                  "snapshot_sha256": snapshot["sha256"], "source_hashes_unchanged": True,
                  "seed": 7, "nf4": True, "compute_dtype": "bfloat16", "double_quant": False,
                  "center_crop": False, "optimizer_updates": 0, "rollouts": 4,
                  "matched_full_initial_states_and_first_images": True, "paths": results,
                  "limits": ["One trained task, two fixed simulator states per path; no full-suite or statistical improvement claim.",
                             "Compatibility jointly changes preparation and BF16 autocast; not a dtype-only ablation.",
                             "Checkpoint has only 50 updates on eight training episodes."]}
        with OUTPUT.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
        progress({"status": report["status"], "output": str(OUTPUT), "completed_paths": list(results)})
        print(json.dumps({"status": report["status"], "successes": {k: v["successes"] for k, v in results.items()}}, indent=2))
    except Exception:
        progress({"status": "failed_requires_diagnosis", "completed_paths": list(results),
                  "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
