"""Immutable fork of clean_task step 050, bounded to cumulative update 200.

Restore the original optimizer/RNG; change training exposure only. Preserve all
original checkpoints. Never launch a policy rollout or the full 500 evaluation.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction import train_clean_task as core
from reproduction.audit_episode_split import sha256
from reproduction.fit_small_sample import atomic_json, completed_snapshot, verify_snapshot
from reproduction.run_clean_task_closed_loop import gpu_preflight

PARENT = ROOT / "runs/clean_task_v1/step_050"
RUN = ROOT / "runs/clean_task_extend_v1"
RESULT = ROOT / "reproduction/results/clean_task_extend_v1.json"
PARENT_VERIFICATION = ROOT / "reproduction/results/clean_task_verification_v1.json"


def verify_parent(record, parent_spec, evidence, report_digest):
    """Anchor the source snapshot to the already verified 50-update result."""
    if (evidence.get("status") != "verified_complete"
            or evidence.get("spec") != parent_spec
            or evidence.get("final_report_sha256") != report_digest
            or evidence.get("snapshots", {}).get("50") != record):
        raise ValueError("Parent differs from the independently verified clean-task result")


def specification(audit):
    parent_spec = core.specification(audit)
    record = verify_snapshot(PARENT, parent_spec)
    if record["step"] != 50:
        raise ValueError("Parent is not the completed 50-update clean pilot")
    evidence = json.loads(PARENT_VERIFICATION.read_text())
    verify_parent(record, parent_spec, evidence, sha256(ROOT / "reproduction/results/clean_task_v1.json"))
    spec = copy.deepcopy(parent_spec)
    spec.update(experiment="clean_task_extend_v1", optimizer_updates=200,
                milestones=[50, 100, 150, 200], parent_updates=50, new_updates=150,
                parent_checkpoint=str(PARENT.resolve()), parent_snapshot_sha256=sha256(PARENT / "snapshot.json"),
                parent_spec=parent_spec,
                initialization="Fork verified clean-task step 050 with saved AdamW and Torch CPU/CUDA RNG; no old full-data adapter")
    assert core.sample_schedule(spec["training_frames"], 200, 200)[:800] == core.sample_schedule(spec["training_frames"])
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.launch and args.execute:
        parser.error("Choose launch or execute")
    audit = json.loads(core.AUDIT.read_text())
    spec = specification(audit)
    if not args.launch and not args.execute:
        print(json.dumps(spec, indent=2))
        return
    if RESULT.exists():
        raise FileExistsError("Final evidence exists; verify, do not retrain")
    gpu_preflight()
    if shutil.disk_usage(ROOT).free < 10 * 2**30:
        raise RuntimeError("Need 10 GiB reserve; no automatic deletion")
    RUN.mkdir(exist_ok=True)
    with (RUN / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Extension worker is already active")
            return
        if any(RUN.glob("*.partial")):
            raise RuntimeError("Partial snapshot preserved; diagnose before retry")
        snapshot = completed_snapshot(RUN)
        if snapshot and not args.resume:
            raise RuntimeError("Completed extension snapshot requires explicit verified resume")
        if args.resume and not snapshot:
            raise RuntimeError("No extension snapshot to resume")
        if not snapshot and (RUN / "progress.json").exists():
            raise RuntimeError("Prior pre-snapshot attempt preserved; no automatic retry")
        if snapshot:
            verify_snapshot(snapshot, spec)
        if args.launch:
            command = [sys.executable, str(Path(__file__).resolve()), "--execute"]
            if args.resume:
                command.append("--resume")
            fcntl.flock(lock, fcntl.LOCK_UN)
            with (RUN / f"worker-{time.time_ns()}.log").open("x") as handle:
                child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            print(json.dumps({"launched_pid": child.pid, "run": str(RUN), "new_update_limit": 150}))
            return
        previous = json.loads((RUN / "progress.json").read_text()) if (RUN / "progress.json").exists() else {}
        state = {"status": "running", "pid": __import__("os").getpid(), "spec": spec,
                 "parent_updates": 50, "completed_updates_in_memory": snapshot and json.loads((snapshot / "snapshot.json").read_text())["step"] or 50,
                 "parent_predictions_verified": previous.get("parent_predictions_verified", False),
                 "started_unix": time.time()}
        atomic_json(RUN / "progress.json", state)
        core.RUN, core.RESULT = RUN, RESULT
        try:
            core.execute(audit, spec, state, snapshot or PARENT)
        except Exception:
            state.update(status="failed", error=traceback.format_exc())
            atomic_json(RUN / "progress.json", state)
            raise


if __name__ == "__main__":
    main()
