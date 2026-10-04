"""Validate per-rollout LIBERO logs before accepting a full-suite metric.

This CPU-only reader never launches evaluation or changes the source log.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def parse_eval_text(text, expected_tasks=10, trials_per_task=50):
    tasks = {}
    suite = None
    pending = None
    awaiting_total = None
    episodes = 0
    successes = 0
    runtime_errors = []
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if line.startswith("Task suite: "):
            if suite is not None:
                raise ValueError(f"line {number}: duplicate suite header")
            suite = line[len("Task suite: "):]
            if suite != "libero_spatial":
                raise ValueError(f"line {number}: unexpected suite {suite}")
        elif line.startswith("Task: "):
            if pending is not None or awaiting_total is not None:
                raise ValueError(f"line {number}: previous rollout/summary is unfinished")
            name = line[len("Task: "):]
            task = tasks.setdefault(name, {"task": name, "episodes": 0, "successes": 0,
                                         "summary_verified": False})
            if task["summary_verified"]:
                raise ValueError(f"line {number}: task repeated after its summary")
            if len(tasks) > expected_tasks:
                raise ValueError(f"line {number}: too many distinct tasks")
            pending = {"task": name}
        elif match := re.fullmatch(r"Starting episode (\d+)\.\.\.", line):
            if pending is None or "trial" in pending:
                raise ValueError(f"line {number}: episode without a fresh task record")
            trial = int(match[1])
            if trial != tasks[pending["task"]]["episodes"] + 1 or trial > trials_per_task:
                raise ValueError(f"line {number}: duplicated, skipped or excess trial")
            pending["trial"] = trial
        elif line.startswith("Caught exception:") or "Traceback (most recent call last)" in line:
            runtime_errors.append({"line": number, "message": line})
        elif match := re.fullmatch(r"Success: (True|False)", line):
            if pending is None or "trial" not in pending or "success" in pending:
                raise ValueError(f"line {number}: success without one unique started trial")
            pending["success"] = match[1] == "True"
        elif match := re.fullmatch(r"# episodes completed so far: (\d+)", line):
            if pending is None or "success" not in pending or "total" in pending:
                raise ValueError(f"line {number}: invalid episode counter")
            if int(match[1]) != episodes + 1:
                raise ValueError(f"line {number}: cumulative episodes are not consecutive")
            pending["total"] = int(match[1])
        elif match := re.fullmatch(r"# successes: (\d+) \(([0-9.]+)%\)", line):
            if pending is None or "total" not in pending:
                raise ValueError(f"line {number}: successes without a complete trial record")
            expected = successes + int(pending["success"])
            if int(match[1]) != expected or match[2] != f"{expected / pending['total'] * 100:.1f}":
                raise ValueError(f"line {number}: cumulative successes/rate disagree with rollouts")
            episodes = pending["total"]
            successes = expected
            task = tasks[pending["task"]]
            task["episodes"] += 1
            task["successes"] += int(pending["success"])
            pending = None
        elif line.startswith("Current task success rate: "):
            if pending is not None or awaiting_total is not None or not tasks:
                raise ValueError(f"line {number}: task summary without complete rollouts")
            task = next(reversed(tasks.values()))
            rate = float(line.split(": ", 1)[1])
            if task["summary_verified"] or task["episodes"] != trials_per_task:
                raise ValueError(f"line {number}: duplicate summary or task does not have {trials_per_task} trials")
            if not math.isclose(rate, task["successes"] / trials_per_task, abs_tol=1e-12, rel_tol=0):
                raise ValueError(f"line {number}: task summary disagrees with individual outcomes")
            task["success_rate"] = rate
            awaiting_total = task
        elif line.startswith("Current total success rate: "):
            if awaiting_total is None or episodes == 0:
                raise ValueError(f"line {number}: total rate without a task summary")
            rate = float(line.split(": ", 1)[1])
            if not math.isclose(rate, successes / episodes, abs_tol=1e-12, rel_tol=0):
                raise ValueError(f"line {number}: total summary disagrees with individual outcomes")
            awaiting_total["summary_verified"] = True
            awaiting_total = None
    complete = bool(suite == "libero_spatial" and len(tasks) == expected_tasks
                    and episodes == expected_tasks * trials_per_task
                    and pending is None and awaiting_total is None
                    and all(t["summary_verified"] and t["episodes"] == trials_per_task
                            for t in tasks.values()) and not runtime_errors)
    return {"schema_version": 1, "status": "complete" if complete else "incomplete",
            "suite": suite, "expected_tasks": expected_tasks, "trials_per_task": trials_per_task,
            "episodes": episodes, "successes": successes,
            "completed_task_summaries": sum(t["summary_verified"] for t in tasks.values()),
            "tasks": list(tasks.values()), "runtime_errors": runtime_errors,
            "pending_trial": pending, "pending_total_summary": awaiting_total is not None,
            "final_success_rate": successes / episodes if complete else None,
            "interim_success_rate": successes / episodes if episodes else None,
            "limits": ["Log completeness and arithmetic only; does not prove experiment configuration parity.",
                       "Incomplete logs have no final success rate."]}


def parse_eval_file(path):
    path = Path(path)
    raw = path.read_bytes()
    report = parse_eval_text(raw.decode("utf-8", errors="strict"))
    report.update(source_log=str(path), source_bytes=len(raw),
                  source_sha256=hashlib.sha256(raw).hexdigest())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = parse_eval_file(args.log)
    if args.output:
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
