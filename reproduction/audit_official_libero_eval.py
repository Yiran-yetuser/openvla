"""Audit one complete official-checkpoint LIBERO-Spatial evaluation log."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from libero_eval_results import parse_eval_text  # noqa: E402


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def normalize(value):
    return " ".join(value.lower().replace("_", " ").split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--task-map-file", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.log.read_bytes()
    text = raw.decode("utf-8", errors="strict")
    report = parse_eval_text(text, expected_tasks=10, trials_per_task=50)
    header = re.search(r"^Selected task IDs: \[([0-9, ]+)\]; trials per task: (\d+)$",
                       text, re.MULTILINE)
    if header is None:
        raise ValueError("official run log is missing the selected-task header")
    task_ids = [int(item) for item in header[1].split(",")]
    if task_ids != list(range(10)) or int(header[2]) != 50:
        raise ValueError("official run must cover canonical task IDs 0-9, 50 trials each")

    config_match = re.search(r"^Evaluation config: (\{.*\})$", text, re.MULTILINE)
    if config_match is None:
        raise ValueError("official run log is missing its serialized GenerateConfig")
    config = json.loads(config_match[1])
    expected_config = {
        "model_family": "openvla",
        "task_suite_name": "libero_spatial",
        "num_trials_per_task": 50,
        "seed": 7,
        "initial_state_offset": 0,
        "center_crop": True,
        "load_in_4bit": True,
        "bnb_double_quant": False,
        "prepare_for_kbit_inference": False,
        "fail_fast": True,
    }
    mismatched = {key: {"expected": expected, "actual": config.get(key)}
                  for key, expected in expected_config.items()
                  if config.get(key) != expected}
    if mismatched:
        raise ValueError(f"official run config mismatch: {mismatched}")
    checkpoint = str(config.get("pretrained_checkpoint", ""))
    if not checkpoint.endswith("cache/official_libero_spatial"):
        raise ValueError(f"not the official full checkpoint: {checkpoint}")
    if config.get("task_ids") is not None:
        raise ValueError("official full control must use the full suite, not a task subset")
    if not re.search(r"^Initial state indices per task: \[0, 50\)$", text, re.MULTILINE):
        raise ValueError("official full control did not log all 50 canonical initial-state indices")
    if report["status"] != "complete" or report["episodes"] != 500 or report["runtime_errors"]:
        raise ValueError("official checkpoint log is incomplete or has runtime errors")

    task_map = runpy.run_path(str(args.task_map_file))["libero_task_map"]["libero_spatial"]
    if len(task_map) != 10:
        raise ValueError("canonical task map must contain exactly 10 tasks")
    if [normalize(row["task"]) for row in report["tasks"]] != [normalize(t) for t in task_map]:
        raise ValueError("official run task order/names do not match canonical task IDs")
    per_task = []
    for task_id, row in enumerate(report["tasks"]):
        per_task.append({"task_id": task_id, "canonical_task": task_map[task_id],
                         "episodes": row["episodes"], "successes": row["successes"],
                         "success_rate": row["success_rate"]})

    checkpoint_manifest = json.loads(args.checkpoint_manifest.read_text())
    if checkpoint_manifest.get("revision") != "962318cec55ac10993ff0f5f43eda9a270b4c873":
        raise ValueError("official checkpoint manifest revision changed")
    map_raw = args.task_map_file.read_bytes()
    revision = subprocess.check_output(
        ["git", "-C", str(args.libero_root), "rev-parse", "HEAD"], text=True
    ).strip()
    project_revision = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        text=True
    ).strip()
    audit = {
        "schema_version": 1,
        "status": "verified_complete_official_checkpoint_single_seed_control",
        "source_log": {"path": str(args.log), "bytes": len(raw), "sha256": digest(raw)},
        "parser_call": {"function": "parse_eval_text", "expected_tasks": 10,
                        "trials_per_task": 50},
        "parser_report": report,
        "metric": {"suite": "libero_spatial", "episodes": 500,
                   "successes": report["successes"],
                   "success_rate": report["successes"] / 500,
                   "per_task": per_task},
        "checks": {
            "complete_500_log_and_task_summaries": True,
            "all_task_ids_0_through_9_once": True,
            "50_trials_per_task": True,
            "serialized_config_matches_requested_control": True,
            "official_checkpoint_revision_matches_manifest": True,
            "zero_runtime_errors": True,
        },
        "run_config": config,
        "checkpoint_manifest": {"path": str(args.checkpoint_manifest),
                                "revision": checkpoint_manifest["revision"],
                                "total_bytes": checkpoint_manifest["total_bytes"],
                                "status_before_control": checkpoint_manifest["status"]},
        "task_map_source": {"file": str(args.task_map_file),
                            "sha256": digest(map_raw),
                            "libero_git_revision": revision},
        "evaluator_project_git_revision": project_revision,
        "paper_comparison_limit": "This is a single-seed, NF4 inference control; it is not directly comparable to the paper's BF16 three-seed mean.",
    }
    with args.output.open("x") as f:
        json.dump(audit, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": audit["status"], **audit["metric"],
                      "source_log_sha256": audit["source_log"]["sha256"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
