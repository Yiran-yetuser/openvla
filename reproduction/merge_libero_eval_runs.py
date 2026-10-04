"""Validate and assemble the process segments that cover LIBERO-Spatial once each.

The source logs stay separate: the prior log is parsed independently, the 7-9
completion prefix is parsed with expected_tasks=3, and the task-9 continuation is
parsed with expected_tasks=1. This CPU-only script never launches a simulator or
modifies source logs.
"""
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


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def norm_task(value):
    return " ".join(value.lower().replace("_", " ").split())


def read_text_and_hash(path):
    path = Path(path)
    raw = path.read_bytes()
    return raw.decode("utf-8", errors="strict"), {
        "path": str(path), "bytes": len(raw), "sha256": sha256(raw)
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-log", type=Path, required=True)
    p.add_argument("--old-audit", type=Path, required=True)
    p.add_argument("--new-log", type=Path, required=True)
    p.add_argument("--continuation-log", type=Path, required=True)
    p.add_argument("--task-map-file", type=Path, required=True)
    p.add_argument("--libero-root", type=Path, required=True)
    p.add_argument("--provenance", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    old_text, old_source = read_text_and_hash(args.old_log)
    new_text, new_source = read_text_and_hash(args.new_log)
    continuation_text, continuation_source = read_text_and_hash(args.continuation_log)
    old = parse_eval_text(old_text, expected_tasks=10, trials_per_task=50)
    prefix = parse_eval_text(new_text, expected_tasks=3, trials_per_task=50)
    continuation = parse_eval_text(continuation_text, expected_tasks=1, trials_per_task=25)
    old["source"] = old_source
    prefix["source"] = new_source
    continuation["source"] = continuation_source

    previous = json.loads(args.old_audit.read_text())
    if old_source["sha256"] != previous.get("source_sha256"):
        raise ValueError("prior log hash does not match its existing independent audit")
    if (old["status"] != "incomplete" or old["episodes"] != 382
            or old["successes"] != 5 or old["completed_task_summaries"] != 7
            or len(old["tasks"]) != 8 or old["runtime_errors"]):
        raise ValueError("prior run no longer matches the audited 382/5/7-plus-partial ledger")
    if any(t["episodes"] != 50 or not t["summary_verified"] for t in old["tasks"][:7]):
        raise ValueError("prior tasks 0-6 are not all complete")
    if (old["tasks"][7]["episodes"] != 32 or old["tasks"][7]["successes"] != 0
            or old["tasks"][7]["summary_verified"]):
        raise ValueError("prior partial task 7 does not match the audited 32-rollout segment")

    selected = re.search(r"^Selected task IDs: \[([0-9, ]+)\]; trials per task: (\d+)$",
                         new_text, re.MULTILINE)
    if selected is None or [int(x) for x in selected[1].split(",")] != [7, 8, 9] or int(selected[2]) != 50:
        raise ValueError("completion prefix must select task IDs [7, 8, 9] at 50 trials each")
    if not re.search(r"^prepare_for_kbit_inference: False$", new_text, re.MULTILINE):
        raise ValueError("completion prefix did not record the expected inference-preparation mode")
    if (prefix["status"] != "incomplete" or prefix["episodes"] != 125
            or prefix["completed_task_summaries"] != 2 or len(prefix["tasks"]) != 3
            or prefix["runtime_errors"]):
        raise ValueError("completion prefix does not match the clean 125-rollout SIGTERM boundary")
    if any(t["episodes"] != 50 or not t["summary_verified"] for t in prefix["tasks"][:2]):
        raise ValueError("completion prefix tasks 7 and 8 are not both complete")
    if (prefix["tasks"][2]["episodes"] != 25 or prefix["tasks"][2]["summary_verified"]):
        raise ValueError("completion prefix task 9 must contain exactly its first 25 states without a summary")

    selected_continuation = re.search(
        r"^Selected task IDs: \[([0-9, ]+)\]; trials per task: (\d+)$",
        continuation_text, re.MULTILINE
    )
    if (selected_continuation is None
            or [int(x) for x in selected_continuation[1].split(",")] != [9]
            or int(selected_continuation[2]) != 25):
        raise ValueError("continuation must select only task ID 9 for 25 trials")
    if not re.search(r"^Initial state indices per task: \[25, 50\)$",
                     continuation_text, re.MULTILINE):
        raise ValueError("continuation did not log the required task-9 initial-state range [25, 50)")
    config_match = re.search(r"^Evaluation config: (\{.*\})$", continuation_text, re.MULTILINE)
    if config_match is None:
        raise ValueError("continuation is missing the serialized GenerateConfig")
    continuation_config = json.loads(config_match[1])
    continuation_settings = {
        "task_suite_name": "libero_spatial", "num_trials_per_task": 25,
        "task_ids": [9], "initial_state_offset": 25, "seed": 7,
        "center_crop": True, "load_in_4bit": True, "bnb_double_quant": True,
        "prepare_for_kbit_inference": False, "fail_fast": True,
    }
    if any(continuation_config.get(key) != value
           for key, value in continuation_settings.items()):
        raise ValueError("continuation config differs from the required 25-state completion settings")
    if not re.search(r"^prepare_for_kbit_inference: False$", continuation_text, re.MULTILINE):
        raise ValueError("continuation did not record the expected inference-preparation mode")
    if (continuation["status"] != "complete" or continuation["episodes"] != 25
            or continuation["completed_task_summaries"] != 1 or continuation["runtime_errors"]):
        raise ValueError("task-9 continuation is not a complete 25-rollout log without runtime errors")

    task_map_ns = runpy.run_path(str(args.task_map_file))
    canonical = task_map_ns["libero_task_map"]["libero_spatial"]
    if len(canonical) != 10:
        raise ValueError("canonical LIBERO-Spatial task map must contain exactly 10 entries")
    expected_names = [norm_task(name) for name in canonical]
    if [norm_task(t["task"]) for t in old["tasks"]] != expected_names[:8]:
        raise ValueError("prior log task names/order do not match canonical task IDs 0-7")
    if [norm_task(t["task"]) for t in prefix["tasks"]] != expected_names[7:]:
        raise ValueError("completion prefix task names/order do not match canonical task IDs 7-9")
    if [norm_task(t["task"]) for t in continuation["tasks"]] != [expected_names[9]]:
        raise ValueError("continuation does not map to canonical task ID 9")

    # A valid 500-trial suite score uses old IDs 0-6 plus the fresh IDs 7-9.
    # The old partial ID 7 segment is fully excluded; the physical-run count
    # still includes those 32 extra attempts. Task 9's two process segments
    # cover disjoint initial-state indices 0-24 and 25-49.
    per_task = []
    for task_id, task in enumerate(old["tasks"][:7]):
        per_task.append({"task_id": task_id, "canonical_task": canonical[task_id],
                         "episodes": 50, "successes": task["successes"],
                         "success_rate": task["successes"] / 50, "source_run": "prior"})
    for offset, task in enumerate(prefix["tasks"][:2]):
        task_id = 7 + offset
        per_task.append({"task_id": task_id, "canonical_task": canonical[task_id],
                         "episodes": 50, "successes": task["successes"],
                         "success_rate": task["successes"] / 50, "source_run": "completion_prefix"})
    task9_successes = prefix["tasks"][2]["successes"] + continuation["tasks"][0]["successes"]
    per_task.append({"task_id": 9, "canonical_task": canonical[9],
                     "episodes": 50, "successes": task9_successes,
                     "success_rate": task9_successes / 50,
                     "source_run": "completion_prefix+continuation",
                     "state_index_ranges": [[0, 25], [25, 50]]})
    if [row["task_id"] for row in per_task] != list(range(10)):
        raise ValueError("merged task ledger has duplicate or missing IDs")

    score_episodes = sum(row["episodes"] for row in per_task)
    score_successes = sum(row["successes"] for row in per_task)
    completion_rollouts = prefix["episodes"] + continuation["episodes"]
    actual_rollouts = old["episodes"] + completion_rollouts
    excluded_rollouts = old["tasks"][7]["episodes"]
    if (score_episodes != 500 or actual_rollouts != 532 or excluded_rollouts != 32
            or score_successes != old["successes"] + prefix["successes"] + continuation["successes"]):
        raise ValueError("merged episode/success arithmetic failed")

    task_map_bytes = args.task_map_file.read_bytes()
    revision = subprocess.check_output(
        ["git", "-C", str(args.libero_root), "rev-parse", "HEAD"], text=True
    ).strip()
    project_revision = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "-C", str(args.libero_root), "status", "--short"], text=True
    ).strip()
    provenance = json.loads(args.provenance.read_text())
    report = {
        "schema_version": 1,
        "status": "verified_complete_segmented_assembly",
        "checks": {
            "prior_source_sha256_matches_previous_audit": True,
            "prior_run_382_rollouts_5_successes_7_complete_summaries": True,
            "prior_partial_task_7_has_32_trials_and_zero_successes": True,
            "completion_prefix_parsed_with_expected_tasks_3": prefix["expected_tasks"] == 3,
            "completion_prefix_is_125_rollouts_for_task_ids_7_8_9": prefix["episodes"] == 125,
            "continuation_parsed_with_expected_tasks_1": continuation["expected_tasks"] == 1,
            "continuation_is_task_9_states_25_through_49": True,
            "task_9_has_50_distinct_initial_state_indices": per_task[-1]["state_index_ranges"] == [[0, 25], [25, 50]],
            "sigterm_after_125_has_no_python_exception_stack": True,
            "all_source_runs_match_canonical_task_id_to_name_map": True,
            "ten_task_ids_appear_exactly_once": [row["task_id"] for row in per_task] == list(range(10)),
            "each_scored_task_has_50_rollouts": all(row["episodes"] == 50 for row in per_task),
            "all_three_logs_have_zero_runtime_errors": not old["runtime_errors"] and not prefix["runtime_errors"] and not continuation["runtime_errors"],
            "raw_logs_parsed_separately_without_concatenation": True,
            "episode_arithmetic_532_executed_500_scored_32_excluded": (
                actual_rollouts == 532 and score_episodes == 500 and excluded_rollouts == 32
            ),
        },
        "metric": {
            "suite": "libero_spatial",
            "score_episodes": score_episodes,
            "successes": score_successes,
            "success_rate": score_successes / score_episodes,
            "per_task": per_task,
        },
        "rollout_ledger": {
            "actual_rollouts_executed": actual_rollouts,
            "valid_unique_score_rollouts": score_episodes,
            "excluded_partial_rollouts": excluded_rollouts,
            "excluded_source_segment": "prior run task ID 7, episodes 1-32; replaced by fresh complete task ID 7",
            "completion_rollouts": completion_rollouts,
            "completion_process_segments": [
                {"source": "completion_prefix", "rollouts": 125,
                 "task_7_state_range": [0, 50], "task_8_state_range": [0, 50],
                 "task_9_state_range": [0, 25], "termination": "SIGTERM, exit code 143; no Python exception traceback"},
                {"source": "continuation", "rollouts": 25,
                 "task_9_state_range": [25, 50], "termination": "completed normally"},
            ],
        },
        "task_id_map": [
            {"task_id": i, "canonical_task": task}
            for i, task in enumerate(canonical)
        ],
        "task_map_source": {
            "file": str(args.task_map_file),
            "file_sha256": sha256(task_map_bytes),
            "libero_git_revision": revision,
            "libero_git_status_short": git_status,
        },
        "evaluator_project_git_revision": project_revision,
        "source_runs": {
            "prior": {"source": old_source, "parser_report": old},
            "completion_prefix": {"source": new_source, "parser_report": prefix},
            "continuation": {"source": continuation_source,
                             "parser_report": continuation,
                             "run_config": continuation_config},
        },
        "provenance": provenance,
        "limits": [
            "The 500-trial score is assembled by canonical task ID from separately parsed process logs; raw logs were never concatenated.",
            "The prior run physically completed 382 rollouts. The completion prefix plus task-9 continuation executed 150 fresh rollouts, for 532 physical executions; only the old partial task-7 segment of 32 is excluded from the 500 scored trials.",
            "The first completion process received SIGTERM (exit code 143) after 125 rollout records. Its log ends after task-9 state index 24 with no Python exception traceback; the separate continuation covers indices 25-49.",
            "The old launch used the adapter path but did not record its SHA-256 at launch; the current digest and pre-run file timestamps are reported separately.",
            "This is a single-seed, local 4-bit LoRA reproduction, not the official full-finetuned checkpoint or the paper's three-seed average.",
            "The merge validates log arithmetic, task mapping, source hashes, and recorded configuration; it cannot recover runtime facts that were not recorded during the old launch.",
        ],
    }
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": report["status"], **report["metric"],
                      **report["rollout_ledger"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
