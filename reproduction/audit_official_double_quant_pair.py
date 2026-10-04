"""Pair official-checkpoint LIBERO state-0 outcomes across NF4 double-quant modes."""
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

OFFICIAL_REVISION = "962318cec55ac10993ff0f5f43eda9a270b4c873"


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def normalize(value):
    return " ".join(value.lower().replace("_", " ").split())


def config_from(text, label):
    match = re.search(r"^Evaluation config: (\{.*\})$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"{label} log is missing its serialized evaluation config")
    return json.loads(match[1])


def selected_ids(text, trials, label):
    match = re.search(r"^Selected task IDs: \[([0-9, ]+)\]; trials per task: (\d+)$",
                      text, re.MULTILINE)
    if match is None:
        raise ValueError(f"{label} log is missing the selected-task header")
    ids = [int(item) for item in match[1].split(",")]
    if ids != list(range(10)) or int(match[2]) != trials:
        raise ValueError(f"{label} must select task IDs 0-9 for {trials} trial(s) each")
    return ids


def first_trial_outcomes(text, label):
    outcomes = {}
    current_task = None
    current_episode = None
    for raw_line in text.replace("\r", "\n").splitlines():
        line = raw_line.strip()
        task_match = re.search(r"Task: (.+)$", line)
        if task_match:
            current_task = task_match[1].strip()
            continue
        episode_match = re.search(r"Starting episode (\d+)\.\.\.$", line)
        if episode_match:
            current_episode = int(episode_match[1])
            continue
        success_match = re.search(r"^Success: (True|False)$", line)
        if success_match and current_task is not None and current_episode is not None:
            key = (normalize(current_task), current_episode)
            if key in outcomes:
                raise ValueError(f"{label} log repeats task/trial record {key}")
            outcomes[key] = success_match[1] == "True"
            current_episode = None
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-log", type=Path, required=True,
                        help="Official full-suite run with double_quant=False")
    parser.add_argument("--double-quant-log", type=Path, required=True,
                        help="Official 1-trial-per-task run with double_quant=True")
    parser.add_argument("--task-map-file", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--full-audit", type=Path, required=True,
                        help="Independent audit JSON for the 500-trial false arm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    full_raw = args.full_log.read_bytes()
    pilot_raw = args.double_quant_log.read_bytes()
    full_audit = json.loads(args.full_audit.read_text())
    if (full_audit.get("status") != "verified_complete_official_checkpoint_single_seed_control"
            or full_audit.get("source_log", {}).get("sha256") != sha256(full_raw)):
        raise ValueError("double_quant=False source log does not match its independent full-suite audit")
    full_text = full_raw.decode("utf-8", errors="strict")
    pilot_text = pilot_raw.decode("utf-8", errors="strict")
    full_report = parse_eval_text(full_text, expected_tasks=10, trials_per_task=50)
    pilot_report = parse_eval_text(pilot_text, expected_tasks=10, trials_per_task=1)
    if (full_report["status"] != "complete" or full_report["episodes"] != 500
            or full_report["runtime_errors"]):
        raise ValueError("double_quant=False full-suite run is incomplete or contains runtime errors")
    if (pilot_report["status"] != "complete" or pilot_report["episodes"] != 10
            or pilot_report["runtime_errors"]):
        raise ValueError("double_quant=True paired run is incomplete or contains runtime errors")

    full_ids = selected_ids(full_text, 50, "double_quant=False full-suite")
    pilot_ids = selected_ids(pilot_text, 1, "double_quant=True paired")
    full_config = config_from(full_text, "double_quant=False full-suite")
    pilot_config = config_from(pilot_text, "double_quant=True paired")
    project_root = Path(__file__).resolve().parents[1]
    evaluator_revision = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if evaluator_revision != full_audit.get("evaluator_project_git_revision"):
        raise ValueError("evaluator checkout revision differs from the audited 500-trial run")
    loader_path = project_root / "experiments" / "robot" / "openvla_utils.py"
    loader_raw = loader_path.read_bytes()
    loader_text = loader_raw.decode("utf-8", errors="strict")
    required_loader_evidence = (
        'bnb_4bit_quant_type="nf4"',
        "bnb_4bit_compute_dtype=torch.bfloat16",
        'bnb_4bit_use_double_quant=getattr(cfg, "bnb_double_quant", True)',
        "torch_dtype=torch.bfloat16",
    )
    if not all(fragment in loader_text for fragment in required_loader_evidence):
        raise ValueError("shared evaluator source does not pin the expected NF4/BF16 quantization path")
    expected_common = {
        "model_family": "openvla",
        "task_suite_name": "libero_spatial",
        "seed": 7,
        "initial_state_offset": 0,
        "center_crop": True,
        "load_in_4bit": True,
        "load_in_8bit": False,
        "prepare_for_kbit_inference": False,
        "fail_fast": True,
    }
    for label, config in (("double_quant=False", full_config),
                          ("double_quant=True", pilot_config)):
        mismatched = {key: {"expected": value, "actual": config.get(key)}
                      for key, value in expected_common.items()
                      if config.get(key) != value}
        if mismatched:
            raise ValueError(f"{label} run has incompatible shared settings: {mismatched}")
        if not str(config.get("pretrained_checkpoint", "")).endswith(
                "cache/official_libero_spatial"):
            raise ValueError(f"{label} run did not use the official checkpoint")
    if full_config.get("bnb_double_quant") is not False:
        raise ValueError("full-suite control must use double_quant=False")
    if pilot_config.get("bnb_double_quant") is not True:
        raise ValueError("paired run must use double_quant=True")
    allowed_config_differences = {
        "bnb_double_quant", "num_trials_per_task", "task_ids", "run_id_note"
    }
    if set(full_config) != set(pilot_config):
        raise ValueError("paired runs serialize different inference configuration fields")
    config_differences = {
        key: {"double_quant_false": full_config[key], "double_quant_true": pilot_config[key]}
        for key in full_config if full_config[key] != pilot_config[key]
    }
    if set(config_differences) != allowed_config_differences:
        raise ValueError(f"paired-run config differences are unexpected: {config_differences}")
    if full_config.get("task_ids") is not None or full_config.get("num_trials_per_task") != 50:
        raise ValueError("full-suite control is not the canonical 10 x 50 run")
    if pilot_config.get("task_ids") != list(range(10)) or pilot_config.get("num_trials_per_task") != 1:
        raise ValueError("paired run must select all 10 tasks for exactly one trial each")
    if not re.search(r"^Initial state indices per task: \[0, 50\)$", full_text, re.MULTILINE):
        raise ValueError("full-suite control did not cover initial states 0-49")
    if not re.search(r"^Initial state indices per task: \[0, 1\)$", pilot_text, re.MULTILINE):
        raise ValueError("paired run did not select initial state index 0")

    full_checkpoint = full_config["pretrained_checkpoint"]
    pilot_checkpoint = pilot_config["pretrained_checkpoint"]
    if full_checkpoint != pilot_checkpoint:
        raise ValueError("paired runs do not use the same checkpoint path")
    full_base = full_config.get("base_model_path")
    pilot_base = pilot_config.get("base_model_path")
    if full_base != pilot_base:
        raise ValueError("paired runs do not use the same base model path")

    task_map = runpy.run_path(str(args.task_map_file))["libero_task_map"]["libero_spatial"]
    if len(task_map) != 10:
        raise ValueError("canonical task map must contain exactly 10 tasks")
    canonical = [normalize(task) for task in task_map]
    full_trials = first_trial_outcomes(full_text, "double_quant=False full-suite")
    pilot_trials = first_trial_outcomes(pilot_text, "double_quant=True paired")
    rows = []
    for task_id, task_name in enumerate(canonical):
        full_key = (task_name, 1)
        pilot_key = (task_name, 1)
        if full_key not in full_trials or pilot_key not in pilot_trials:
            raise ValueError(f"missing paired state-0 outcome for task ID {task_id}")
        full_success = full_trials[full_key]
        pilot_success = pilot_trials[pilot_key]
        rows.append({
            "task_id": task_id,
            "canonical_task": task_map[task_id],
            "initial_state_index": 0,
            "double_quant_false_success": full_success,
            "double_quant_true_success": pilot_success,
            "outcome": ("both_success" if full_success and pilot_success else
                        "false_only_success" if full_success else
                        "true_only_success" if pilot_success else "both_failure"),
        })

    manifest = json.loads(args.checkpoint_manifest.read_text())
    if manifest.get("revision") != OFFICIAL_REVISION:
        raise ValueError("official checkpoint manifest revision changed")
    task_map_raw = args.task_map_file.read_bytes()
    task_map_revision = subprocess.check_output(
        ["git", "-C", str(args.libero_root), "rev-parse", "HEAD"], text=True
    ).strip()
    totals = {
        "paired_states": len(rows),
        "double_quant_false_successes": sum(row["double_quant_false_success"] for row in rows),
        "double_quant_true_successes": sum(row["double_quant_true_success"] for row in rows),
        "both_success": sum(row["outcome"] == "both_success" for row in rows),
        "false_only_success": sum(row["outcome"] == "false_only_success" for row in rows),
        "true_only_success": sum(row["outcome"] == "true_only_success" for row in rows),
        "both_failure": sum(row["outcome"] == "both_failure" for row in rows),
    }
    audit = {
        "schema_version": 1,
        "status": "verified_official_double_quant_paired_states",
        "scope": "10 matched state-0 outcomes for the same official checkpoint; not a success-rate estimate",
        "source_logs": {
            "double_quant_false_full_suite": {"path": str(args.full_log),
                                               "bytes": len(full_raw),
                                               "sha256": sha256(full_raw)},
            "double_quant_true_paired": {"path": str(args.double_quant_log),
                                         "bytes": len(pilot_raw),
                                         "sha256": sha256(pilot_raw)},
        },
        "checkpoint": {"path": full_checkpoint, "revision": manifest["revision"],
                       "manifest_total_bytes": manifest["total_bytes"]},
        "shared_settings": expected_common,
        "allowed_config_differences": sorted(allowed_config_differences),
        "observed_config_differences": config_differences,
        "arms": {"double_quant_false": False, "double_quant_true": True},
        "evaluator_code": {
            "git_revision": evaluator_revision,
            "nf4_bf16_loader_file": str(loader_path),
            "nf4_bf16_loader_sha256": sha256(loader_raw),
            "runtime_log_omits_loader_banner": True,
        },
        "totals": totals,
        "per_task": rows,
        "checks": {
            "full_suite_complete_500": True,
            "paired_run_complete_10": True,
            "canonical_task_ids_and_order_match": full_ids == pilot_ids == list(range(10)),
            "state_zero_ranges_match": True,
            "same_checkpoint_and_shared_inference_settings": True,
            "shared_evaluator_source_pins_nf4_bf16_compute": True,
            "only_double_quant_and_eval_window_keys_differ": True,
            "full_suite_log_matches_independent_audit": True,
            "evaluator_revision_matches_full_suite_audit": True,
            "checkpoint_revision_matches_manifest": True,
            "no_runtime_errors": True,
        },
        "task_map_source": {"file": str(args.task_map_file),
                            "sha256": sha256(task_map_raw),
                            "libero_git_revision": task_map_revision},
    }
    if not all(audit["checks"].values()):
        raise ValueError("paired-state audit check failed")
    with args.output.open("x") as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": audit["status"], **totals,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
