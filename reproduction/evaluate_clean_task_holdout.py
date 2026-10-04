"""One-shot offline evaluation on unused, content-unique same-task episodes.

The verified step-200 adapter is frozen. This script evaluates full-frame
teacher-forced action tokens plus three fixed autoregressive stages per episode.
It does not update weights or run the simulator. Output paths are exclusive.
"""
import argparse
import gc
import hashlib
import json
import time
import traceback
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction import extend_clean_task as extension
from reproduction.audit_episode_split import sha256, episode_fingerprint
from reproduction.fit_small_sample import atomic_json, verify_snapshot

AUDIT = ROOT / "reproduction/results/clean_task_split_audit_v1.json"
CHECKPOINT = ROOT / "runs/clean_task_extend_v1/step_200"
RUN = ROOT / "runs/clean_task_holdout_eval_v1"
RESULT = ROOT / "reproduction/results/clean_task_holdout_eval_v1.json"
SUMMARY = ROOT / "reproduction/results/clean_task_holdout_summary_v1.json"
KEY = "libero_spatial_no_noops"


def plan(audit):
    if audit.get("status") != "cpu_split_verified_training_not_started" or not all(audit["checks"].values()):
        raise ValueError("The source episode split audit is not verified")
    candidates = [row for row in audit["episode_inventory"]
                  if row["instruction"] == audit["task_instruction"]]
    if len(candidates) != 46 or len({row["content_sha256"] for row in candidates}) != len(candidates):
        raise ValueError("Expected 46 content-unique episodes for this instruction")
    selected = audit["selection"]["train"] + audit["selection"]["validation"]
    selected_ids = {row["episode_index"] for row in selected}
    selected_hashes = {row["content_sha256"] for row in selected}
    if (len(selected_ids) != 10 or len(selected_hashes) != 10
            or not selected_ids.issubset({row["episode_index"] for row in candidates})):
        raise ValueError("Training and validation selection is not 10 distinct episodes")
    heldout = [row for row in candidates if row["episode_index"] not in selected_ids]
    if len(heldout) != 36 or any(row["content_sha256"] in selected_hashes for row in heldout):
        raise ValueError("The held-out episode set overlaps train/validation or is incomplete")
    if len({row["content_sha256"] for row in heldout}) != len(heldout):
        raise ValueError("Held-out episodes contain duplicate content")
    return heldout


def fixed_stage_indices(count):
    if count < 3:
        raise ValueError("Each episode needs at least three eligible frames")
    return {0, (count - 1) // 2, count - 1}


def batch_rows(rows, stats, transform, collator, tokenizer, instruction):
    examples = []
    from reproduction.action_metrics import training_target
    for row in rows:
        target = training_target(row["raw_action"], stats["action"])
        example = transform({"dataset_name": KEY, "action": target[None],
                             "observation": {"image_primary": row["image"][None]},
                             "task": {"language_instruction": instruction.encode()}})
        supervised = example["labels"][example["labels"] != -100]
        if len(supervised) != 8 or int(supervised[-1]) != tokenizer.tokenizer.eos_token_id:
            raise ValueError("Unexpected teacher-forced action/EOS supervision")
        if int((supervised > tokenizer.action_token_begin_idx).sum()) != 7:
            raise ValueError("Expected seven action tokens per frame")
        examples.append(example)
    return collator(examples)


def evaluate_episode(rows, model, processor, tokenizer, transform, collator, stats, audit, torch, np):
    from contextlib import nullcontext
    from reproduction.action_metrics import metrics
    from experiments.robot.openvla_utils import get_vla_action

    stages = fixed_stage_indices(len(rows))
    teacher_rows, ar_rows, losses = [], [], []
    patches = model.vision_backbone.featurizer.patch_embed.num_patches
    for offset in range(0, len(rows), 2):
        group = rows[offset:offset + 2]
        data = batch_rows(group, stats, transform, collator, tokenizer, audit["task_instruction"])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids=data["input_ids"].cuda(),
                           attention_mask=data["attention_mask"].cuda(),
                           pixel_values=data["pixel_values"].to("cuda", torch.bfloat16),
                           labels=data["labels"].cuda())
        if not torch.isfinite(output.loss):
            raise FloatingPointError("Non-finite teacher-forced loss")
        losses.append((float(output.loss), len(group)))
        pred = output.logits[:, patches:-1].argmax(-1)
        gt = data["labels"][:, 1:].cuda()
        for j, row in enumerate(group):
            mask = gt[j] > tokenizer.action_token_begin_idx
            predicted = pred[j][mask].cpu().numpy()
            target = gt[j][mask].cpu().numpy()
            if len(target) != 7:
                raise ValueError("Teacher-forced row did not contain seven action tokens")
            teacher_rows.append({"episode_index": row["episode_index"], "timestep": row["timestep"],
                                 "image_sha256": row["image_sha256"], "predicted_tokens": predicted.tolist(),
                                 "target_tokens": target.tolist(),
                                 "token_accuracy": float((predicted == target).mean())})
        del output, data, pred, gt

    for index in sorted(stages):
        row = rows[index]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            action = get_vla_action(model, processor, audit["base"]["path"],
                                    {"full_image": row["image"]}, audit["task_instruction"],
                                    KEY, center_crop=False)
        ar_rows.append({"episode_index": row["episode_index"], "timestep": row["timestep"],
                        "stage": ("early", "middle", "late")[sorted(stages).index(index)],
                        "image_sha256": row["image_sha256"], "raw_action": row["raw_action"],
                        "action": action.tolist(), **metrics(action, row["raw_action"], stats["action"])})

    episode = rows[0]["episode_index"]
    if any(row["episode_index"] != episode for row in teacher_rows + ar_rows):
        raise ValueError("Episode evaluator mixed data from different episodes")
    return {"episode_index": episode, "eligible_frames": len(rows),
            "stage_timesteps": [ar["timestep"] for ar in ar_rows],
            "teacher_loss": sum(value * count for value, count in losses) / len(teacher_rows),
            "teacher_token_accuracy": float(np.mean([row["token_accuracy"] for row in teacher_rows])),
            "autoregressive_frames": len(ar_rows),
            "autoregressive_motion_mae": float(np.mean([row["motion_mae_6d"] for row in ar_rows])),
            "autoregressive_normalized_l1": float(np.mean([row["normalized_l1_7d"] for row in ar_rows])),
            "gripper_correct": sum(row["gripper_correct"] for row in ar_rows),
            "teacher_rows": teacher_rows, "autoregressive_rows": ar_rows}


def load_and_evaluate(audit, heldout, model, processor, tokenizer, transform, collator, stats, progress, torch, np):
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds
    import dlimp as dl

    targets = {row["episode_index"]: row for row in heldout}
    builder = tfds.builder(KEY, data_dir=audit["dataset"]["path"])
    dataset = builder.as_dataset(split="train", shuffle_files=False, decoders={
        "steps": {"observation": {"image": tfds.decode.SkipDecoding(),
                                   "wrist_image": tfds.decode.SkipDecoding()}}})
    results, seen = [], set()
    for index, data_episode in enumerate(dataset):
        if index not in targets:
            continue
        expected = targets[index]
        steps = list(data_episode["steps"].as_numpy_iterator())
        fingerprint = episode_fingerprint(steps)
        if fingerprint != expected["content_sha256"]:
            raise ValueError(f"Content fingerprint changed for held-out episode {index}")
        rows = []
        for timestep, step in enumerate(steps):
            if bool(step["is_last"]) or bool(step["is_terminal"]):
                continue
            instruction = step["language_instruction"].decode()
            if instruction != audit["task_instruction"]:
                raise ValueError(f"Instruction changed in episode {index}")
            image = tf.io.decode_image(step["observation"]["image"], channels=3)
            image = dl.transforms.resize_image(image, size=(224, 224)).numpy()
            rows.append({"episode_index": index, "timestep": timestep,
                         "raw_action": step["action"].tolist(), "image": image,
                         "image_sha256": hashlib.sha256(image.tobytes()).hexdigest()})
        if len(rows) != expected["eligible_frames"]:
            raise ValueError(f"Eligible-frame count changed for held-out episode {index}")
        results.append(evaluate_episode(rows, model, processor, tokenizer, transform,
                                        collator, stats, audit, torch, np))
        seen.add(index)
        progress.update(status="running", episodes_done=len(seen), episodes_total=len(targets),
                        latest_episode=index, latest_teacher_accuracy=results[-1]["teacher_token_accuracy"])
        atomic_json(RUN / "progress.json", progress)
        print(f"episode {index}: frames={len(rows)} teacher_acc={results[-1]['teacher_token_accuracy']:.4f} "
              f"AR_motion_MAE={results[-1]['autoregressive_motion_mae']:.6f}", flush=True)
        del rows, steps, data_episode
    if seen != set(targets):
        raise ValueError(f"TFDS enumeration omitted held-out episodes: {sorted(set(targets)-seen)}")
    results.sort(key=lambda row: row["episode_index"])
    return results


def summarize(episodes, np):
    teacher_rows = [row for episode in episodes for row in episode["teacher_rows"]]
    ar_rows = [row for episode in episodes for row in episode["autoregressive_rows"]]
    correct = sum(sum(a == b for a, b in zip(row["predicted_tokens"], row["target_tokens"]))
                  for row in teacher_rows)
    total_tokens = sum(len(row["target_tokens"]) for row in teacher_rows)
    per_episode_teacher = [row["teacher_token_accuracy"] for row in episodes]
    per_episode_ar = [float(np.mean([row["motion_mae_6d"] for row in ep["autoregressive_rows"]]))
                      for ep in episodes]
    return {"episodes": len(episodes), "teacher_frames": len(teacher_rows),
            "teacher_action_tokens": total_tokens, "teacher_correct_tokens": correct,
            "teacher_token_accuracy": correct / total_tokens,
            "teacher_loss_mean_episode": float(np.mean([row["teacher_loss"] for row in episodes])),
            "teacher_accuracy_mean_episode": float(np.mean(per_episode_teacher)),
            "teacher_accuracy_median_episode": float(np.median(per_episode_teacher)),
            "teacher_accuracy_sd_episode": float(np.std(per_episode_teacher, ddof=1)),
            "autoregressive_frames": len(ar_rows),
            "autoregressive_motion_mae": float(np.mean([row["motion_mae_6d"] for row in ar_rows])),
            "autoregressive_motion_mae_mean_episode": float(np.mean(per_episode_ar)),
            "autoregressive_motion_mae_sd_episode": float(np.std(per_episode_ar, ddof=1)),
            "autoregressive_normalized_l1": float(np.mean([row["normalized_l1_7d"] for row in ar_rows])),
            "autoregressive_gripper_correct": sum(row["gripper_correct"] for row in ar_rows)}


def execute(audit, spec, heldout, progress, np, torch):
    from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets.datasets import RLDSBatchTransform
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction

    base_path = spec["base_path"]
    for name, record in audit["base"]["files"].items():
        if sha256(Path(base_path) / name) != record["sha256"]:
            raise ValueError(f"Base file hash mismatch before evaluation: {name}")
    checkpoint_record = verify_snapshot(CHECKPOINT, spec)
    if checkpoint_record["step"] != 200:
        raise ValueError("Expected the verified step-200 checkpoint")
    checkpoint_manifest_sha = sha256(CHECKPOINT / "snapshot.json")
    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    tokenizer = ActionTokenizer(processor.tokenizer)
    transform = RLDSBatchTransform(tokenizer, processor.tokenizer,
                                   processor.image_processor.apply_transform, PurePromptBuilder)
    collator = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length,
                                                  processor.tokenizer.pad_token_id)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
    base = AutoModelForVision2Seq.from_pretrained(base_path, torch_dtype=torch.bfloat16,
        quantization_config=quant, attn_implementation="sdpa", low_cpu_mem_usage=True,
        trust_remote_code=True, device_map={"": 0})
    base = prepare_model_for_kbit_training(base)
    model = PeftModel.from_pretrained(base, CHECKPOINT, is_trainable=False)
    stats = {KEY: audit["training_statistics"]}
    model.norm_stats = stats
    model.base_model.model.norm_stats = stats
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Frozen holdout evaluation unexpectedly has trainable parameters")
    progress.update(status="evaluating", model_loaded=True, optimizer_updates=0,
                     checkpoint_snapshot_sha256=checkpoint_manifest_sha)
    atomic_json(RUN / "progress.json", progress)
    episodes = load_and_evaluate(audit, heldout, model, processor, tokenizer, transform,
                                 collator, stats[KEY], progress, torch, np)
    for name, record in audit["base"]["files"].items():
        if sha256(Path(base_path) / name) != record["sha256"]:
            raise ValueError(f"Base file hash changed during evaluation: {name}")
    if sha256(CHECKPOINT / "snapshot.json") != checkpoint_manifest_sha:
        raise ValueError("Step-200 checkpoint manifest changed during evaluation")
    report = {"schema_version": 1, "status": "completed_offline_holdout_only",
              "experiment": "clean_task_holdout_eval_v1", "audit_sha256": sha256(AUDIT),
              "checkpoint": str(CHECKPOINT.resolve()), "checkpoint_snapshot_sha256": checkpoint_manifest_sha,
              "checkpoint_step": 200, "optimizer_updates": 0, "simulator_rollouts": 0,
              "inference_path": "fresh OXE base + PEFT k-bit training preparation + BF16 autocast; frozen adapter",
              "task_instruction": audit["task_instruction"],
              "train_episode_indices": [row["episode_index"] for row in audit["selection"]["train"]],
              "validation_episode_indices": [row["episode_index"] for row in audit["selection"]["validation"]],
              "heldout_episode_indices": [row["episode_index"] for row in heldout],
              "heldout_episode_sha256": {str(row["episode_index"]): row["content_sha256"] for row in heldout},
              "frame_policy": "all frames except is_last or is_terminal for teacher forcing; 3 fixed autoregressive stages per episode",
              "statistics_source": "training episodes only", "summary": summarize(episodes, np),
              "episodes": episodes,
              "limits": ["This is a one-shot held-out evaluation for the LIBERO fine-tuning split, not evidence that OXE pretraining never saw these demonstrations.",
                         "Teacher-forced token accuracy and sparse autoregressive action error are offline metrics, not simulator success rate.",
                         "The inference path matches the training-prepared BF16-autocast path; production no-autocast parity is a separate unresolved comparison.",
                         "No model updates, optimizer state changes, checkpoint writes, or simulator rollouts were performed."]}
    if RESULT.exists() or SUMMARY.exists():
        raise FileExistsError("Holdout evidence exists; refusing to overwrite")
    summary = {key: value for key, value in report.items() if key not in ("episodes", "heldout_episode_sha256")}
    summary["result_sha256"] = None
    with RESULT.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    summary["result_sha256"] = sha256(RESULT)
    with SUMMARY.open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    progress.update(status=report["status"], result=str(RESULT), summary=str(SUMMARY),
                    optimizer_updates=0, simulator_rollouts=0)
    atomic_json(RUN / "progress.json", progress)
    print(json.dumps({"status": report["status"], "summary": report["summary"],
                      "result_sha256": summary["result_sha256"]}, indent=2))
    del model, base
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute and (RESULT.exists() or SUMMARY.exists()):
        raise FileExistsError("Holdout evidence already exists; do not rerun")
    audit = json.loads(AUDIT.read_text())
    heldout = plan(audit)
    spec = extension.specification(audit)
    checkpoint_record = verify_snapshot(CHECKPOINT, spec)
    if checkpoint_record["step"] != 200:
        raise ValueError("Expected the completed step-200 checkpoint")
    plan_report = {"status": "planned", "checkpoint_step": 200,
                   "checkpoint_snapshot_sha256": sha256(CHECKPOINT / "snapshot.json"),
                   "train_episode_count": len(audit["selection"]["train"]),
                   "validation_episode_count": len(audit["selection"]["validation"]),
                   "heldout_episode_count": len(heldout),
                   "heldout_episode_indices": [row["episode_index"] for row in heldout],
                   "heldout_eligible_frames": sum(row["eligible_frames"] for row in heldout),
                   "autoregressive_frames": 3 * len(heldout), "optimizer_updates": 0,
                   "simulator_rollouts": 0, "result": str(RESULT), "summary": str(SUMMARY)}
    if not args.execute:
        print(json.dumps(plan_report, indent=2))
        return
    from reproduction.run_clean_task_closed_loop import gpu_preflight
    import numpy as np
    import torch
    if RUN.exists() and any(RUN.iterdir()):
        raise FileExistsError("Holdout run directory is non-empty; preserve existing evidence")
    if shutil.disk_usage(ROOT).free < 10 * 2**30:
        raise RuntimeError("Need 10 GiB reserve; no automatic cleanup")
    gpu_preflight()
    RUN.mkdir(exist_ok=True)
    progress = {**plan_report, "status": "running", "started_unix": time.time(),
                "episodes_done": 0, "episodes_total": len(heldout)}
    atomic_json(RUN / "progress.json", progress)
    try:
        execute(audit, spec, heldout, progress, np, torch)
    except Exception:
        progress.update(status="failed", error=traceback.format_exc())
        atomic_json(RUN / "progress.json", progress)
        raise


if __name__ == "__main__":
    import shutil
    main()
