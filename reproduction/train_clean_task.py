"""Fresh OXE-base QLoRA, bounded to 50 updates on 8 whole train episodes.

Two separate episodes are validation only. Default is CPU verification; --launch
starts one locked worker. Checkpoints are immutable, hash-verified and resumable.
No simulator rollout or paper success-rate claim is made.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.audit_episode_split import sha256, episode_fingerprint, action_statistics
from reproduction.fit_small_sample import atomic_json, completed_snapshot, verify_snapshot

AUDIT = ROOT / "reproduction/results/clean_task_split_audit_v1.json"
RUN = ROOT / "runs/clean_task_v1"
RESULT = ROOT / "reproduction/results/clean_task_v1.json"
KEY = "libero_spatial_no_noops"


def specification(audit):
    if audit["status"] != "cpu_split_verified_training_not_started" or not all(audit["checks"].values()):
        raise ValueError("Split audit has not passed")
    train, validation = audit["selection"]["train"], audit["selection"]["validation"]
    if len(train) != 8 or len(validation) != 2:
        raise ValueError("Expected fixed 8/2 episode partition")
    if not {e["content_sha256"] for e in train}.isdisjoint(e["content_sha256"] for e in validation):
        raise ValueError("Training and validation overlap")
    return {"experiment": "clean_task_v1", "audit_sha256": sha256(AUDIT),
            "base_path": audit["base"]["path"], "base_revision": audit["base"]["revision"],
            "initialization": "fresh LoRA on OXE pretrained base; no historical adapter",
            "optimizer_updates": 50, "learning_rate": 1e-4, "rank": 32, "lora_alpha": 16,
            "lora_dropout": 0., "targets": "all-linear", "seed": 7, "microbatch": 2,
            "accumulation": 8, "nf4": True, "compute_dtype": "bfloat16", "double_quant": False,
            "augmentation": False, "center_crop": False, "milestones": [0, 10, 25, 50],
            "train_episodes": [e["episode_index"] for e in train],
            "validation_episodes": [e["episode_index"] for e in validation],
            "training_frames": audit["training_statistics"]["num_transitions"],
            "validation_frames": audit["validation"]["num_transitions"],
            "sampling": "seeded shuffled training epochs; 16 frames per update",
            "validation_teacher": "all eligible validation frames",
            "autoregressive": "three fixed stage frames per episode; not full-frame or simulator evaluation"}


def sample_schedule(frame_count, updates=50):
    if frame_count < 2 or not 1 <= updates <= 50:
        raise ValueError("Invalid bounded sampling budget")
    result, epoch = [], 0
    while len(result) < updates * 16:
        order = list(range(frame_count))
        random.Random(7 + epoch).shuffle(order)
        result.extend(order)
        epoch += 1
    return result[:updates * 16]


def load_frames(audit):
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds
    import dlimp as dl
    frames = {"train": [], "validation": []}
    selected = {e["episode_index"]: (split, e) for split in frames for e in audit["selection"][split]}
    # TFDS slicing changes which shards participate in interleaving. Enumerate
    # exactly the full stream used by the audit; never interpret [:] as its index.
    builder = tfds.builder(KEY, data_dir=audit["dataset"]["path"])
    dataset = builder.as_dataset(split="train", shuffle_files=False, decoders={
        "steps": {"observation": {"image": tfds.decode.SkipDecoding(),
                                   "wrist_image": tfds.decode.SkipDecoding()}}})
    seen = set()
    for idx, data_episode in enumerate(dataset):
        if idx in selected:
            split, episode = selected[idx]
            seen.add(idx)
            steps = list(data_episode["steps"].as_numpy_iterator())
            assert episode_fingerprint(steps) == episode["content_sha256"]
            for t, step in enumerate(steps):
                if bool(step["is_last"]) or bool(step["is_terminal"]):
                    continue
                assert step["language_instruction"].decode() == audit["task_instruction"]
                image = tf.io.decode_image(step["observation"]["image"], channels=3)
                image = dl.transforms.resize_image(image, size=(224, 224)).numpy()
                frames[split].append({"episode_index": idx, "timestep": t,
                                     "raw_action": step["action"].tolist(), "image": image,
                                     "image_sha256": hashlib.sha256(image.tobytes()).hexdigest()})
    assert seen == set(selected)
    # Preserve the audit's concatenation order as well as identities: float64
    # mean/std reductions can otherwise differ in their final rounding bit.
    for split in frames:
        order = {e["episode_index"]: i for i, e in enumerate(audit["selection"][split])}
        frames[split].sort(key=lambda r: (order[r["episode_index"]], r["timestep"]))
    all_training_actions = [r["raw_action"] for r in frames["train"]]
    assert action_statistics(all_training_actions) == audit["training_statistics"]["action"]
    assert len(frames["train"]) == audit["training_statistics"]["num_transitions"]
    assert len(frames["validation"]) == audit["validation"]["num_transitions"]
    return frames


def execute(audit, spec, state, resume):
    import numpy as np
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets.datasets import RLDSBatchTransform
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from reproduction.action_metrics import training_target, metrics
    from experiments.robot.openvla_utils import get_vla_action, get_vla, get_processor
    from types import SimpleNamespace
    torch.manual_seed(7)
    np.random.seed(7)
    frames = load_frames(audit)
    base_path = spec["base_path"]
    for name, record in audit["base"]["files"].items():
        assert sha256(Path(base_path) / name) == record["sha256"]
    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    tokenizer = ActionTokenizer(processor.tokenizer)
    transform = RLDSBatchTransform(tokenizer, processor.tokenizer,
                                   processor.image_processor.apply_transform, PurePromptBuilder)
    collator = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length,
                                               processor.tokenizer.pad_token_id)
    stats = {KEY: audit["training_statistics"]}

    def batch(rows):
        examples = []
        for row in rows:
            target = training_target(row["raw_action"], stats[KEY]["action"])
            example = transform({"dataset_name": KEY, "action": target[None],
                                 "observation": {"image_primary": row["image"][None]},
                                 "task": {"language_instruction": audit["task_instruction"].encode()}})
            labels = example["labels"]
            supervised = labels[labels != -100]
            assert len(supervised) == 8 and int(supervised[-1]) == processor.tokenizer.eos_token_id
            assert int((supervised > tokenizer.action_token_begin_idx).sum()) == 7
            examples.append(example)
        return collator(examples)

    # Validate every supervision frame before any GPU update; discard temporary tensors.
    for split in frames:
        for row in frames[split]:
            batch([row])
    state.update(supervision_frames_verified=sum(map(len, frames.values())))
    atomic_json(RUN / "progress.json", state)
    record = verify_snapshot(resume, spec) if resume else None
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
    base = AutoModelForVision2Seq.from_pretrained(base_path, torch_dtype=torch.bfloat16,
        quantization_config=quant, attn_implementation="sdpa", low_cpu_mem_usage=True,
        trust_remote_code=True, device_map={"": 0})
    base = prepare_model_for_kbit_training(base)
    model = (PeftModel.from_pretrained(base, resume, is_trainable=True) if resume else
             get_peft_model(base, LoraConfig(r=32, lora_alpha=16, lora_dropout=0., target_modules="all-linear",
                                           init_lora_weights="gaussian")))
    model.norm_stats = stats
    model.base_model.model.norm_stats = stats
    assert all("lora_" in name for name, p in model.named_parameters() if p.requires_grad)
    if not resume:
        assert all(not bool(torch.count_nonzero(p)) for name, p in model.named_parameters() if "lora_B" in name)
    trainable = [p for p in model.parameters() if p.requires_grad]
    state["trainable_parameters"] = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    start = record["step"] if record else 0
    if record:
        saved = torch.load(resume / "optimizer_rng.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        del saved
    schedule = sample_schedule(len(frames["train"]))

    def forward(data, production=False):
        from contextlib import nullcontext
        with (nullcontext() if production else torch.autocast("cuda", dtype=torch.bfloat16)):
            return model(input_ids=data["input_ids"].cuda(), attention_mask=data["attention_mask"].cuda(),
                         pixel_values=data["pixel_values"].to("cuda", torch.bfloat16), labels=data["labels"].cuda())

    def evaluate(step, production=False):
        from contextlib import nullcontext
        model.eval()
        result = {"step": step, "loader": "production_no_autocast" if production else "training_BF16_autocast",
                  "center_crop": False, "splits": {}}
        with torch.inference_mode():
            for split in frames:
                fixed = {(r["episode_index"], r["timestep"]) for r in audit["fixed_frames"][split]}
                selected = [r for r in frames[split] if (r["episode_index"], r["timestep"]) in fixed]
                # Validation uses ALL 246 eligible frames; train teacher metrics use 24 fixed monitors only.
                teacher = frames[split] if split == "validation" else selected
                teacher_rows, ar_rows, losses = [], [], []
                for i in range(0, len(teacher), 2):
                    group = teacher[i:i + 2]
                    data = batch(group)
                    output = forward(data, production)
                    assert torch.isfinite(output.loss)
                    losses.append((float(output.loss), len(group)))
                    patches = model.vision_backbone.featurizer.patch_embed.num_patches
                    pred = output.logits[:, patches:-1].argmax(-1)
                    gt = data["labels"][:, 1:].cuda()
                    for j, row in enumerate(group):
                        mask = gt[j] > tokenizer.action_token_begin_idx
                        a, b = pred[j][mask].cpu().numpy(), gt[j][mask].cpu().numpy()
                        assert len(b) == 7
                        teacher_rows.append({"episode_index": row["episode_index"], "timestep": row["timestep"],
                            "image_sha256": row["image_sha256"], "predicted_tokens": a.tolist(),
                            "target_tokens": b.tolist(), "token_accuracy": float((a == b).mean())})
                    del output, data, pred, gt
                for row in selected:
                    with (nullcontext() if production else torch.autocast("cuda", dtype=torch.bfloat16)):
                        action = get_vla_action(model, processor, base_path, {"full_image": row["image"]},
                                               audit["task_instruction"], KEY, center_crop=False)
                    ar_rows.append({k: v for k, v in row.items() if k != "image"})
                    ar_rows[-1].update(action=action.tolist(), **metrics(action, row["raw_action"], stats[KEY]["action"]))
                result["splits"][split] = {"teacher_frames": len(teacher_rows),
                    "teacher_loss": sum(x * n for x, n in losses) / len(teacher_rows),
                    "teacher_token_accuracy": float(np.mean([r["token_accuracy"] for r in teacher_rows])),
                    "autoregressive_frames": len(ar_rows),
                    "autoregressive_motion_mae": float(np.mean([r["motion_mae_6d"] for r in ar_rows])),
                    "autoregressive_normalized_l1": float(np.mean([r["normalized_l1_7d"] for r in ar_rows])),
                    "gripper_correct": sum(r["gripper_correct"] for r in ar_rows),
                    "teacher_rows": teacher_rows, "autoregressive_rows": ar_rows}
        return result

    def save(step, evaluation):
        final, partial = RUN / f"step_{step:03d}", RUN / f"step_{step:03d}.partial"
        if final.exists() or partial.exists():
            raise FileExistsError("Preserve existing checkpoint evidence")
        if shutil.disk_usage(RUN).free < 4 * 2**30:
            raise RuntimeError("Insufficient checkpoint headroom")
        partial.mkdir()
        model.save_pretrained(partial)
        processor.save_pretrained(partial)
        atomic_json(partial / "dataset_statistics.json", stats)
        torch.save({"optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()}, partial / "optimizer_rng.pt")
        atomic_json(partial / "evaluation.json", evaluation)
        names = ["adapter_model.safetensors", "adapter_config.json", "dataset_statistics.json",
                 "optimizer_rng.pt", "evaluation.json"]
        atomic_json(partial / "snapshot.json", {"step": step, "spec": spec,
                                               "sha256": {n: sha256(partial / n) for n in names}})
        partial.rename(final)
        state.update(latest_completed_checkpoint=str(final), completed_updates_in_memory=step)
        atomic_json(RUN / "progress.json", state)
        print(f"Committed checkpoint {step}", flush=True)

    if not record:
        save(0, evaluate(0))
    with (RUN / f"losses-{time.time_ns()}.jsonl").open("x") as log:
        for step in range(start + 1, 51):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            indices = schedule[(step - 1) * 16:step * 16]
            for micro in range(8):
                output = forward(batch([frames["train"][i] for i in indices[micro * 2:micro * 2 + 2]]))
                assert torch.isfinite(output.loss)
                losses.append(float(output.loss.detach()))
                (output.loss / 8).backward()
                del output
            gradients = [p.grad for p in trainable if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert any(bool(torch.count_nonzero(g)) for g in gradients)
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            del gradients
            state.update(completed_updates_in_memory=step, last_mean_loss=float(np.mean(losses)))
            atomic_json(RUN / "progress.json", state)
            log.write(json.dumps({"optimizer_step": step, "sample_indices": indices,
                                  "microbatch_losses": losses, "mean_loss": state["last_mean_loss"]}, allow_nan=False) + "\n")
            log.flush()
            print(f"Update {step}/50 loss={state['last_mean_loss']:.6f}", flush=True)
            if step in spec["milestones"]:
                save(step, evaluate(step))
    peak = torch.cuda.max_memory_allocated() / 2**30
    del optimizer, trainable, model, base
    gc.collect()
    torch.cuda.empty_cache()
    cfg = SimpleNamespace(pretrained_checkpoint=str(RUN / "step_050"), base_model_path=base_path,
                          load_in_4bit=True, load_in_8bit=False, bnb_double_quant=False)
    model, processor = get_vla(cfg), get_processor(cfg)
    production = evaluate(50, production=True)
    evaluations = [json.loads((p.parent / "evaluation.json").read_text()) for p in sorted(RUN.glob("step_???/snapshot.json"))]
    equal = all([r["action"] for r in evaluations[-1]["splits"][s]["autoregressive_rows"]] ==
                [r["action"] for r in production["splits"][s]["autoregressive_rows"]] for s in frames)
    for name, record in audit["base"]["files"].items():
        assert sha256(Path(base_path) / name) == record["sha256"]
    report = {"status": "completed" if equal else "completed_loader_difference_requires_diagnosis",
              "spec": spec, "evaluations": evaluations, "production_evaluation": production,
              "production_actions_equal": equal, "base_checkpoint_modified": False,
              "training_peak_allocated_gib": peak,
              "limits": ["Single task, 8/2 episodes, one seed, only 50 updates; not full-paper reproduction.",
                         "Episode validation isolation holds for this fresh adapter, not proof about OXE pretraining overlap.",
                         "Train teacher uses 24 monitor frames; validation teacher uses all eligible frames.",
                         "Autoregressive uses fixed 24 train/6 validation frames, not all-frame action metrics.",
                         "No simulator rollout or task success-rate measurement.",
                         "50 updates draw 800 of the training frames, less than one full epoch."]}
    with RESULT.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    state.update(status=report["status"], result=str(RESULT))
    atomic_json(RUN / "progress.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-preupdate", action="store_true",
                        help="One explicit retry of a failed CPU-only attempt; retain failure evidence")
    args = parser.parse_args()
    audit = json.loads(AUDIT.read_text())
    spec = specification(audit)
    if not args.launch and not args.execute:
        print(json.dumps(spec, indent=2))
        return
    if RESULT.exists():
        print("Final evidence exists; verify it instead of repeating training")
        return
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Worker already active; no duplicate launch")
            return
        occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                                           "--format=csv,noheader"], text=True).strip()
        if occupied:
            raise RuntimeError(f"Other GPU work is active; no preemption: {occupied}")
        if shutil.disk_usage(RUN).free < 10 * 2**30:
            raise RuntimeError("Need at least 10GiB free; do not delete other data automatically")
        snapshot = completed_snapshot(RUN)
        if snapshot and not args.resume:
            raise RuntimeError("Existing completed snapshot: explicit verified resume required")
        if args.resume and not snapshot:
            raise RuntimeError("No complete snapshot to resume")
        if snapshot:
            verify_snapshot(snapshot, spec)
        previous = json.loads((RUN / "progress.json").read_text()) if (RUN / "progress.json").exists() else None
        if previous and not snapshot:
            if not args.retry_preupdate:
                raise RuntimeError("Prior attempt has no completed snapshot: diagnose retained progress/log first")
            if (previous["status"] != "failed" or previous["completed_updates_in_memory"] != 0
                    or "trainable_parameters" in previous or (RUN / "retry-preupdate.json").exists()):
                raise RuntimeError("Only one failed pre-model CPU attempt can be explicitly retried")
        if any(RUN.glob("*.partial")):
            raise RuntimeError("Partial checkpoint evidence needs diagnosis before retry")
        if args.launch:
            command = [sys.executable, str(Path(__file__).resolve()), "--execute"]
            if args.resume:
                command.append("--resume")
            if args.retry_preupdate:
                command.append("--retry-preupdate")
            # The child takes the lifetime lock; it must not race this launcher's lock.
            fcntl.flock(lock, fcntl.LOCK_UN)
            with (RUN / f"worker-{time.time_ns()}.log").open("x") as handle:
                process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            print(json.dumps({"launched_pid": process.pid, "run": str(RUN)}))
            return
        if args.retry_preupdate:
            with (RUN / "retry-preupdate.json").open("x") as handle:
                json.dump({"prior_failure": previous, "repair": "Select from the full TFDS stream; retain fingerprint check",
                           "retry_unix": time.time()}, handle, indent=2, allow_nan=False)
        state = {"status": "running", "pid": os.getpid(), "spec": spec,
                 "completed_updates_in_memory": 0, "started_unix": time.time()}
        atomic_json(RUN / "progress.json", state)
        try:
            execute(audit, spec, state, snapshot)
        except Exception:
            state.update(status="failed", error=traceback.format_exc())
            atomic_json(RUN / "progress.json", state)
            raise


if __name__ == "__main__":
    main()
