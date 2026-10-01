"""Bounded real-action fitting probe with immutable resumable checkpoints.

Default: dry run. --launch starts an independent local worker (no Codex API).
--execute runs it; --resume resumes only a verified, completed snapshot.
This is deliberate fitting of 16 previously seen frames, NOT generalization.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.audit_training_runtime import file_hash, preflight

SOURCE = ROOT / "adapter-tmp/libero_spatial_full_paper/openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0+q-4bit--image_aug"
RUN = ROOT / "runs/small_fit_v1"
RESULT = ROOT / "reproduction/results/small_fit_v1.json"


def atomic_json(path, value):
    """Progress can change atomically; immutable evidence/checkpoints cannot."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def completed_snapshot(run):
    paths = sorted(run.glob("step_[0-9][0-9][0-9]/snapshot.json"))
    return paths[-1].parent if paths else None


def verify_snapshot(path, expected_spec):
    record = json.loads((path / "snapshot.json").read_text())
    if record["spec"] != expected_spec:
        raise ValueError("Resume specification differs; choose a new run")
    for filename, digest in record["sha256"].items():
        if file_hash(path / filename) != digest:
            raise ValueError(f"Corrupt snapshot: {filename}")
    return record


def plan_spec(checkpoint, updates, learning_rate):
    if not 1 <= updates <= 50:
        raise ValueError("This probe is bounded to 1..50 optimizer updates")
    if learning_rate != 1e-4:
        raise ValueError("v1 uses fixed lr1e-4; changing it requires a new version")
    return {"schema_version": 1, "source_checkpoint": str(checkpoint.resolve()),
            "optimizer_updates": updates, "learning_rate": learning_rate,
            "fit_frames": 16, "microbatch": 2, "accumulation": 8,
            "augmentation": False, "nf4": True, "double_quant": False,
            "seed": 7, "milestones": sorted({0, min(10, updates), min(25, updates), updates}),
            "claim": "memorization probe; no generalization or rollout success claim"}


def worker(args, spec):
    args.run.mkdir(parents=True, exist_ok=True)
    with (args.run / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This fitting worker is already active; do not duplicate")
        if args.output.exists():
            print("Final report already exists; verify it, do not retrain", flush=True)
            return
        snapshots = completed_snapshot(args.run)
        if snapshots and not args.resume:
            raise FileExistsError("Completed snapshot exists; use --resume")
        # Allocate headroom for four immutable adapters + optimizer/RNG and working copies.
        preflight(args.checkpoint, args.run)
        if shutil.disk_usage(args.run).free < 12 * 2**30:
            raise RuntimeError("Fitting probe requires at least 12GiB free; do not delete user data")
        hashes = {name: file_hash(args.checkpoint / name) for name in
                  ("adapter_model.safetensors", "adapter_config.json", "dataset_statistics.json")}
        spec = {**spec, "original_sha256": hashes}
        if (args.run / "spec.json").exists():
            if json.loads((args.run / "spec.json").read_text()) != spec:
                raise ValueError("Run specification changed")
        else:
            atomic_json(args.run / "spec.json", spec)
        state = {"status": "running", "pid": os.getpid(), "spec": spec,
                 "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "completed_updates_in_memory": 0, "latest_completed_checkpoint": None}
        atomic_json(args.run / "progress.json", state)
        try:
            execute_fit(args, spec, state, snapshots)
        except Exception as error:
            state.update(status="failed", error=repr(error), traceback=traceback.format_exc())
            atomic_json(args.run / "progress.json", state)
            raise


def execute_fit(args, spec, state, snapshot):
    import gc
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds
    import dlimp as dl
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets.datasets import RLDSBatchTransform
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from reproduction.action_metrics import training_target, metrics
    from experiments.robot.openvla_utils import get_vla_action, get_vla, get_processor
    from types import SimpleNamespace

    torch.manual_seed(7)
    np.random.seed(7)
    stats = json.loads((args.checkpoint / "dataset_statistics.json").read_text())
    key = "libero_spatial_no_noops"
    base_path = json.loads((args.checkpoint / "adapter_config.json").read_text())["base_model_name_or_path"]
    audit = json.loads((ROOT / "reproduction/results/training_chain_cpu_audit_v1.json").read_text())
    assert all(x == "pass" for x in audit["checks"].values())
    selected = audit["rows"]
    assert len(selected) == 24
    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    tokenizer = ActionTokenizer(processor.tokenizer)
    transform = RLDSBatchTransform(tokenizer, processor.tokenizer,
                                   processor.image_processor.apply_transform, PurePromptBuilder)
    collator = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length,
                                               processor.tokenizer.pad_token_id)
    frames, batches = [], []
    episodes = tfds.load(key, data_dir="/home/yyz/modified_libero_rlds",
                         split="train[:8]", shuffle_files=False)
    for ep_idx, episode in enumerate(episodes):
        steps = list(episode["steps"].as_numpy_iterator())
        for row in [r for r in selected if r["episode"] == ep_idx]:
            raw = steps[row["t"]]
            image = raw["observation"]["image"]
            if not isinstance(image, np.ndarray):
                image = tf.io.decode_image(image, channels=3).numpy()
            image = dl.transforms.resize_image(tf.convert_to_tensor(image), size=(224, 224)).numpy()
            assert hashlib.sha256(image.tobytes()).hexdigest() == row["image_sha256"]
            instruction = raw["language_instruction"].decode()
            assert instruction == row["instruction"]
            target = training_target(raw["action"], stats[key]["action"])
            assert np.allclose(target, row["target"], atol=2e-6, rtol=0)
            batches.append(transform({"dataset_name": key, "action": target[None],
                                      "observation": {"image_primary": image[None]},
                                      "task": {"language_instruction": instruction.encode()}}))
            frames.append({"episode": ep_idx, "t": row["t"], "image_sha256": row["image_sha256"],
                           "instruction": instruction, "raw_target": raw["action"].tolist(), "image": image})
    assert len(frames) == 24
    fixed_batches = [collator(batches[i:i + 2]) for i in range(0, 24, 2)]
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
    record = verify_snapshot(snapshot, spec) if snapshot else None
    checkpoint = snapshot if snapshot else args.checkpoint
    base = AutoModelForVision2Seq.from_pretrained(base_path, torch_dtype=torch.bfloat16,
                quantization_config=quant, attn_implementation="sdpa", low_cpu_mem_usage=True,
                trust_remote_code=True)
    base = prepare_model_for_kbit_training(base)
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
    model.norm_stats = stats
    model.base_model.model.norm_stats = stats
    trainable = [p for n, p in model.named_parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable) == audit["lora_parameter_count"]
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(trainable, lr=spec["learning_rate"])
    start_step = record["step"] if record else 0
    if record:
        # Only load optimizer/RNG files created by this local run and SHA256-verified above.
        resume_state = torch.load(snapshot / "optimizer_rng.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(resume_state["optimizer"])
        torch.set_rng_state(resume_state["cpu_rng"])
        torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
        del resume_state
        state["latest_completed_checkpoint"] = str(snapshot)
        print(f"Resume from committed update {start_step}; uncheckpointed work is not claimed retained", flush=True)

    def forward(batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(input_ids=batch["input_ids"].cuda(), attention_mask=batch["attention_mask"].cuda(),
                         pixel_values=batch["pixel_values"].to("cuda", torch.bfloat16), labels=batch["labels"])

    def evaluate(step, production=False):
        model.eval()
        rows = []
        with torch.inference_mode():
            for idx, batch in enumerate(fixed_batches):
                output = forward(batch) if not production else model(
                    input_ids=batch["input_ids"].cuda(), attention_mask=batch["attention_mask"].cuda(),
                    pixel_values=batch["pixel_values"].to("cuda", torch.bfloat16), labels=batch["labels"].cuda())
                patches = model.vision_backbone.featurizer.patch_embed.num_patches
                predicted = output.logits[:, patches:-1].argmax(-1)
                gt = batch["labels"][:, 1:].cuda()
                for j in range(2):
                    mask = gt[j] > tokenizer.action_token_begin_idx
                    a = predicted[j][mask].cpu().numpy()
                    b = gt[j][mask].cpu().numpy()
                    assert len(b) == 7
                    frame = frames[2 * idx + j]
                    rows.append({k: v for k, v in frame.items() if k != "image"})
                    rows[-1].update(split="fit" if 2 * idx + j < 16 else "probe_previously_seen",
                                    teacher_token_accuracy=float((a == b).mean()),
                                    teacher_token_l1=float(np.abs(tokenizer.decode_token_ids_to_actions(a) -
                                                                 tokenizer.decode_token_ids_to_actions(b)).mean()))
                del output
            for i, frame in enumerate(frames):
                from contextlib import nullcontext
                context = nullcontext() if production else torch.autocast("cuda", dtype=torch.bfloat16)
                with context:
                    action = get_vla_action(model, processor, base_path, {"full_image": frame["image"]},
                                            frame["instruction"], key, center_crop=False)
                rows[i]["autoregressive"] = {"action": action.tolist(), **metrics(action, frame["raw_target"], stats[key]["action"])}
        summaries = {}
        for split in ("fit", "probe_previously_seen"):
            subset = [r for r in rows if r["split"] == split]
            summaries[split] = {"samples": len(subset),
                "teacher_token_accuracy": float(np.mean([r["teacher_token_accuracy"] for r in subset])),
                "teacher_token_l1": float(np.mean([r["teacher_token_l1"] for r in subset])),
                "autoregressive_motion_mae": float(np.mean([r["autoregressive"]["motion_mae_6d"] for r in subset])),
                "autoregressive_normalized_l1": float(np.mean([r["autoregressive"]["normalized_l1_7d"] for r in subset])),
                "gripper_correct": sum(r["autoregressive"]["gripper_correct"] for r in subset),
                "small_translation_count": sum(r["autoregressive"]["translation_norm"] < .02 for r in subset)}
        return {"step": step, "loader": "production_no_autocast" if production else "training_BF16_autocast",
                "center_crop": False, "summary": summaries, "rows": rows}

    def save(step, evaluation):
        final = args.run / f"step_{step:03d}"
        working = args.run / f"step_{step:03d}.partial"
        if final.exists() or working.exists():
            raise FileExistsError(f"Preserve existing checkpoint/partial evidence: {final}")
        if shutil.disk_usage(args.run).free < 4 * 2**30:
            raise RuntimeError("Not enough checkpoint headroom; leave existing evidence intact")
        working.mkdir()
        model.save_pretrained(working)
        processor.save_pretrained(working)
        atomic_json(working / "dataset_statistics.json", stats)
        torch.save({"optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()}, working / "optimizer_rng.pt")
        atomic_json(working / "evaluation.json", evaluation)
        protected = ("adapter_model.safetensors", "adapter_config.json", "dataset_statistics.json",
                     "optimizer_rng.pt", "evaluation.json")
        atomic_json(working / "snapshot.json", {"spec": spec, "step": step,
                    "sha256": {name: file_hash(working / name) for name in protected}})
        working.rename(final)  # Commit all files before advertising resume progress.
        state["latest_completed_checkpoint"] = str(final)
        atomic_json(args.run / "progress.json", state)
        print(f"Checkpoint committed: {step}; {json.dumps(evaluation['summary'])}", flush=True)

    if not record:
        save(0, evaluate(0))
    state["completed_updates_in_memory"] = start_step
    attempt = f"attempt-{time.time_ns()}"
    with (args.run / f"{attempt}-losses.jsonl").open("x") as log:
        for step in range(start_step + 1, spec["optimizer_updates"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for batch in fixed_batches[:8]:
                output = forward(batch)
                assert torch.isfinite(output.loss)
                losses.append(float(output.loss.detach()))
                (output.loss / 8).backward()
                del output
            grads = [p.grad for p in trainable if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state["completed_updates_in_memory"] = step
            state["last_mean_loss"] = float(np.mean(losses))
            atomic_json(args.run / "progress.json", state)
            log.write(json.dumps({"attempt": attempt, "optimizer_step": step, "losses": losses,
                                  "mean_loss": state["last_mean_loss"]}, allow_nan=False) + "\n")
            log.flush()
            print(f"update {step}/{spec['optimizer_updates']} mean loss={state['last_mean_loss']:.6f}", flush=True)
            if step in spec["milestones"]:
                save(step, evaluate(step))
    peak = torch.cuda.max_memory_allocated() / 2**30
    del optimizer, trainable, model, base
    if "grads" in locals():
        del grads
    gc.collect()
    torch.cuda.empty_cache()
    final_checkpoint = args.run / f"step_{spec['optimizer_updates']:03d}"
    cfg = SimpleNamespace(pretrained_checkpoint=str(final_checkpoint), base_model_path=base_path,
                          load_in_4bit=True, load_in_8bit=False, bnb_double_quant=False)
    model, processor = get_vla(cfg), get_processor(cfg)
    production_evaluation = evaluate(spec["optimizer_updates"], production=True)
    after = {name: file_hash(args.checkpoint / name) for name in spec["original_sha256"]}
    assert after == spec["original_sha256"]
    evaluations = [json.loads((p.parent / "evaluation.json").read_text())
                   for p in sorted(args.run.glob("step_[0-9][0-9][0-9]/snapshot.json"))]
    final_training = evaluations[-1]
    equal = [r["autoregressive"]["action"] for r in final_training["rows"]] == [r["autoregressive"]["action"] for r in production_evaluation["rows"]]
    report = {"schema_version": 1, "spec": spec, "evaluations": evaluations,
              "production_evaluation": production_evaluation, "production_actions_equal": equal,
              "original_checkpoint_modified": False, "training_peak_allocated_gib": peak,
              "final_checkpoint": str(final_checkpoint), "optimizer_updates": spec["optimizer_updates"],
              "status": "completed" if equal else "completed_loader_difference_requires_diagnosis",
              "limits": ["16-frame memorization probe, not policy improvement", "no random augmentation; no center crop",
                         "8 probe frames were seen by the original full-data adapter, not held out",
                         "lr1e-4 is a diagnostic choice, not a learning-rate ablation",
                         "resume uses completed adapter+optimizer+RNG snapshots; bitwise determinism not guaranteed",
                         "no simulator rollout or paper success-rate claim"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    state.update(status=report["status"], result=str(args.output), production_actions_equal=equal)
    atomic_json(args.run / "progress.json", state)
    print(json.dumps({"status": report["status"], "result": str(args.output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=SOURCE)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    spec = plan_spec(args.checkpoint, args.updates, args.learning_rate)
    if args.launch:
        args.run.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            print("Final report exists; verify it instead of launching again")
            return
        # Keep the launch path idempotent too; the worker also holds a lifetime lock.
        with (args.run / "worker.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("Worker already active; not launching another")
                return
        command = [sys.executable, str(Path(__file__).resolve()), "--checkpoint", str(args.checkpoint),
                   "--run", str(args.run), "--output", str(args.output), "--updates", str(args.updates),
                   "--learning-rate", str(args.learning_rate), "--execute"]
        if args.resume:
            command.append("--resume")
        with (args.run / f"worker-{time.time_ns()}.log").open("x") as handle:
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        print(json.dumps({"launched_pid": process.pid, "run": str(args.run)}))
    elif args.execute:
        worker(args, spec)
    else:
        print(json.dumps(spec, indent=2))
        print("DRY RUN: use --launch only when host GPU is idle; no long/full-suite training")


if __name__ == "__main__":
    main()
