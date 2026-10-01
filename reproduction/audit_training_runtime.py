"""One-update real-action QLoRA audit, never overwriting original weights.

Default is a CPU-only dry run. --execute requires an idle GPU and sufficient
disk space. Temporary adapter copies are cleaned even if an exception occurs.
This is an engineering verification, NOT continued training or policy improvement.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def preflight(checkpoint, temporary_root):
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
        text=True).strip()
    if occupied:
        raise RuntimeError(f"GPU compute process is active; do not preempt: {occupied}")
    required = (checkpoint / "adapter_model.safetensors").stat().st_size + 2**30 + 64 * 2**20
    available = shutil.disk_usage(temporary_root).free
    if available < required:
        raise RuntimeError(f"Temporary checkpoint plus 1GiB reserve requires {required} bytes; free={available}")
    return {"free_disk_bytes": available, "required_disk_bytes": required}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cpu-report", type=Path, default=ROOT / "reproduction/results/training_chain_cpu_audit_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction/results/training_chain_runtime_audit_v1.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    assert (args.checkpoint / "adapter_config.json").is_file()
    if not args.execute:
        print("DRY RUN: idle GPU only; real 16 fixed frames, batch2 x accumulation8, one AdamW update lr5e-4.")
        print("Original checkpoint is read-only. Save/reload in a private temporary directory; clean afterwards.")
        print("Checks both same-training-loader reload and production inference reload; no full-suite eval.")
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    conditions = preflight(args.checkpoint, Path(tempfile.gettempdir()))
    original_hashes = {name: file_hash(args.checkpoint / name) for name in
                       ("adapter_model.safetensors", "adapter_config.json", "dataset_statistics.json")}

    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds
    import dlimp as dl
    import torch
    from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets.datasets import RLDSBatchTransform
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from reproduction.action_metrics import training_target
    from experiments.robot.openvla_utils import get_vla, get_processor, get_vla_action
    from types import SimpleNamespace

    torch.manual_seed(7)
    np.random.seed(7)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    adapter_config = json.loads((args.checkpoint / "adapter_config.json").read_text())
    base_path = adapter_config["base_model_name_or_path"]
    statistics = json.loads((args.checkpoint / "dataset_statistics.json").read_text())
    key = "libero_spatial_no_noops"
    audit = json.loads(args.cpu_report.read_text())
    assert all(value == "pass" for value in audit["checks"].values())
    selected = audit["rows"][:16]
    assert len(selected) == 16
    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(action_tokenizer, processor.tokenizer,
                                       processor.image_processor.apply_transform, PurePromptBuilder)
    frames, batches = [], []
    episodes = tfds.load(key, data_dir=str(Path.home() / "modified_libero_rlds"),
                         split="train[:8]", shuffle_files=False)
    for ep_idx, episode in enumerate(episodes):
        steps = list(episode["steps"].as_numpy_iterator())
        for row in [row for row in selected if row["episode"] == ep_idx]:
            step = steps[row["t"]]
            image = step["observation"]["image"]
            if not isinstance(image, np.ndarray):
                image = tf.io.decode_image(image, channels=3).numpy()
            image = dl.transforms.resize_image(tf.convert_to_tensor(image), size=(224, 224)).numpy()
            assert hashlib.sha256(image.tobytes()).hexdigest() == row["image_sha256"]
            assert step["language_instruction"].decode() == row["instruction"]
            target = training_target(step["action"], statistics[key]["action"])
            assert np.allclose(target, row["target"], atol=2e-6, rtol=0)
            batches.append(batch_transform({"dataset_name": key, "action": target[None],
                           "observation": {"image_primary": image[None]},
                           "task": {"language_instruction": step["language_instruction"]}}))
            frames.append({"image": image, "instruction": row["instruction"]})
    assert len(batches) == 16
    collator = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length,
                                               processor.tokenizer.pad_token_id)
    microbatches = [collator(batches[i:i + 2]) for i in range(0, 16, 2)]
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)

    def training_loader(path):
        base = AutoModelForVision2Seq.from_pretrained(base_path, torch_dtype=torch.bfloat16,
                quantization_config=quantization, attn_implementation="sdpa",
                low_cpu_mem_usage=True, trust_remote_code=True)
        base = prepare_model_for_kbit_training(base)
        model = PeftModel.from_pretrained(base, path, is_trainable=True)
        model.norm_stats = statistics
        model.base_model.model.norm_stats = statistics
        return model

    def prediction(model, chosen_processor):
        model.eval()
        with torch.inference_mode():
            return [get_vla_action(model, chosen_processor, base_path,
                                   {"full_image": frames[i]["image"]}, frames[i]["instruction"], key,
                                   center_crop=False).tolist() for i in (0, 1, 2)]

    model = training_loader(args.checkpoint)
    trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
    assert trainable and all("lora_" in name for name in trainable)
    assert sum(param.numel() for param in trainable.values()) == audit["lora_parameter_count"]
    before = {name: param.detach().cpu().clone() for name, param in trainable.items()}
    before_action = prediction(model, processor)
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=5e-4)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    losses = []
    for i, batch in enumerate(microbatches):
        assert torch.all((batch["labels"] > action_tokenizer.action_token_begin_idx).sum(dim=1) == 7)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids=batch["input_ids"].cuda(), attention_mask=batch["attention_mask"].cuda(),
                           pixel_values=batch["pixel_values"].to("cuda", torch.bfloat16),
                           labels=batch["labels"])
        loss = output.loss
        assert torch.isfinite(loss), "Non-finite real-action loss"
        losses.append(float(loss.detach()))
        (loss / 8).backward()
        del output, loss
        print(f"Real-action microbatch {i+1}/8 completed; optimizer not yet stepped", flush=True)
    gradients = []
    for name, param in trainable.items():
        grad = param.grad
        if grad is not None:
            assert torch.isfinite(grad).all(), name
        gradients.append({"name": name, "grad_present": grad is not None,
                          "grad_nonzero": bool(grad.count_nonzero()) if grad is not None else False,
                          "grad_norm": float(grad.float().norm()) if grad is not None else None})
    assert any(row["grad_nonzero"] for row in gradients)
    assert all(param.grad is None for param in model.parameters() if not param.requires_grad)
    optimizer.step()  # The only optimizer update in this audit.
    optimizer.zero_grad(set_to_none=True)
    changes = []
    for name, param in trainable.items():
        value = param.detach().cpu()
        assert torch.isfinite(value).all(), name
        changes.append({"name": name, "max_abs_change": float((value - before[name]).abs().max())})
    assert any(row["max_abs_change"] > 0 for row in changes)
    del before, optimizer, trainable
    gc.collect()
    after_action = prediction(model, processor)
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    # Save with the same default PEFT behavior as finetune.py, including auto-saved lm_head.
    with tempfile.TemporaryDirectory(prefix="openvla-runtime-audit-") as directory:
        temporary_checkpoint = Path(directory)
        model.save_pretrained(temporary_checkpoint)
        processor.save_pretrained(temporary_checkpoint)
        (temporary_checkpoint / "dataset_statistics.json").write_text(json.dumps(statistics))
        saved_bytes = sum(path.stat().st_size for path in temporary_checkpoint.iterdir() if path.is_file())
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = training_loader(temporary_checkpoint)
        same_loader_action = prediction(model, processor)
        same_loader_equal = bool(np.array_equal(after_action, same_loader_action))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        cfg = SimpleNamespace(pretrained_checkpoint=str(temporary_checkpoint), base_model_path=base_path,
                              load_in_4bit=True, load_in_8bit=False, bnb_double_quant=False)
        model = get_vla(cfg)
        production_action = prediction(model, get_processor(cfg))
        production_equal = bool(np.array_equal(after_action, production_action))
        production_max_difference = float(np.abs(np.asarray(after_action) - production_action).max())
        del model
        gc.collect()
        torch.cuda.empty_cache()
    assert not temporary_checkpoint.exists(), "Temporary checkpoint must be cleaned"
    assert all(file_hash(args.checkpoint / name) == digest for name, digest in original_hashes.items())
    # Do not mask production dtype/loader differences behind an unconditional PASS.
    report = {"schema_version": 1, "scope": "one real-action optimizer update; no policy improvement claim",
              "checkpoint": str(args.checkpoint), "original_checkpoint_modified": False,
              "original_checkpoint_sha256": original_hashes,
              "conditions": conditions, "samples": 16, "batch_size": 2, "accumulation_steps": 8,
              "optimizer_updates": 1, "learning_rate": 5e-4, "new_optimizer_not_exact_resume": True,
              "random_augmentation": False, "center_crop_for_prediction": False,
              "quantization": {"nf4": True, "double_quant": False, "compute": "bfloat16"},
              "losses": losses, "gradients": gradients, "parameter_changes": changes,
              "gradient_tensors_nonzero": sum(row["grad_nonzero"] for row in gradients),
              "changed_parameter_tensors": sum(row["max_abs_change"] > 0 for row in changes),
              "prediction_frame_indices": [0, 1, 2],
              "before_update_actions": before_action, "after_update_actions": after_action,
              "same_training_loader_reloaded_actions": same_loader_action,
              "production_loader_reloaded_actions": production_action,
              "same_training_loader_prediction_equal": same_loader_equal,
              "production_prediction_equal": production_equal,
              "production_max_abs_action_difference": production_max_difference,
              "saved_temporary_bytes": saved_bytes, "temporary_checkpoint_cleaned": True,
              "peak_allocated_gib_training_phase": peak_gib,
              "status": "pass" if same_loader_equal and production_equal else "reload_difference_requires_diagnosis"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k not in {"gradients", "parameter_changes"}}, indent=2))


if __name__ == "__main__":
    main()
