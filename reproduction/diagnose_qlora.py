"""Bounded, deterministic diagnostic using production inference + training labels.

Example: python reproduction/diagnose_qlora.py --checkpoint PATH --output PATH
No training or simulator rollouts. Full-data adapter samples are NOT held out.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--trained-episodes", type=int, default=432)
    parser.add_argument("--double-quant", action="store_true")
    args = parser.parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic evidence: {output_path}")

    import numpy as np
    import tensorflow as tf
    # TF is used for data/image operations only; never compete with PyTorch for VRAM.
    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds
    import torch
    import dlimp as dl
    from PIL import Image
    from experiments.robot.openvla_utils import get_vla, get_processor, get_vla_action, crop_and_resize
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets.datasets import RLDSBatchTransform
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from reproduction.action_metrics import metrics, training_target

    torch.manual_seed(7)
    np.random.seed(7)
    cfg = SimpleNamespace(pretrained_checkpoint=args.checkpoint, base_model_path="openvla/openvla-7b",
                          load_in_4bit=True, load_in_8bit=False, bnb_double_quant=args.double_quant)
    model, processor = get_vla(cfg), get_processor(cfg)
    has_adapter = hasattr(model, "disable_adapter")
    key = "libero_spatial_no_noops" if "libero_spatial_no_noops" in model.norm_stats else "libero_spatial"
    stats = model.norm_stats[key]["action"]
    tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(tokenizer, processor.tokenizer,
                                       processor.image_processor.apply_transform, PurePromptBuilder)
    split = f"train[{args.episode_start}:{args.episode_start + args.episodes}]"
    dataset = tfds.load("libero_spatial_no_noops", data_dir=str(Path.home() / "modified_libero_rlds"),
                        split=split, shuffle_files=False)
    rows = []
    started = time.monotonic()
    with torch.inference_mode():
        for episode_index, episode in enumerate(dataset):
            steps = list(episode["steps"].as_numpy_iterator())
            # Cover start, middle and end; omit terminal action (often placeholder).
            indices = sorted(set([0, (len(steps) - 2) // 2, len(steps) - 2]))
            for t in indices:
                step = steps[t]
                image = step["observation"]["image"]
                if not isinstance(image, np.ndarray):
                    image = tf.io.decode_image(image, channels=3).numpy()
                image = dl.transforms.resize_image(tf.convert_to_tensor(image), size=(224, 224)).numpy()
                instruction = step["language_instruction"].decode().lower()
                raw = step["action"].astype(float)
                target = training_target(raw, stats)
                row = {"episode": args.episode_start + episode_index, "t": t, "length": len(steps),
                       "instruction": instruction, "raw_target": raw.tolist(),
                       "training_target": target.tolist(), "image_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
                variants = [("adapter", nullcontext(), True), ("adapter_no_crop", nullcontext(), False)]
                if has_adapter:
                    variants.insert(0, ("base", model.disable_adapter(), True))
                for name, context, crop in variants:
                    with context:
                        action = get_vla_action(model, processor, cfg.base_model_path,
                                                {"full_image": image}, instruction, key, center_crop=crop)
                    row[name] = {"action": action.tolist(), **metrics(action, raw, stats)}
                # Teacher forcing: same cropped pixels but true previous action tokens.
                cropped = crop_and_resize(tf.image.convert_image_dtype(image, tf.float32), .9, 1)
                cropped = tf.image.convert_image_dtype(tf.clip_by_value(cropped, 0, 1), tf.uint8).numpy()
                batch = batch_transform({"dataset_name": key, "action": target[None],
                                         "observation": {"image_primary": cropped[None]},
                                         "task": {"language_instruction": instruction.encode()}})
                labels = batch["labels"][None].cuda()
                result = model(input_ids=batch["input_ids"][None].cuda(),
                               attention_mask=torch.ones_like(labels, dtype=torch.bool),
                               pixel_values=batch["pixel_values"][None].to("cuda", torch.bfloat16),
                               labels=labels)
                patches = model.vision_backbone.featurizer.patch_embed.num_patches
                logits = result.logits[:, patches:-1]
                gt = labels[:, 1:]
                mask = gt > tokenizer.action_token_begin_idx
                tokens = logits.argmax(-1)[mask].cpu().numpy()
                gt_tokens = gt[mask].cpu().numpy()
                assert len(gt_tokens) == 7, "Training prompt must supervise exactly seven action tokens"
                row["teacher_forced"] = {"loss": float(result.loss),
                    "token_accuracy": float((tokens == gt_tokens).mean()),
                    "normalized_l1": float(np.abs(tokenizer.decode_token_ids_to_actions(tokens) -
                                                  tokenizer.decode_token_ids_to_actions(gt_tokens)).mean())}
                rows.append(row)
                print(f"ep={row['episode']} t={t} adapter_normL1={row['adapter']['normalized_l1_7d']:.4f} "
                      f"teacher_acc={row['teacher_forced']['token_accuracy']:.3f}", flush=True)

    summary = {}
    for name in rows[0].keys() & {"base", "adapter", "adapter_no_crop"}:
        summary[name] = {metric: float(np.mean([r[name][metric] for r in rows]))
                         for metric in ("motion_mae_6d", "simulator_mae_7d", "normalized_l1_7d",
                                        "gripper_correct", "translation_norm")}
        summary[name]["normalized_mae_per_dim"] = np.mean([r[name]["normalized_abs_error"] for r in rows], axis=0).tolist()
    summary["teacher_forced"] = {metric: float(np.mean([r["teacher_forced"][metric] for r in rows]))
                                  for metric in ("loss", "token_accuracy", "normalized_l1")}
    report = {"schema_version": 1, "checkpoint": args.checkpoint, "split": split,
              "model_type": "local_peft_adapter" if has_adapter else "official_full_checkpoint",
              "held_out_from_training": args.episode_start >= args.trained_episodes,
              "sampling": "three fixed stages per episode, excluding terminal action; not representative success rate",
              "quantization": {"nf4": True, "double_quant": args.double_quant, "compute": "bfloat16"},
              "gpu": torch.cuda.get_device_name(0), "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
              "seconds": time.monotonic()-started, "samples": len(rows), "summary": summary, "rows": rows}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"samples": len(rows), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
