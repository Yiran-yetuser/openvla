"""CPU-only real-data audit. Never loads GPU weights or modifies checkpoints.

Checks the production RLDS trajectory/frame transforms against raw TFDS steps,
then checks real action supervision, collator padding and inference prompt/pixels.
GPU gradient/update/prediction round-trip checks are explicitly not claimed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert args.episodes > 0
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OPENVLA_TRAIN_EPISODES"] = str(args.episodes)
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds
    import torch
    import dlimp as dl
    from PIL import Image
    from transformers import AutoProcessor
    from safetensors import safe_open
    from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs
    from prismatic.vla.datasets.rlds.dataset import (
        make_dataset_from_rlds, apply_trajectory_transforms, apply_frame_transforms,
    )
    from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
    from prismatic.vla.datasets.datasets import RLDSBatchTransform
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from reproduction.action_metrics import training_target

    torch.set_num_threads(2)
    tf.random.set_seed(7)
    np.random.seed(7)
    name = "libero_spatial_no_noops"
    data_root = Path.home() / "modified_libero_rlds"
    statistics = json.loads((args.checkpoint / "dataset_statistics.json").read_text())
    config = json.loads((args.checkpoint / "adapter_config.json").read_text())
    stats = statistics[name]["action"]
    processor = AutoProcessor.from_pretrained(config["base_model_name_or_path"], trust_remote_code=True)
    tokenizer = ActionTokenizer(processor.tokenizer)
    transform = RLDSBatchTransform(tokenizer, processor.tokenizer,
                                   processor.image_processor.apply_transform, PurePromptBuilder)
    kwargs = make_oxe_dataset_kwargs(name, data_root, load_proprio=False,
                                    action_proprio_normalization_type=NormalizationType.BOUNDS_Q99)
    pipeline, _ = make_dataset_from_rlds(**kwargs, train=True, shuffle=False,
                                        dataset_statistics=statistics[name],
                                        num_parallel_reads=1, num_parallel_calls=1)
    pipeline = apply_trajectory_transforms(pipeline, train=True, window_size=1,
                                          future_action_window_size=0, skip_unlabeled=True,
                                          goal_relabeling_strategy="uniform", num_parallel_calls=1)
    # Same decoder/resize as training. Disable random augmentation for alignment;
    # this does not verify every random crop/color sample from historical training.
    pipeline = apply_frame_transforms(pipeline, train=True, resize_size=(224, 224),
                                     image_augment_kwargs={}, num_parallel_calls=1)
    raw_dataset = tfds.load(name, data_dir=str(data_root),
                           split=f"train[:{args.episodes}]", shuffle_files=False)
    transformed_iterator = iter(pipeline.as_numpy_iterator())
    counts = dict(episodes=0, paired_frames=0, supervised_frames=0,
                  supervised_action_tokens=0, supervised_eos_tokens=0,
                  terminal_frames=0, raw_small_translation_frames=0)
    maximum_normalization_difference = 0.0
    maximum_token_decode_error = 0.0
    sampled_examples, rows = [], []
    for episode_idx, episode in enumerate(raw_dataset):
        steps = list(episode["steps"].as_numpy_iterator())
        trajectory = next(transformed_iterator)
        assert len(steps) == len(trajectory["action"])
        for t, step in enumerate(steps):
            raw = np.asarray(step["action"], dtype=float)
            expected = training_target(raw, stats)
            actual = trajectory["action"][t, 0]
            diff = float(np.abs(expected - actual).max())
            maximum_normalization_difference = max(maximum_normalization_difference, diff)
            assert np.allclose(actual, expected, atol=2e-6, rtol=0)
            assert trajectory["observation"]["timestep"][t, 0] == t
            instruction = step["language_instruction"]
            assert trajectory["task"]["language_instruction"][t] == instruction
            assert instruction.strip()
            raw_image = step["observation"]["image"]
            if not isinstance(raw_image, np.ndarray):
                raw_image = tf.io.decode_image(raw_image, channels=3).numpy()
            image = dl.transforms.resize_image(tf.convert_to_tensor(raw_image), size=(224, 224)).numpy()
            assert np.array_equal(image, trajectory["observation"]["image_primary"][t, 0])
            counts["paired_frames"] += 1
            counts["terminal_frames"] += int(step["is_last"])
            counts["raw_small_translation_frames"] += int(np.linalg.norm(raw[:3]) < .02)

            # Beginning/middle/penultimate: exactly the prior 24 diagnostic frames.
            if t not in {0, (len(steps) - 2) // 2, len(steps) - 2}:
                continue
            frame = tf.nest.map_structure(lambda value: value[t], trajectory)
            batch = transform(frame)
            labels = batch["labels"].numpy()
            supervised = labels[labels != -100]
            action_ids = labels[labels > tokenizer.action_token_begin_idx]
            direct_ids = processor.tokenizer.vocab_size - np.digitize(expected, tokenizer.bins)
            assert len(supervised) == 8 and len(action_ids) == 7
            assert supervised[-1] == processor.tokenizer.eos_token_id
            assert np.array_equal(action_ids, direct_ids)
            assert np.all(labels[:len(labels) - 8] == -100)
            decode_error = float(np.abs(tokenizer.decode_token_ids_to_actions(action_ids) - expected).max())
            maximum_token_decode_error = max(maximum_token_decode_error, decode_error)
            assert decode_error <= 2 / 255 + 2e-6
            prompt = f"In: What action should the robot take to {instruction.decode().lower()}?\nOut:"
            inference = processor(prompt, Image.fromarray(image), return_tensors="pt")
            inference_ids = inference["input_ids"][0].tolist()
            # Production predict_action appends this empty token before generation.
            if inference_ids[-1] != 29871:
                inference_ids.append(29871)
            assert inference_ids == batch["input_ids"][:-8].tolist()
            assert torch.equal(inference["pixel_values"][0], batch["pixel_values"])
            counts["supervised_frames"] += 1
            counts["supervised_action_tokens"] += 7
            counts["supervised_eos_tokens"] += 1
            sampled_examples.append(batch)
            rows.append({"episode": episode_idx, "t": t, "instruction": instruction.decode(),
                         "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                         "target": actual.tolist(), "action_token_ids": action_ids.tolist(),
                         "sequence_length": len(labels)})
        counts["episodes"] += 1
        print(f"episode={episode_idx} checked {len(steps)} paired frames", flush=True)
    assert next(transformed_iterator, None) is None
    assert counts["episodes"] == args.episodes
    collator = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length,
                                               processor.tokenizer.pad_token_id)
    collated = collator(sampled_examples)
    assert collated["pixel_values"].shape[0] == len(sampled_examples)
    for i, sample in enumerate(sampled_examples):
        n = len(sample["labels"])
        assert torch.equal(collated["labels"][i, :n], sample["labels"])
        assert torch.all(collated["labels"][i, n:] == -100)
        assert torch.all(collated["attention_mask"][i, :n])
        assert not torch.any(collated["attention_mask"][i, n:])
    # Stored tensor evidence, not proof of runtime gradients or optimizer updates.
    tensors = []
    weights = args.checkpoint / "adapter_model.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as handle:
        for key in sorted(handle.keys()):
            value = handle.get_tensor(key)
            assert torch.isfinite(value).all()
            tensors.append({"key": key, "shape": list(value.shape),
                            "dtype": str(value.dtype), "nonzero": bool(value.count_nonzero()),
                            "numel": value.numel()})
    assert tensors
    lora_tensors = [item for item in tensors if "lora_" in item["key"]]
    extra_tensors = [item for item in tensors if "lora_" not in item["key"]]
    assert lora_tensors
    # PEFT auto-saves the full output layer when lm_head is targeted. It is not
    # an unexpected trainable base parameter merely because it is serialized.
    assert all(item["key"] == "base_model.model.language_model.lm_head.base_layer.weight"
               for item in extra_tensors), extra_tensors
    base_dir = Path.home() / ".cache/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f"
    weight_map = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]
    head_key = "language_model.lm_head.weight"
    head_equal = None
    if extra_tensors:
        with safe_open(base_dir / weight_map[head_key], framework="pt", device="cpu") as base_handle:
            base_head = base_handle.get_tensor(head_key).float()
        with safe_open(weights, framework="pt", device="cpu") as adapter_handle:
            saved_head = adapter_handle.get_tensor(extra_tensors[0]["key"])
        head_equal = torch.equal(base_head, saved_head)
        del base_head, saved_head
    report = {"schema_version": 1, "scope": "CPU bounded current-pipeline audit; not original HDF5 verification",
              "checkpoint": str(args.checkpoint), "split": f"train[:{args.episodes}]", "counts": counts,
              "max_normalization_abs_difference": maximum_normalization_difference,
              "max_token_decode_abs_error": maximum_token_decode_error,
              "checks": {"raw_to_production_frame_alignment": "pass", "bounds_q99_and_gripper": "pass",
                         "seven_action_tokens_plus_eos": "pass", "collator_padding": "pass",
                         "training_vs_inference_prompt_prefix": "pass", "same_image_pixel_transform": "pass",
                         "checkpoint_tensors_finite": "pass"},
              "checkpoint_tensor_count": len(tensors), "checkpoint_parameter_count": sum(x["numel"] for x in tensors),
              "lora_parameter_count": sum(x["numel"] for x in lora_tensors),
              "extra_saved_tensor_count": len(extra_tensors), "saved_lm_head_equals_base_fp32": head_equal,
              "checkpoint_bytes": weights.stat().st_size, "saved_config_inference_mode": config["inference_mode"],
              "checkpoint_tensors": tensors, "rows": rows,
              "pending": ["runtime_LoRA_gradients", "actual_optimizer_update", "save_reload_prediction_equivalence",
                          "original_HDF5_image_action_alignment", "historical_random_augmentation_samples"],
              "notes": ["Saved inference_mode=True is normal for exported PEFT adapters; training load uses is_trainable=True.",
                        "Nonzero saved tensors alone do not establish a correct historical optimizer trajectory.",
                        "Terminal frames are included by the current loader; this count alone is not proof of a bug."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"counts": counts, "checks": report["checks"],
                      "checkpoint_parameters": report["checkpoint_parameter_count"],
                      "pending": report["pending"]}, indent=2))


if __name__ == "__main__":
    main()
