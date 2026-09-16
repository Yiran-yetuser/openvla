import os
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = Path.home() / "modified_libero_rlds"
DATASET_NAME = "libero_spatial_no_noops"

ORIGINAL_MODEL = "openvla/openvla-7b"

FINETUNED_MODEL = (
    "runs/libero_spatial_100step/"
    "openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+"
    "lora-r32+dropout-0.0+q-4bit--image_aug"
)

OUTPUT_DIR = Path("libero_compare_100samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 100
SEED = 42

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 80)
print("OpenVLA LIBERO Spatial — 100 Sample Evaluation")
print("=" * 80)
print("Device:", DEVICE)
print("Number of samples:", NUM_SAMPLES)
print("Dataset:", DATA_ROOT / DATASET_NAME)
print()

# ============================================================
# TensorFlow
# ============================================================

import tensorflow as tf

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

print("TensorFlow:", tf.__version__)

import tensorflow_datasets as tfds

# ============================================================
# Dataset
# ============================================================

dataset_dir = DATA_ROOT / DATASET_NAME / "1.0.0"

if not dataset_dir.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{dataset_dir}"
    )

print("Dataset directory:", dataset_dir)

builder = tfds.builder(
    DATASET_NAME,
    data_dir=str(DATA_ROOT),
)

print("TFDS builder:", builder.name)
print("Splits:", builder.info.splits)
print()

# ============================================================
# Extract candidate samples
# ============================================================

print("=" * 80)
print("Extracting real LIBERO samples")
print("=" * 80)

ds = builder.as_dataset(
    split="train",
    shuffle_files=False,
)

candidates = []

for episode_idx, episode in enumerate(ds):

    steps = episode["steps"]

    for step_idx, step in enumerate(steps):

        observation = step["observation"]

        # ----------------------------
        # image
        # ----------------------------

        image = None

        if "image" in observation:
            image = observation["image"]

        elif "image_primary" in observation:
            image = observation["image_primary"]

        if image is None:
            continue

        image = image.numpy()

        # ----------------------------
        # language
        # ----------------------------

        language = None

        if "language_instruction" in step:
            language = step["language_instruction"]

        elif "task" in step:
            task = step["task"]

            if "language_instruction" in task:
                language = task["language_instruction"]

        if language is None:
            continue

        language = language.numpy()

        if isinstance(language, bytes):
            language = language.decode("utf-8")
        elif hasattr(language, "item"):
            language = language.item()
            if isinstance(language, bytes):
                language = language.decode("utf-8")

        language = str(language)

        # ----------------------------
        # action
        # ----------------------------

        action = step["action"].numpy().astype(np.float32)
        action = np.asarray(action).reshape(-1)

        if action.shape[0] != 7:
            continue

        # ----------------------------
        # image validation
        # ----------------------------

        if image.ndim != 3:
            continue

        if image.shape[-1] != 3:
            continue

        candidates.append({
            "episode_idx": int(episode_idx),
            "step_idx": int(step_idx),
            "image": image,
            "language": language,
            "action": action,
        })

        if len(candidates) >= NUM_SAMPLES:
            break

    if len(candidates) >= NUM_SAMPLES:
        break

print("Collected samples:", len(candidates))

if len(candidates) < NUM_SAMPLES:
    raise RuntimeError(
        f"Only found {len(candidates)} valid samples."
    )

print()

# ============================================================
# Deterministic selection
# ============================================================

rng = random.Random(SEED)

rng.shuffle(candidates)

samples = candidates[:NUM_SAMPLES]

print("Selected", len(samples), "samples.")
print()

# ============================================================
# Save sample images
# ============================================================

sample_dir = OUTPUT_DIR / "samples"
sample_dir.mkdir(exist_ok=True)

for i, sample in enumerate(samples):

    image_np = sample["image"]

    if image_np.dtype != np.uint8:
        image_np = np.clip(
            image_np,
            0,
            255,
        ).astype(np.uint8)

    Image.fromarray(image_np).save(
        sample_dir / f"sample_{i:03d}.png"
    )

# ============================================================
# Action tokenizer
# ============================================================

class SimpleActionTokenizer:

    def __init__(
        self,
        tokenizer,
        bins=256,
        min_action=-1.0,
        max_action=1.0,
    ):

        self.tokenizer = tokenizer
        self.bins = bins

        self.bin_edges = np.linspace(
            min_action,
            max_action,
            bins + 1,
        )

        self.bin_centers = (
            self.bin_edges[:-1]
            + self.bin_edges[1:]
        ) / 2

    def decode(self, token_ids):

        actions = []

        for token_id in token_ids:

            token_id = int(token_id)

            discretized_action = (
                self.tokenizer.vocab_size
                - token_id
                - 1
            )

            discretized_action = int(
                np.clip(
                    discretized_action,
                    0,
                    self.bins - 1,
                )
            )

            actions.append(
                self.bin_centers[
                    discretized_action
                ]
            )

        return np.asarray(
            actions,
            dtype=np.float32,
        )

# ============================================================
# Load model helper
# ============================================================

from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def load_model(path, name):

    print("=" * 80)
    print("Loading", name)
    print("=" * 80)

    processor = AutoProcessor.from_pretrained(
        path,
        trust_remote_code=True,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    model.eval()

    print(
        "Model:",
        type(model).__name__,
    )

    print(
        "VRAM allocated:",
        f"{torch.cuda.memory_allocated()/1024**3:.2f} GB",
    )

    print()

    return processor, model


# ============================================================
# Prediction helper
# ============================================================

@torch.inference_mode()
def predict_action(
    model,
    processor,
    image,
    language,
):

    prompt = (
        "What action should the robot take to "
        + language
        + "?"
    )

    inputs = processor(
        prompt,
        image,
        return_tensors="pt",
    )

    for key in inputs:

        if not torch.is_tensor(inputs[key]):
            continue

        if inputs[key].dtype.is_floating_point:

            inputs[key] = inputs[key].to(
                DEVICE,
                dtype=torch.bfloat16,
            )

        else:

            inputs[key] = inputs[key].to(
                DEVICE
            )

    generated = model.generate(
        **inputs,
        max_new_tokens=7,
        do_sample=False,
    )

    input_len = inputs["input_ids"].shape[1]

    action_token_ids = generated[
        0,
        input_len:
    ].detach().cpu().numpy()

    action_token_ids = action_token_ids[:7]

    if len(action_token_ids) != 7:

        raise RuntimeError(
            "Expected 7 action tokens, "
            f"got {len(action_token_ids)}"
        )

    tokenizer = SimpleActionTokenizer(
        processor.tokenizer
    )

    action = tokenizer.decode(
        action_token_ids
    )

    return action


# ============================================================
# Run evaluation
# ============================================================

all_results = []

# ============================================================
# ORIGINAL
# ============================================================

original_processor, original_model = load_model(
    ORIGINAL_MODEL,
    "ORIGINAL OpenVLA-7B",
)

print("=" * 80)
print("Evaluating ORIGINAL model")
print("=" * 80)

original_predictions = []

for i, sample in enumerate(samples):

    if i % 10 == 0:
        print(
            f"Original: {i}/{NUM_SAMPLES}"
        )

    image_np = sample["image"]

    if image_np.dtype != np.uint8:
        image_np = np.clip(
            image_np,
            0,
            255,
        ).astype(np.uint8)

    image = Image.fromarray(image_np)

    action = predict_action(
        original_model,
        original_processor,
        image,
        sample["language"],
    )

    original_predictions.append(action)

    print(
        f"\rOriginal: {i+1}/{NUM_SAMPLES}",
        end="",
        flush=True,
    )

print()
print()

# ============================================================
# Free ORIGINAL
# ============================================================

del original_model
del original_processor

torch.cuda.empty_cache()

print(
    "Original model unloaded."
)

print(
    "VRAM:",
    f"{torch.cuda.memory_allocated()/1024**3:.2f} GB",
)

print()

# ============================================================
# FINE-TUNED
# ============================================================

finetuned_processor, finetuned_model = load_model(
    FINETUNED_MODEL,
    "100-STEP FINE-TUNED OpenVLA",
)

print("=" * 80)
print("Evaluating 100-step FINE-TUNED model")
print("=" * 80)

finetuned_predictions = []

for i, sample in enumerate(samples):

    if i % 10 == 0:
        print(
            f"Fine-tuned: {i}/{NUM_SAMPLES}"
        )

    image_np = sample["image"]

    if image_np.dtype != np.uint8:
        image_np = np.clip(
            image_np,
            0,
            255,
        ).astype(np.uint8)

    image = Image.fromarray(image_np)

    action = predict_action(
        finetuned_model,
        finetuned_processor,
        image,
        sample["language"],
    )

    finetuned_predictions.append(action)

    print(
        f"\rFine-tuned: {i+1}/{NUM_SAMPLES}",
        end="",
        flush=True,
    )

print()
print()

# ============================================================
# Statistics
# ============================================================

print("=" * 80)
print("Computing statistics")
print("=" * 80)

original_maes = []
finetuned_maes = []

original_mses = []
finetuned_mses = []

original_dim_errors = []
finetuned_dim_errors = []

gt_actions = []
original_actions = []
finetuned_actions = []

for i, sample in enumerate(samples):

    gt = sample["action"].astype(np.float64)

    original = (
        original_predictions[i]
        .astype(np.float64)
    )

    finetuned = (
        finetuned_predictions[i]
        .astype(np.float64)
    )

    gt_actions.append(gt)
    original_actions.append(original)
    finetuned_actions.append(finetuned)

    orig_error = np.abs(original - gt)
    ft_error = np.abs(finetuned - gt)

    original_dim_errors.append(orig_error)
    finetuned_dim_errors.append(ft_error)

    original_maes.append(
        np.mean(orig_error)
    )

    finetuned_maes.append(
        np.mean(ft_error)
    )

    original_mses.append(
        np.mean((original - gt) ** 2)
    )

    finetuned_mses.append(
        np.mean((finetuned - gt) ** 2)
    )

    all_results.append({
        "sample": i,
        "episode": sample["episode_idx"],
        "step": sample["step_idx"],
        "language": sample["language"],
        "gt": gt.tolist(),
        "original": original.tolist(),
        "finetuned": finetuned.tolist(),
        "original_mae": float(
            np.mean(orig_error)
        ),
        "finetuned_mae": float(
            np.mean(ft_error)
        ),
        "original_mse": float(
            np.mean((original - gt) ** 2)
        ),
        "finetuned_mse": float(
            np.mean((finetuned - gt) ** 2)
        ),
    })

# ============================================================
# Arrays
# ============================================================

gt_actions = np.asarray(gt_actions)
original_actions = np.asarray(original_actions)
finetuned_actions = np.asarray(finetuned_actions)

original_dim_errors = np.asarray(
    original_dim_errors
)

finetuned_dim_errors = np.asarray(
    finetuned_dim_errors
)

# ============================================================
# Aggregate statistics
# ============================================================

mean_original_mae = np.mean(original_maes)
mean_finetuned_mae = np.mean(finetuned_maes)

median_original_mae = np.median(original_maes)
median_finetuned_mae = np.median(finetuned_maes)

mean_original_mse = np.mean(original_mses)
mean_finetuned_mse = np.mean(finetuned_mses)

dim_original_mae = np.mean(
    original_dim_errors,
    axis=0,
)

dim_finetuned_mae = np.mean(
    finetuned_dim_errors,
    axis=0,
)

# ============================================================
# Gripper statistics
#
# Dimension 6 is the gripper action.
# ============================================================

gt_gripper = gt_actions[:, 6]
orig_gripper = original_actions[:, 6]
ft_gripper = finetuned_actions[:, 6]

# The tokenizer produces values around +/- 1.
# Treat >= 0 as positive and < 0 as negative.

gt_sign = gt_gripper >= 0
orig_sign = orig_gripper >= 0
ft_sign = ft_gripper >= 0

original_gripper_accuracy = np.mean(
    orig_sign == gt_sign
)

finetuned_gripper_accuracy = np.mean(
    ft_sign == gt_sign
)

# ============================================================
# Improvement
# ============================================================

if mean_original_mae != 0:

    mae_improvement = (
        (mean_original_mae - mean_finetuned_mae)
        / mean_original_mae
        * 100
    )

else:
    mae_improvement = 0.0

if mean_original_mse != 0:

    mse_improvement = (
        (mean_original_mse - mean_finetuned_mse)
        / mean_original_mse
        * 100
    )

else:
    mse_improvement = 0.0

# ============================================================
# Print final results
# ============================================================

print()
print("=" * 80)
print("FINAL 100-SAMPLE RESULTS")
print("=" * 80)

print()
print("Overall:")
print(
    f"Original mean MAE   : "
    f"{mean_original_mae:.6f}"
)

print(
    f"Fine-tuned mean MAE : "
    f"{mean_finetuned_mae:.6f}"
)

print(
    f"MAE improvement     : "
    f"{mae_improvement:+.2f}%"
)

print()

print(
    f"Original median MAE   : "
    f"{median_original_mae:.6f}"
)

print(
    f"Fine-tuned median MAE : "
    f"{median_finetuned_mae:.6f}"
)

print()

print(
    f"Original mean MSE   : "
    f"{mean_original_mse:.6f}"
)

print(
    f"Fine-tuned mean MSE : "
    f"{mean_finetuned_mse:.6f}"
)

print(
    f"MSE improvement     : "
    f"{mse_improvement:+.2f}%"
)

print()

print("=" * 80)
print("PER-DIMENSION MAE")
print("=" * 80)

print(
    f"{'Dim':<8}"
    f"{'Original':>16}"
    f"{'Fine-tuned':>16}"
    f"{'Change':>16}"
)

print("-" * 56)

for d in range(7):

    if dim_original_mae[d] != 0:

        change = (
            (dim_original_mae[d]
             - dim_finetuned_mae[d])
            / dim_original_mae[d]
            * 100
        )

    else:
        change = 0.0

    print(
        f"{d:<8}"
        f"{dim_original_mae[d]:>16.6f}"
        f"{dim_finetuned_mae[d]:>16.6f}"
        f"{change:>15.2f}%"
    )

print()

print("=" * 80)
print("GRIPPER")
print("=" * 80)

print(
    f"Original gripper accuracy   : "
    f"{original_gripper_accuracy * 100:.2f}%"
)

print(
    f"Fine-tuned gripper accuracy : "
    f"{finetuned_gripper_accuracy * 100:.2f}%"
)

print()

print("=" * 80)
print("WIN / LOSS")
print("=" * 80)

wins = np.sum(
    np.asarray(finetuned_maes)
    < np.asarray(original_maes)
)

losses = np.sum(
    np.asarray(finetuned_maes)
    > np.asarray(original_maes)
)

ties = NUM_SAMPLES - wins - losses

print(
    f"Fine-tuned better : {wins}/{NUM_SAMPLES}"
)

print(
    f"Original better   : {losses}/{NUM_SAMPLES}"
)

print(
    f"Tie               : {ties}/{NUM_SAMPLES}"
)

print()

# ============================================================
# Save JSON
# ============================================================

summary = {

    "num_samples": NUM_SAMPLES,

    "seed": SEED,

    "dataset": DATASET_NAME,

    "original_model": ORIGINAL_MODEL,

    "finetuned_model": FINETUNED_MODEL,

    "original_mean_mae":
        float(mean_original_mae),

    "finetuned_mean_mae":
        float(mean_finetuned_mae),

    "original_median_mae":
        float(median_original_mae),

    "finetuned_median_mae":
        float(median_finetuned_mae),

    "original_mean_mse":
        float(mean_original_mse),

    "finetuned_mean_mse":
        float(mean_finetuned_mse),

    "mae_improvement_percent":
        float(mae_improvement),

    "mse_improvement_percent":
        float(mse_improvement),

    "original_dimension_mae":
        dim_original_mae.tolist(),

    "finetuned_dimension_mae":
        dim_finetuned_mae.tolist(),

    "original_gripper_accuracy":
        float(original_gripper_accuracy),

    "finetuned_gripper_accuracy":
        float(finetuned_gripper_accuracy),

    "finetuned_better_count":
        int(wins),

    "original_better_count":
        int(losses),

    "tie_count":
        int(ties),

    "samples": all_results,
}

json_path = (
    OUTPUT_DIR /
    "evaluation_100samples.json"
)

with open(json_path, "w") as f:

    json.dump(
        summary,
        f,
        indent=2,
    )

# ============================================================
# Save CSV
# ============================================================

import csv

csv_path = (
    OUTPUT_DIR /
    "evaluation_100samples.csv"
)

with open(
    csv_path,
    "w",
    newline="",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "sample",
        "episode",
        "step",
        "language",

        "gt_0",
        "gt_1",
        "gt_2",
        "gt_3",
        "gt_4",
        "gt_5",
        "gt_6",

        "original_0",
        "original_1",
        "original_2",
        "original_3",
        "original_4",
        "original_5",
        "original_6",

        "finetuned_0",
        "finetuned_1",
        "finetuned_2",
        "finetuned_3",
        "finetuned_4",
        "finetuned_5",
        "finetuned_6",

        "original_mae",
        "finetuned_mae",

        "original_mse",
        "finetuned_mse",
    ])

    for r in all_results:

        writer.writerow([
            r["sample"],
            r["episode"],
            r["step"],
            r["language"],

            *r["gt"],
            *r["original"],
            *r["finetuned"],

            r["original_mae"],
            r["finetuned_mae"],

            r["original_mse"],
            r["finetuned_mse"],
        ])

# ============================================================
# Save numpy arrays
# ============================================================

np.save(
    OUTPUT_DIR / "ground_truth.npy",
    gt_actions,
)

np.save(
    OUTPUT_DIR / "original_actions.npy",
    original_actions,
)

np.save(
    OUTPUT_DIR / "finetuned_actions.npy",
    finetuned_actions,
)

# ============================================================
# Cleanup
# ============================================================

del finetuned_model
del finetuned_processor

torch.cuda.empty_cache()

print()
print("=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)

print()
print("Results:")
print(json_path)
print(csv_path)

print()
print(
    "Final VRAM allocated:",
    f"{torch.cuda.memory_allocated()/1024**3:.2f} GB",
)

print()
