import os
import csv
import glob
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
import torch

from PIL import Image
from io import BytesIO

from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import PeftModel


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "openvla/openvla-7b"
REPO_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = os.environ.get(
    "LIBERO_DATA_DIR",
    str(Path.home() / "modified_libero_rlds/libero_spatial_no_noops/1.0.0"),
)

ADAPTER_DIR = (
    REPO_DIR / "adapter-tmp/libero_spatial_1000step/"
    "openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+"
    "lora-r32+dropout-0.0+q-4bit--image_aug"
)

STATS_FILE = (
    REPO_DIR / "runs/libero_spatial_1000step/"
    "openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+"
    "lora-r32+dropout-0.0+q-4bit--image_aug/"
    "dataset_statistics.json"
)

OUTPUT_CSV = REPO_DIR / "planB2_libero_gt.csv"

NUM_SAMPLES = 50
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# Environment
# ============================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=" * 80)
print("PLAN B-2: REAL LIBERO GROUND-TRUTH ACTION COMPARISON")
print("=" * 80)

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

if torch.cuda.is_available():
    print(
        "VRAM:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )


# ============================================================
# Load statistics
# ============================================================

import json

with open(STATS_FILE, "r") as f:
    stats_all = json.load(f)

stats = stats_all["libero_spatial_no_noops"]["action"]

q01 = np.asarray(stats["q01"], dtype=np.float32)
q99 = np.asarray(stats["q99"], dtype=np.float32)
mask = np.asarray(stats["mask"], dtype=bool)

print("\nLIBERO action statistics:")
print("q01 :", q01)
print("q99 :", q99)
print("mask:", mask)


# ============================================================
# Action normalization used by OpenVLA
# ============================================================

def normalize_action(action):
    """
    Convert LIBERO ground-truth action into OpenVLA normalized space.

    For masked dimensions:
        normalized = 2 * (action - low) / (high - low) - 1

    For unmasked dimensions:
        leave unchanged.
    """
    action = np.asarray(action, dtype=np.float32)

    normalized = action.copy()

    normalized[mask] = (
        2.0 * (action[mask] - q01[mask])
        / (q99[mask] - q01[mask])
        - 1.0
    )

    return normalized


def unnormalize_action(normalized):
    """
    Convert OpenVLA normalized action back into LIBERO action space.
    """
    normalized = np.asarray(normalized, dtype=np.float32)

    action = normalized.copy()

    action[mask] = (
        0.5 * (normalized[mask] + 1.0)
        * (q99[mask] - q01[mask])
        + q01[mask]
    )

    return action


# ============================================================
# Decode TFRecord episode
# ============================================================

def parse_episode(path):
    ds = tf.data.TFRecordDataset(path)

    raw = next(iter(ds.take(1)))

    example = tf.train.Example()
    example.ParseFromString(raw.numpy())

    features = example.features.feature

    # Images
    image_bytes = list(
        features["steps/observation/image"].bytes_list.value
    )

    wrist_bytes = list(
        features["steps/observation/wrist_image"].bytes_list.value
    )

    # Language
    language_bytes = list(
        features["steps/language_instruction"].bytes_list.value
    )

    # Actions
    action_values = np.asarray(
        features["steps/action"].float_list.value,
        dtype=np.float32
    )

    num_steps = len(image_bytes)

    actions = action_values.reshape(num_steps, 7)

    instructions = [
        x.decode("utf-8", errors="replace")
        for x in language_bytes
    ]

    return {
        "images": image_bytes,
        "wrist_images": wrist_bytes,
        "instructions": instructions,
        "actions": actions,
    }


# ============================================================
# Collect candidate samples
# ============================================================

files = sorted(
    glob.glob(
        os.path.join(
            DATA_DIR,
            "libero_spatial-train.tfrecord-*"
        )
    )
)

print("\nTFRecord files:", len(files))

if not files:
    raise RuntimeError("No TFRecord files found.")


# ============================================================
# Read all episode metadata
# ============================================================

episodes = []

print("\nReading episode metadata...")

for idx, path in enumerate(files):
    print(
        f"  [{idx + 1:02d}/{len(files):02d}] "
        f"{os.path.basename(path)}"
    )

    episode = parse_episode(path)

    episodes.append(
        (
            path,
            episode
        )
    )

print("\nEpisodes loaded:", len(episodes))


# ============================================================
# Build transition list
# ============================================================

candidates = []

for file_idx, (path, episode) in enumerate(episodes):

    n = len(episode["images"])

    for t in range(n):

        candidates.append(
            {
                "file_idx": file_idx,
                "path": path,
                "t": t,
            }
        )

print("Total transitions:", len(candidates))


# ============================================================
# Sample transitions
# ============================================================

if NUM_SAMPLES > len(candidates):
    raise RuntimeError(
        f"NUM_SAMPLES={NUM_SAMPLES} > "
        f"available transitions={len(candidates)}"
    )

samples = random.sample(
    candidates,
    NUM_SAMPLES
)

print("Selected samples:", len(samples))


# ============================================================
# Load processor
# ============================================================

print("\n[1/4] Loading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    local_files_only=True,
)

print("Processor loaded.")


# ============================================================
# Quantization config
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# Load Base model
# ============================================================

print("\n[2/4] Loading Base OpenVLA...")

base_model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    local_files_only=True,
    device_map="auto",
)

base_model.eval()

print("Base model loaded.")


# ============================================================
# Load LoRA model
# ============================================================

print("\n[3/4] Loading LoRA adapter...")

lora_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
    is_trainable=False,
)

lora_model.eval()

print("LoRA model loaded.")


# ============================================================
# Model devices
# ============================================================

model_device = next(
    base_model.parameters()
).device

print("\nModel device:", model_device)


# ============================================================
# Prediction helper
# ============================================================

def predict_model(model, image, instruction):

    prompt = (
        f"In: What action should the robot take to {instruction}?\n"
        "Out:"
    )

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    )

    # Move tensors to model device
    for key in inputs:
        if torch.is_tensor(inputs[key]):
            inputs[key] = inputs[key].to(model_device)

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(
            dtype=torch.bfloat16
        )

    input_ids = inputs["input_ids"]

    # OpenVLA expects trailing token 29871
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (
                input_ids,
                torch.tensor(
                    [[29871]],
                    dtype=torch.long,
                    device=model_device,
                ),
            ),
            dim=1,
        )

    # Need 7 action tokens
    with torch.inference_mode():

        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=inputs["pixel_values"],
            max_new_tokens=7,
        )

    predicted_token_ids = (
        generated_ids[0, -7:]
        .detach()
        .cpu()
        .numpy()
    )

    # Exact OpenVLA decoding
    discretized_actions = (
        model.vocab_size
        - predicted_token_ids
    )

    # OpenVLA bin centers
    bin_centers = model.bin_centers

    # Exact OpenVLA behavior:
    # valid index range is [0, len(bin_centers) - 1]
    discretized_actions = np.clip(
        discretized_actions - 1,
        a_min=0,
        a_max=bin_centers.shape[0] - 1,
    )

    normalized_action = bin_centers[
        discretized_actions
    ]

    normalized_action = np.asarray(
        normalized_action,
        dtype=np.float32
    )

    # LIBERO unnormalization
    action = unnormalize_action(
        normalized_action
    )

    return action, normalized_action


# ============================================================
# Evaluation
# ============================================================

print("\n[4/4] Running evaluation...")
print("=" * 80)

rows = []

for idx, sample in enumerate(samples):

    episode = episodes[sample["file_idx"]][1]

    t = sample["t"]

    image_bytes = episode["images"][t]

    instruction = episode["instructions"][t]

    gt_action = episode["actions"][t]

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    print(
        f"\n[{idx + 1:02d}/{len(samples):02d}] "
        f"t={t} | {instruction}"
    )

    # Base
    base_action, base_norm = predict_model(
        base_model,
        image,
        instruction,
    )

    # LoRA
    lora_action, lora_norm = predict_model(
        lora_model,
        image,
        instruction,
    )

    # Errors in original LIBERO action space
    base_error = base_action - gt_action
    lora_error = lora_action - gt_action

    base_mae = np.mean(np.abs(base_error))
    lora_mae = np.mean(np.abs(lora_error))

    base_mse = np.mean(base_error ** 2)
    lora_mse = np.mean(lora_error ** 2)

    base_l2 = np.linalg.norm(base_error)
    lora_l2 = np.linalg.norm(lora_error)

    print("  GT   :", np.round(gt_action, 4))
    print("  Base :", np.round(base_action, 4))
    print("  LoRA :", np.round(lora_action, 4))

    print(
        f"  MAE  Base={base_mae:.6f} "
        f"LoRA={lora_mae:.6f}"
    )

    print(
        f"  MSE  Base={base_mse:.6f} "
        f"LoRA={lora_mse:.6f}"
    )

    rows.append(
        {
            "sample": idx,
            "file": os.path.basename(sample["path"]),
            "t": t,
            "instruction": instruction,

            "gt_dx": gt_action[0],
            "gt_dy": gt_action[1],
            "gt_dz": gt_action[2],
            "gt_drx": gt_action[3],
            "gt_dry": gt_action[4],
            "gt_drz": gt_action[5],
            "gt_gripper": gt_action[6],

            "base_dx": base_action[0],
            "base_dy": base_action[1],
            "base_dz": base_action[2],
            "base_drx": base_action[3],
            "base_dry": base_action[4],
            "base_drz": base_action[5],
            "base_gripper": base_action[6],

            "lora_dx": lora_action[0],
            "lora_dy": lora_action[1],
            "lora_dz": lora_action[2],
            "lora_drx": lora_action[3],
            "lora_dry": lora_action[4],
            "lora_drz": lora_action[5],
            "lora_gripper": lora_action[6],

            "base_mae": base_mae,
            "lora_mae": lora_mae,

            "base_mse": base_mse,
            "lora_mse": lora_mse,

            "base_l2": base_l2,
            "lora_l2": lora_l2,
        }
    )


# ============================================================
# Save CSV
# ============================================================

fieldnames = list(rows[0].keys())

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# Summary
# ============================================================

base_mae_all = np.array(
    [r["base_mae"] for r in rows]
)

lora_mae_all = np.array(
    [r["lora_mae"] for r in rows]
)

base_mse_all = np.array(
    [r["base_mse"] for r in rows]
)

lora_mse_all = np.array(
    [r["lora_mse"] for r in rows]
)

base_l2_all = np.array(
    [r["base_l2"] for r in rows]
)

lora_l2_all = np.array(
    [r["lora_l2"] for r in rows]
)


print("\n")
print("=" * 80)
print("PLAN B-2 RESULT")
print("=" * 80)

print(f"Samples: {len(rows)}")

print("\nMAE")
print(
    f"  Base : {base_mae_all.mean():.6f}"
)
print(
    f"  LoRA : {lora_mae_all.mean():.6f}"
)

print("\nMSE")
print(
    f"  Base : {base_mse_all.mean():.6f}"
)
print(
    f"  LoRA : {lora_mse_all.mean():.6f}"
)

print("\nL2")
print(
    f"  Base : {base_l2_all.mean():.6f}"
)
print(
    f"  LoRA : {lora_l2_all.mean():.6f}"
)

mae_improvement = (
    1.0
    - lora_mae_all.mean()
    / base_mae_all.mean()
) * 100.0

mse_improvement = (
    1.0
    - lora_mse_all.mean()
    / base_mse_all.mean()
) * 100.0

l2_improvement = (
    1.0
    - lora_l2_all.mean()
    / base_l2_all.mean()
) * 100.0

print("\nRelative improvement:")
print(
    f"  MAE : {mae_improvement:+.2f}%"
)
print(
    f"  MSE : {mse_improvement:+.2f}%"
)
print(
    f"  L2  : {l2_improvement:+.2f}%"
)

base_better = np.sum(
    base_mse_all < lora_mse_all
)

lora_better = np.sum(
    lora_mse_all < base_mse_all
)

same = len(rows) - base_better - lora_better

print("\nPer-sample MSE winner:")
print("  Base better :", base_better)
print("  LoRA better :", lora_better)
print("  Same        :", same)

print("\nCSV saved to:")
print(OUTPUT_CSV)

print("\nDONE")
