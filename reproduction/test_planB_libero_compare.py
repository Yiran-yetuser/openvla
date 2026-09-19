import os
import gc
import json
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ============================================================
# Configuration
# ============================================================

MODEL_ID = "openvla/openvla-7b"
REPO_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = REPO_DIR / "test_image.jpg"

ADAPTER_PATH = (
    REPO_DIR / "adapter-tmp/libero_spatial_1000step/"
    "openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+"
    "lora-r32+dropout-0.0+q-4bit--image_aug"
)

STATS_PATH = (
    REPO_DIR / "runs/libero_spatial_1000step/"
    "openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+"
    "lora-r32+dropout-0.0+q-4bit--image_aug/"
    "dataset_statistics.json"
)

OUTPUT_CSV = REPO_DIR / "planB_libero_compare.csv"

DATASET_KEY = "libero_spatial_no_noops"


INSTRUCTIONS = [
    "pick up the object",
    "move the object to the left",
    "move the object to the right",
    "move the object forward",
    "move the object backward",
    "move the object upward",
    "move the object downward",
    "move the object closer",
    "move the object farther away",
    "place the object in the container",
    "move the object to the center",
    "move the object away from the container",
]


# ============================================================
# Utility
# ============================================================


def print_vram(tag):
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3

    print(f"[VRAM] {tag}: " f"allocated={allocated:.2f} GB, " f"reserved={reserved:.2f} GB, " f"peak={peak:.2f} GB")


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ============================================================
# Load LIBERO statistics
# ============================================================


def load_libero_stats():
    print("[1/7] Loading LIBERO statistics...")

    with open(STATS_PATH, "r") as f:
        all_stats = json.load(f)

    if DATASET_KEY not in all_stats:
        raise KeyError(f"{DATASET_KEY} not found in statistics.\n" f"Available keys: {list(all_stats.keys())}")

    stats = all_stats[DATASET_KEY]["action"]

    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats["mask"], dtype=bool)

    print(f"Dataset key: {DATASET_KEY}")
    print(f"q01 : {q01}")
    print(f"q99 : {q99}")
    print(f"mask: {mask}")

    return q01, q99, mask


# ============================================================
# Prompt
# ============================================================


def make_prompt(instruction):
    return f"In: What action should the robot take to {instruction}?\nOut:"


# ============================================================
# Prepare model inputs
# ============================================================


def prepare_inputs(processor, image, instruction, model):
    prompt = make_prompt(instruction)

    inputs = processor(
        prompt,
        image,
        return_tensors="pt",
    )

    # Move every tensor to the model device.
    #
    # IMPORTANT:
    # OpenVLA's vision tower in this 4-bit configuration uses
    # bfloat16, so pixel_values must match that dtype.
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if key == "pixel_values":
                inputs[key] = value.to(
                    device=model.device,
                    dtype=torch.bfloat16,
                )
            else:
                inputs[key] = value.to(model.device)

    return inputs


# ============================================================
# OpenVLA action decoding
# ============================================================


@torch.inference_mode()
def predict_action_libero(
    model,
    processor,
    image,
    instruction,
    q01,
    q99,
    mask,
):
    """
    Run OpenVLA generation and decode the 7 action tokens.

    This intentionally does NOT call model.predict_action(),
    because predict_action() requires the dataset statistics to
    exist inside model.norm_stats.

    Instead we directly reproduce the decoding logic from
    OpenVLA's modeling_prismatic.py.
    """

    inputs = prepare_inputs(
        processor=processor,
        image=image,
        instruction=instruction,
        model=model,
    )

    input_ids = inputs["input_ids"]

    # --------------------------------------------------------
    # OpenVLA predict_action() adds token 29871 if needed.
    # --------------------------------------------------------

    if not torch.all(input_ids[:, -1] == 29871):
        extra_token = torch.tensor(
            [[29871]],
            dtype=torch.long,
            device=input_ids.device,
        )

        input_ids = torch.cat(
            (input_ids, extra_token),
            dim=1,
        )

    # --------------------------------------------------------
    # OpenVLA action dimension = 7
    # --------------------------------------------------------

    action_dim = 7

    # --------------------------------------------------------
    # Generate action tokens
    # --------------------------------------------------------

    generated_ids = model.generate(
        input_ids,
        max_new_tokens=action_dim,
        do_sample=False,
    )

    # --------------------------------------------------------
    # Extract last 7 tokens
    # --------------------------------------------------------

    predicted_token_ids = generated_ids[0, -action_dim:].detach().cpu().numpy()

    # --------------------------------------------------------
    # OpenVLA action-token decoding
    #
    # Source:
    #
    # discretized_actions = vocab_size - token_ids
    # discretized_actions -= 1
    # clip to [0,255]
    # bin_centers[index]
    # --------------------------------------------------------

    discretized_actions = model.vocab_size - predicted_token_ids

    discretized_actions = np.clip(
        discretized_actions - 1,
        a_min=0,
        a_max=model.bin_centers.shape[0] - 1,
    )

    normalized_actions = model.bin_centers[discretized_actions]

    normalized_actions = np.asarray(
        normalized_actions,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # LIBERO unnormalization
    #
    # normalized [-1,1]
    #
    # action = 0.5 * (norm + 1)
    #          * (q99 - q01)
    #          + q01
    #
    # For gripper:
    # mask=False
    # => keep normalized value directly.
    # --------------------------------------------------------

    action_high = q99
    action_low = q01

    libero_action = np.where(
        mask,
        0.5 * (normalized_actions + 1.0) * (action_high - action_low) + action_low,
        normalized_actions,
    )

    return (
        libero_action.astype(np.float32),
        normalized_actions.astype(np.float32),
        predicted_token_ids.astype(np.int64),
    )


# ============================================================
# Load 4-bit base model
# ============================================================


def load_base_model(processor):
    print()
    print("Loading 4-bit OpenVLA...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    model.eval()

    print("BASE model loaded.")

    print_vram("after BASE model load")

    return model


# ============================================================
# Load LoRA model
# ============================================================


def load_lora_model(processor):
    print()
    print("Loading base model for LoRA...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    print("Base model for LoRA loaded.")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        local_files_only=True,
        is_trainable=False,
    )

    model.eval()

    print("1000-step LoRA adapter loaded.")

    print_vram("after LoRA model load")

    return model


# ============================================================
# Main
# ============================================================


def main():

    print("=" * 72)
    print("OpenVLA Plan B")
    print("Base OpenVLA vs 1000-step LIBERO Spatial LoRA")
    print("=" * 72)

    print(f"Model      : {MODEL_ID}")
    print(f"Image      : {IMAGE_PATH}")
    print(f"Adapter    : {ADAPTER_PATH}")
    print(f"Statistics : {STATS_PATH}")
    print(f"Output     : {OUTPUT_CSV}")

    # --------------------------------------------------------
    # 1. Statistics
    # --------------------------------------------------------

    q01, q99, mask = load_libero_stats()

    # --------------------------------------------------------
    # 2. Image
    # --------------------------------------------------------

    print()
    print("[2/7] Loading image...")

    image = Image.open(IMAGE_PATH).convert("RGB")

    print(f"Image size: {image.size}")

    # --------------------------------------------------------
    # 3. Processor
    # --------------------------------------------------------

    print()
    print("[3/7] Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        local_files_only=True,
    )

    print("Processor loaded.")

    # --------------------------------------------------------
    # 4. Decoder
    # --------------------------------------------------------

    print()
    print("[4/7] Using OpenVLA action decoding...")
    print("Action decoding will reproduce " "OpenVLA's predict_action() logic.")

    # ========================================================
    # BASE
    # ========================================================

    print()
    print("[5/7] Loading BASE OpenVLA...")

    base_model = load_base_model(processor)

    base_results = []

    print()
    print("=" * 72)
    print("BASE MODEL INFERENCE")
    print("=" * 72)

    for i, instruction in enumerate(INSTRUCTIONS, start=1):

        print()
        print(f"BASE [{i:02d}/{len(INSTRUCTIONS)}] " f"{instruction}")

        action, normalized, tokens = predict_action_libero(
            model=base_model,
            processor=processor,
            image=image,
            instruction=instruction,
            q01=q01,
            q99=q99,
            mask=mask,
        )

        base_results.append(
            {
                "instruction": instruction,
                "action": action,
                "normalized": normalized,
                "tokens": tokens,
            }
        )

        print(
            "  tokens:",
            tokens.tolist(),
        )

        print(
            "  normalized:",
            np.round(normalized, 6),
        )

        print(
            "  LIBERO action:",
            np.round(action, 6),
        )

    print_vram("after BASE inference")

    # --------------------------------------------------------
    # Free BASE
    # --------------------------------------------------------

    print()
    print("Freeing BASE model...")

    del base_model
    cleanup()

    print_vram("after freeing BASE")

    # ========================================================
    # LoRA
    # ========================================================

    print()
    print("[6/7] Loading 1000-step LoRA...")

    lora_model = load_lora_model(processor)

    lora_results = []

    print()
    print("=" * 72)
    print("1000-STEP LoRA MODEL INFERENCE")
    print("=" * 72)

    for i, instruction in enumerate(INSTRUCTIONS, start=1):

        print()
        print(f"LoRA [{i:02d}/{len(INSTRUCTIONS)}] " f"{instruction}")

        action, normalized, tokens = predict_action_libero(
            model=lora_model,
            processor=processor,
            image=image,
            instruction=instruction,
            q01=q01,
            q99=q99,
            mask=mask,
        )

        lora_results.append(
            {
                "instruction": instruction,
                "action": action,
                "normalized": normalized,
                "tokens": tokens,
            }
        )

        print(
            "  tokens:",
            tokens.tolist(),
        )

        print(
            "  normalized:",
            np.round(normalized, 6),
        )

        print(
            "  LIBERO action:",
            np.round(action, 6),
        )

    print_vram("after LoRA inference")

    # ========================================================
    # Comparison
    # ========================================================

    print()
    print("[7/7] Comparing BASE vs LoRA...")

    rows = []

    print()
    print("=" * 72)
    print("BASE vs 1000-STEP LoRA")
    print("=" * 72)

    for base, lora in zip(
        base_results,
        lora_results,
    ):

        instruction = base["instruction"]

        base_action = base["action"]
        lora_action = lora["action"]

        delta = lora_action - base_action

        delta_l2 = float(np.linalg.norm(delta))

        token_changed = not np.array_equal(
            base["tokens"],
            lora["tokens"],
        )

        print()
        print(instruction)

        print(
            "  BASE:",
            np.round(base_action, 6),
        )

        print(
            "  LoRA:",
            np.round(lora_action, 6),
        )

        print(
            "  Δ:",
            np.round(delta, 6),
        )

        print(
            "  Δ L2:",
            f"{delta_l2:.6f}",
        )

        print(
            "  token changed:",
            token_changed,
        )

        row = {
            "instruction": instruction,
            "base_token_0": int(base["tokens"][0]),
            "base_token_1": int(base["tokens"][1]),
            "base_token_2": int(base["tokens"][2]),
            "base_token_3": int(base["tokens"][3]),
            "base_token_4": int(base["tokens"][4]),
            "base_token_5": int(base["tokens"][5]),
            "base_token_6": int(base["tokens"][6]),
            "lora_token_0": int(lora["tokens"][0]),
            "lora_token_1": int(lora["tokens"][1]),
            "lora_token_2": int(lora["tokens"][2]),
            "lora_token_3": int(lora["tokens"][3]),
            "lora_token_4": int(lora["tokens"][4]),
            "lora_token_5": int(lora["tokens"][5]),
            "lora_token_6": int(lora["tokens"][6]),
            "token_changed": token_changed,
            "base_a0": float(base_action[0]),
            "base_a1": float(base_action[1]),
            "base_a2": float(base_action[2]),
            "base_a3": float(base_action[3]),
            "base_a4": float(base_action[4]),
            "base_a5": float(base_action[5]),
            "base_a6": float(base_action[6]),
            "lora_a0": float(lora_action[0]),
            "lora_a1": float(lora_action[1]),
            "lora_a2": float(lora_action[2]),
            "lora_a3": float(lora_action[3]),
            "lora_a4": float(lora_action[4]),
            "lora_a5": float(lora_action[5]),
            "lora_a6": float(lora_action[6]),
            "delta_a0": float(delta[0]),
            "delta_a1": float(delta[1]),
            "delta_a2": float(delta[2]),
            "delta_a3": float(delta[3]),
            "delta_a4": float(delta[4]),
            "delta_a5": float(delta[5]),
            "delta_a6": float(delta[6]),
            "delta_l2": delta_l2,
        }

        rows.append(row)

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    fieldnames = list(rows[0].keys())

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print(f"CSV saved to:")
    print(OUTPUT_CSV)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    all_l2 = np.asarray(
        [r["delta_l2"] for r in rows],
        dtype=np.float32,
    )

    changed_count = sum(r["token_changed"] for r in rows)

    print()
    print("Summary:")
    print(f"  Instructions tested : {len(rows)}")
    print(f"  Token outputs changed: " f"{changed_count}/{len(rows)}")
    print(f"  Mean action Δ L2    : " f"{all_l2.mean():.6f}")
    print(f"  Max action Δ L2     : " f"{all_l2.max():.6f}")
    print(f"  Min action Δ L2     : " f"{all_l2.min():.6f}")

    del lora_model
    cleanup()


if __name__ == "__main__":
    main()
