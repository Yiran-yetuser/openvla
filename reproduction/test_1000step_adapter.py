import os
from pathlib import Path
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
)
from peft import PeftModel

BASE_MODEL = "openvla/openvla-7b"
REPO_DIR = Path(__file__).resolve().parent.parent

ADAPTER = str(
    REPO_DIR / "adapter-tmp/libero_spatial_1000step/"
    "openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+"
    "lora-r32+dropout-0.0+q-4bit--image_aug"
)

DEVICE = "cuda:0"

print("=" * 70)
print("OpenVLA 1000-step QLoRA adapter smoke test")
print("=" * 70)

print("\n[1/5] Loading processor...")

processor = AutoProcessor.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)

print("Processor loaded.")

print("\n[2/5] Loading 4-bit base OpenVLA...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForVision2Seq.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map={"": 0},
    attn_implementation="sdpa",
)

print(
    f"Base loaded. VRAM allocated: "
    f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
)

print("\n[3/5] Loading 1000-step LoRA adapter...")
print("Adapter:", ADAPTER)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER,
    is_trainable=False,
)

model.eval()

print("Adapter loaded successfully.")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print(f"Trainable parameters : {trainable:,}")
print(f"Total parameters     : {total:,}")
print(
    f"VRAM allocated      : "
    f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
)
print(
    f"VRAM reserved       : "
    f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
)

print("\n[4/5] Checking adapter state...")

print("PEFT config:")
for name, cfg in model.peft_config.items():
    print(f"  adapter name : {name}")
    print(f"  rank         : {cfg.r}")
    print(f"  alpha        : {cfg.lora_alpha}")
    print(f"  inference    : {cfg.inference_mode}")

print("\n[5/5] Running one inference...")

image = Image.open(REPO_DIR / "test.jpg").convert("RGB")

prompt = "In: What action should the robot take to pick up the object?\nOut:"

inputs = processor(
    prompt,
    image,
    return_tensors="pt",
)

inputs = {
    k: (
        v.to(DEVICE, dtype=torch.bfloat16)
        if torch.is_floating_point(v)
        else v.to(DEVICE)
    )
    for k, v in inputs.items()
}

torch.cuda.reset_peak_memory_stats()

with torch.inference_mode():
    action = model.predict_action(
        **inputs,
        unnorm_key="bridge_orig",
        do_sample=False,
    )

print("\nPredicted action:")
print(action)

print("\nAction shape:")
print(action.shape)

print(
    f"\nPeak VRAM: "
    f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
)

print("\n" + "=" * 70)
print("SUCCESS: 1000-step QLoRA adapter loaded and inference completed")
print("=" * 70)
