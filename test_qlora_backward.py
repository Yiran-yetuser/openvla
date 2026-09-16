import gc
import os
import torch

from PIL import Image

from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

MODEL_ID = "openvla/openvla-7b"
IMAGE_PATH = "test_image.jpg"

print("=" * 70)
print("OpenVLA 4-bit + LoRA Forward / Backward Test")
print("=" * 70)

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
print(f"GPU     : {torch.cuda.get_device_name(0)}")

total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"VRAM    : {total_vram:.2f} GB")

# ------------------------------------------------------------
# Clean GPU memory
# ------------------------------------------------------------
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

device = torch.device("cuda")

# ------------------------------------------------------------
# 1. Processor
# ------------------------------------------------------------
print("\n[1/6] Loading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

print("Processor loaded.")

# ------------------------------------------------------------
# 2. 4-bit configuration
# ------------------------------------------------------------
print("\n[2/6] Creating 4-bit configuration...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("4-bit NF4 configuration ready.")

# ------------------------------------------------------------
# 3. Load model
# ------------------------------------------------------------
print("\n[3/6] Loading OpenVLA in 4-bit...")

vla = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto",
)

print("OpenVLA loaded.")

# ------------------------------------------------------------
# 4. Prepare QLoRA
# ------------------------------------------------------------
print("\n[4/6] Preparing QLoRA...")

vla = prepare_model_for_kbit_training(vla)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules="all-linear",
    init_lora_weights="gaussian",
)

vla = get_peft_model(vla, lora_config)

vla.print_trainable_parameters()

# ------------------------------------------------------------
# 5. Prepare one fake training sample
# ------------------------------------------------------------
print("\n[5/6] Preparing one training sample...")

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Cannot find {IMAGE_PATH}. "
        f"Expected: {os.path.abspath(IMAGE_PATH)}"
    )

image = Image.open(IMAGE_PATH).convert("RGB")

instruction = "pick up the object"

# OpenVLA expects a language instruction together with the image.
prompt = f"In: What action should the robot take to {instruction}?\nOut:"

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt",
)

# Move tensors to GPU.
for key in inputs:
    if torch.is_tensor(inputs[key]):
        inputs[key] = inputs[key].to(device)

# The OpenVLA vision backbone in this environment expects fp16.
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

print("Image size:", image.size)
print("input_ids shape:", inputs["input_ids"].shape)
print("pixel_values shape:", inputs["pixel_values"].shape)
print("pixel_values dtype:", inputs["pixel_values"].dtype)

# ------------------------------------------------------------
# Create labels
# ------------------------------------------------------------
#
# This is NOT a real robot-action label.
# It only exists to test the backward/optimizer path.
#
# We use the input sequence itself as labels so that the model
# can calculate a normal causal-language-model loss.
#
labels = inputs["input_ids"].clone()

# Ignore padding tokens if any.
if processor.tokenizer.pad_token_id is not None:
    labels[labels == processor.tokenizer.pad_token_id] = -100

inputs["labels"] = labels

print("labels shape:", inputs["labels"].shape)

# ------------------------------------------------------------
# 6. Forward + backward + optimizer
# ------------------------------------------------------------
print("\n[6/6] Running forward + backward...")

vla.train()

# Only LoRA parameters require gradients.
trainable_params = [
    p for p in vla.parameters()
    if p.requires_grad
]

print("Trainable tensors:", len(trainable_params))

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=1e-4,
)

torch.cuda.reset_peak_memory_stats()

optimizer.zero_grad(set_to_none=True)

# Forward
print("\n>>> Forward pass...")

with torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16,
):
    outputs = vla(**inputs)

loss = outputs.loss

print(f"Loss: {loss.item():.6f}")

forward_memory = torch.cuda.memory_allocated() / 1024**3
print(f"GPU allocated after forward: {forward_memory:.2f} GB")

# Backward
print("\n>>> Backward pass...")

loss.backward()

backward_memory = torch.cuda.memory_allocated() / 1024**3
peak_memory = torch.cuda.max_memory_allocated() / 1024**3

print(f"GPU allocated after backward: {backward_memory:.2f} GB")
print(f"GPU peak memory: {peak_memory:.2f} GB")

# Check gradients
num_grads = 0
grad_norm = 0.0

for p in trainable_params:
    if p.grad is not None:
        num_grads += 1
        grad_norm += p.grad.detach().float().norm().item() ** 2

grad_norm = grad_norm ** 0.5

print(f"LoRA tensors with gradients: {num_grads}")
print(f"Gradient norm: {grad_norm:.6f}")

# Optimizer step
print("\n>>> Optimizer step...")

optimizer.step()

print("Optimizer step completed.")

# ------------------------------------------------------------
# Final memory
# ------------------------------------------------------------
final_allocated = torch.cuda.memory_allocated() / 1024**3
final_reserved = torch.cuda.memory_reserved() / 1024**3
final_peak = torch.cuda.max_memory_allocated() / 1024**3

print("\n" + "=" * 70)
print("QLoRA BACKWARD TEST RESULT")
print("=" * 70)

print("Forward       : SUCCESS")
print("Backward      : SUCCESS")
print("Optimizer     : SUCCESS")

print(f"Final allocated : {final_allocated:.2f} GB")
print(f"Final reserved  : {final_reserved:.2f} GB")
print(f"Peak allocated  : {final_peak:.2f} GB")

print("=" * 70)
print("SUCCESS: OpenVLA 4-bit + LoRA can complete one training step.")
print("=" * 70)
