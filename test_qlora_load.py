import gc
import torch

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

print("=" * 60)
print("OpenVLA 4-bit + LoRA Load Test")
print("=" * 60)

print(f"PyTorch      : {torch.__version__}")
print(f"CUDA         : {torch.version.cuda}")
print(f"GPU          : {torch.cuda.get_device_name(0)}")
print(
    f"VRAM total   : "
    f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
)

torch.cuda.empty_cache()
gc.collect()

# ------------------------------------------------------------
# 1. Processor
# ------------------------------------------------------------
print("\n[1/4] Loading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

print("Processor loaded.")

# ------------------------------------------------------------
# 2. 4-bit configuration
# ------------------------------------------------------------
print("\n[2/4] Creating 4-bit configuration...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("4-bit NF4 configuration ready.")

# ------------------------------------------------------------
# 3. Load OpenVLA
# ------------------------------------------------------------
print("\n[3/4] Loading OpenVLA in 4-bit...")

vla = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto",
)

print("OpenVLA loaded.")

# ------------------------------------------------------------
# 4. Prepare + LoRA
# ------------------------------------------------------------
print("\n[4/4] Preparing model for QLoRA...")

vla = prepare_model_for_kbit_training(vla)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules="all-linear",
    init_lora_weights="gaussian",
)

vla = get_peft_model(vla, lora_config)

print("\n" + "=" * 60)
print("QLoRA model ready")
print("=" * 60)

vla.print_trainable_parameters()

# ------------------------------------------------------------
# Parameter statistics
# ------------------------------------------------------------
total_params = sum(p.numel() for p in vla.parameters())
trainable_params = sum(
    p.numel() for p in vla.parameters()
    if p.requires_grad
)

print()
print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")
print(
    f"Trainable ratio      : "
    f"{100 * trainable_params / total_params:.4f}%"
)

# ------------------------------------------------------------
# GPU memory
# ------------------------------------------------------------
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3

    print()
    print("GPU memory:")
    print(f"Allocated : {allocated:.2f} GB")
    print(f"Reserved  : {reserved:.2f} GB")
    print(f"Peak      : {peak:.2f} GB")

print("\nSUCCESS: 4-bit + LoRA model loaded successfully.")
