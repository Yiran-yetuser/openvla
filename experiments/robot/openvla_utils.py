"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def prepare_vla_for_kbit_inference(vla):
    """Opt in to the same k-bit preparation used by the QLoRA training path."""
    from peft import prepare_model_for_kbit_training

    vla = prepare_model_for_kbit_training(vla, use_gradient_checkpointing=True)
    vla._openvla_kbit_inference_prepared = True
    vla.eval()
    return vla


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading with BF16 compute and SDPA attention")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # 原始代码：直接对 checkpoint 调 from_pretrained，并强制使用 flash_attention_2
    # vla = AutoModelForVision2Seq.from_pretrained(
    #     cfg.pretrained_checkpoint,
    #     attn_implementation="flash_attention_2",
    #     torch_dtype=torch.bfloat16,
    #     load_in_8bit=cfg.load_in_8bit,
    #     load_in_4bit=cfg.load_in_4bit,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    # )
    #
    # 修改原因：
    #   1) LoRA 微调产物（adapter_config.json + adapter_model.safetensors）不能直接用
    #      from_pretrained 加载，必须先加载 base 模型再套 PeftModel；
    #   2) 本机没有安装 flash-attn，改用 PyTorch 原生 sdpa 注意力实现；
    #   3) 12GB 显存下 base 模型需要用 4-bit NF4 量化加载。
    checkpoint_dir = Path(cfg.pretrained_checkpoint)
    is_lora_checkpoint = (checkpoint_dir / "adapter_config.json").is_file()
    prepare_for_kbit_inference = getattr(cfg, "prepare_for_kbit_inference", False)
    if prepare_for_kbit_inference and not cfg.load_in_4bit:
        raise ValueError("prepare_for_kbit_inference requires load_in_4bit=True")
    if prepare_for_kbit_inference and not is_lora_checkpoint:
        raise ValueError("prepare_for_kbit_inference currently supports LoRA adapter checkpoints only")
    # Match the adapter and official-checkpoint controls: bare load_in_4bit
    # otherwise silently defaults to FP4 instead of the training NF4 quantizer.
    quantization_config = None
    if cfg.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=getattr(cfg, "bnb_double_quant", True),
        )
    elif cfg.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    if is_lora_checkpoint:

        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.base_model_path,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": 0} if quantization_config is not None else None,
        )
        vla = PeftModel.from_pretrained(base_vla, cfg.pretrained_checkpoint, is_trainable=False)
        vla.eval()
    else:
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.pretrained_checkpoint,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # Move model to device.
        # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
        #       already be set to the right devices and casted to the correct dtype upon loading.
        if not cfg.load_in_8bit and not cfg.load_in_4bit:
            vla = vla.to(DEVICE)

    if prepare_for_kbit_inference:
        vla = prepare_vla_for_kbit_inference(vla)

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        # 原始代码：vla.norm_stats = norm_stats
        # 修改原因：PeftModel 只是外包装，predict_action() 仍然绑定在内部 base 模型上，
        #          只给 wrapper 赋值时内部模型读不到 norm_stats，会在反归一化时报错。
        vla.norm_stats = norm_stats
        inner_vla = getattr(getattr(vla, "base_model", None), "model", None)
        if inner_vla is not None:
            inner_vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    vla.eval()
    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    # 原始代码：processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    # 修改原因：LoRA adapter 目录里只有 adapter 权重，没有 config.json 与 processor 文件，
    #          直接加载会报 "does not appear to have a file named config.json"。
    #          adapter 使用的是与 base 模型相同的 processor，因此回退到 base 模型加载。
    checkpoint_dir = Path(cfg.pretrained_checkpoint)
    is_lora_checkpoint = (checkpoint_dir / "adapter_config.json").is_file()
    processor_path = cfg.base_model_path if is_lora_checkpoint else cfg.pretrained_checkpoint
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # Get action.
    context = (torch.autocast("cuda", dtype=torch.bfloat16)
               if getattr(vla, "_openvla_kbit_inference_prepared", False) else nullcontext())
    with torch.inference_mode(), context:
        action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action
