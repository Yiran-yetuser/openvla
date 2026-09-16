# OpenVLA 本地复现与 QLoRA 训练 Context

> 项目：论文  
> 更新时间：2026-09-06  
> 当前阶段：1000 steps QLoRA 正式训练中  
> 目标：在 RTX 5070 Ti Laptop（约 12 GB VRAM）上复现 OpenVLA-7B，并使用 QLoRA 在 LIBERO Spatial 上微调与评估。

---

## 1. 项目目标

1. 本地运行 OpenVLA-7B。
2. 实现 4-bit 推理。
3. 验证 QLoRA + LoRA 训练。
4. 使用 `libero_spatial_no_noops` 进行真实 action-token 微调。
5. 比较原始模型、100-step、1000-step 模型。
6. 最终进行 LIBERO simulator evaluation，以任务成功率作为核心指标。

---

## 2. 硬件与系统

### GPU

```text
NVIDIA GeForce RTX 5070 Ti Laptop GPU
VRAM: 12227 MiB
实际可用显存约：11.47 GiB
Compute Capability: (12, 0)
SM: 120 / Blackwell
```

### NVIDIA / CUDA

```text
Driver: 595.84
PyTorch: 2.8.0+cu128
PyTorch CUDA runtime: 12.8
torch.cuda.is_available(): True
BF16 supported: True
```

### 系统

```text
Ubuntu 22.04
Python 3.10.20
Kernel: 6.8.0-138-generic
glibc: 2.35
```

---

## 3. OpenVLA 仓库

本地：

```text
~/openvla
```

分支：

```text
main
```

之前确认仓库 clean 且 synced with `origin/main`。

当时看到的最新 commit：

```text
c8f03f4 Update README: Remove code block typo
```

主要目录：

```text
~/openvla/
├── experiments/
├── prismatic/
├── scripts/
├── vla-scripts/
├── Makefile
├── README.md
├── pyproject.toml
├── requirements-min.txt
├── test.jpg
├── test_openvla_4bit.py
├── test_openvla_backward.py
├── test_openvla_lora.py
├── test_openvla_inference.py
├── test_openvla_real_action_backward.py
└── test_openvla_alllinear_checkpoint.py
```

---

## 4. 关键 Python 包版本

```text
transformers 4.40.1
tokenizers 0.19.1
timm 0.9.10
accelerate 0.30.1
bitsandbytes 0.50.2
PEFT 0.11.1
numpy 2.2.6
safetensors 0.8.0
TensorFlow 2.21.0
PyTorch 2.8.0+cu128
```

### 重要环境保护原则

不要盲目运行：

```bash
pip install -e .
pip install -r requirements.txt
```

因为 OpenVLA 当前 `pyproject.toml` 中包含：

```text
torch==2.2.0
torchvision==0.17.0
torchaudio==2.2.0
```

可能破坏当前已经正常工作的 Blackwell / PyTorch 2.8 + CUDA 12.8 环境。

OpenVLA 已通过：

```bash
pip install -e . --no-deps
```

安装 editable package。

---

## 5. bitsandbytes

当前：

```text
bitsandbytes 0.50.2
```

已经验证可以在：

```text
RTX 5070 Ti Laptop
Blackwell SM120
PyTorch 2.8.0+cu128
```

上工作。

曾出现：

```text
inner dimension (4304) is not aligned for fast kernel with blocksize=64,
falling back to slower implementation.
```

这是性能 fallback，不是错误，不会导致 OOM。

---

## 6. Hugging Face 模型

模型：

```text
openvla/openvla-7b
```

缓存：

```text
~/.cache/huggingface/hub/models--openvla--openvla-7b
```

约：

```text
15 GB
```

---

## 7. HF Mirror / Proxy

曾使用：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

代理：

```text
HTTPS_PROXY=http://127.0.0.1:7890/
HTTP_PROXY=http://127.0.0.1:7890/
http_proxy=http://127.0.0.1:7890/
```

曾因：

```text
ALL_PROXY=socks://127.0.0.1:7890/
```

导致 Hugging Face 下载错误。

解决：

```bash
unset ALL_PROXY all_proxy
```

---

## 8. 4-bit 推理验证

已成功完成 OpenVLA-7B 4-bit inference。

配置：

```text
load_in_4bit=True
bnb_4bit_quant_type="nf4"
bnb_4bit_compute_dtype=torch.bfloat16
bnb_4bit_use_double_quant=True
device_map={"": 0}
torch_dtype=torch.bfloat16
attn_implementation="sdpa"
```

一次成功结果：

```text
SUCCESS: OpenVLA inference completed

Predicted action:
[ 1.48869192e-05
 -1.07164215e-03
  8.85264086e-03
 -1.80195192e-02
 -6.80938021e-03
 -2.26318844e-02
  9.96078431e-01 ]

Action shape:
(7,)

Peak VRAM:
4.40 GB
```

---

## 9. Accelerate 版本问题

最初使用较新的 accelerate 时，4-bit 加载出现：

```text
dispatch_model
→ model.to(device)
→ bitsandbytes 4-bit ValueError
```

后来调整为：

```text
accelerate==0.30.1
```

之后 4-bit 加载成功。

因此当前环境不要随意升级 accelerate。

---

## 10. LoRA 验证

PEFT：

```text
peft==0.11.1
```

最初测试：

```text
r=8
lora_alpha=16
lora_dropout=0.05
bias="none"
task_type="CAUSAL_LM"

target_modules:
q_proj
k_proj
v_proj
o_proj
```

结果：

```text
trainable params:
8,388,608

all params:
7,549,625,792

trainable:
0.1111%
```

显存：

```text
allocated: 4.11 GB
reserved: 4.27 GB
```

---

## 11. QLoRA backward 验证

已完成：

```text
4-bit base model
+
LoRA
+
forward
+
loss
+
backward
+
optimizer step
```

结果：

```text
input_ids:
(1,18)

pixel_values:
(1,6,224,224)

dtype:
bf16

Loss:
22.028207778778...

Peak allocated:
6.05 GB

Peak reserved:
6.24 GB

VRAM headroom:
5.43 GB

STATUS:
GOOD
```

---

## 12. OpenVLA Action Tokenizer

官方当前逻辑：

```text
n_bins = 256
min_action = -1
max_action = 1
```

核心：

```python
self.bins = np.linspace(
    self.min_action,
    self.max_action,
    self.n_bins
)

self.bin_centers = (
    self.bins[:-1] + self.bins[1:]
) / 2
```

action token 起始位置：

```python
action_token_begin_idx = tokenizer.vocab_size - (bins + 1)
```

当前：

```text
Action token begin index:
31743

Action vocabulary size:
256
```

### 重要

当前官方 ActionTokenizer 没有：

```python
action_token_end_idx
```

不要再使用这个不存在的属性。

---

## 13. OpenVLA action-token label mask

官方逻辑：

```python
labels = input_ids.clone()
labels[:, :-(len(action) + 1)] = IGNORE_INDEX
```

原因是 HuggingFace causal LM forward 内部会进行 shift。

因此不要简单改成：

```python
labels[:, :-len(action)] = -100
```

真实测试中：

```text
7 action tokens
```

最终 non-ignored labels 为：

```text
8
```

这是符合官方实现的。

---

## 14. 真实 action-token backward

使用真实 OpenVLA action tokenizer 后成功：

```text
trainable params:
8,388,608

input_ids:
(1,25)

pixel_values:
(1,6,224,224)

Action-token loss:
17.306367874...

Peak allocated:
6.10 GB

Peak reserved:
6.28 GB

STATUS:
EXCELLENT
```

---

## 15. All-linear LoRA + Gradient Checkpointing

进一步验证了更接近官方 finetune 的配置：

```text
4-bit NF4
LoRA rank = 8
LoRA alpha = 8
LoRA dropout = 0.05
target_modules = all-linear
gradient checkpointing = enabled
```

训练参数：

```text
trainable params:
27,707,072

all params:
7,568,944,256

trainable:
0.3661%
```

成功完成：

```text
forward
backward
optimizer step
```

显存：

```text
After forward:
5.26 GB

After backward:
5.33 GB

Peak allocated:
5.33 GB

Peak reserved:
5.60 GB

VRAM headroom:
6.14 GB

STATUS:
EXCELLENT
```

结论：

> RTX 5070 Ti Laptop 约 12 GB 显存可以进行 OpenVLA-7B 的 4-bit + all-linear LoRA + gradient checkpointing 训练。

---

## 16. 为什么不做 Full Fine-tune

OpenVLA-7B 在当前约 11.47 GiB VRAM 环境下不适合 full fine-tuning。

Full fine-tuning 需要：

```text
model weights
+
gradients
+
optimizer states
+
activations
```

显存远超当前能力。

因此当前路线：

```text
4-bit QLoRA
```

而不是 full fine-tune。

---

# 17. 当前训练数据集

当前训练数据：

```text
libero_spatial_no_noops
```

本地：

```text
~/modified_libero_rlds/libero_spatial_no_noops/
```

完整目录：

```text
~/modified_libero_rlds/
├── libero_spatial_no_noops/
│   └── 1.0.0/
│       ├── features.json
│       ├── dataset_info.json
│       ├── libero_spatial-train.tfrecord-00000-of-00016
│       ├── ...
│       └── libero_spatial-train.tfrecord-00015-of-00016
└── .cache/
```

大小：

```text
约 1.8 GB
```

---

## 18. Dataset 信息

成功通过 OpenVLA RLDSDataset 加载：

```text
Dataset:
libero_spatial_no_noops

Episodes:
432

Shards:
16

Transformed dataset length:
52970
```

日志曾显示：

```text
INFO Load dataset info...
INFO Creating a tf.data.Dataset reading 16 files...
INFO Constructing tf.data.Dataset libero_spatial...
INFO Computing dataset statistics...
100% ... 432/432
INFO Constructing datasets...
INFO Applying frame transforms...
Dataset loaded!
Length: 52970
```

---

## 19. 数据内容

主要包括：

```text
observation
task
action
```

视觉：

```text
observation["image_primary"]
```

语言：

```text
task["language_instruction"]
```

OpenVLA prompt：

```text
What action should the robot take to {lang}?
```

动作：

```text
7D action
```

训练目标：

```text
图像 + 语言指令
        ↓
7 action tokens
        ↓
7D robot action
```

---

## 20. No-Noops 数据集注意事项

当前使用：

```text
libero_spatial_no_noops
```

这是过滤 no-op transitions 后的数据集。

最终 simulator evaluation 通常使用：

```text
libero_spatial
```

并正确处理对应 normalization statistics。

因此：

> 训练数据集与最终 simulator evaluation 数据配置不是完全同一个概念。

---

# 21. finetune.py 单 GPU 修改

官方 finetune.py 原本使用 DDP：

```python
vla = DDP(
    vla,
    device_ids=[device_id],
    find_unused_parameters=True,
    gradient_as_bucket_view=True,
)
```

单 GPU 直接运行曾遇到：

```text
Default process group not initialized
```

并且 DDP + reentrant gradient checkpointing + LoRA 曾出现：

```text
Expected to mark a variable ready only once
```

---

## 22. 当前 finetune.py 修改

备份：

```bash
cp vla-scripts/finetune.py \
   vla-scripts/finetune.py.bak_single_gpu
```

当前使用：

```python
if dist.is_initialized():
    vla = DDP(
        vla,
        device_ids=[device_id],
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )

vla_raw = vla.module if hasattr(vla, "module") else vla
```

并将：

```text
vla.module.config.image_sizes
```

改为：

```text
vla_raw.config.image_sizes
```

将：

```text
vla.module.vision_backbone...
```

改为：

```text
vla_raw.vision_backbone...
```

保存模型时使用：

```text
vla_raw.save_pretrained(...)
```

所有 barrier 改为：

```python
if dist.is_initialized():
    dist.barrier()
```

---

# 23. 100-step QLoRA 训练

配置：

```text
dataset:
libero_spatial_no_noops

use_lora:
True

use_quantization:
True

lora_rank:
32

batch_size:
2

gradient accumulation:
8

effective batch:
16

learning rate:
5e-4

max_steps:
100
```

训练成功：

```text
100/100 [20:06, 11.96s/it]
Saving Model Checkpoint for Step 100
Max step 100 reached!
```

---

# 24. 100-step checkpoint

输出目录：

```text
~/openvla/runs/libero_spatial_100step/
```

checkpoint 约：

```text
15 GB
```

包含：

```text
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json
config.json
generation_config.json
tokenizer files
dataset_statistics.json
added_tokens.json
```

没有：

```text
adapter_config.json
adapter_model.safetensors
```

这是因为当前 OpenVLA 保存流程会把 LoRA merge 回基础模型后保存 merged model。

因此：

```text
15 GB checkpoint
≠
LoRA 没有训练
```

不要因此删除 checkpoint。

---

# 25. 100-step checkpoint 验证

使用：

```python
AutoConfig.from_pretrained(
    checkpoint,
    trust_remote_code=True
)
```

成功：

```text
Model type:
openvla

Model class:
OpenVLAForActionPrediction

Image size:
[224, 224]
```

4-bit reload：

```text
Loading checkpoint shards: 100% 4/4
```

结果：

```text
Device:
cuda:0

Dtype:
torch.bfloat16

VRAM allocated:
4.02 GB

VRAM reserved:
4.18 GB
```

---

# 26. 100-step 单样本比较

某个样本结果：

```text
Dim       GT          Original       Fine-tuned
0         0.131250    0.019531       -0.003906
1        -0.040179   -0.042969       -0.144531
2        -0.000000   -0.136719        0.113281
3         0.000000   -0.128906        0.011719
4        -0.049286    0.003906        0.011719
5        -0.000000   -0.074219        0.996094
6        -1.000000    0.996094        0.996094
```

```text
Original MAE:
0.357663

Fine-tuned MAE:
0.488243

Original MSE:
0.577218

Fine-tuned MSE:
0.717492
```

单样本不能判断整体效果。

---

# 27. 100-sample 对比

脚本：

```text
compare_libero_100samples.py
```

比较：

```text
Original OpenVLA
vs
100-step fine-tuned OpenVLA
```

同一批样本。

结果：

```text
Collected samples: 100
Selected 100 samples.

Original model:
100/100

Fine-tuned model:
100/100
```

---

## 28. 100-sample Overall

### MAE

```text
Original mean MAE:
0.408058

Fine-tuned mean MAE:
0.397743

MAE improvement:
+2.53%
```

Median：

```text
Original:
0.392934

Fine-tuned:
0.372239
```

### MSE

```text
Original mean MSE:
0.432656

Fine-tuned mean MSE:
0.454866

MSE improvement:
-5.13%
```

---

## 29. Per-dimension MAE

```text
Dim       Original      Fine-tuned       Change

0         0.399855      0.348471        +12.85%
1         0.349766      0.276259        +21.02%
2         0.414529      0.382442         +7.74%
3         0.122817      0.053722        +56.26%
4         0.151446      0.047076        +68.92%
5         0.157524      0.385997       -145.04%
6         1.260469      1.290234        -2.36%
```

其中：

```text
Dim 3:
+56.26%

Dim 4:
+68.92%

Dim 5:
-145.04%

Dim 6:
-2.36%
```

---

# 30. Gripper Accuracy

```text
Original:
20.00%

Fine-tuned:
18.00%
```

目前没有提升。

---

# 31. Win / Loss

```text
Fine-tuned better:
50 / 100

Original better:
50 / 100

Tie:
0
```

因此：

> 100-step 模型尚未表现出明确的整体优势。

---

# 32. 100-step 结果的正确解释

不能简单说：

```text
100 steps 训练失败
```

也不能说：

```text
100 steps 已经成功提升
```

正确结论：

> 100 steps 已经证明训练流程有效，模型行为发生了变化，部分动作维度明显改善，但整体指标尚未表现出稳定、显著的提升。

原因之一：

```text
100 steps × effective batch 16
≈ 1600 samples
```

相对于：

```text
52970 transformed samples
```

训练量仍然较小。

因此继续训练到 1000 steps 是合理的。

---

# 33. 当前 1000-step 正式训练

当前阶段：

```text
1000 steps QLoRA
```

数据：

```text
libero_spatial_no_noops
```

配置：

```text
LoRA rank = 32
LoRA target = all-linear
4-bit NF4
BF16 compute
batch_size = 2
gradient accumulation = 8
effective batch = 16
learning rate = 5e-4
gradient checkpointing = enabled
max_steps = 1000
save_steps = 1000
```

命令：

```bash
cd ~/openvla

python vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir ~/modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --run_root_dir runs/libero_spatial_1000step \
    --adapter_tmp_dir adapter-tmp/libero_spatial_1000step \
    --use_lora True \
    --use_quantization True \
    --lora_rank 32 \
    --batch_size 2 \
    --grad_accumulation_steps 8 \
    --learning_rate 5e-4 \
    --save_steps 1000 \
    --max_steps 1000
```

---

# 34. 当前 1000-step 训练状态

2026-09-06 15:24 左右已经完成 dataset initialization。

最新日志：

```text
INFO | >> [*] Threads per Dataset: [1]
INFO | >> [*] Reads per Dataset: [1]
INFO | >> [*] Constructing datasets...
INFO | >> [*] Applying frame transforms on dataset...
INFO | >> [*] Saved dataset statistics file at path
runs/libero_spatial_1000step/.../dataset_statistics.json
```

随后出现：

```text
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice:
```

已经选择：

```text
3
```

即：

```text
Don't visualize my results
```

W&B 不需要登录。

---

# 35. 1000-step 训练预期

上一轮 100 steps：

```text
约 20 分钟
约 12 秒 / step
```

因此 1000 steps 粗略：

```text
约 3～3.5 小时
```

实际时间可能有所变化。

正常情况下应该出现：

```text
0/1000
...
100/1000
...
1000/1000
```

并最终：

```text
Saving Model Checkpoint for Step 1000
Max step 1000 reached!
```

---

# 36. 1000-step 保存策略

当前：

```text
save_steps = 1000
```

不要每 100 steps 保存，因为每个 merged checkpoint 约 15 GB。

如果每 100 steps 保存一次：

```text
10 × 15 GB
≈ 150 GB
```

因此当前只保存 Step 1000。

---

# 37. 训练过程中显存预期

之前 all-linear QLoRA + checkpointing：

```text
Peak allocated:
5.33 GB

Peak reserved:
5.60 GB
```

因此正常训练预计仍然在约：

```text
5～6 GB
```

范围。

如果出现 OOM，优先检查：

```text
batch_size
gradient checkpointing
use_quantization
```

不要首先改变整个训练方案。

---

# 38. 1000-step 完成后的计划

### Step 1

确认：

```text
~/openvla/runs/libero_spatial_1000step/
```

checkpoint 成功。

### Step 2

加载 1000-step checkpoint。

### Step 3

进行同一 evaluation set：

```text
Original
vs
100-step
vs
1000-step
```

### Step 4

比较：

```text
Mean MAE
Median MAE
Mean MSE
Per-dimension MAE
Gripper accuracy
Win/Loss
```

### Step 5

再决定是否训练：

```text
3000 steps
```

---

# 39. 最终必须做 LIBERO Simulator Evaluation

Action MAE 不是最终目标。

最终要做：

```text
OpenVLA
↓
LIBERO environment
↓
执行 action
↓
任务成功 / 失败
```

核心指标：

```text
Task Success Rate
```

而不是只看：

```text
Action MAE
```

---

# 40. Simulator Evaluation 注意事项

OpenVLA 官方 LIBERO evaluation 会处理：

```text
cfg.unnorm_key = cfg.task_suite_name
```

如果 checkpoint 包含对应 no-noops normalization statistics，可能 fallback 到：

```text
libero_spatial_no_noops
```

另外 gripper action 有特殊处理：

```text
[0,1]
→
[-1,+1]
→
invert_gripper_action()
```

因此：

> TFRecord 原始 action 与 simulator 最终执行 action 不能简单逐维直接比较。

---

# 41. 当前 100-sample 脚本的限制

目前的 100-sample 脚本不是严格意义上的全数据集均匀随机采样。

实际更接近：

```text
读取前面的候选 samples
↓
收集 100 个
↓
seed=42 shuffle
```

所以是：

```text
shuffled subset
```

而不是：

```text
globally uniform random 100 samples
```

以后做严谨 benchmark 时需要重新设计采样方式，或者直接使用 LIBERO simulator evaluation。

---

# 42. 重要文件路径

### OpenVLA

```text
~/openvla
```

### finetune.py

```text
~/openvla/vla-scripts/finetune.py
```

### finetune.py 备份

```text
~/openvla/vla-scripts/finetune.py.bak_single_gpu
```

### Dataset

```text
~/modified_libero_rlds
```

### Model cache

```text
~/.cache/huggingface/hub/models--openvla--openvla-7b
```

### 100-step output

```text
~/openvla/runs/libero_spatial_100step
```

### 100-step temporary adapter

```text
~/openvla/adapter-tmp/libero_spatial_100step
```

### 1000-step output

```text
~/openvla/runs/libero_spatial_1000step
```

### 1000-step temporary adapter

```text
~/openvla/adapter-tmp/libero_spatial_1000step
```

---

# 43. 已验证成功的能力

```text
OpenVLA-7B 4-bit inference                 ✅
Blackwell / RTX 5070 Ti compatibility     ✅
bitsandbytes 4-bit                         ✅
PEFT LoRA                                  ✅
QLoRA forward                              ✅
QLoRA backward                             ✅
QLoRA optimizer step                       ✅
Real OpenVLA action-token loss             ✅
All-linear LoRA                            ✅
Gradient checkpointing                    ✅
LIBERO RLDS loading                        ✅
LIBERO Spatial dataset                    ✅
Single-GPU finetune                        ✅
100-step training                          ✅
100-step checkpoint                        ✅
Checkpoint 4-bit reload                    ✅
100-sample offline comparison              ✅
1000-step training initialization          ✅
```

---

# 44. 尚未完成

```text
1000-step QLoRA training                   🔄
1000-step checkpoint                       ⏳
1000-step offline evaluation               ⏳
100 vs 1000 step comparison                ⏳
LIBERO simulator setup                     ⏳
LIBERO Spatial task success evaluation     ⏳
是否训练 3000 steps                        ⏳
最终模型选择                               ⏳
```

---

# 45. 不要做的事情

不要：

```text
重新安装 PyTorch
```

不要：

```bash
pip install -r requirements.txt
pip install -e .
```

不要随意升级：

```text
accelerate
transformers
bitsandbytes
peft
```

不要同时改变：

```text
LoRA rank
learning rate
batch size
quantization
dataset
```

当前配置已经经过实际训练验证。

---

# 46. 当前实验路线

```text
OpenVLA-7B
    ↓
4-bit NF4
    ↓
LoRA rank=32
    ↓
LIBERO Spatial No-Noops
    ↓
100 steps ────────────────→ 已完成
    ↓
100-sample comparison
    ↓
1000 steps ───────────────→ 当前
    ↓
1000-step evaluation
    ↓
比较 100 / 1000
    ↓
LIBERO simulator evaluation
    ↓
Success Rate
    ↓
决定是否 3000 steps
```

---

# 47. 核心实验配置速查

```text
GPU:
RTX 5070 Ti Laptop
~11.47 GiB VRAM

Model:
openvla/openvla-7b

Quantization:
4-bit NF4

Compute dtype:
BF16

LoRA:
all-linear

LoRA rank:
32

LoRA alpha:
16

LoRA dropout:
0.0（正式 finetune 当前配置）

Batch:
2

Gradient accumulation:
8

Effective batch:
16

Learning rate:
5e-4

Gradient checkpointing:
enabled

Dataset:
libero_spatial_no_noops

Dataset size:
52970 transformed samples

Episodes:
432

Shards:
16

Previous:
100 steps completed

Current:
1000 steps training

Next:
1000-step evaluation

Final:
LIBERO simulator success rate
```

---

# 48. 下一步操作

如果 1000-step 训练仍在运行：

```text
不要重新启动
不要修改参数
不要关闭训练进程
```

训练结束后提供最后约 30～50 行终端输出。

下一步：

```text
验证 1000-step checkpoint
↓
加载原始 OpenVLA
↓
加载 1000-step 模型
↓
同一批 LIBERO 样本评估
↓
比较 100 / 1000 steps
↓
决定是否 3000 steps
```

---

# END OF CONTEXT
