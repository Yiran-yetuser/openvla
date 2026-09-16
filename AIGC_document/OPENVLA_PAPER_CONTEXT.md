# OpenVLA 项目 — 论文讨论 Context

> 用途：将本窗口已经讨论过的 OpenVLA 论文理解内容，整理为可交给新窗口 / Agent 的项目上下文。
>
> **范围约定：**
> - 本文件主要记录“论文理解”相关内容。
> - 工程复现（CUDA、PyTorch、bitsandbytes、LoRA/QLoRA、训练命令、GPU 显存等）应放到单独的 reproduction context。
> - 本文件区分“论文内容”和“为了理解而做的解释/抽象”，避免 Agent 把解释误认为论文原文。

---

## 1. 项目基本信息

### 论文

**OpenVLA: An Open-Source Vision-Language-Action Model for Robotic Manipulation**

核心研究对象：OpenVLA，一个面向机器人操作（robotic manipulation）的开源 Vision-Language-Action（VLA）模型。

### 当前学习目标

目前主要目标是：

1. 理解 OpenVLA 为什么被设计成这样；
2. 理解其 VLA / VLM / robot policy 的关系；
3. 理解训练数据 Open X-Embodiment（OXE）；
4. 理解连续机器人动作如何被离散化并转换成 token；
5. 理解 OpenVLA 的训练目标；
6. 后续继续阅读论文的模型结构、设计决策、实验和 ablation。

---

# 2. VLA（Vision-Language-Action）基本概念

## 2.1 什么是 VLA

VLA = **Vision-Language-Action**

最简单的抽象：

```text
Image + Language Instruction
              ↓
             VLA
              ↓
           Action
```

即：

> 输入视觉观察和自然语言任务描述，模型预测机器人应该执行的动作。

与普通 Vision-Language Model（VLM）相比，VLA 不仅需要理解“看到了什么”和“应该做什么”，还要进一步输出可以控制机器人的 action。

---

## 2.2 与传统 Robot Policy 的区别

传统机器人策略通常更加针对：

- 特定机器人
- 特定任务
- 特定环境
- 特定 action space

例如：

```text
Camera Image
     ↓
Robot Policy
     ↓
Robot Action
```

而 VLA 希望利用大型视觉-语言预训练模型已经获得的知识：

```text
Internet-scale Vision + Language Knowledge
                    ↓
             Pretrained VLM
                    ↓
        Robot Demonstration Data
                    ↓
                 VLA
                    ↓
                Action
```

因此 VLA 的一个核心思想是：

> 将已有的大规模视觉-语言知识迁移到机器人控制任务。

---

# 3. Related Work 中的 VLA 发展脉络

之前重点讨论过以下工作之间的关系。

## 3.1 RT-2

RT-2 可以理解为：

> 将 Vision-Language Model 扩展到机器人控制。

核心思想：

```text
Web-scale VLM knowledge
          +
Robot data
          ↓
Robot-capable VLM
```

一个重要思想是：

> 把机器人动作也作为模型可以生成的 token / 输出序列来处理。

这为后续 VLA 的发展提供了重要思路。

---

## 3.2 RT-2-X / Open X-Embodiment

进一步的发展方向是：

> 不希望机器人模型只针对一种机器人，而希望利用来自不同机器人、不同任务的数据进行训练。

因此出现了多机器人、多任务的数据和 generalist robot policy 方向。

---

## 3.3 OpenVLA 的定位

OpenVLA 的核心特点：

- 开源；
- 基于 pretrained VLM；
- 使用大规模机器人 demonstration data；
- 支持多机器人 / 多任务；
- 将 robot action 转换为 token；
- 使用语言模型式的 next-token prediction 进行训练；
- 关注 efficient fine-tuning。

可以粗略理解成：

```text
Pretrained VLM
      +
Large-scale robot demonstrations
      ↓
    OpenVLA
      ↓
General-purpose robot manipulation policy
```

---

# 4. Generalist Robot Policy 与 VLA

之前讨论过 Octo 等 generalist robot policy。

重要理解：

> “Generalist robot policy”和“VLA”不是完全相同的概念。

Generalist robot policy 强调：

- 一个模型；
- 多机器人；
- 多任务；
- 多环境。

VLA 则更强调：

- Vision；
- Language；
- Action；
- 以及将视觉语言模型能力连接到机器人动作预测。

OpenVLA 的特点在于：

> 直接以 pretrained VLM 为基础，并将机器人动作表示成模型能够生成的 token。

---

# 5. Section 3.2 — OpenVLA Training Procedure

这一节是目前已经重点理解过的部分。

## 5.1 整体训练流程

论文中的基本思想可以抽象为：

```text
Pretrained VLM backbone
          +
Robot demonstration trajectories
          ↓
      OpenVLA
```

论文使用约 **970k robot trajectories** 进行训练。

---

# 6. Robot Trajectory 是什么

一个 trajectory / episode 不是单独的一张图片。

它是一串连续的 time steps：

```text
Episode
│
├── Step 0
│    ├── Observation
│    └── Action
│
├── Step 1
│    ├── Observation
│    └── Action
│
├── Step 2
│    ├── Observation
│    └── Action
│
└── ...
```

可以概念化为：

\[
(o_t, l, a_t)
\]

其中：

- \(o_t\)：robot observation / visual observation
- \(l\)：language instruction
- \(a_t\)：robot action

注意：

> 这里是帮助理解的抽象形式，不应直接当成 OXE 所有 dataset 的精确字段 schema。

---

# 7. Robot Observation

一个 observation 可能包含：

- RGB camera image；
- robot proprioception / state；
- 其他具体机器人数据集提供的 observation。

不同机器人 / dataset 的 observation 可能不同。

因此不能简单认为所有 OXE 数据都具有完全相同的 observation 字段。

---

# 8. Robot Action

在 OpenVLA 的典型理解中，robot action 可以表示为 7D：

\[
a_t =
[
\Delta x,
\Delta y,
\Delta z,
\Delta r_x,
\Delta r_y,
\Delta r_z,
gripper
]
\]

也就是：

1. x 方向位置变化；
2. y 方向位置变化；
3. z 方向位置变化；
4. x 方向旋转变化；
5. y 方向旋转变化；
6. z 方向旋转变化；
7. gripper 状态 / 控制。

### 注意

这里是 OpenVLA 中常见 action representation 的概念解释。

不同原始机器人数据集的 action representation 不一定天然完全相同。

---

# 9. 为什么要把连续 Action 离散化

这是 OpenVLA 非常重要的设计。

机器人动作原本是连续值：

```text
Δx = 0.013
Δy = -0.027
Δz = 0.004
...
```

但是 Llama 本身是一个 language model。

语言模型天然擅长做：

```text
token → token → token → ...
```

因此 OpenVLA 采取：

```text
Continuous Action
      ↓
Discretization
      ↓
Integer 0 ... 255
      ↓
Action Token
```

---

# 10. Action Discretization

对于一个 N-dimensional action：

\[
a \in \mathbb{R}^{N}
\]

OpenVLA 将每一个 action dimension 离散成 **256 个 bins**。

因此：

\[
a_i
\rightarrow
k_i,\quad k_i\in\{0,\ldots,255\}
\]

最终：

\[
N\text{-dimensional continuous action}
\rightarrow
N\text{ discrete integers}
\]

每个整数范围：

\[
[0,255]
\]

---

# 11. 为什么使用 1%-99% Quantile Range

之前讨论过论文中的这一设计。

如果直接使用 action 数据的绝对最小值和最大值：

```text
min ---------------------------- max
```

极端 outlier 可能占据很大的范围。

这样会导致：

> 256 个 bins 被迫覆盖非常大的数值区间，而大多数正常 action 的有效分辨率下降。

因此使用约：

```text
1% quantile ---------------- 99% quantile
```

作为主要离散化范围。

目的：

> 减少 outlier 对 action discretization resolution 的影响。

---

# 12. Action Tokenization

这是 Section 3.2 中非常关键的一部分。

## 12.1 问题

经过离散化之后，一个 N-dimensional action 得到 N 个整数：

```text
[23, 147, 92, 201, 17, 128, 255]
```

每个整数属于：

```text
0 ... 255
```

因此需要 **256 个 action tokens**。

---

# 13. Llama Tokenizer 的问题

OpenVLA 的 language backbone 使用 Llama tokenizer。

论文指出：

> Llama tokenizer 只为 fine-tuning 时新加入的 token 预留了 100 个 special tokens。

但是 OpenVLA 需要：

```text
256 action tokens
```

所以：

```text
100 reserved special tokens
        <
256 action tokens
```

不够。

---

# 14. OpenVLA 如何解决

OpenVLA 没有继续简单地增加 tokenizer 的 special tokens。

而是采取了一个简单的方法：

> 覆盖 Llama vocabulary 中最少使用的 256 个 tokens。

论文说明：

> 这对应 vocabulary 的最后 256 个 tokens。

概念上可以理解成：

```text
原始 Llama vocabulary

...
token_x
token_y
token_z
[最后 256 个低频 tokens]
        ↓
    覆盖掉
        ↓
[action_0]
[action_1]
[action_2]
...
[action_255]
```

---

# 15. 为什么覆盖最后 256 个 Token

核心原因：

> 这些 token 本身是 vocabulary 中最少使用的 token。

因此将它们改造成 action tokens，可以尽可能减少对原始语言能力的影响。

这体现了 OpenVLA 的一个设计取向：

> 尽可能简单地把 robot action 接入已有语言模型。

---

# 16. 一个完整 Action Tokenization 示例

假设：

```text
连续 action：

[
  0.013,
 -0.027,
  0.004,
  ...
]
```

经过 discretization：

```text
[
  23,
  147,
  92,
  ...
]
```

再经过 action token mapping：

```text
[
  <action_23>,
  <action_147>,
  <action_92>,
  ...
]
```

于是语言模型看到的就不再是连续浮点数，而是一串 token。

因此：

```text
Image
  +
Language Instruction
  +
Context
  ↓
VLM
  ↓
Action Tokens
  ↓
Continuous Robot Action
```

---

# 17. OpenVLA 的训练目标

经过 action tokenization 后：

> OpenVLA 可以使用标准的 next-token prediction objective。

即类似语言模型：

```text
token_1
  ↓
token_2
  ↓
token_3
  ↓
...
```

模型预测下一个 token。

但是有一个关键区别：

> Cross-Entropy loss 只计算 action token 的位置。

---

# 18. 为什么只对 Action Tokens 计算 Loss

训练序列中可能同时包含：

- image information；
- language instruction；
- context；
- action tokens。

但 OpenVLA 的训练目标是学习：

> 根据 observation + language instruction 预测机器人动作。

因此真正需要优化的是：

```text
Action Token Prediction
```

概念上：

\[
\mathcal{L}
=
-\sum_{t\in action}
\log p(a_t|context)
\]

而不是让普通语言 token 也承担同样的训练 loss。

---

# 19. 这件事非常重要：OpenVLA 本质上仍然是 Token Prediction

可以把 OpenVLA 的一个核心思想总结成：

```text
传统 Robot Policy：

Observation
     ↓
Neural Network
     ↓
Continuous Action


OpenVLA：

Image + Language
       ↓
   Pretrained VLM
       ↓
  Action Tokens
       ↓
  Discrete Actions
       ↓
Continuous Robot Action
```

所以它不是单独设计一个完全不同的机器人控制网络，而是：

> **把 robot action 转换成 language-model 能预测的 token。**

---

# 20. Section 3.3 — Training Data

OpenVLA 使用的核心训练数据来源：

## Open X-Embodiment（OXE）

规模约：

**970k robot trajectories**

主要特点：

- 多机器人；
- 多任务；
- 多环境；
- 多种 robot embodiment；
- 大规模 demonstration trajectories。

---

# 21. 为什么需要 Open X-Embodiment

如果只使用一个机器人：

```text
Robot A
Task 1
Task 2
Task 3
```

模型很容易过度适应：

- Robot A；
- 特定 camera；
- 特定 workspace；
- 特定 task distribution。

而 OXE 提供：

```text
Robot A ─┐
Robot B ─┤
Robot C ─┤
Robot D ─┤
         ├──> Large-scale robot data
Robot E ─┤
Robot F ─┘
```

因此模型可以学习更加一般性的：

- object interaction；
- manipulation patterns；
- visual grounding；
- language-conditioned behavior。

---

# 22. OXE 数据的概念结构

之前我们用以下抽象理解 OXE：

```text
Episode
│
├── Step 0
│   ├── Observation
│   ├── Language
│   └── Action
│
├── Step 1
│   ├── Observation
│   ├── Language
│   └── Action
│
├── Step 2
│   ├── Observation
│   ├── Language
│   └── Action
│
└── ...
```

可以进一步概念化：

\[
\tau =
\{(o_0,l,a_0),
(o_1,l,a_1),
\ldots,
(o_T,l,a_T)\}
\]

这里：

- \(\tau\)：trajectory；
- \(o_t\)：observation；
- \(l\)：language instruction；
- \(a_t\)：action。

---

# 23. OXE 与普通 Image Dataset 的区别

可以这样对比：

### ImageNet

```text
Image
  ↓
Label
```

### OXE

```text
Robot Episode
     ↓
Time sequence
     ↓
Observation + Language + Action
     ↓
Robot behavior
```

因此：

> OXE 的核心不是“图片分类”，而是机器人 demonstration trajectory。

---

# 24. 需要注意：OXE 精确 Schema 尚未在本窗口完全展开

之前关于 OXE 数据结构的解释主要是**概念层面的抽象**。

如果后续需要回答：

- RLDS 的精确字段；
- episode / step 的实际 JSON / tensor structure；
- image 的具体 key；
- language 的具体 key；
- action tensor shape；
- 不同 OXE dataset 的 schema；
- OpenVLA dataloader 如何读取这些数据；

应该进一步查看：

1. Open X-Embodiment 官方数据说明；
2. OpenVLA 官方代码；
3. RLDS 数据格式。

不要把前面的概念化结构直接当成官方完整 schema。

---

# 25. 当前已经建立的核心知识链

目前对 OpenVLA 的理解可以串成：

```text
                 Internet-scale knowledge
                          │
                          ▼
                  Pretrained VLM
                          │
                          │
             + Robot Demonstrations
                          │
                          ▼
                       OpenVLA
                          │
              ┌───────────┴───────────┐
              │                       │
            Image                 Language
              │                       │
              └───────────┬───────────┘
                          ▼
                    VLM reasoning
                          │
                          ▼
                Discrete Action Tokens
                          │
                     0 ... 255
                          │
                          ▼
                 Continuous Actions
                          │
                          ▼
                       Robot
```

---

# 26. 当前最重要的几个概念

如果后续继续阅读论文，应该始终记住下面几个核心点：

### ① VLA

\[
Vision + Language \rightarrow Action
\]

---

### ② OXE

大规模、多机器人、多任务 demonstration trajectories。

---

### ③ Action Discretization

\[
Continuous\ Action
\rightarrow
256\ bins
\]

---

### ④ Action Tokenization

\[
0...255
\rightarrow
256\ Action\ Tokens
\]

---

### ⑤ Llama Vocabulary 修改

由于原本只预留 100 个 special tokens：

```text
100 < 256
```

所以覆盖最少使用的最后 256 个 vocabulary tokens。

---

### ⑥ Training Objective

使用标准 next-token prediction：

\[
P(token_{t+1}|token_{\leq t})
\]

但是：

> CE loss 只计算 action token positions。

---

# 27. 当前论文理解进度

目前已经讨论：

```text
Related Work
    │
    ├── VLA
    ├── RT-2
    ├── RT-2-X
    ├── Open X-Embodiment
    └── Generalist Robot Policy / Octo
             │
             ▼
Section 3
    │
    ├── 3.2 OpenVLA Training Procedure
    │      ├── robot trajectory
    │      ├── action representation
    │      ├── action discretization
    │      ├── action tokens
    │      ├── Llama tokenizer
    │      └── next-token prediction
    │
    └── 3.3 Training Data
           └── Open X-Embodiment
```

---

# 28. 尚未深入讨论的论文内容

后续论文窗口可以继续按照论文顺序深入：

- Section 3.4 中的关键设计决策；
- OpenVLA 模型架构；
- vision encoder；
- language backbone；
- projector / multimodal fusion；
- action prediction；
- inference procedure；
- data preprocessing；
- training details；
- experiments；
- LIBERO；
- real-world evaluation；
- ablation studies；
- 与其他 VLA / robot policy 的比较；
- scaling / fine-tuning 结果；
- 论文 limitations。

---

# 29. “论文内容”和“复现内容”的边界

## 本文件负责

```text
论文：
├── 为什么这样设计
├── 模型结构
├── 训练目标
├── 数据
├── action tokenization
├── 实验
├── ablation
└── 论文结论
```

## 另一个 reproduction context 负责

```text
工程：
├── Ubuntu
├── NVIDIA Driver
├── CUDA
├── PyTorch
├── Transformers
├── bitsandbytes
├── OpenVLA GitHub
├── 4-bit inference
├── LoRA / QLoRA
├── GPU memory
├── Dataset download
├── LIBERO
├── Training
└── Evaluation
```

---

# 30. 给后续 Agent 的工作原则

如果一个新的 Agent 读取本文件，请遵循：

1. 不要把本文件中的“概念抽象”自动当成论文的逐字原文。
2. 如果用户问“论文到底怎么写的”，应优先以论文原文为准。
3. 如果论文原文与这里的解释存在差异，应以论文原文为准，并明确指出差异。
4. 不要自行补全 OXE 的精确 schema。
5. 如果讨论最新 OpenVLA 官方实现，则需要另外查询当前官方代码。
6. 论文理解和工程复现应保持分离。
7. 用户希望逐步理解 OpenVLA，而不是只获得结论，因此解释时优先说明“为什么”。

---

# 31. 一句话总结

OpenVLA 的核心可以压缩成：

> **以 pretrained VLM 为基础，用大规模多机器人 demonstration data 训练，将连续 robot action 离散成 256 个 bins，再把这些离散 action 映射成 token，使机器人动作预测可以直接转化为语言模型式的 next-token prediction，并只在 action token 上计算训练 loss。**

