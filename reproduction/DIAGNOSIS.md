## 25. 12GB 单卡 QLoRA：纠错与分阶段继续（2026-10-01）

### 已确认的问题与未证实的推断

| 位置 | 证据 | 影响与边界 |
|---|---|---|
| 旧离线 GT 评测 | policy gripper 0/1 与 raw gripper +1/-1 直接相减 | 旧误差结论失效；不在 simulator 路径上，不能解释其 1.3% |
| 旧视觉对比 | generate 未传 pixel_values | 不能证明视觉条件下的微调效果；已修复 |
| 梯度累积循环 | 按微批次保存、最终更新后额外 backward | 重复保存、记录不清；已修复，尚无导致低分的证据 |
| 训练规模与量化 | 本地 3500 × 16 / NF4；官方 50000 × 128 / 无量化 | 学习不足和量化损失是高优先级假设，不是已验证因果结论 |
| 训练/评测 double quant | 训练 False，旧部署 True | 新诊断显式使用 False；单帧无差别不足以排除全局影响 |
| 失败动作形态 | 部分视频有停滞、抓起后未正确放下 | 怀疑运动精度、偏离演示后的恢复或释放时机；需动作 trace 核验 |

### 实验安排

1. `diagnose_qlora.py`：每个 episode 固定起始、中间、末段非终止帧，比较 base / adapter / 不裁剪 adapter，并对相同裁剪画面计算 teacher-forcing 指标。保存动作、图像 hash、动作空间说明和 GPU 峰值。全量模型的训练帧诊断不是泛化测评，也不是任务成功率。
2. 在小数据 adapter 的 episode 12–15 重测 corrected held-out 指标。固定 12 帧，不是旧随机 50 帧，二者不可当作同一实验比较。原归一化统计来自全量数据，因此不是严格完全隔离的验证流程。
3. 从 3500-step adapter **权重热启动**：新目录、新优化器、lr=1e-4，10 次优化更新（80 微批次）。这是训练/保存链路验证，不是模型改进证据，不是精确断点续训；核对 `training_state.json`。
4. 同一帧集合重测 pilot。完成后审核，再设计学习率/时长的受控实验，不自动启动长训练。10 步指标的波动不能证明因果关系或策略改善。
5. 闭环入口支持 `--task_ids '[0]' --num_trials_per_task 2 --trace_actions True --fail_fast True --bnb_double_quant False`：记录每步动作、末端位置和夹爪状态，异常立即报告。默认完整评测协议不变。先核验两个固定初始状态，失败后按停滞、抓取、搬运、释放阶段定位，再决定是否扩大到每任务少量 rollout。

### 如何读诊断

- 训练帧 teacher-forcing 与自回归都差：优先检查学习量、目标动作/图像配对、量化加载和权重。
- teacher-forcing 好但自回归差：优先查解码/提示格式、前序动作 token 依赖和误差累积；不直接等同于训练不足。
- 离线动作好但闭环差：优先查图像/相机与仿真接口、动作频率、精确放置和偏离演示轨迹后的恢复。
- 不裁剪和裁剪差异显著：提示图像预处理敏感，但少量帧不足以证明应该关闭裁剪；需闭环受控对照。

运行阶段 1：

```bash
# 默认只打印计划，不占 GPU、不训练
python reproduction/run_diagnostic_stage.py
# Fast3R 结束、GPU 空闲后才运行；不会终止其他 GPU 进程
python reproduction/run_diagnostic_stage.py --execute
```

CPU 回归测试：`python -m unittest reproduction.test_action_metrics -v`，5 项通过（夹爪往返、旧指标反例、运动裁剪、量化夹爪端点与反归一化）。首次 GPU 诊断在加载时 OOM，Fast3R 当时占约 5.7GB 显存；这不是策略失败证据。按用户要求保留 Fast3R，等待完成。结果未产生时明确标记 pending。

官方配方：[LIBERO-Spatial model card](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial)。官方建议核对训练画面上的部署预测及演示回放：[performance troubleshooting](https://github.com/openvla/openvla#vla-performance-troubleshooting)。

## 首轮实测结果

- 全量 adapter：24 张固定训练帧，六维运动 MAE 0.085184，归一化连续目标 L1 0.163627，夹爪方向正确 24/24；teacher-forcing token accuracy 36.31%，量化 token 目标 L1 0.117180。裁剪比不裁剪的离线 L1 更低，但不能据此推出闭环因果关系。
- 小数据 adapter：12 张未用于训练的固定帧，六维运动 MAE 0.149595、归一化 L1 0.267444、夹爪正确 9/12。不是旧随机50帧实验，不能恢复旧19.2%指标。
- 本地/官方动作反归一化 q99 和 mask 完全一致，q01 最大差约5.4e-9。暂未发现明显统计范围错误。
- task0 的两个固定状态闭环：0/2，无运行异常，各220个动作。99.55–100% 的平移命令范数 <0.02，夹爪始终张开，末端离起点最大距离2.94/3.65毫米。失败在起始停滞，尚未进入抓取阶段。
- 训练帧起始状态也有同类现象：8帧中7帧预测相同近零平移，而GT平移并不全为零。说明停滞不只出现在simulator分布中；仍不能证明是训练量、量化或优化设置导致。
- 官方完整模型对照：固定 revision `962318cec55ac10993ff0f5f43eda9a270b4c873`，14个文件（15,085,047,523 bytes）已下载并逐文件核验大小，见 `results/official_checkpoint_manifest.json`。Fast3R结束后已运行，完整模型和adapter均显式使用NF4/BF16计算/double quant=False，避免FP4默认值混入控制组。
- 10-update热启动暂缓，优先完成官方对照；保存新checkpoint需另有2GiB磁盘余量。原始HDF5演示及完整sim初始状态在本机未找到，演示回放尚未完成，不用TFDS中的末端状态冒充完整sim状态。

## 官方 NF4 对照：固定训练画面（2026-10-01）

`results/official_diagnostic_v1.json` 已完成。与本地全量诊断逐项核验：24张图像SHA256、episode/step、指令、原始目标动作及量化配置全部相同。这些是训练分布内的诊断帧，不是泛化或任务成功率。

| 指标 | 本地3500-step adapter | 官方完整checkpoint，NF4部署 |
|---|---:|---:|
| 六维运动 MAE | 0.085184 | 0.013161 |
| 连续归一化七维 L1 | 0.163627 | 0.028775 |
| 夹爪方向正确数 | 24/24 | 24/24 |
| teacher-forcing token accuracy | 36.31% | 94.05% |
| teacher-forcing decoded-token L1 | 0.117180 | 0.006956 |

官方模型不裁剪时连续L1为0.087103，高于同模型裁剪的0.028775；不能据此推广到整个评测分布。官方PyTorch峰值allocated为4.918GiB，诊断计算28.08秒（不含模型加载），不是训练峰值或系统总显存。

官方模型在起始帧上没有复现本地7/8样本的相同近零平移输出。这削弱了“NF4部署本身必然让模型停滞”的解释，问题更集中于本地训练后的策略；并不证明QLoRA训练等价于官方训练，也不能排除量化训练、训练量、优化参数或其他训练链路的影响。

## 官方 NF4 对照：task0 两个固定状态

`results/official_task0_trace_summary.json`：成功 **1/2**，不是整个suite的50%成功率估计。两次无运行异常；与本地trace的task、指令、trial索引、seed=7、center crop=True和double quant=False一致，首次策略查询时的末端位置和夹爪qpos逐值相同。这是可观察状态核验，不是完整MuJoCo状态的独立hash核验。

| 模型 / 初始状态 | 成功 | 动作数 | 近零平移命令比例 | 末端最大位移 | 夹爪命令切换 |
|---|---|---:|---:|---:|---:|
| 本地 / 0 | 否 | 220 | 100.00% | 2.94mm | 0 |
| 官方NF4 / 0 | 是 | 80 | 3.75% | 383.40mm | 2 |
| 本地 / 1 | 否 | 220 | 99.55% | 3.65mm | 0 |
| 官方NF4 / 1 | 否 | 220 | 88.64% | 15.01mm | 0 |

官方在状态0成功，说明当前NF4/提示/相机/动作接口组合**至少能完成一次此任务**，不支持“共同部署链路必然失败”的解释。状态1两模型仍停滞，不能宣布共同环境、预处理或量化影响已全部排除。没有同官方模型的BF16对照，无法测量NF4性能损失；推理量化对照也不能证明QLoRA量化训练没有影响。

下一步诊断优先级：核查本地训练数据配对、监督token、优化更新和学习量，并单独检查状态1的图像与初始状态敏感性；每次仅改变一个因素。原始HDF5演示回放与10-update保存链路验证仍未完成，系统盘约2GiB空余，后者继续暂缓。此轮有界官方对照已完成，不自动扩大到长训练或全套评测。

## 训练链路体检：CPU阶段（2026-10-01）

新增可重复执行的 `audit_training_chain.py`，真实输出为 `results/training_chain_cpu_audit_v1.json`。没有加载GPU模型、修改权重或执行优化更新。当前Fast3R作业占约8.4GiB显存，遵照用户要求不抢占。

- 前8条TFDS演示共951帧，经生产标准化、BOUNDS_Q99归一化、goal relabeling、window_size=1 / future_action_window_size=0、解码与resize后，与直接读取的同一TFDS step逐项对照：图像像素、指令和动作索引一致，归一化最大误差1.83e-7。
- 24个固定阶段帧与前述诊断hash相同。每帧7个动作token和1个EOS，提示词部分屏蔽；动作token与独立离散化计算一致，最大量化解码误差0.003922。collator没有截断标签，padding不参与loss。
- 相同图像下，训练image_transform与推理processor的pixel_values逐值一致；补上生产predict_action的29871空白token后，推理提示前缀与训练提示前缀一致。此检查禁用随机增强，不能替代历史增强画面的核验。
- checkpoint共879个张量均finite：878个LoRA张量，参数量110,828,288；另含完整lm_head，PEFT因为target_modules包含lm_head会自动保存该层。其FP32权重与基础模型逐值相同，不是基础输出层被误训练的证据。导出配置inference_mode=True是保存行为，热启动使用is_trainable=True；字段本身不是冻结训练的证据。
- 439个LoRA B中426个非零，13个全零仅位于DINO最后block23、SigLIP最后block26及attention pool。当前模型返回倒数第二层patch特征，末层和池化不参与该输出，因而这些零矩阵不直接说明有效分支没有学习。非零矩阵也不证明历史优化轨迹正确。
- 951帧中5帧原始平移范数<0.02；当前训练包含8个terminal帧。样本范围有限，既不证明全局空闲动作比例，也不证明terminal一定是无效占位标签。

**第一步尚未全部完成：** runtime LoRA梯度是否finite/非零、一次真实optimizer update前后参数变化、保存后重新加载的预测一致性仍pending。不能用旧 `test_qlora_backward.py` 代替：该脚本把提示词input_ids直接当labels，自认是假样本，未验证真实机器人动作监督。原始HDF5配对也未验证；本次仅证明“存储TFDS → 当前生产转换”在上述样本一致。GPU空闲且有足够保存余量后，才在独立临时产物中做有界验证，保留原checkpoint。没有发现新的已证实低成功率根因。
