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
- 官方完整模型对照：固定 revision `962318cec55ac10993ff0f5f43eda9a270b4c873`，14个文件（15,085,047,523 bytes）已下载并逐文件核验大小，见 `results/official_checkpoint_manifest.json`。这是下载完成证据，不是性能结果。准备运行时 Fast3R 再次占用约8GiB显存，因此按用户要求等待，不抢占GPU。加载完整模型和adapter均显式使用NF4/BF16计算/double quant=False，避免FP4默认值混入控制组。
- 10-update热启动暂缓，优先完成官方对照；保存新checkpoint需另有2GiB磁盘余量。原始HDF5演示及完整sim初始状态在本机未找到，演示回放尚未完成，不用TFDS中的末端状态冒充完整sim状态。
