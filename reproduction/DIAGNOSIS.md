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

## 训练链路体检：CPU阶段（2026-10-01，当时状态快照；运行时结果见下节）

新增可重复执行的 `audit_training_chain.py`，真实输出为 `results/training_chain_cpu_audit_v1.json`。没有加载GPU模型、修改权重或执行优化更新。当前Fast3R作业占约8.4GiB显存，遵照用户要求不抢占。

- 前8条TFDS演示共951帧，经生产标准化、BOUNDS_Q99归一化、goal relabeling、window_size=1 / future_action_window_size=0、解码与resize后，与直接读取的同一TFDS step逐项对照：图像像素、指令和动作索引一致，归一化最大误差1.83e-7。
- 24个固定阶段帧与前述诊断hash相同。每帧7个动作token和1个EOS，提示词部分屏蔽；动作token与独立离散化计算一致，最大量化解码误差0.003922。collator没有截断标签，padding不参与loss。
- 相同图像下，训练image_transform与推理processor的pixel_values逐值一致；补上生产predict_action的29871空白token后，推理提示前缀与训练提示前缀一致。此检查禁用随机增强，不能替代历史增强画面的核验。
- checkpoint共879个张量均finite：878个LoRA张量，参数量110,828,288；另含完整lm_head，PEFT因为target_modules包含lm_head会自动保存该层。其FP32权重与基础模型逐值相同，不是基础输出层被误训练的证据。导出配置inference_mode=True是保存行为，热启动使用is_trainable=True；字段本身不是冻结训练的证据。
- 439个LoRA B中426个非零，13个全零仅位于DINO最后block23、SigLIP最后block26及attention pool。当前模型返回倒数第二层patch特征，末层和池化不参与该输出，因而这些零矩阵不直接说明有效分支没有学习。非零矩阵也不证明历史优化轨迹正确。
- 951帧中5帧原始平移范数<0.02；当前训练包含8个terminal帧。样本范围有限，既不证明全局空闲动作比例，也不证明terminal一定是无效占位标签。

**第一步尚未全部完成：** runtime LoRA梯度是否finite/非零、一次真实optimizer update前后参数变化、保存后重新加载的预测一致性仍pending。不能用旧 `test_qlora_backward.py` 代替：该脚本把提示词input_ids直接当labels，自认是假样本，未验证真实机器人动作监督。原始HDF5配对也未验证；本次仅证明“存储TFDS → 当前生产转换”在上述样本一致。GPU空闲且有足够保存余量后，才在独立临时产物中做有界验证，保留原checkpoint。没有发现新的已证实低成功率根因。

运行时验证入口已准备：`audit_training_runtime.py` 默认dry-run；`--execute`才会在空闲GPU上执行。固定16帧、microbatch2×累积8、AdamW lr5e-4、仅一次更新，记录每个LoRA矩阵的梯度与参数差。以默认PEFT保存方式写独立临时adapter，分别用相同训练加载器和生产推理加载器比较3个阶段帧的动作；两种加载器的差异不能隐藏在统一PASS中。原始权重/config/stats前后SHA256一致，临时adapter退出时清理。保存前要求现有adapter大小+1GiB保留空间+64MiB辅助文件缓冲，不保存优化器、不合并完整模型。当前只有dry-run和4项安全门控CPU回归测试通过，不能据此声称GPU链路已通过。

## 训练链路体检：运行时已完成（2026-10-01）

GPU空闲后执行，未抢占Fast3R。真实报告 `results/training_chain_runtime_audit_v1.json`，状态 `pass`。此前CPU快照中的前三项runtime待办在本轮完成；原始HDF5配对和历史随机增强仍未验证。

- 16个与CPU体检一致的真实动作帧，无随机增强；batch2×累积8，AdamW lr5e-4，仅1次更新，新优化器，非精确断点续训。
- 8个微批次loss均finite，范围2.262401–3.528154，平均2.819472。878个LoRA张量中852个有finite、非零梯度，且852个更新后发生变化，最大参数绝对差0.000501126。冻结参数均无梯度。
- 其余26个LoRA A/B张量无梯度、无变化，恰对应CPU审查中13个不参与返回patch输出的视觉末层/池化分支；不是全模型没有学习。按张量名称核验，而非忽略全部无梯度情况。
- 3个阶段帧中2个预测在更新后改变；不据此宣称误差减小或策略改进。同训练加载器重载和生产加载器重载，均与保存前更新后的3×7动作逐值一致，生产最大绝对差0。
- 训练准备加载器的预测使用BF16 CUDA autocast；生产加载器保持实际入口的无autocast。比较的是最终离散解码动作，不证明两种加载器内部dtype、logits或所有输入完全相同。
- 训练阶段PyTorch峰值allocated 7.957GiB，不是系统总显存。临时保存971,136,009 bytes，退出后已清理；原adapter/config/stats的SHA256前后不变。没有保存新训练checkpoint或优化器。

第一尝试在更新前失败：体检预测给 `prepare_model_for_kbit_training` 转为FP32的视觉层传入BF16图像却没有autocast，错误 `Input type (c10::BFloat16) and bias type (float) should be the same`。证据 `results/training_chain_runtime_attempt1_error.json`；补上与真实训练一致的混合精度上下文后安全重试通过。它是新体检脚本的错误，不是历史低成功率的根因证据。

**本轮第一步（当前数据→动作监督→反传→一次更新→临时保存重载）完成。** 当前样本不支持“LoRA完全没梯度 / optimizer不更新 / 保存加载丢失adapter”这些解释；不能还原历史训练全过程，也不能推出充分收敛或达到论文成功率。下一轮建议先设计有界少样本拟合验证，查看训练目标能否持续下降及自回归运动是否摆脱近零；本轮不自动启动该实验、长训练或500评测。低成功率唯一根因仍未确定。

## 扩容后继续：有界真实样本拟合（2026-10-02，启动时快照；最终结果见下节）

用户授权继续；磁盘扩容后检查可用约122GB，宿主GPU无compute作业。新增 `fit_small_sample.py`，从原3500-step adapter热启动，固定前16个已核验真实帧，batch2×累积8、AdamW lr1e-4、最多50次更新，无随机增强。0/10/25/50节点测同24帧的teacher-forcing与自回归动作；后8帧只作本轮probe，原全量adapter已见过它们，**不是held-out**。为对齐拟合画面，本轮训练/诊断都不裁剪，不能与之前中心裁剪的指标直接当成同一实验比较。lr1e-4是诊断选择，不是学习率单因素消融。

运行记录位于 `runs/small_fit_v1`（Git忽略）：独立后台worker、原子progress JSON、不可覆盖的完成checkpoint、逐步loss日志。节点保存adapter/optimizer/Torch CPU和CUDA RNG及SHA256；只有完整发布的manifest才能resume，未完成partial保留作为错误证据。最多保留4个完成节点，不覆盖原权重；脚本有空闲GPU、重复worker锁、更新上限、磁盘与hash守卫。最终小报告目标 `results/small_fit_v1.json`，未产生前不填性能结论；生产加载器会另外复测24帧动作，差异单列。

14项CPU回归测试（原9项+5项新拟合/恢复守卫）和Python AST通过；GPU拟合刚启动，**不能把CPU测试通过当作模型已拟合成功**。后台接手说明见 `CONTINUE.md`，每小时在本聊天检查一次，忙或无变化静默。本地Python计算不调用Codex/API；定时接手仍需要机器开机、桌面应用运行和可用额度，不能保证额度刷新时无缝自动恢复，不使用重置券或购买额度。

### 第10步中间证据（不是最终50步结果）

`results/small_fit_milestone10_v1.json` 记录已SHA256核验的0/10节点；24帧episode/step/图像hash与CPU报告相同，原权重hash未变。
16个拟合帧的teacher-forcing token准确率从38.39%升至97.32%，自回归六维运动MAE从0.112703降至0.005097，归一化L1从0.214299降至0.006506；夹爪正确15/16→16/16，近零平移样本9→1。
8个probe帧token准确率均为32.14%，运动MAE仅从0.100045变为0.092145。当前证据支持“该QLoRA/监督/优化组合可以拟合小批真实动作”，不证明泛化或闭环改善，也不能据此将历史失败唯一归因于训练量。50步实验仍在运行，生产loader最终重测待完成。

## small_fit_v1：50步有界拟合完成（2026-10-02）

真实完整结果为 `results/small_fit_v1.json`，状态 `completed`；只读核验记录 `results/small_fit_verification_v1.json`。没有重跑GPU实验。四个完整snapshot的spec、adapter/config/stats/optimizer-RNG/evaluation SHA256均通过；24帧身份与CPU审查一致，逐帧动作指标与分组汇总独立重算通过。50条loss记录连续、每条8个微批次，均finite；原adapter/config/stats SHA256未变。检查点保留在Git忽略的 `runs/small_fit_v1`；未实际执行中断后的GPU恢复，因此不声称resume已端到端验证。

| 更新后节点 | fit16 token准确率 | fit运动MAE | fit归一化L1 | fit近零平移数 | probe8 token准确率 | probe运动MAE | probe归一化L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 38.39% | 0.112703 | 0.214299 | 9 | 32.14% | 0.100045 | 0.166197 |
| 10 | 97.32% | 0.005097 | 0.006506 | 1 | 32.14% | 0.092145 | 0.162852 |
| 25 | 100.00% | 0.001350 | 0.002039 | 0 | 32.14% | 0.087087 | 0.170275 |
| 50 | 100.00% | 0.001350 | 0.002039 | 0 | 32.14% | 0.091907 | 0.177316 |

近零阈值是平移命令范数<0.02，不是机械臂实际位移。第50步fit夹爪正确16/16、probe7/8（起始probe8/8）。fit离散目标已全对，但连续运动MAE仍非零，连续GT与动作token量化解码不能混为一谈。

日志中的平均训练loss在更新1/10/25/50**之前**分别为2.819472、0.283406、0.000280078、0.0000210585；动作表在相应更新**之后**计算。没有单独测量第0步loss，不把更新1的loss冒充第0步。训练阶段PyTorch峰值allocated为9.217GiB，不是系统总显存，已在12GB单卡完成此限定实验。

第50步实际生产loader（无autocast）重载后的24×7自回归动作与训练准备loader（BF16 autocast）逐值一致。**teacher-forcing并非全等：** probe decoded-token L1为训练0.127591、生产0.124930，虽然两者token准确率均32.14%；fit两者token准确率100%、token L1为0。报告只证明这24帧最终离散自回归动作重载一致，不隐藏teacher差异、不宣称全部logits/dtype相同。

**诊断结论：** 当前数据监督、优化和QLoRA组合能把16帧拟合至动作token全对，并摆脱这些帧的近零运动；“该配置根本无法学到动作”不受此证据支持。probe没有对应提高且连续L1后期变差，提示有限样本记忆/跨帧迁移不足，而非已验证泛化。probe也曾被原全量adapter见过，不能称held-out。此结果不能确认历史失败的唯一原因、不能测量4bit训练的损失、不能把lr1e-4当学习率消融；无随机增强/无中心裁剪/新AdamW的联合改变尚未分离。

本轮完成后关闭接手自动化，保留结果、四个checkpoint和所有日志，不启动长训练或500评测。下一步建议用户选择一个单因素有界实验：保持这16帧、同一源adapter与所有设置，只将lr从1e-4改为历史5e-4，使用独立run检查同节点拟合趋势与迁移；它只能比较本实验内的学习率影响，不能直接复原历史训练。真正泛化验证仍需按episode独立划分训练/验证集，不能用这8个probe替代。

## 用户授权继续：lr5e-4 单因素有界对照（2026-10-02，启动快照；最终见下节）

启动 `small_fit_lr5e4_v1`，独立run/result，不从前轮step050继续；同原源adapter与SHA256、同帧/监督/新AdamW/seed7/NF4/BF16/doubleFalse/无增强/无裁剪，仅lr5e-4替代1e-4，仍最多50更新。启动时GPU空闲，磁盘约46GB可用，没有抢占Fast3R。

`results/small_fit_lr5e4_start_v1.json` 已核验第0步完整snapshot全部保护文件hash、原权重hash、spec仅lr不同、24帧全部输出逐值匹配参考。脚本在第一次更新前强制匹配，失败即停止；已进入真实参数更新，但不把进行中的loss当最终结果。原small_fit_v1报告与checkpoint不变。

新增只读 `compare_small_fit_lr.py`，最终校验frame/spec/hash、独立重算动作指标与分组汇总，比较0/10/25/50和生产loader差异；缺少最终报告就报错，不补造。输出独占创建，不覆盖证据。22项CPU测试通过，包括控制组配置/起点匹配、生产loader标识和真实参考摘要反篡改检查；这不是尚在运行的GPU对照已经完成。

已核验第0/10步完整snapshot文件hash、frame身份及原始目标，见 `results/small_fit_lr5e4_milestone10_v1.json`。第10步lr5e-4 fit token准确率100%、运动MAE0.001350（参考lr1e-4同节点97.32%、0.005097）；probe33.93%、0.070397（参考32.14%、0.092145）。在这个中间节点较大学习率更快拟合，不支持“小样本中5e-4必然学不会”的解释；不是最终对照，更不是历史全量训练或独立泛化结论。

仅一seed/每lr一run，最多只能判断固定样本在该对照内的敏感性；不能证明历史5e-4训练必然有问题，也不能推断泛化/任务成功率。8probe曾被原adapter见过，不能标held-out。最终报告尚待 `results/small_fit_lr5e4_v1.json`，真实比较输出目标 `results/small_fit_lr_comparison_v1.json`。新每小时接手heartbeat已创建，沿用额度边界、静默策略；收尾删除，不自动扩大实验。

## 学习率对照最终完成：两组50更新，证据已核验（2026-10-02）

真实报告 `results/small_fit_lr5e4_v1.json` 状态completed；只读完整比较 `results/small_fit_lr_comparison_v1.json` 状态verified_comparison、local_snapshot_verification=True。没有新GPU运行。两组spec除lr外全部一致，第0步完整结果一致；24帧身份/真实目标、逐帧指标与摘要独立重算，8个完整snapshot的保护文件SHA256及源adapter/config/stats通过。每组一个loss日志、50条连续更新、每条8微批次均finite；存储比较与重新计算的只读比较一致。保留所有原始产物，不覆盖此前报告，不删除模型数据。

| 更新后节点 | fit token准确率1e-4 / 5e-4 | probe token准确率1e-4 / 5e-4 | probe运动MAE 1e-4 / 5e-4 | probe归一化L1 1e-4 / 5e-4 |
|---|---|---|---|---|
| 0 | 38.39% / 38.39% | 32.14% / 32.14% | 0.100045 / 0.100045 | 0.166197 / 0.166197 |
| 10 | 97.32% / 100.00% | 32.14% / 33.93% | 0.092145 / 0.070397 | 0.162852 / 0.152872 |
| 25 | 100.00% / 100.00% | 32.14% / 28.57% | 0.087087 / 0.084092 | 0.170275 / 0.202182 |
| 50 | 100.00% / 100.00% | 32.14% / 28.57% | 0.091907 / 0.078811 | 0.177316 / 0.188476 |

两组第50步fit运动MAE均0.001350、归一化L1均0.002039、夹爪16/16、近零平移0/16；量化token全对不等于连续GT零误差。probe第50步夹爪1e-4为7/8、5e-4为8/8，近零平移分别3/8、0/8。运动MAE不含夹爪且量纲不同于归一化L1，不能混成单一总体优劣；近零是命令范数<0.02，不是实际位移。

| loss测量在该更新之前 | lr1e-4 | lr5e-4 |
|---|---:|---:|
| 1 | 2.819472 | 2.819472 |
| 10 | 0.283406 | 0.0000489532 |
| 25 | 0.000280078 | 0.00000622744 |
| 50 | 0.0000210585 | 0.000000450759 |

loss在更新之前测量，动作表在更新之后；第0步未单独测loss。5e-4训练阶段PyTorch峰值allocated9.219GiB（参考9.217GiB），非系统总显存。两组实际生产loader重载24×7自回归动作均与各自训练loader逐值一致。teacher-forcing仍有差异：第50步probe decoded-token L1，1e-4训练/生产0.127591/0.124930，5e-4为0.123669/0.124510；各组token accuracy分别32.14%/28.57%，训练与生产同准确率但不是相同logits/token解码L1。

**解释：** 此单seed固定帧实验中5e-4较快记忆16帧，没有出现“较大学习率必然不能学习”的现象；后期probe token准确率较低、归一化L1较高，但运动MAE较低、夹爪正确数更多，不能简单宣布整体更好/更差。两组probe都未显示随训练loss持续下降的全面迁移改善；与过拟合/跨帧迁移不足相容，但这8帧被原全量adapter见过，不是独立泛化证据。不能将历史全量失败唯一归因于lr或4bit，也不能据此推出论文或闭环成功率。

22项CPU测试通过，Notebook schema/AST及最终reader输出核验；本阶段完成后删除接手自动化，不继续训练/500评测。下一阶段需用户选择：建议从未微调LIBERO的基础权重开始，按episode隔离训练/验证并只用训练集计算统计，设计有界小规模试验。若继续从已见全部LIBERO的3500-step adapter开始，新的episode划分也不能称从未见过的held-out。另一条路线是先核查历史随机增强/裁剪影响；本轮不执行任何新GPU阶段。

## 新阶段：干净基础权重与 episode 隔离（2026-10-02，尚无最终训练指标）

用户授权继续后新增 `audit_episode_split.py` 和 `train_clean_task.py`，不改历史训练脚本、权重、统计或结果。完整扫描TFDS训练432个episode，目标任务“pick up the black bowl next to the cookie box and place it on the plate”有46个候选；按seed7从去除完整内容重复的候选中选择训练8条、验证2条。有效帧排除is_last或is_terminal，训练992、验证246。独占创建证据 `results/clean_task_split_audit_v1.json`，验证全部动作/state finite、episode语言一致和首末标记、全量清单、内容指纹隔离及train-only统计。它是CPU数据准备结果，不是训练完成或成功率。

训练 `[185,343,79,110,72,400,75,113]`；验证 `[394,212]`。指纹包含有序双视角编码图像、raw action、state、语言和终止标记；源HDF5路径是任务文件共享metadata，不能把同路径当相同演示，也不能把不同episode index直接当独立内容。原HDF5仍未验证。本轮是探索性同任务episode验证，不是跨任务泛化，也不保证OXE预训练语料无相关数据。

动作先按LIBERO规则将gripper转换为1-clip(raw,0,1)，仅训练992帧计算mean/std/min/max/q01/q99，六维运动用BOUNDS_Q99，gripper mask=False。验证六维超出训练q01/q99区间的比例依次为3.66%、0%、6.50%、2.85%、0%、4.88%；不使用验证分布重算边界。这是归一化裁剪诊断，不是预测误差。

原OXE基础权重缓存revision `47a0ec7fc4ec123775a391911046cf33cf9ed83f` 的3个分片、配置、tokenizer/processor和custom code已核验哈希，索引没有LoRA张量且无adapter_config。训练新建rank32/alpha16 all-linear LoRA，不从见过全量LIBERO的旧adapter热启动。NF4/BF16/doubleFalse、batch2×累积8、lr1e-4、seed7、无增强无中心裁剪、最多50更新，独立run `runs/clean_task_v1`。抽样仅训练帧；50更新800帧少于一轮992帧，所以即使验证不佳也不能据此证明无法泛化或量化有问题。

计划0/10/25/50节点：teacher-forcing评估全部246验证帧和24训练monitor帧，自回归只评估每条演示3个阶段帧（训练24/验证6）。两类指标覆盖不同，分开标注；最终生产loader重载并记录teacher与自回归差异。不自动rollout，没有本轮任务成功率。

**本轮读取故障与修复：** 首次worker PID30196在任何GPU模型加载/更新前，被episode完整指纹assert拦截。TFDS全流由分片交错读取，`train[i:i+1]`不是全流第i条；切片改变参与的分片交错顺序。改为与audit相同完整流枚举，再筛选指定编号，指纹标准未放宽。全部10条指纹重新CPU核验；按manifest顺序拼接动作后train-only统计逐值一致（顺序变化可能影响mean/std浮点末位）。保留原失败日志/progress副本 `retry-preupdate.json`，仅一次有界preupdate重试。此新脚本错误不是历史3500-step失败的已证实根因。

新增10项CPU守卫测试（episode去重/隔离/统计/配置/训练抽样），总32项通过；它们不是GPU训练成功证据。每小时低频接手已恢复，无变化不通知，额度不足不绕过。当前训练最终指标仍待 `results/clean_task_v1.json`，不得填造。

**第0步真实证据已核验：** `results/clean_task_start_v1.json` 状态verified_start_not_final。`verify_clean_task.py --start` 验证完整snapshot文件hash/spec、原base文件hash，重新CPU读取全部10条演示后核验teacher目标token/身份、AR原动作/身份与指标独立重算。未更新时训练24monitor teacher token13.10%、AR运动MAE0.150306；验证全部246帧teacher token11.67%、loss11.527381，6个验证阶段帧AR运动MAE0.134617、归一化L1 0.274609、夹爪6/6。验证teacher和AR覆盖不同，不能据此推断完整策略成绩。全部1238有效帧7动作+EOS监督检查通过；新LoRA参数110,828,288，worker已开始真实更新。最终结果尚待产生，不用逐步训练loss冒充验证改善。

## 干净起点试运行完成与生产差异诊断（2026-10-02）

最终真实报告 `results/clean_task_v1.json`：50次更新已完成，状态completed_loader_difference_requires_diagnosis，production_actions_equal=False；不修改原证据为PASS。`results/clean_task_verification_v1.json` 为verified_complete，仅表示四节点hash/spec、base未变、重新读取完整真实数据后的teacher身份/独立目标tokens、AR动作指标/摘要、全部50条连续loss/训练抽样通过，不表示两个loader动作全等。没有重复训练、rollout或更改历史权重。

| 更新后节点 | 训练24monitor teacher准确率 | 验证246帧teacher准确率 | 验证teacher loss | 验证6帧AR运动MAE | 验证6帧AR归一化L1 | 验证6帧夹爪 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 13.10% | 11.67% | 11.527381 | 0.134617 | 0.274609 | 6/6 |
| 10 | 14.29% | 14.34% | 6.681424 | 0.161278 | 0.331879 | 5/6 |
| 25 | 24.40% | 23.23% | 3.714302 | 0.302624 | 0.579720 | 4/6 |
| 50 | 32.14% | 32.69% | 3.251718 | 0.132556 | 0.226822 | 6/6 |

以上为训练准备loader（BF16 autocast）。最终实际生产loader验证teacher准确率32.75%、loss3.253109；验证6帧AR与训练loader逐值一致。训练24monitor中3个AR动作不同，整体30帧27/30相同；差异帧为episode185/t104、79/t0、72/t0，其中185/t104六维差异最大绝对值1.108100，不把所有差异轻描淡写为小数舍入。生产训练monitor AR运动MAE0.136861，训练loader为0.132429，不能混用。

真实平均训练loss在更新1/10/25/50**之前**为11.414975/7.326590/4.041071/2.797524；表中teacher/AR在对应更新后测量。第0步是评估teacher loss，不是新优化器训练loss。训练峰值PyTorch allocated8.773GiB，非系统总显存。50次更新抽样800个不同训练帧，小于一轮992帧；有效验证统计单位只有2条演示，不把246相关帧当246独立任务。验证teacher目标预测改善，6帧AR运动误差仅略降且中途变差，不能推出收敛、闭环提升或历史唯一根因。

**有界只读重载诊断：** `diagnose_clean_reload.py` 在宿主GPU空闲时仅推理6帧（3差异训练帧+3验证控制），无optimizer、梯度更新、rollout或checkpoint写入，保存 `results/clean_task_reload_diagnosis_v1.json`。同训练准备loader重载6/6精确复现训练输出，同生产loader6/6复现生产输出；在同一生产实例仅加BF16 autocast，5/6匹配训练（并未全修复）；应用完整prepare_model_for_kbit_training后配合autocast，6/6匹配训练。原snapshot保护文件SHA前后不变，两路径模型/config class一致。

训练准备路径非量化vision参数全部FP32（31,323,840个），生产路径有2,733,760个vision和264,108,544个其他非量化参数仍BF16；准备路径gradient_checkpointing=True，生产=False。这里计数按模型parameter张量numel，不作为实际存储/显存估计。完整准备同时改变dtype与checkpointing，未执行纯dtype/纯flag独立消融，所以不能唯一归因某层精度。证据支持本轮加载准备差异而非保存损坏；自回归早期token分叉可扩大后续动作差异是合理机制推断，未直接记录每步logit margin，不作为已测证据。

**边界与下一步：** 本轮已完成有限训练、完整性核验和加载差异诊断，但未修复生产实现、未测闭环或全任务成功率。建议下一阶段先做显式可选的训练准备兼容推理入口并验证全30帧，再决定有界闭环或追加训练；需用户选择，不自动扩大。Notebook第36节记录真实结果；全部原权重/数据/日志及test.jpg保留，收尾关闭本轮自动化。

收尾已通过32项CPU测试、26个reproduction Python AST、67-cell Notebook schema/全部code AST；数据划分与最终reader的执行输出与存储值精确一致，重载诊断四条匹配标志独立重算通过。应用确认heartbeat `openvla` deleteStatus=deleted；没有新的训练任务。
