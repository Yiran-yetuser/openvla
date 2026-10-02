# OpenVLA 接手记录（2026-10-02）

## 当前授权阶段：clean_task_v1（优先于下方历史说明）

**本阶段已完成并核验，禁止再次launch/resume。** 最终 `results/clean_task_v1.json` 保留原状态completed_loader_difference_requires_diagnosis，独立完整核验 `clean_task_verification_v1.json` 为verified_complete（是完整性/指标核验，不是生产动作全等PASS）。四节点SHA/spec、全部真实帧/目标tokens/AR指标及50条loss/训练抽样、base不变均通过。验证teacher246帧准确率0/10/25/50为11.67%/14.34%/23.23%/32.69%；6个固定验证AR运动MAE0.134617/0.161278/0.302624/0.132556，不是稳定单调改善或闭环成绩。生产验证teacher32.75%，AR与训练路径6/6一致；训练monitor有3/24个AR不同，完整30帧27/30一致。

`diagnose_clean_reload.py --execute`已完成，**不得重复运行**。只读6帧（3差异训练帧+3验证控制）、无optimizer/参数更新/rollout，结果 `results/clean_task_reload_diagnosis_v1.json`。相同训练准备loader重载6/6复现训练输出；实际生产loader6/6复现生产输出；同生产实例仅加autocast5/6匹配训练，应用完整prepare_model_for_kbit_training+autocast后6/6匹配训练。非量化FP32/BF16 dtype及gradient checkpointing标志不同；准备操作联合改变这些，不是纯dtype单因素。支持加载准备差异，不支持保存损坏，也不把这6帧结论扩展为历史唯一根因。未修改生产实现，原不一致报告保留。

下一阶段需要用户选择：建议先设计显式、可选的训练准备兼容推理入口并验证全部30帧，再决定有界闭环对照或追加训练；不自动修改所有模型加载策略，不启动新训练/500评测。原checkpoint、optimizer、数据、日志/test.jpg保留。下方启动说明仅为历史记录。

收尾核验：32项CPU测试、26个reproduction Python AST文件、67-cell Notebook schema/全部code语法及两个clean reader存储输出精确一致；诊断结果的四路径匹配布尔值独立重算一致。应用已确认本轮heartbeat `openvla` 删除（deleteStatus=deleted），不再定时启动此已完成阶段。

用户已授权从干净 OXE 基础权重开始、同任务完整 episode 隔离验证。新 CPU 证据为 `results/clean_task_split_audit_v1.json`，完整432个episode扫描通过。同目标任务46个候选，seed7选训练episode `[185,343,79,110,72,400,75,113]`、验证 `[394,212]`；有效帧992/246（排除is_last或is_terminal），内容指纹不重叠，动作统计仅来自训练992帧。源HDF5 metadata是共享路径而非独立episode身份，也未找到/验证原HDF5文件。

基础权重是本地HF缓存 `openvla/openvla-7b` revision `47a0ec7fc4ec123775a391911046cf33cf9ed83f`，3个权重分片及配置/processor/tokenizer/custom code均已SHA256核验。新建LoRA，不加载3500-step或两组small_fit adapter。这里只保证本轮LIBERO微调的episode隔离，未重新审计OXE预训练语料重叠。

- 唯一新训练入口 `reproduction/train_clean_task.py`，默认dry-run。`--launch`启动独立worker，最多50次更新；lr1e-4、seed7、rank32/alpha16、all-linear、batch2×累积8、NF4/BF16/doubleFalse、无增强/无裁剪。每个训练epoch独立seed打乱，仅训练帧被抽样，50更新共800帧，不到完整一轮；这是探针，不是足够训练或论文配方。第0步LoRA B全0守卫。
- 状态 `runs/clean_task_v1/progress.json`、日志 `worker-*.log`、四节点checkpoint `step_000/010/025/050`；最终结果 `results/clean_task_v1.json`。已有结果先核验，禁止覆盖/重训。检查GPU必须宿主沙箱外只读nvidia-smi和ps；不抢占其他任务。
- 第0/10/25/50步：验证teacher-forcing覆盖全部246帧；训练teacher仅24固定monitor帧。自回归仅24训练/6验证阶段帧，不可冒充全帧自回归或闭环成功率。最终实际生产loader重载，单独保留teacher差异，不把相同token accuracy当全部logits相等。
- 首次worker PID30196在模型加载/更新前失败：按TFDS位置切片不等于全流episode编号，被完整指纹守卫拦截。已改完整流枚举筛选，重新CPU核验全部10条指纹及train-only统计逐值一致；拼接顺序保持manifest顺序，避免mean/std末位舍入差。保留原log与 `runs/clean_task_v1/retry-preupdate.json`，仅允许一次 `--retry-preupdate`。不降低检查标准、不删除失败证据。
- worker结束且未完成时，只有完整snapshot spec/hash验证通过才能 `--launch --resume`；partial保留且停止等待诊断。resume尚未端到端验证；不宣称bitwise保证，不反复启动。原始base SHA前后必须一致，checkpoint必须独立。
- 每次先查额度、progress/result；额度不可用不绕过、不用API/重置券/改模型，等下次定时接手。每小时heartbeat id=`openvla`已应用确认创建；无变化/忙保持安静，仅新证据/完成/失败/必要操作通知。收尾后删除，不扩大训练/500评测。
- 完成后只读核验四snapshot哈希、全部50条loss/训练抽样身份、真实teacher/AR指标和生产差异，更新notebook/DIAGNOSIS/CONTINUE，运行32项CPU测试、AST、Notebook schema/reader校验；提交推送origin/codex/complete-openvla-reproduction并更新Yiran-yetuser/openvla PR #1（gh显式指定repo）。保留原权重、数据、所有日志和test.jpg。

独立Python worker不调用Codex/API，但依赖电脑开机不休眠；本地定时接手依赖应用运行与可用额度，不保证额度刷新瞬间恢复。GPU忙或worker运行时不启动第二份。

第0步完整snapshot已核验：`results/clean_task_start_v1.json` 为verified_start_not_final，含snapshot保护文件hash、原base hash未变、重新加载数据后逐帧身份/目标tokens以及动作指标独立重算。训练monitor24 token13.10%；验证246 teacher token11.67%、6个固定验证阶段帧AR运动MAE0.134617，均是未更新起点，不是50步结果。全部1238帧7动作+EOS监督守卫已通过，新LoRA可训练参数110,828,288；worker PID33063已进入真实参数更新，状态以progress/宿主ps为准。

最终只读校验入口：`python -m reproduction.verify_clean_task --output reproduction/results/clean_task_verification_v1.json`，必须使用上述离线环境/Python。有输出先只读核验，不覆盖；最终结果缺失则不执行final验证，也不补造。它重载CPU真实帧，核验每个teacher目标token/身份、AR目标/指标/摘要、四节点SHA/spec、生产loader阶段以及全部50条loss/训练抽样。若有resume产生重复loss histories会明确失败，需按attempt/已提交snapshot审查，不能默默当50条连续日志。仅起点用`--start`的已完成记录不得反复跑。

## 当前目标与已完成结果

继续12GB单卡QLoRA近似复现，先解决低成功率的证据链，不把论文成绩当本地结果。
已有1000-step/12-episode、3500-step/432-episode权重；旧19.2%离线改善已撤回。
官方NF4对照与当前训练链路体检已完成，见DIAGNOSIS.md和notebook第25–30节；禁止重跑或覆盖这些结果。
原权重、数据、test.jpg和历史日志必须保留。GPU被其他作业占用时不抢占。

## 当前阶段已完成：small_fit_lr5e4_v1（禁止重新启动）

最终状态completed、50更新；`results/small_fit_lr5e4_v1.json` 和 `results/small_fit_lr_comparison_v1.json` 已真实产生，完整比较verified_comparison且local_snapshot_verification=True。两组8个snapshot、源hash、24帧身份/真实动作/指标摘要及各50条loss已核验；原权重未变。两组fit最终token100%、运动MAE0.001350；probe5e-4 token28.57%、MAE0.078811/L1 0.188476，对照1e-4为32.14%、0.091907/0.177316，不能简单认定整体优劣或历史根因。两个loader自回归动作重载一致，teacher差异已记录。

**GPU worker已结束，不再launch/resume，不重跑已有结果。下一实验等待用户选择。** 下列配置与启动入口仅保留作历史说明；不要把已完成阶段重复运行。22项CPU测试、Notebook schema/AST及最终reader校验在收尾再核验。

用户2026-10-02再次授权继续。GPU空闲、磁盘约46GB可用后，启动独立worker（初始PID42189，02:48:09 UTC即上海10:48:09）；当前状态以progress/宿主ps为准，不凭旧PID认定仍在运行。

- 历史GPU入口：`fit_small_sample.py --experiment small_fit_lr5e4_v1 --launch`。本阶段已完成，不要再运行；恢复机制只适用于未完成且已核验的snapshot，不能当继续超过50更新的入口。
- 同原3500-step源adapter、同16拟合帧、同8个原adapter见过的probe、batch2×累积8、NF4/BF16/doubleFalse/seed7、新AdamW、无增强/无裁剪，最多50更新；**仅学习率改为5e-4**。不能从上轮step050继续，必须以同源权重开始对照。
- `runs/small_fit_lr5e4_v1/progress.json` / 独立日志及step_XXX；最终 `results/small_fit_lr5e4_v1.json`。不要读取旧run当新run。结果存在先核验，不覆盖、不重训。
- 起始完整snapshot已核验，24帧所有第0步结果与参考完全相同，见 `results/small_fit_lr5e4_start_v1.json`；此文件不是最终结果。spec除lr外全部一致，源hash核验，22项CPU测试通过。
- 第10步中间报告 `results/small_fit_lr5e4_milestone10_v1.json` 已核验hash/frame：fit token100%、运动MAE0.001350，probe token33.93%、运动MAE0.070397；同节点参考fit97.32%、运动0.005097、probe32.14%/0.092145。这是中间节点，不能替代最终50步结论。
- 最终运行只读比较：`python reproduction/compare_small_fit_lr.py --verify-local --output reproduction/results/small_fit_lr_comparison_v1.json`。输出存在则只读核验、禁止覆盖。校验两report/frame/metric、8个完整snapshot、源hash与真实loss；loss在更新前，动作指标在更新后，不把update1的loss冒充step0。
- 基于真实0/10/25/50趋势更新notebook/DIAGNOSIS。区分teacher-forcing/自回归/生产loader；8probe不是held-out，本实验不能充分解释历史失败。仅一seed/每lr一run，不能声称统计显著性。
- 本阶段曾创建每小时heartbeat，id=`openvla`；2026-10-02收尾时应用工具已确认deleteStatus=deleted。不扩大训练/500评测，下一实验待用户选择。额度不可用不绕过、不用重置券/付费API/改模型。

## 已完成阶段：small_fit_v1（禁止重新启动）

最终结果 `results/small_fit_v1.json` 状态completed、50更新；核验记录 `results/small_fit_verification_v1.json`。0/10/25/50节点hash/spec、24帧身份、逐帧动作指标/汇总、50条loss及原权重SHA256均核验通过。fit16在25/50步token准确率100%、运动MAE0.001350；probe8准确率仍32.14%、第50步连续L1反而高于起始。24帧生产重载自回归动作逐值一致，但probe teacher-token L1训练0.127591/生产0.124930，不宣称两loader全等。没有泛化或闭环改善证据。

下面保留历史实验配置和恢复说明，**不是下一次接手的启动指令**。本阶段GPU worker已结束；用户现已授权上节独立学习率对照，不扩展长训练/500评测。

- 入口 `reproduction/fit_small_sample.py`，默认dry-run，`--launch`启动独立本地worker。
- 16个CPU体检已核验真实帧，batch2×累积8，lr1e-4，最多50次新AdamW更新，从本地3500-step adapter热启动。
- 无随机增强、无中心裁剪；不是原历史训练的精确续训，也不是泛化或任务成功率实验。
- 在0/10/25/50步记录同24帧teacher-forcing和自回归动作；前16帧用于本轮拟合，其余8帧只作probe，但原全量adapter已经见过它们，不得标held-out。
- 状态 `runs/small_fit_v1/progress.json`，日志 `runs/small_fit_v1/worker-*.log`，证据 `reproduction/results/small_fit_v1.json`。
- 第10步中间报告 `results/small_fit_milestone10_v1.json` 保留；最终完整证据见上述报告。禁止重跑已完成节点。
- 检查点 `runs/small_fit_v1/step_XXX` 保存adapter、optimizer、torch CPU/CUDA RNG、evaluation和SHA256 manifest；完整目录原子发布，保留所有完成节点。
- `.partial`目录是失败证据，不能直接当作完成检查点，也不能自动删除。resume仅使用完成manifest并核验hash/spec。

## 每次接手的顺序

1. 查询当前额度；不可用时不设法绕过、不使用付费API或免费重置券、不改模型设置。
2. 先读当前small_fit_lr5e4_v1的progress/result，宿主沙箱外只读nvidia-smi和ps核实运行状态；禁止把沙箱不可见当作结束。GPU忙/worker仍在运行/无变化时保持安静。
3. 有最终JSON先核验0/10/25/50节点、24帧身份、finite指标、源权重hash和生产加载器差异；不重复启动。
4. 已结束但无最终结果时，读失败状态和log。安全修复有界重试，不无限循环；完整检查点可用时增加`--resume`，无完整checkpoint时先诊断，不覆盖partial。
5. 用 `/home/yyz/miniconda3/envs/openvla/bin/python` 和 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/home/yyz/openvla`。
6. 依据真实拟合趋势更新DIAGNOSIS/notebook，运行CPU测试、AST、Notebook schema和reader输出核验。提交推送origin/codex/complete-openvla-reproduction，更新Yiran-yetuser/openvla PR #1，gh必须显式指定repo。
7. 不直接扩大到长训练或500评测。若拟合通过，先提出单因素下一实验；若未通过，定位数据/监督/优化/加载差异。完成本阶段并汇报后删除相应接手自动化。

历史启动方式（本阶段已完成，不要再次运行）：

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/home/yyz/openvla \
  /home/yyz/miniconda3/envs/openvla/bin/python reproduction/fit_small_sample.py --launch
# 中断后仅从核验完成的snapshot接着跑；不重复启动活跃worker：
# 同命令增加 --resume
```

## 额度恢复边界

small_fit_v1的旧heartbeat已删除；本轮small_fit_lr5e4_v1也已完成，应用已确认删除其heartbeat（id=`openvla`）。不自动扩展下一实验；接手记录和checkpoint保留。
定时任务是后续接手机会，不是额度绕过，也不保证刷新瞬间或原中断点无缝恢复。
独立Python worker不调用Codex/API，但仍依赖机器不关机、不休眠；checkpoint用于进程中断后的恢复。
接手定时任务依赖电脑开机、桌面应用运行及可用额度。忙或无进展不通知，仅新证据、完成、失败或必要用户选择时通知。
