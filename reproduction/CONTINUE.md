# OpenVLA 接手记录（2026-10-02）

## 当前目标与已完成结果

继续12GB单卡QLoRA近似复现，先解决低成功率的证据链，不把论文成绩当本地结果。
已有1000-step/12-episode、3500-step/432-episode权重；旧19.2%离线改善已撤回。
官方NF4对照与当前训练链路体检已完成，见DIAGNOSIS.md和notebook第25–30节；禁止重跑或覆盖这些结果。
原权重、数据、test.jpg和历史日志必须保留。GPU被其他作业占用时不抢占。

## 正在推进的下一阶段：small_fit_v1

- 入口 `reproduction/fit_small_sample.py`，默认dry-run，`--launch`启动独立本地worker。
- 16个CPU体检已核验真实帧，batch2×累积8，lr1e-4，最多50次新AdamW更新，从本地3500-step adapter热启动。
- 无随机增强、无中心裁剪；不是原历史训练的精确续训，也不是泛化或任务成功率实验。
- 在0/10/25/50步记录同24帧teacher-forcing和自回归动作；前16帧用于本轮拟合，其余8帧只作probe，但原全量adapter已经见过它们，不得标held-out。
- 状态 `runs/small_fit_v1/progress.json`，日志 `runs/small_fit_v1/worker-*.log`，证据 `reproduction/results/small_fit_v1.json`。
- 第10步已核验中间报告 `results/small_fit_milestone10_v1.json`：fit token accuracy38.39%→97.32%、运动MAE0.112703→0.005097；probe token仍32.14%，不可称泛化改善。禁止重跑已完成节点。
- 检查点 `runs/small_fit_v1/step_XXX` 保存adapter、optimizer、torch CPU/CUDA RNG、evaluation和SHA256 manifest；完整目录原子发布，保留所有完成节点。
- `.partial`目录是失败证据，不能直接当作完成检查点，也不能自动删除。resume仅使用完成manifest并核验hash/spec。

## 每次接手的顺序

1. 查询当前额度；不可用时不设法绕过、不使用付费API或免费重置券、不改模型设置。
2. 先读progress/result，宿主沙箱外只读nvidia-smi和ps核实运行状态；禁止把沙箱不可见当作结束。GPU忙/worker仍在运行/无变化时保持安静。
3. 有最终JSON先核验0/10/25/50节点、24帧身份、finite指标、源权重hash和生产加载器差异；不重复启动。
4. 已结束但无最终结果时，读失败状态和log。安全修复有界重试，不无限循环；完整检查点可用时增加`--resume`，无完整checkpoint时先诊断，不覆盖partial。
5. 用 `/home/yyz/miniconda3/envs/openvla/bin/python` 和 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/home/yyz/openvla`。
6. 依据真实拟合趋势更新DIAGNOSIS/notebook，运行CPU测试、AST、Notebook schema和reader输出核验。提交推送origin/codex/complete-openvla-reproduction，更新Yiran-yetuser/openvla PR #1，gh必须显式指定repo。
7. 不直接扩大到长训练或500评测。若拟合通过，先提出单因素下一实验；若未通过，定位数据/监督/优化/加载差异。完成本阶段并汇报后删除相应接手自动化。

启动（先确认宿主GPU空闲）：

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/home/yyz/openvla \
  /home/yyz/miniconda3/envs/openvla/bin/python reproduction/fit_small_sample.py --launch
# 中断后仅从核验完成的snapshot接着跑；不重复启动活跃worker：
# 同命令增加 --resume
```

## 额度恢复边界

本聊天每小时一次heartbeat已创建，automation id=`openvla`；仅新进展/失败/完成/必要选择通知。
定时任务是后续接手机会，不是额度绕过，也不保证刷新瞬间或原中断点无缝恢复。
独立Python worker不调用Codex/API，但仍依赖机器不关机、不休眠；checkpoint用于进程中断后的恢复。
接手定时任务依赖电脑开机、桌面应用运行及可用额度。忙或无进展不通知，仅新证据、完成、失败或必要用户选择时通知。
