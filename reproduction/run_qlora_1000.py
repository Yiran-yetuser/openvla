import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
# 离线缓存里的本地模型目录，仅用于确认 base 模型已下载（不直接作为 --vla_path 传入）。
MODEL_DIR = (
    Path.home() / ".cache/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f"
)


# exp_id = base_vlm 的最后一段 + 数据集 + 超参，因此这里必须用 HF 仓库 ID。
# 如果传本地快照目录，exp_id 会变成 47a0ec7f... 这种哈希前缀，
# 导致 test_1000step_adapter.py / test_planB2_libero_gt.py 里写死的 adapter 路径找不到。
VLA_PATH = "openvla/openvla-7b"
EXP_ID = (
    "openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+"
    "lora-r32+dropout-0.0+q-4bit--image_aug"
)

# 默认配置 = 12-episode 实验（保持脚本原有行为）；
# 论文口径（全量 432 条 episode、更多步数）用命令行参数覆盖，例如：
#   python reproduction/run_qlora_1000.py \
#       --train-episodes 432 --max-steps 5000 \
#       --run-name libero_spatial_full_paper
DEFAULT_TRAIN_EPISODES = 12
DEFAULT_EVAL_EPISODE_START = 12
DEFAULT_EVAL_EPISODES = 4
DEFAULT_MAX_STEPS = 1000
DEFAULT_RUN_NAME = "libero_spatial_12episode_1000step"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenVLA LoRA 微调启动脚本（可配置数据规模与步数）")
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES,
                        help="训练使用前 N 个 episode（432 = 官方全量 LIBERO-Spatial）")
    parser.add_argument("--eval-episode-start", type=int, default=DEFAULT_EVAL_EPISODE_START,
                        help="held-out 起始 episode（仅被评测脚本使用，训练不会读）")
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES,
                        help="held-out episode 数量")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="训练步数")
    parser.add_argument("--save-steps", type=int, default=None,
                        help="每多少步保存一次 adapter，默认等于 --max-steps；设小一点可以在长跑中断时保留进度")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME,
                        help="输出目录名，run 输出到 runs/<name>/，adapter 输出到 adapter-tmp/<name>/")
    parser.add_argument("--log-file", default=None, help="训练日志路径，默认 reproduction/finetune_<name>.log")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    run_root = f"runs/{args.run_name}"
    adapter_tmp_dir = f"adapter-tmp/{args.run_name}"
    log_file = Path(args.log_file) if args.log_file else REPO_DIR / "reproduction" / f"finetune_{args.run_name}.log"

    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"本地 OpenVLA 模型目录不存在: {MODEL_DIR}")

    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "TOKENIZERS_PARALLELISM": "false",
            "OPENVLA_TRAIN_EPISODES": str(args.train_episodes),
            "OPENVLA_EVAL_EPISODE_START": str(args.eval_episode_start),
            "OPENVLA_EVAL_EPISODES": str(args.eval_episodes),
        }
    )

    finetune_args = [
        "--vla_path",
        VLA_PATH,
        "--data_root_dir",
        str(Path.home() / "modified_libero_rlds"),
        "--dataset_name",
        "libero_spatial_no_noops",
        "--run_root_dir",
        run_root,
        "--adapter_tmp_dir",
        adapter_tmp_dir,
        "--batch_size",
        "2",
        "--grad_accumulation_steps",
        "8",
        "--max_steps",
        str(args.max_steps),
        "--save_steps",
        str(args.save_steps or args.max_steps),
        "--learning_rate",
        "5e-4",
        "--use_lora",
        "true",
        "--use_quantization",
        "true",
        "--image_aug",
        "true",
    ]

    command = [sys.executable, str(REPO_DIR / "vla-scripts/finetune.py"), *finetune_args]

    # 训练期间阻止系统休眠。
    # 原因：笔记本进入 suspend 后 CUDA 上下文会失效，训练进程会卡住不再前进
    #       （2026-09-21 00:45 的训练就是这样停在第 20 步，唤醒后无法恢复）。
    # 说明：--what 里同时包含 handle-lid-switch，合盖也不会触发休眠，方便过夜长跑。
    # 说明：systemd-inhibit 需要用户会话的 D-Bus，若不可用则退回普通启动。
    bus_socket = Path(f"/run/user/{os.getuid()}/bus")
    if shutil.which("systemd-inhibit") and bus_socket.exists():
        env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={bus_socket}")
        command = [
            "systemd-inhibit",
            "--what=idle:sleep:handle-lid-switch",
            "--why=OpenVLA QLoRA training",
            "--mode=block",
            *command,
        ]
        print("已启用休眠抑制：空闲/主动休眠与合盖都不会中断训练", flush=True)
    else:
        print("警告：未启用休眠抑制，训练期间请不要让电脑休眠", flush=True)

    print(f"启动 OpenVLA-7B QLoRA 训练：train[:{args.train_episodes}] / {args.max_steps} steps", flush=True)
    print(f"日志文件        : {log_file}", flush=True)
    print(f"run 输出目录    : {REPO_DIR / run_root / EXP_ID}", flush=True)
    print(f"adapter 输出目录: {REPO_DIR / adapter_tmp_dir / EXP_ID}", flush=True)

    with log_file.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()

    return_code = process.wait()
    print(f"训练进程结束，退出码: {return_code}", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
