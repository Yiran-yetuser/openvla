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
LOG_FILE = REPO_DIR / "reproduction" / "finetune_12episode_1000step.log"

# exp_id = base_vlm 的最后一段 + 数据集 + 超参，因此这里必须用 HF 仓库 ID。
# 如果传本地快照目录，exp_id 会变成 47a0ec7f... 这种哈希前缀，
# 导致 test_1000step_adapter.py / test_planB2_libero_gt.py 里写死的 adapter 路径找不到。
VLA_PATH = "openvla/openvla-7b"
RUN_ROOT = "runs/libero_spatial_12episode_1000step"
ADAPTER_TMP_DIR = "adapter-tmp/libero_spatial_12episode_1000step"
EXP_ID = (
    "openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+"
    "lora-r32+dropout-0.0+q-4bit--image_aug"
)


def main() -> int:
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"本地 OpenVLA 模型目录不存在: {MODEL_DIR}")

    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "TOKENIZERS_PARALLELISM": "false",
            "OPENVLA_TRAIN_EPISODES": "12",
            "OPENVLA_EVAL_EPISODE_START": "12",
            "OPENVLA_EVAL_EPISODES": "4",
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
        RUN_ROOT,
        "--adapter_tmp_dir",
        ADAPTER_TMP_DIR,
        "--batch_size",
        "2",
        "--grad_accumulation_steps",
        "8",
        "--max_steps",
        "1000",
        "--save_steps",
        "1000",
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
    # 说明：systemd-inhibit 需要用户会话的 D-Bus，若不可用则退回普通启动。
    bus_socket = Path(f"/run/user/{os.getuid()}/bus")
    if shutil.which("systemd-inhibit") and bus_socket.exists():
        env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={bus_socket}")
        command = [
            "systemd-inhibit",
            "--what=idle:sleep",
            "--why=OpenVLA QLoRA 12-episode training",
            "--mode=block",
            *command,
        ]
        print("已启用休眠抑制：训练期间系统不会因空闲而休眠", flush=True)
    else:
        print("警告：未启用休眠抑制，训练期间请不要让电脑休眠", flush=True)

    print("启动 OpenVLA-7B QLoRA 12-episode / 1000-step 训练", flush=True)
    print(f"日志文件: {LOG_FILE}", flush=True)
    print(f"run 输出目录    : {REPO_DIR / RUN_ROOT / EXP_ID}", flush=True)
    print(f"adapter 输出目录: {REPO_DIR / ADAPTER_TMP_DIR / EXP_ID}", flush=True)

    with LOG_FILE.open("w", encoding="utf-8") as log_file:
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
            log_file.write(line)
            log_file.flush()

    return_code = process.wait()
    print(f"训练进程结束，退出码: {return_code}", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
