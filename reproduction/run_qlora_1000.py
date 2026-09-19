import os
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = (
    Path.home() / ".cache/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f"
)
LOG_FILE = REPO_DIR / "reproduction" / "finetune_1000step.log"


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
        }
    )

    command = [
        sys.executable,
        str(REPO_DIR / "vla-scripts/finetune.py"),
        "--vla_path",
        str(MODEL_DIR),
        "--data_root_dir",
        str(Path.home() / "modified_libero_rlds"),
        "--dataset_name",
        "libero_spatial_no_noops",
        "--run_root_dir",
        "runs",
        "--adapter_tmp_dir",
        "adapter-tmp",
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

    print("启动 OpenVLA-7B QLoRA 1000-step 训练", flush=True)
    print(f"日志文件: {LOG_FILE}", flush=True)

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
