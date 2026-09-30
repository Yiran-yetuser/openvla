"""Stage 1: corrected offline diagnostics + a ten-update warm-start smoke test.

Default is a dry run. --execute requires an idle GPU and never stops other jobs.
The smoke checkpoint is an engineering validation, not a better trained model.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
EXP = "openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0+q-4bit--image_aug"
FULL = ROOT / "adapter-tmp/libero_spatial_full_paper" / EXP
SMALL = ROOT / "adapter-tmp/libero_spatial_12episode_1000step" / EXP
RUN = "libero_spatial_diagnostic_stage1_10steps"
PILOT = ROOT / "adapter-tmp" / RUN / EXP.replace("lr-0.0005", "lr-0.0001")
RESULTS = ROOT / "reproduction/results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="offline",
               TOKENIZERS_PARALLELISM="false", PYTHONPATH=str(ROOT))
    jobs = [
        (["reproduction/diagnose_qlora.py", "--checkpoint", str(FULL),
          "--output", str(RESULTS / "full3500_diagnostic_v1.json")], RESULTS / "full3500_diagnostic_v1.json"),
        (["reproduction/diagnose_qlora.py", "--checkpoint", str(SMALL), "--episode-start", "12",
          "--episodes", "4", "--trained-episodes", "12", "--output", str(RESULTS / "small1000_heldout_v1.json")],
         RESULTS / "small1000_heldout_v1.json"),
        (["reproduction/run_qlora_1000.py", "--train-episodes", "432", "--max-steps", "10",
          "--save-steps", "10", "--run-name", RUN, "--learning-rate", "0.0001", "--init-adapter", str(FULL)],
         PILOT / "training_state.json"),
        (["reproduction/diagnose_qlora.py", "--checkpoint", str(PILOT),
          "--output", str(RESULTS / "warmstart10_diagnostic_v1.json")], RESULTS / "warmstart10_diagnostic_v1.json"),
    ]
    for command, evidence in jobs:
        print(" ".join([sys.executable, *command]), flush=True)
    if not args.execute:
        print("Dry run only. Execute after Fast3R completes: add --execute")
        return
    # Do not run two models concurrently on a 12GB GPU. Do not kill anything.
    gpu_processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], text=True).strip()
    if gpu_processes:
        raise RuntimeError(f"GPU compute process still active; defer this stage: {gpu_processes}")
    for command, evidence in jobs:
        if evidence.exists():
            with evidence.open() as handle:
                previous = json.load(handle)
            if evidence.name == "training_state.json":
                assert previous["optimizer_step"] == 10 and previous["microbatches"] == 80
            else:
                assert previous["samples"] > 0 and previous["summary"]
            print(f"Already complete; preserving {evidence}", flush=True)
            continue
        if command[0] == "reproduction/run_qlora_1000.py":
            cache = ROOT / "cache/official_libero_spatial"
            if cache.exists() and not (RESULTS / "official_checkpoint_manifest.json").exists():
                raise RuntimeError("Official checkpoint download pending; defer training smoke test")
            if shutil.disk_usage(ROOT).free < 3 * 2**30:
                raise RuntimeError("Training smoke test needs checkpoint space plus 2GiB disk reserve")
        subprocess.run([sys.executable, *command], cwd=ROOT, env=env, check=True)
    print("Stage 1 complete. Review JSON evidence before any larger training or simulation run.", flush=True)


if __name__ == "__main__":
    main()
