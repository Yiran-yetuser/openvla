"""Compare the opt-in k-bit inference preparation on the saved 30 pilot frames.

--execute performs inference only (24 training monitor + 6 episode-validation
stage frames). It does not update weights, run environments, or overwrite prior
evidence. The default prints a bounded plan.
"""
import argparse
from contextlib import nullcontext
import gc
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from reproduction.train_clean_task import AUDIT, RESULT, RUN, KEY, load_frames, specification
from reproduction.fit_small_sample import verify_snapshot
from reproduction.audit_episode_split import sha256


def config(checkpoint, base, prepare):
    return SimpleNamespace(pretrained_checkpoint=str(checkpoint), base_model_path=str(base),
                           load_in_4bit=True, load_in_8bit=False, bnb_double_quant=False,
                           prepare_for_kbit_inference=prepare, center_crop=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = ROOT / "reproduction/results/clean_task_kbit_inference_v1.json"
    if not args.execute:
        print(json.dumps({"mode": "inference_only_plan", "frames": 30,
                          "paths": ["production_default", "prepare_for_kbit_inference=True"],
                          "output": str(output)}))
        return
    if output.exists():
        raise FileExistsError("Existing inference evidence must be verified, never overwritten")
    occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                                       "--format=csv,noheader"], text=True).strip()
    if occupied:
        raise RuntimeError(f"GPU busy; do not preempt another job: {occupied}")
    audit, final = json.loads(AUDIT.read_text()), json.loads(RESULT.read_text())
    assert final["status"] == "completed_loader_difference_requires_diagnosis"
    spec = specification(audit)
    checkpoint, snapshot = RUN / "step_050", verify_snapshot(RUN / "step_050", spec)
    frames = load_frames(audit)
    records = []
    for split in ("train", "validation"):
        by_identity = {(r["episode_index"], r["timestep"]): r for r in frames[split]}
        saved = {split: {"training": [], "production": []}}
        for path_name, target in (("training_BF16_autocast", "training"),
                                  ("production_no_autocast", "production")):
            source = final["evaluations"][-1] if target == "training" else final["production_evaluation"]
            for row in source["splits"][split]["autoregressive_rows"]:
                identity = (row["episode_index"], row["timestep"])
                assert identity in by_identity
                if target == "training":
                    saved[split]["training"].append({"identity": identity,
                        "action": row["action"], "image_sha256": row["image_sha256"]})
                else:
                    match = next(x for x in saved[split]["training"] if x["identity"] == identity)
                    assert row["image_sha256"] == match["image_sha256"]
                    match["production_action"] = row["action"]
        for row in saved[split]["training"]:
            identity = row["identity"]
            records.append({"split": split, "episode_index": identity[0], "timestep": identity[1],
                            "image_sha256": row["image_sha256"], "training_action": row["action"],
                            "production_action": row["production_action"],
                            "production_matches_training": row["production_action"] == row["action"]})
    assert len(records) == 30 and sum(r["split"] == "train" for r in records) == 24
    assert sum(r["split"] == "validation" for r in records) == 6
    from experiments.robot.openvla_utils import get_vla, get_processor, get_vla_action
    import torch
    base = Path(spec["base_path"])

    def run_path(prepare):
        model, processor = get_vla(config(checkpoint, base, prepare)), get_processor(config(checkpoint, base, prepare))
        results = []
        model.eval()
        inference_context = torch.autocast("cuda", dtype=torch.bfloat16) if prepare else nullcontext()
        with torch.inference_mode(), inference_context:
            for row in records:
                frame = next(x for x in frames[row["split"]]
                             if x["episode_index"] == row["episode_index"] and x["timestep"] == row["timestep"])
                action = get_vla_action(model, processor, str(base), {"full_image": frame["image"]},
                                        audit["task_instruction"], KEY, center_crop=False)
                results.append(action.tolist())
        signature = {"vision_fp32_params": sum(p.numel() for n, p in model.named_parameters()
                          if "vision_backbone" in n and p.dtype == torch.float32),
                     "vision_bf16_params": sum(p.numel() for n, p in model.named_parameters()
                          if "vision_backbone" in n and p.dtype == torch.bfloat16),
                     "gradient_checkpointing": bool(model.base_model.model.is_gradient_checkpointing)}
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
        return results, signature

    default_actions, default_signature = run_path(False)
    compatible_actions, compatible_signature = run_path(True)
    for name, base_record in audit["base"]["files"].items():
        assert sha256(base / name) == base_record["sha256"]
    verify_snapshot(checkpoint, spec)
    for row, actual_default, actual_compatible in zip(records, default_actions, compatible_actions):
        row.update(default_reproduces_saved=row["production_action"] == actual_default,
                   compatible_reproduces_training=row["training_action"] == actual_compatible,
                   default_action=actual_default, compatible_action=actual_compatible)
    default_matches = sum(r["default_reproduces_saved"] for r in records)
    compatible_matches = sum(r["compatible_reproduces_training"] for r in records)
    report = {"status": "verified" if default_matches == 30 and compatible_matches == 30 else "mismatch_requires_diagnosis",
              "experiment": "clean_task_kbit_inference_v1", "spec": spec,
              "base_hashes_unchanged": True, "snapshot_sha256": snapshot["sha256"],
              "production_checkpoint_sha256": sha256(checkpoint / "adapter_model.safetensors"),
              "default_matches_saved_production": default_matches,
              "compatible_matches_saved_training": compatible_matches,
              "default_signature": default_signature, "compatible_signature": compatible_signature,
              "optimizer_updates": 0, "rollouts": 0, "checkpoint_modified": False,
              "frames": records,
              "limits": ["30 recorded demonstration frames: 24 train monitors and 6 frames from 2 validation episodes.",
                         "Inference-path parity only; no simulator success rate, full-suite claim or dtype-only causal claim.",
                         "prepare_model_for_kbit_training jointly changes dtype, freezing and checkpointing setup."]}
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in ("status", "default_matches_saved_production",
                                              "compatible_matches_saved_training", "optimizer_updates", "rollouts")}, indent=2))


if __name__ == "__main__":
    main()
