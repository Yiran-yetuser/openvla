"""Inference-only investigation of the three clean-pilot reload discrepancies.

No optimizer, training, rollout, or checkpoint write. Reproduce both recorded
paths; on the SAME production instance vary autocast, then k-bit preparation.
The latter jointly changes nonquantized dtypes and checkpointing flags; do not
call it a pure dtype-only ablation. Six frames maximum (3 discrepant, 3 controls).
"""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.train_clean_task import AUDIT, RESULT, RUN, KEY, load_frames, specification
from reproduction.fit_small_sample import verify_snapshot
from reproduction.audit_episode_split import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("DRY RUN: at most six frames, read-only adapter inference, no update/rollout")
        return
    output = ROOT / "reproduction/results/clean_task_reload_diagnosis_v1.json"
    if output.exists():
        raise FileExistsError("Do not repeat completed inference evidence")
    occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                                       "--format=csv,noheader"], text=True).strip()
    if occupied:
        raise RuntimeError(f"GPU busy; no preemption: {occupied}")
    audit, final = json.loads(AUDIT.read_text()), json.loads(RESULT.read_text())
    spec = specification(audit)
    checkpoint = RUN / "step_050"
    snapshot = verify_snapshot(checkpoint, spec)
    previous = final["evaluations"][-1]
    recorded_production = final["production_evaluation"]
    ids = []
    reference_train, reference_production = [], []
    for split in ("train", "validation"):
        rows = previous["splits"][split]["autoregressive_rows"]
        prod_rows = recorded_production["splits"][split]["autoregressive_rows"]
        for a, b in zip(rows, prod_rows):
            assert (a["episode_index"], a["timestep"]) == (b["episode_index"], b["timestep"])
            if a["action"] != b["action"] or (split == "validation" and len(ids) < 6):
                ids.append((split, a["episode_index"], a["timestep"]))
                reference_train.append(a["action"])
                reference_production.append(b["action"])
    assert len(ids) == 6 and sum(a != b for a, b in zip(reference_train, reference_production)) == 3
    frames = load_frames(audit)
    chosen = [next(r for r in frames[s] if (r["episode_index"], r["timestep"]) == (e, t)) for s, e, t in ids]
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training
    from experiments.robot.openvla_utils import get_vla, get_processor, get_vla_action
    from types import SimpleNamespace
    base_path = spec["base_path"]
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
    base = AutoModelForVision2Seq.from_pretrained(base_path, torch_dtype=torch.bfloat16,
        quantization_config=quant, attn_implementation="sdpa", low_cpu_mem_usage=True,
        trust_remote_code=True, device_map={"": 0})
    base = prepare_model_for_kbit_training(base)
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
    stats = json.loads((checkpoint / "dataset_statistics.json").read_text())
    model.norm_stats = stats
    model.base_model.model.norm_stats = stats
    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)

    def prediction(autocast):
        model.eval()
        with torch.inference_mode(), (torch.autocast("cuda", dtype=torch.bfloat16) if autocast else nullcontext()):
            return [get_vla_action(model, processor, base_path, {"full_image": r["image"]},
                                  audit["task_instruction"], KEY, center_crop=False).tolist() for r in chosen]

    def signature():
        inner = model.base_model.model
        dtype_parameters = {}
        for name, p in model.named_parameters():
            category = "vision" if "vision_backbone" in name else "other"
            key = category + ":" + str(p.dtype) + ":" + type(p).__name__
            dtype_parameters[key] = dtype_parameters.get(key, 0) + p.numel()
        return {"model_class": str(type(inner)), "config_class": str(type(inner.config)),
                "parameter_dtype_counts": dtype_parameters,
                "gradient_checkpointing": bool(inner.is_gradient_checkpointing),
                "config_use_cache": getattr(inner.config, "use_cache", None)}

    paths = {"prepared_reload_BF16_autocast": {"signature": signature(), "actions": prediction(True)}}
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    cfg = SimpleNamespace(pretrained_checkpoint=str(checkpoint), base_model_path=base_path,
                          load_in_4bit=True, load_in_8bit=False, bnb_double_quant=False)
    model, processor = get_vla(cfg), get_processor(cfg)
    paths["production_reload_no_autocast"] = {"signature": signature(), "actions": prediction(False)}
    paths["same_production_instance_BF16_autocast"] = {"signature": signature(), "actions": prediction(True)}
    # Apply the whole preparation procedure on the same in-memory instance.
    model = prepare_model_for_kbit_training(model)
    paths["same_production_instance_prepared_BF16_autocast"] = {"signature": signature(), "actions": prediction(True)}
    verify_snapshot(checkpoint, spec)
    evidence = {"status": "bounded_reload_diagnosis_complete", "spec": spec,
                "snapshot_sha256": snapshot["sha256"], "final_report_sha256": sha256(RESULT),
                "frames": [{"split": s, "episode_index": e, "timestep": t,
                            "image_sha256": row["image_sha256"]} for (s, e, t), row in zip(ids, chosen)],
                "recorded_training_actions": reference_train, "recorded_production_actions": reference_production,
                "paths": paths,
                "prepared_reload_matches_recorded_training": paths["prepared_reload_BF16_autocast"]["actions"] == reference_train,
                "production_reload_matches_recorded_production": paths["production_reload_no_autocast"]["actions"] == reference_production,
                "production_autocast_matches_training": paths["same_production_instance_BF16_autocast"]["actions"] == reference_train,
                "production_prepared_autocast_matches_training": paths["same_production_instance_prepared_BF16_autocast"]["actions"] == reference_train,
                "optimizer_updates": 0, "checkpoint_modified": False,
                "limits": ["Six selected frames only; no claim about all inputs or closed-loop behavior.",
                           "Preparation changes nonquantized dtypes and checkpointing together, not a dtype-only causal ablation.",
                           "No production implementation changed; original mismatch evidence retained."]}
    with output.open("x") as handle:
        json.dump(evidence, handle, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in evidence.items() if k.endswith("training") or k.endswith("production") or k == "status"}), flush=True)


if __name__ == "__main__":
    main()
